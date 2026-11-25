"""Empirical A/B experiment: KV-cache-friendly prefix-first prompts vs. Leximeter's
current item-first randomized prompts on ``deepseek-chat``.

Usage (from worktree root ``D:\\worktrees\\leximeter-kv-cache-exp``):
    uv run python scripts/kv_cache_experiment.py --phase sanity
    uv run python scripts/kv_cache_experiment.py --phase full
    uv run python scripts/kv_cache_experiment.py --phase analyze
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from jinja2 import Environment, PackageLoader
from openai import AsyncOpenAI
from scipy import stats

from leximeter.client.base import LLMClient
from leximeter.tasks.classify import Classify, ClassifyConfig
from leximeter.tasks.codify import ChunkResult, Codify, CodifyConfig
from leximeter.tasks.rate import Rate, RateConfig
from leximeter.templates import load_prompt

# ---------------------------------------------------------------------------
# Strict budget guard & directories
# ---------------------------------------------------------------------------
MAX_CALLS = 220

WORKTREE_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = WORKTREE_ROOT / "runs" / "kv_cache_exp"
LEDGER_PATH = RUNS_DIR / "call_ledger.jsonl"

# DeepSeek official API pricing for deepseek-flash / deepseek-v4-flash (DeepSeek-V4.1-Flash)
# Verified against DeepSeek official pricing docs (USD per 1M tokens):
#   Peak hours: 01:00–04:00 & 06:00–10:00 UTC Mon–Fri (excl. Chinese public holidays)
#   Off-Peak:   All other times (50% discount)
#     Off-Peak: Input Cache Hit $0.003 | Input Cache Miss $0.150 | Output $0.600
#     Peak:     Input Cache Hit $0.006 | Input Cache Miss $0.300 | Output $1.200
PRICING_OFF_PEAK_USD_PER_1M = {
    "cache_hit": 0.003,
    "cache_miss": 0.150,
    "output": 0.600,
}
PRICING_PEAK_USD_PER_1M = {
    "cache_hit": 0.006,
    "cache_miss": 0.300,
    "output": 1.200,
}
PRICING_USD_PER_1M = (
    PRICING_OFF_PEAK_USD_PER_1M  # Active window at ~13:30 UTC (Off-Peak)
)

# ---------------------------------------------------------------------------
# Faithful prefix-first Jinja2 templates
#
# Identical wording, scale, rules, and JSON format to Leximeter's shipped
# prompts (`ratings_prompt.jinja2`, `classification_prompt.jinja2`,
# `codify_prompt.jinja2`). ONLY two changes:
#   1. Static instructions + rubric + JSON schema + rules moved to TOP;
#      dynamic `snip.single_entry(modality, text)` moved to BOTTOM.
#   2. Deterministic attribute/label/code order (`ordered_dict_json` / `list`)
#      instead of per-item `shuffled_dict` / `shuffled`.
# ---------------------------------------------------------------------------


def ordered_dict_json(d: dict[str, Any]) -> str:
    """JSON-formatted dict string preserving insertion order (same formatting as shuffled_dict)."""
    return json.dumps(OrderedDict(d.items()), ensure_ascii=False, indent=2)


_PREFIX_ENV = Environment(loader=PackageLoader("leximeter", "prompts"))
_PREFIX_ENV.filters["ordered_dict_json"] = ordered_dict_json

RATE_PREFIX_TEMPLATE = _PREFIX_ENV.from_string(
    """{%- import "snippets.jinja2" as snip -%}
Your task: for each attribute below, rate how strongly the provided content manifests it.

BEGIN ATTRIBUTES
{{ attributes | ordered_dict_json }}
END ATTRIBUTES
Each dictionary key is an attribute. If a definition is provided, use it to anchor judgment; otherwise use your best consistent definition.

BEGIN RATING SCALE
{% if scale %}
{{ scale }}
{% else %}
Use integers 0-100 (inclusive). low = absent; high = extreme; mid = moderate.
Use the full range and every increment; do not round to 5s/10s.
Extremes are rare: use near 0 only if truly absent and near 100 only if overwhelming.
Use moderate intermediates (e.g. 19, 67, 32) to account for nuance where applicable.
{% endif %}
END RATING SCALE

Method (per attribute): pick one exact integer. Stick to provided scale. Double-check before choosing extremes.
Interpret gradations as: absent→faint→moderate→abundant→extreme.
Don't overlook subtlety; don't default to extremes. Consider full spectrum, including intermediate gradations.
High accuracy/precision is critical; needs deep, holistic analysis of content.

Rules:
- Judge each attribute independently and separately from each other
- Absolutely no indirect inference from other attributes or cross-attribute leakage (example of bad: inferring environmentalism from general political lean; good: measuring direct evidence of environmental opinions, no inference/biasing)
- Only measure the direct signal of each attribute alone in the content, NOT what is implied by other attributes; CRUCIAL each attribute measured independently on its own direct, specifically relevant signal

Output JSON only, in following format:
{
\t"<insert attribute name here>": <insert corresponding rating here>,
\t"<attribute name>": <corresponding rating>,
\t...
}

Attributes you are measuring in the content are: {{ attributes.keys() | list }}
Assess EVERY attribute; no drops. Use these attribute names verbatim, with absolutely no modification.
Same case, same spelling, same punctuation, same formatting.

{% if additional_instructions %}
The user has provided the following additional instructions, clarifications, or labelled examples (if these provided, rely heavily on them to calibrate your judgment).
If they conflict with the other instructions, these user instructions take precedence.

BEGIN ADDITIONAL INSTRUCTIONS/CLARIFICATIONS
{{ additional_instructions }}
END ADDITIONAL INSTRUCTIONS/CLARIFICATIONS
{% endif %}
{{ snip.single_entry(modality, text) }}"""
)

CLASSIFY_PREFIX_TEMPLATE = _PREFIX_ENV.from_string(
    """{%- import "snippets.jinja2" as snip -%}
Task: For each label below, decide whether it applies to the provided content.

BEGIN LABELS
{{ attributes | ordered_dict_json }}
END LABELS
Each key is a label. If a definition is provided, use it to anchor judgment; otherwise use your best consistent definition.

Rules (per label):
- Output True if label applies to the content; False if not
- Zero, one, or many labels may apply; mark True for all that apply
- Independently judge each label; no cross-label inference

Output JSON only, in following format:
{
    "<insert label name here>": <insert True if this label applies, False if not>,
    "<label name>": <insert False|True>,
    ...
}

Labels you are evaluating are: {{ attributes.keys() | list }}
Assess EVERY label (False or True); no drops. Use these label names verbatim, with absolutely no modification.
Same case, same spelling, same punctuation, same formatting.

{% if additional_instructions %}
The user has provided the following additional instructions, clarifications, or labelled examples (if these provided, rely heavily on them to calibrate your judgment).
If they conflict with the other instructions, these user instructions take precedence.
Important: if user has explicitly asked for only the best label, output True for ONLY the one label that best applies, and False for all others.
BEGIN ADDITIONAL INSTRUCTIONS/CLARIFICATIONS
{{ additional_instructions }}
END ADDITIONAL INSTRUCTIONS/CLARIFICATIONS
{% endif %}
{{ snip.single_entry(modality, text) }}"""
)

CODIFY_PREFIX_TEMPLATE = _PREFIX_ENV.from_string(
    """{%- import "snippets.jinja2" as snip -%}
{% if modality != "text" %}
Since these inputs are not text, you should not output verbatim excerpts.
Instead, describe the relevant parts of the entity being coded, in your own words.
{% endif %}

Your task is to act as a passage coder, as is done in the social sciences.

{% if categories %}
You are to carefully read and dissect the text, marking the boundaries of snippets/subsections of the text that belong to the following categories/attributes:

BEGIN CATEGORIES/ATTRIBUTES
{{ categories }}
END CATEGORIES/ATTRIBUTES
Follow the specific definitions of the categories in assigning them to snippets/subsections of the text.
{% else %}
Consider the broad topic alluded to in these user instructions:
BEGIN USER INSTRUCTIONS
{{ additional_instructions }}
END USER INSTRUCTIONS
Consider the specific category labels that fall under the broad topic alluded to in the user instructions.
For example, if the broad topic is Bible verses, the specific category labels would be specific Bible verses like "John 3:16" or "Psalm 23:1".
Or if the topic is Supreme Court decisions, the specific category labels would be specific Supreme Court decisions like "Marbury v. Madison" or "Brown v. Board of Education".
In your output, you would only output as category labels the specific category labels (under the broad topic) that are present in the text and should be marked.
So, in the example above, if the text alludes to the Bible verse "John 3:16", you would output "John 3:16" as the category label and mark the snippets/subsections that allude to the John 3:16.
You would not output every single Bible verse; just every Bible verse that is present in the text.
Be thorough and exhaustive in your output, denoting every specific category label that is applicable in the text and marking every snippet/subsection that is applicable to each category label.
{% endif %}

You mark the boundaries of the snippets/subsections by verbatim outputting the first 8 words and the last 8 words of the snippet/subsection (more if necessary to ensure uniqueness in programmatically identifying the snippets/subsections).
If the snippet/subsection is less than 20 words, you should write out the entire snippet/subsection as the beginning excerpt and leave the ending excerpt blank.
It is crucial that these beginning and ending excerpts are completely verbatim, same exact spelling/mistakes/punctuation/capitalization/line breaks/etc.
We will programmatically use these excerpts to identify and pull out the whole snippets/subsections that you are marking.
Be sure that the beginning and ending excerpts are completely verbatim, same exact everything, same exact characters.

Output a JSON in the following format:

{
    "<insert category name here>":
        [
            {
                "beginning excerpt": "<insert the first 8 words of the snippet/subsection here verbatim (or the entire snippet/subsection if it is less than 20 words)>",
                "ending excerpt": "<insert the last 8 words of the snippet/subsection here verbatim (or leave blank if the snippet/subsection is less than 20 words)>",
            },
            {
                "beginning excerpt": "<insert the first 8 words of a different snippet/subsection here verbatim (or the entire snippet/subsection if it is less than 20 words)>",
                "ending excerpt": "<insert the last 8 words of that snippet/subsection here verbatim (or leave blank if the snippet/subsection is less than 20 words)>",
            },
            ...
        ]
    "<category name>",
        [
            {
                "beginning excerpt": "...",
                "ending excerpt": "...",
            },
            ...
        ],
    ...
}

{% if categories %}
Again: the categories you are marking are: {{ categories.keys() | list }}
Ensure you input these category names verbatim into the JSON.
Exact spelling/capitalization/underscoring/any other formatting of the category names must be preserved.
Carefully consider the entire text for each and every category; do not skip any (though it is possible that some categories are not present in the text, in which case that category name must still be in the JSON but with an empty list of snippets/subsections).
{% endif %}
It is possible and often the case that text will overlap between categories, in which case you should mark the text for each applicable category separately. The same area of text can be marked for multiple categories.
Often in this case, it is not the exact same snippet/subsection that is being marked for each category, but rather a slightly different snippet/subsection that perhaps starts or ends at a different word, or is a different length.
You must carefully consider extremely precise boundaries for the snippets/subsections distinctly for each category.
Be precise with your identified snippets/subsections; don't include any words/chunks of text that are outside the essential boundaries of the snippet/subsection where the category applies.
Be very thorough. Every single applicable snippet/subsection (include small ones) throughout the text, beginning to end, must be marked.
This requires meticulosity and attention to detail.
Patiently find every single snippet/subsection that fits each category, and be willing to mark a large number of snippets/subsections if necessary. 
Also be willing to mark the same or similar snippets/subsections under multiple categories separately if applicable.
Don't be lazy and mark giant chunks of text where most of it doesn't fit the category. In this case, only mark narrowly scoped excerpts of the text that fit the category.
Rigorously consider every single category. Remember that the same or similar snippets/subsections can be marked separately under multiple categories.
It is absolutely crucial that you look through the entire text. Do not skim through the middle and end; don't only mark snippets/subsections at the beginning. Full, thorough coverage is key.
Scour every single sentence, phrase, and paragraph of the text. Double check that you have not missed any snippets/subsections and that you have not neglected any portions of the text in your thinking and marking.
Make extra effort to check each of the beginning, middle, and end of the text to ensure you have not missed any snippets/subsections there. Cover everything, beginning to end.
It is crucial that you think through each portion of the text carefully and thoroughly, otherwise you will miss snippets/subsections.

{% if additional_instructions %}
These are additional instructions from the user, follow them carefully. They take precedence over the general instructions above, if there is a conflict.
In particular, consider any instructions, if provided, on the scope of the text to be marked (should you be marking very specific phrases or full paragraphs/sections?).
If the user indicates a different scope of how large the snippets/subsections should be/how much context to include, follow their instructions over the general instructions above, if there is a conflict.
BEGIN ADDITIONAL INSTRUCTIONS
{{ additional_instructions }}
END ADDITIONAL INSTRUCTIONS
{% endif %}
{{ snip.single_entry(modality, text) }}"""
)

# ---------------------------------------------------------------------------
# Datasets (20 paragraphs for rate/classify; 8 multi-paragraph passages for codify)
# ---------------------------------------------------------------------------

PARAGRAPHS_20: list[dict[str, str]] = [
    {
        "id": "p01_monetary_policy",
        "topic": "Central bank briefing",
        "text": (
            "Minutes released Wednesday by the Federal Reserve's monetary policy committee "
            "indicated that most voting members favor holding the benchmark overnight rate "
            "steady between 4.25 and 4.50 percent through the third quarter. Staff econometric "
            "models projected core personal consumption inflation easing gradually toward 2.3 "
            "percent over the next six quarters, provided shelter costs continue moderating in "
            "line with new-lease indices. Committee participants noted that three-month moving "
            "averages of payroll gains remain consistent with stable labor-force absorption, "
            "reducing the probability of an abrupt contraction. Several regional presidents "
            "emphasized that transparent forward guidance and disciplined adherence to the "
            "dual-mandate framework remain essential for anchoring long-term bond expectations "
            "without unsettling credit markets."
        ),
    },
    {
        "id": "p02_populist_lobbyists",
        "topic": "Populist op-ed",
        "text": (
            "Every working family paying forty dollars more a week at the grocery checkout "
            "knows the truth that Washington consultants pretend not to see: this economy is "
            "rigged from top to bottom by corporate monopolies and the captured regulators who "
            "dine with them. While three meatpacking conglomerates posted record operating "
            "margins last year, the federal agencies tasked with enforcing antitrust law sat "
            "on their hands and issued toothless memos. Enough is enough! Ordinary citizens "
            "are being bled dry to finance billionaire stock buybacks while career bureaucrats "
            "protect their own revolving-door futures. We must break up these predatory "
            "cartels immediately and strip agency insiders of the power to rubber-stamp "
            "mergers behind closed doors."
        ),
    },
    {
        "id": "p03_permafrost_grid",
        "topic": "Climate science commentary",
        "text": (
            "Satellite interferometry across eastern Siberia and northern Alaska shows "
            "permafrost active-layer depths expanding 18 percent faster between 2019 and 2025 "
            "than mid-range IPCC scenarios anticipated. Because thawing organic soils release "
            "both methane and carbon dioxide in a self-reinforcing feedback loop, delaying "
            "power-sector decarbonization by even a single decade substantially narrows the "
            "remaining carbon budget for limiting warming below two degrees Celsius. Recent "
            "modeling by national laboratory grid engineers demonstrates that accelerating "
            "high-voltage interstate transmission permitting and utility-scale storage "
            "interconnection could retire 65 percent of remaining coal capacity by 2033 at "
            "manageable ratepayer cost. Lawmakers face a closing window to pass binding "
            "clean-electricity standards before irreversible ecosystem tipping points lock in."
        ),
    },
    {
        "id": "p04_zoning_hearing",
        "topic": "Municipal zoning debate",
        "text": (
            "During Tuesday's packed city council hearing, longtime Oakridge residents "
            "challenged the planning department's proposal to allow four-story apartment "
            "buildings within half a mile of commuter rail stations. Homeowners argued that "
            "the city's traffic study severely underestimated peak-hour congestion on two-lane "
            "feeder streets and ignored strained stormwater mains already backing up after "
            "heavy rain. 'Nobody from City Hall ever asked the families who built this "
            "neighborhood whether our schools can absorb four hundred extra students,' said "
            "homeowner association president Denise Alvarez. Housing advocates countered that "
            "restrictive single-family zoning has driven median rents up 34 percent in five "
            "years, pricing out teachers and nurses who serve the very same district."
        ),
    },
    {
        "id": "p05_ai_governance",
        "topic": "Technology policy essay",
        "text": (
            "Evaluating frontier artificial intelligence models requires standardized, "
            "reproducible testing protocols rather than ad hoc vendor demonstrations. "
            "Over the past eighteen months, the National Institute of Standards and "
            "Technology has collaborated with university computer-science labs and private "
            "developers to build open benchmark suites measuring cybersecurity exploitation "
            "capabilities, hallucination rates, and demographic calibration across clinical "
            "triage tasks. Empirical audits show that pre-deployment red-teaming catches "
            "roughly two-thirds of high-severity failure modes when evaluators receive "
            "structured weight or API access. Rather than imposing rigid licensing moratoria "
            "that could entrench incumbent firms, policymakers should fund independent "
            "evaluation consortia and mandate transparent incident reporting for systems "
            "deployed in critical infrastructure."
        ),
    },
    {
        "id": "p06_port_strike",
        "topic": "Labor union rally speech",
        "text": (
            "Look out at those cranes on the waterfront—every single container that kept "
            "supermarkets stocked during the pandemic moved because our dockworkers stood "
            "twelve-hour shifts in freezing rain! Now the shipping barons sitting in overseas "
            "boardrooms want to replace half our crew with automated gantries while pocketing "
            "four billion dollars in windfall freight profits. When federal mediators tell us "
            "to be patient and accept a two-percent raise below inflation, they show exactly "
            "whose side they are on. We refuse to watch our livelihoods sold off by "
            "executives who have never worn steel-toed boots. Stand on this picket line "
            "tonight until management guarantees enforceable job protections, full hazard "
            "pay, and a binding moratorium on unmanned terminal equipment!"
        ),
    },
    {
        "id": "p07_ag_trade_bulletin",
        "topic": "Agricultural trade bulletin",
        "text": (
            "November soybean futures settled 14 cents higher at $11.42 per bushel on the "
            "Chicago Board of Trade after Panama Canal authorities extended draft "
            "restrictions through mid-autumn due to persistently low water levels in Gatun "
            "Lake. Export elevators along the lower Mississippi reported vessel rerouting "
            "surcharges averaging $8.60 per metric ton as grain shippers shifted cargoes "
            "around the Cape of Good Hope toward East Asian crushers. Meanwhile, USDA weekly "
            "crop progress data indicated that 68 percent of Midwestern acreage remains in "
            "good-to-excellent condition following timely July rainfall. Analysts expect "
            "Brazilian harvest competition to cap upside price rallies unless El Nino "
            "anomalies persist into southern hemisphere planting."
        ),
    },
    {
        "id": "p08_rural_hospitals",
        "topic": "Healthcare policy editorial",
        "text": (
            "Since 2018, forty-seven rural community hospitals have closed their inpatient "
            "wards, forcing expectant mothers and cardiac patients in agricultural counties "
            "to drive more than an hour for emergency stabilization. Operating margins at "
            "remaining critical-access facilities fell to negative 3.1 percent last fiscal "
            "year as traveling-nurse staffing expenses outpaced flat Medicaid and private "
            "insurer reimbursements. State hospital associations warn that another two dozen "
            "facilities face imminent insolvency before next winter unless legislatures act "
            "now. Expanding targeted rural stabilization grants, modernizing telehealth "
            "reimbursement parity, and streamlining residency pipelines for family physicians "
            "would cost a fraction of the uncompensated emergency transfers that counties "
            "currently absorb."
        ),
    },
    {
        "id": "p09_industrial_subsidies",
        "topic": "Free-market policy critique",
        "text": (
            "History offers little encouragement for governments attempting to pick winners "
            "in capital-intensive manufacturing through multi-billion-dollar subsidy "
            "packages and protective tariffs. Recent evaluations of domestic semiconductor "
            "and battery tax credits reveal unit job-creation costs exceeding $420,000 per "
            "permanent position, while import duties on upstream equipment raise input "
            "prices for downstream American fabricators and consumers alike. Central "
            "planners inevitably lack the dispersed price signals that competitive capital "
            "markets process daily, causing subsidized plants to lag behind shifting "
            "process technologies. Congress would achieve faster productivity growth by "
            "repealing protectionist trade barriers, simplifying corporate expensing rules, "
            "and allowing private investors—not commerce department committees—to allocate "
            "industrial capital."
        ),
    },
    {
        "id": "p10_env_justice",
        "topic": "Environmental justice essay",
        "text": (
            "Walk three blocks from the petrochemical fence line in St. Bernard Parish at "
            "dawn and you can taste the sulfur stinging the back of your throat before the "
            "school buses even roll. Pediatric hospital records document childhood asthma "
            "hospitalization rates nearly triple the state average, yet state environmental "
            "regulators renewed three flaring permits last month without holding a single "
            "evening hearing for working parents. For decades, multibillion-dollar refinery "
            "operators have treated low-income Gulf Coast communities as disposable "
            "sacrifice zones while federal inspectors look the other way. We cannot wait "
            "another five years for voluntary industry pledges: regulators must enforce "
            "continuous fence-line benzene monitoring and shut down chronic violators today."
        ),
    },
    {
        "id": "p11_phonics_rct",
        "topic": "Education research summary",
        "text": (
            "A four-year randomized controlled trial across 14 public school districts "
            "evaluated structured systematic phonics instruction against balanced-literacy "
            "curricula among 6,420 first- and second-grade students. Independent psychometric "
            "assessments administered by the state education research consortium found that "
            "students assigned to structured phonics scored 0.28 standard deviations higher "
            "on standardized decoding fluency and 0.19 standard deviations higher on reading "
            "comprehension by the end of third grade, with the largest gains concentrated "
            "among students entering below the twenty-fifth percentile. Attrition across "
            "treatment and control schools remained below 4 percent and showed no differential "
            "bias. District curriculum directors noted that sustained teacher coaching during "
            "the first implementation year accounted for most classroom fidelity differences."
        ),
    },
    {
        "id": "p12_lng_exports",
        "topic": "Energy industry commentary",
        "text": (
            "Expanding North American liquefied natural gas export capacity serves both "
            "allied energy security and domestic industrial competitiveness. Over the past "
            "three winters, flexible Gulf Coast shipments helped European utilities replace "
            "piped Russian gas without triggering rolling factory blackouts, while vibrant "
            "upstream production kept U.S. wholesale power prices among the lowest in the "
            "developed world. Pausing export terminal approvals in pursuit of abrupt fossil "
            "phase-outs risks ceding long-term Asian supply contracts to higher-emission "
            "coal producers and less reliable state-owned exporters. A pragmatic energy "
            "strategy should streamline pipeline and terminal permitting, harness market "
            "pricing to spur methane-leak reductions, and avoid regulatory mandates that "
            "penalize reliable baseload generation."
        ),
    },
    {
        "id": "p13_water_contamination",
        "topic": "Investigative journalism",
        "text": (
            "Internal laboratory logs obtained through public-records requests show that "
            "municipal water engineers detected perfluoroalkyl chemicals at 42 parts per "
            "trillion—more than ten times the federal drinking-water limit—in three wells "
            "serving Riverbend as early as October 2024. Despite statutory notice "
            "requirements, the state Department of Environmental Quality waited eleven "
            "months before issuing a public health advisory, citing ongoing 'data "
            "validation' while residents continued cooking and mixing infant formula with "
            "tap water. Independent toxicologists who reviewed the logs warned that "
            "prolonged exposure elevates thyroid and kidney disease risks and urged "
            "immediate distribution of point-of-use carbon filters alongside emergency "
            "interconnection to neighboring aquifers."
        ),
    },
    {
        "id": "p14_election_reforms",
        "topic": "Civic institutions commentary",
        "text": (
            "Quietly and with minimal partisan fanfare, twelve states have modernized their "
            "election administration over the past four years through bipartisan legislation "
            "combining risk-limiting post-election paper audits, upgraded voter-roll "
            "maintenance, and transparent chain-of-custody tracking for mail ballots. "
            "Surveys conducted by nonpartisan civic institutes show that county clerks "
            "across both rural and urban jurisdictions report smoother canvass timelines "
            "and higher volunteer poll-worker retention after adopting these standardized "
            "procedures. Independent redistricting commissions in Michigan and Colorado have "
            "similarly demonstrated that structured public hearings and open mapping "
            "software can earn broad cross-party legitimacy. Strengthening durable "
            "institutional guardrails builds public confidence far more reliably than "
            "performative rhetorical battles."
        ),
    },
    {
        "id": "p15_anti_establishment_podcast",
        "topic": "Anti-establishment transcript",
        "text": (
            "Wake up, folks, because the coastal elites and unelected central bankers are "
            "telling you right to your face what they have planned next! First they destroy "
            "your savings with reckless money printing, then the legacy media tells you "
            "everything is fine, and now they want programmable digital currency so some "
            "faceless bureaucrat in a glass tower can freeze your account if you step out "
            "of line. Do not trust a single word coming out of these corrupt institutions—"
            "the agencies, the big banks, and the corporate press are all playing for the "
            "same club, and everyday Americans aren't invited. It's time for regular people "
            "to tune out the propaganda and take our country back!"
        ),
    },
    {
        "id": "p16_congestion_pricing",
        "topic": "Urban transport brief",
        "text": (
            "Longitudinal traffic telemetry from Stockholm and London offers clear evidence "
            "that peak-hour congestion pricing simultaneously reduces inner-city vehicle "
            "volumes by 18 to 22 percent, cuts fine particulate emissions near schools, and "
            "generates predictable dedicated revenue for frequent bus rapid transit. "
            "Although commuter skepticism typically peaks in the six months preceding launch, "
            "municipal transit authorities observed approval ratings climbing above 60 "
            "percent once commuters experienced faster bus headways and shorter ambulance "
            "response times. Critics rightly highlight distributional impacts on outer-ring "
            "shift workers with limited rail access; however, pairing variable tolling with "
            "low-income commuter rebates and upfront bus-lane expansion resolves most equity "
            "concerns while preserving fiscal sustainability."
        ),
    },
    {
        "id": "p17_bookstores_essay",
        "topic": "Cultural reflection",
        "text": (
            "On rainy Sunday afternoons, the narrow wooden aisles of Maple Street Books "
            "still smell faintly of cedar shelving and ink from small-press poetry "
            "chapbooks. Regulars linger beside hand-lettered staff recommendation cards, "
            "trading quiet impressions of translated novels and out-of-print natural history "
            "essays that rarely surface in algorithmic storefronts. Over the past two "
            "decades, several beloved quarterly literary journals have folded quietly as "
            "reading habits drifted toward shorter digital formats, leaving a noticeable "
            "hush in local letters. Yet watching a teenager carry a dog-eared paperback "
            "toward the reading bench by the front window reminds you that unhurried "
            "literary conversation endures in small, patient rooms."
        ),
    },
    {
        "id": "p18_measles_bulletin",
        "topic": "Public health alert",
        "text": (
            "County public health officials issued an urgent immunization alert Friday "
            "following twenty-three confirmed measles cases across three elementary schools, "
            "including four unvaccinated children hospitalized for respiratory complications. "
            "Epidemiological contact tracing conducted with the Centers for Disease Control "
            "indicates that kindergarten MMR vaccination coverage in the affected zip codes "
            "dropped to 84 percent last fall, well below the 95 percent herd-immunity "
            "threshold needed to halt airborne transmission. Because measles remains "
            "infectious in shared airspaces for up to two hours, the health department is "
            "opening walk-in evening vaccination clinics immediately at six community "
            "centers. Parents of exposed or under-immunized children are strongly urged to "
            "contact their pediatricians or visit a county clinic within forty-eight hours."
        ),
    },
    {
        "id": "p19_eurozone_debt",
        "topic": "Sovereign debt analysis",
        "text": (
            "Updated fiscal sustainability projections for major Eurozone economies "
            "underscore the arithmetic challenge created by aging dependency ratios and "
            "higher structural refinancing costs. With ten-year sovereign yields averaging "
            "210 basis points above their 2015–2021 norm, primary budget balances in France, "
            "Italy, and Belgium must improve by 1.4 to 2.2 percent of GDP over the next "
            "five years merely to stabilize public debt-to-GDP trajectories. Under the "
            "European Commission's revised expenditure-rule framework, member states face "
            "enforceable multi-year net spending ceilings calibrated through stochastic debt "
            "sustainability analysis. Economists caution that front-loaded austerity risks "
            "dampening near-term investment, whereas credible pension indexation reforms "
            "and broadened tax bases better preserve growth while reassuring bond markets."
        ),
    },
    {
        "id": "p20_oyster_restoration",
        "topic": "Conservation feature",
        "text": (
            "Before sunrise on the Choptank River, commercial watermen and marine "
            "ecologists from the state natural resources agency hoist cages of spat-on-shell "
            "oysters together over thirty acres of restored sanctuary reef. A decade ago, "
            "disputes over harvest closures and sediment runoff divided bay towns, as "
            "declining catches squeezed family boatyards while nutrient pollution triggered "
            "summer dead zones. Today, joint monitoring transects show adult oyster biomass "
            "rebounding fourfold on protected bars, filtering millions of gallons of murky "
            "estuary water daily and seeding adjacent public harvest grounds with juvenile "
            "larvae. Watermen say predictable rotational closures combined with shell-recycling "
            "cost shares have turned former adversaries into practical partners invested in "
            "a healthier bay."
        ),
    },
]

RATE_ATTRIBUTES_FWD: dict[str, str] = OrderedDict(
    [
        (
            "emotional_intensity",
            "Degree to which the passage relies on charged, affective, outraged, or impassioned language rather than detached, matter-of-fact prose.",
        ),
        (
            "analytical_rigor",
            "Extent to which claims are grounded in empirical data, causal reasoning, concrete mechanisms, or systematic evidence rather than assertion or anecdote.",
        ),
        (
            "populist_rhetoric",
            "Degree to which the passage frames issues as a moral struggle between ordinary citizens or workers and corrupt, self-serving, or out-of-touch elites.",
        ),
        (
            "economic_focus",
            "Prominence of economic concepts, markets, prices, fiscal policy, labor conditions, trade, or financial incentives as central themes of the text.",
        ),
        (
            "environmental_concern",
            "Strength of concern expressed for ecological health, climate change, pollution, resource conservation, or environmental harm.",
        ),
        (
            "institutional_trust",
            "Degree of confidence or deference expressed toward established public institutions, regulators, courts, scientific bodies, or formal governance processes (0 = deep distrust/hostility, 50 = neutral/absent, 100 = strong confidence).",
        ),
        (
            "policy_advocacy",
            "Strength with which the text advocates for specific laws, regulations, public programs, or governmental action (or explicit repeal of them) rather than merely describing events.",
        ),
        (
            "urgency",
            "Degree to which the text portrays the issue as an immediate crisis requiring rapid, time-sensitive intervention rather than a routine or long-term matter.",
        ),
        (
            "conflict_framing",
            "Extent to which the passage frames its subject around adversarial clash, polarization, blame, or competing camps rather than consensus or neutral exposition.",
        ),
        (
            "pro_market_orientation",
            "Strength of endorsement for private enterprise, market pricing, deregulation, or commercial competitiveness over state intervention or regulatory constraint.",
        ),
    ]
)

RATE_ATTRIBUTES_REV: dict[str, str] = OrderedDict(
    reversed(list(RATE_ATTRIBUTES_FWD.items()))
)

CLASSIFY_LABELS_FWD: dict[str, str] = OrderedDict(
    [
        (
            "cites_quantitative_evidence",
            "True if the passage cites at least one specific number, statistic, percentage, monetary amount, or quantitative study finding as evidence.",
        ),
        (
            "advocates_policy_change",
            "True if the passage explicitly calls for enacting, reforming, funding, or repealing a specific public policy, law, or regulation.",
        ),
        (
            "criticizes_institutions",
            "True if the passage directly criticizes the competence, integrity, responsiveness, or legitimacy of government agencies, regulators, or formal public institutions.",
        ),
        (
            "mentions_economic_costs",
            "True if the passage discusses financial costs, prices, budgets, wages, subsidies, tax revenues, or economic trade-offs.",
        ),
        (
            "addresses_environment_or_climate",
            "True if the passage substantively addresses climate change, emissions, pollution, ecosystem conservation, or natural-resource degradation.",
        ),
        (
            "uses_populist_framing",
            "True if the passage contrasts ordinary citizens, workers, or local communities against powerful corporate, political, or bureaucratic elites.",
        ),
        (
            "expresses_time_urgency",
            "True if the passage explicitly frames the problem as time-critical, worsening rapidly, or requiring immediate action.",
        ),
        (
            "references_local_community_impact",
            "True if the passage highlights tangible consequences for specific towns, neighborhoods, local workers, schools, or municipal residents.",
        ),
        (
            "presents_competing_tradeoffs",
            "True if the passage explicitly acknowledges tension, downsides, or competing priorities between two or more legitimate goals or stakeholder groups.",
        ),
        (
            "adopts_persuasive_advocacy_voice",
            "True if the passage uses first-person advocacy ('we must', 'our communities') or overtly persuasive editorial rhetoric rather than third-person reportage.",
        ),
    ]
)

CLASSIFY_LABELS_REV: dict[str, str] = OrderedDict(
    reversed(list(CLASSIFY_LABELS_FWD.items()))
)

CODIFY_PASSAGES_8: list[dict[str, str]] = [
    {
        "id": "c01_housing_transit",
        "topic": "Transit corridor redevelopment",
        "text": (
            "Over the past six years, rising land values along the Eastline light-rail "
            "corridor have transformed once-affordable duplex blocks into speculative "
            "redevelopment targets. Small business owners operating auto-repair shops and "
            "family bakeries report commercial lease renewals jumping 45 to 60 percent, "
            "forcing several decades-old storefronts to relocate outside city limits. "
            "Meanwhile, working-class tenants face monthly rent increases that outpace "
            "regional wage growth by more than three to one, pushing municipal eviction "
            "filings to their highest level since 2011. Local pediatricians and elementary "
            "principals note that mid-year housing displacement has driven student mobility "
            "above 28 percent at two neighborhood schools, disrupting reading intervention "
            "schedules and draining per-pupil state attendance funding.\n\n"
            "City housing officials initially promoted the corridor upzoning as an "
            "engine for transit ridership and property-tax expansion, approving twelve "
            "market-rate apartment towers while deferring promised inclusionary-housing "
            "rules for three consecutive budget cycles. An audit by the city comptroller "
            "revealed that the housing trust fund left $19 million unspent for twenty "
            "months due to cumbersome application forms and chronic staffing vacancies "
            "in the permitting division. Neighborhood tenant unions organized monthly "
            "marches outside City Hall and formed a volunteer eviction-defense clinic "
            "that helped 140 households negotiate extended move-out agreements and "
            "relocation stipends. Longtime merchants also pooled funds to hire a bilingual "
            "legal aid attorney to challenge opaque triple-net lease pass-throughs.\n\n"
            "In response to sustained civic pressure, a bipartisan council working group "
            "has proposed pairing automatic density bonuses near stations with a mandatory "
            "twenty-percent affordable set-aside and a municipal commercial-rent "
            "stabilization pilot for legacy ground-floor tenants. Developers warn that "
            "overly stringent affordability mandates without low-interest public gap "
            "financing could stall new construction altogether, worsening regional "
            "housing scarcity. Urban planners argue that combining streamlined "
            "by-right approvals with dedicated land-trust acquisitions and faster trust-fund "
            "disbursement offers the most durable balance between expanding housing supply "
            "and preventing displacement."
        ),
    },
    {
        "id": "c02_coastal_insurance",
        "topic": "Coastal flood adaptation and insurance",
        "text": (
            "Homeowners along the Pamlico Sound have watched annual wind-and-flood "
            "insurance premiums triple since 2021 after three major private underwriters "
            "withdrew from coastal zip codes citing escalating storm-surge losses. "
            "Retirees living on fixed pensions say escrow payments now exceed their "
            "original mortgage principal, while small marina operators struggle to secure "
            "commercial hazard coverage required by marine lenders. County tax assessors "
            "warn that falling waterfront sales prices could erode 14 percent of local "
            "property-tax revenue needed to maintain rural roads and volunteer fire "
            "stations. Local realtors report that out-of-state cash investors are buying "
            "distressed cottages at steep discounts while longtime working families abandon "
            "homes passed down across three generations.\n\n"
            "State insurance regulators expanded the residual high-risk underwriting pool "
            "to keep policies available, but legislative delays stalled a proposed "
            "statewide resilience grant fund for two consecutive sessions. Furthermore, "
            "updated federal flood-plain maps lagged local tidal-gauge records by nearly "
            "a decade, allowing subdivisions to be permitted in low-lying marshes that "
            "now flood during routine king tides. Frustrated residents formed the Sound "
            "Shoreline Coalition, pooling neighborhood elevation surveys and hosting "
            "packed town halls where families shared photos of ruined drywall and "
            "mold-damaged crawl spaces to press county commissioners for transparent "
            "drainage engineering. Several churches converted fellowship halls into dry "
            "supply depots and organized volunteer muck-out crews after autumn nor'easters.\n\n"
            "Coastal engineers and county officials are now weighing a hybrid adaptation "
            "strategy that pairs fortified living shorelines and elevated primary roads "
            "with voluntary buyouts for the most repeatedly flooded parcels. Local "
            "mayors acknowledge a painful trade-off: preserving every historic waterfront "
            "street through seawalls would bankrupt municipal bond capacity, yet "
            "managed retreat fractures multi-generational neighborhoods and shrinks the "
            "tax base. Economists suggest offering transferable development credits to "
            "inland town centers alongside verified premium discounts for fortified-roof "
            "retrofits and elevated HVAC platforms so households can reduce risk without "
            "abrupt financial ruin."
        ),
    },
    {
        "id": "c03_steel_decarbonization",
        "topic": "Industrial decarbonization and steelworkers",
        "text": (
            "At the Monongahela Valley integrated steelworks, blast-furnace margins have "
            "tightened sharply over the past five years as volatile coking-coal prices, "
            "aging refractory linings, and low-cost imported slab squeezed annual plant "
            "earnings. Company executives told investors that maintaining century-old "
            "coal-fired furnaces requires $380 million in overdue capital repairs that "
            "no longer clear corporate hurdle rates, especially as automotive buyers "
            "demand lower-carbon steel to satisfy European and corporate procurement "
            "rules. Local diners, machine shops, and municipal school budgets—all "
            "anchored by 1,900 union steelworker paychecks—face acute anxiety over a "
            "potential mill shutdown. County treasurers estimate that losing the facility's "
            "industrial property assessments would immediately trigger a $6.8 million deficit "
            "across three river-valley boroughs.\n\n"
            "Federal energy officials announced a conditional $500 million industrial "
            "demonstration grant to help replace two coal blast furnaces with a "
            "direct-reduced iron unit and electric arc furnace capable of running on "
            "clean hydrogen. However, regional grid interconnection queues delayed the "
            "required high-voltage substation upgrade by thirty months, and state "
            "environmental permitting boards spent a year resolving conflicting air-quality "
            "modeling rules. Union shop stewards held emergency rank-and-file meetings, "
            "warning that electric arc furnaces typically employ 35 percent fewer "
            "operators per ton and insisting that workers must have a binding seat at "
            "the transition planning table rather than receiving sudden layoff notices. "
            "Retired millwrights joined younger crane operators in testifying before state "
            "utility regulators to demand accelerated grid upgrades tied to union jobs.\n\n"
            "Negotiators from the union, plant management, and the county economic "
            "development authority have drafted a transition compact combining funded "
            "electrical-maintenance apprenticeships, early-retirement bridge pensions "
            "for senior furnace crews, and preferential hiring at an adjacent "
            "transformer-manufacturing facility. Labor economists note the core tension: "
            "subsidizing legacy blast furnaces indefinitely risks losing both climate "
            "targets and export competitiveness, whereas rapid automation without "
            "retraining guarantees severe community scarring. Tying public grant "
            "disbursements directly to audited wage floors and local workforce retention "
            "benchmarks is emerging as a viable compromise."
        ),
    },
    {
        "id": "c04_rural_broadband",
        "topic": "Municipal broadband and digital divide",
        "text": (
            "Across four mountainous counties in southern Appalachia, sluggish DSL lines "
            "and spotty cellular reception have long imposed steep economic penalties on "
            "remote clinics, family farms, and small e-commerce workshops. Telehealth "
            "nurses report dropped video consultations during winter storms, while high "
            "school students routinely sit in fast-food parking lots after dusk to upload "
            "homework assignments. Private telecom carriers declined to extend fiber-optic "
            "trunk lines down low-density hollow roads, explaining that rocky trenching "
            "costs of $65,000 per mile cannot be recouped from twelve subscribers per "
            "square mile. Small timber processors and precision machine shops say unreliable "
            "cloud connectivity has cost them lucrative supply-chain contracts with regional "
            "manufacturers requiring real-time inventory tracking.\n\n"
            "When federal broadband infrastructure grants allocated $42 million to the "
            "region, state mapping agencies initially relied on carrier-submitted "
            "census-block reports that falsely marked entire valleys as 'fully served' "
            "whenever a single address near the highway had cable access. That flawed "
            "map disqualified six rural townships from the first funding round and "
            "sparked widespread anger across county courthouses. In response, retired "
            "lineworkers, librarians, and high-school robotics clubs launched a "
            "door-to-door speed-test campaign, logging 11,400 geolocated bandwidth "
            "measurements that formally overturned the state's eligibility map during "
            "the administrative challenge window. County clerks stayed open on Saturdays "
            "so older residents without smartphones could file paper affidavits documenting "
            "chronic service outages.\n\n"
            "Armed with corrected coverage data, an electric cooperative partnered with "
            "county officials to string aerial fiber along existing power poles, cutting "
            "deployment costs nearly in half while guaranteeing open-access wholesale "
            "rates. Fiscal watchdogs caution that publicly backed networks still carry "
            "long-term maintenance and pole-attachment liability risks if subscription "
            "take-rates fall below 55 percent. Yet local leaders contend that treating "
            "middle-mile fiber like rural electrification—pairing targeted capital "
            "subsidies with cooperative operational governance—resolves a market failure "
            "that private competition alone never bridged."
        ),
    },
    {
        "id": "c05_teacher_retention",
        "topic": "School district funding and teacher retention",
        "text": (
            "Mid-sized suburban and rural school districts across the Midwest are "
            "confronting an unprecedented staffing crunch as starting teacher salaries "
            "trail entry-level private-sector pay by nearly 24 percent after inflation. "
            "District finance officers explain that expiring federal pandemic relief "
            "grants, rising special-education transportation contracts, and steeper "
            "health-insurance premiums have left operating budgets with zero cushion for "
            "across-the-board raises. Last August, Riverdale Unified opened the academic "
            "year with thirty-eight unfilled math, science, and special-education posts, "
            "relying on long-term substitutes and overcrowded combined classrooms. "
            "Principals report that nearly a third of newly hired teachers now resign "
            "within three years, saddling the district with recurring recruitment, onboarding, "
            "and emergency substitute expenses exceeding $1.4 million annually.\n\n"
            "State legislators enacted a modest two-percent base-funding increase in "
            "April, but simultaneously imposed strict property-tax levy caps that "
            "prevented local school boards from raising supplemental revenue without "
            "supermajority referenda. Meanwhile, the state education department's new "
            "online licensure portal suffered months of processing backlogs, stranding "
            "out-of-state veteran teachers in bureaucratic limbo until November. Exhausted "
            "educators packed school-board meetings wearing red shirts, describing "
            "sixty-hour workweeks spent covering absent colleagues' prep periods, while "
            "parent-teacher councils circulated petitions warning that high turnover was "
            "eroding math proficiency and student mental-health support. Local coaches "
            "and paraprofessionals organized weekend canvasses to explain school levy "
            "finances to skeptical taxpayers.\n\n"
            "To break the impasse, a joint task force of superintendents, union "
            "representatives, and local business leaders recommended restructuring "
            "salary schedules to lift early-career pay, creating paid year-long teacher "
            "residencies for paraprofessionals already rooted in the community, and "
            "eliminating redundant paperwork mandates. Board members candidly debated "
            "the fiscal trade-off: consolidating two under-enrolled elementary buildings "
            "would free $4.2 million annually for classroom salaries, but closing "
            "neighborhood schools draws fierce community resistance and lengthens bus "
            "rides for young children. Pairing phased facility sharing with state-matched "
            "hard-to-staff stipends emerged as the most workable middle path."
        ),
    },
    {
        "id": "c06_groundwater_irrigation",
        "topic": "Aquifer depletion and agricultural irrigation",
        "text": (
            "Beneath the High Plains of western Kansas, saturated thicknesses in the "
            "Ogallala Aquifer have declined by more than thirty feet over the past "
            "twenty years, forcing irrigators to drill deeper wells and burn far more "
            "diesel and natural gas to lift each acre-foot of water. Family grain farms "
            "operating on razor-thin margins report irrigation energy bills doubling "
            "during consecutive drought summers, while regional feedlots and grain "
            "elevators worry that shrinking corn yields will trigger a wider rural "
            "economic contraction. Agricultural lenders have begun tightening operating "
            "credit lines for parcels where well-flow rates can no longer sustain peak "
            "July evapotranspiration. Municipal water managers in three county-seat towns "
            "similarly warn that falling water tables have forced emergency bond issues "
            "to redrill public drinking wells.\n\n"
            "For decades, state water-rights statutes grounded in prior appropriation "
            "rewarded senior pumpers with 'use it or lose it' incentives, while "
            "underfunded groundwater management boards hesitated to enforce mandatory "
            "metering or pumping caps. A legislative oversight committee found that "
            "nearly one-third of high-capacity wells lacked verified calibration records "
            "and that voluntary conservation easements reached less than 4 percent of "
            "critically depleted townships. Recognizing that inaction threatened their "
            "children's inheritance, a coalition of third-generation growers in Sheridan "
            "County petitioned to create a Local Enhanced Management Area, voting "
            "themselves a binding 20-percent reduction in five-year groundwater allocations "
            "and hosting peer-to-peer field days to share soil-moisture sensor data.\n\n"
            "Agronomists tracking the pilot report that shifting from thirsty continuous "
            "corn to drought-tolerant sorghum, subsurface drip lines, and deficit "
            "irrigation cut pumping by 23 percent while keeping net farm profitability "
            "nearly unchanged thanks to lower fuel and fertilizer outlays. Policymakers "
            "still wrestle with a difficult trade-off between near-term farm cash flow "
            "and multi-decade aquifer lifespan: aggressive immediate cuts could push "
            "highly leveraged young farmers into bankruptcy, whereas gradual voluntary "
            "targets fail to stabilize water tables. Expanding state cost-share grants "
            "for precision telemetry and flexible five-year water banking allows growers "
            "to adapt across wet and dry seasons."
        ),
    },
    {
        "id": "c07_hospital_merger",
        "topic": "Regional hospital consolidation",
        "text": (
            "When two neighboring nonprofit hospital systems announced plans to merge "
            "across the tri-county metro area, independent health economists warned that "
            "combining 72 percent of regional inpatient beds would give the merged "
            "entity substantial leverage in commercial insurer negotiations. Self-insured "
            "municipal employers and small manufacturers testified that hospital facility "
            "fees had already climbed 19 percent over three years, forcing firms to raise "
            "employee deductibles and freeze wage increases. At the same time, leaders of "
            "the smaller community hospital argued that without the merger's shared "
            "electronic-health-record infrastructure and capital borrowing power, their "
            "obstetrics and oncology wings faced $28 million in cumulative losses and "
            "probable closure. Local bond-rating analysts noted that two consecutive years "
            "of negative cash flow had already pushed the smaller system dangerously close "
            "to violating debt covenants.\n\n"
            "The state attorney general's antitrust division opened a nine-month review "
            "and subpoenaed internal pricing memos, while the state health planning "
            "commission conducted public hearings on certificate-of-need filings. Critics "
            "noted, however, that state regulators lacked dedicated actuarial staff to "
            "audit complex bundled-payment contracts and had approved four similar "
            "regional consolidations during the previous decade with minimal post-merger "
            "monitoring. Nurses' associations and community health clinics mobilized "
            "hundreds of patients to submit written comments demanding binding guarantees "
            "against maternity-ward closures, bilingual interpreter layoffs, and aggressive "
            "medical-debt collection lawsuits. Unionized technicians also staged an "
            "informational picket highlighting safe nurse-to-patient staffing ratios.\n\n"
            "Rather than either an unconditional approval or an outright court injunction "
            "that might precipitate insolvency, state negotiators drafted a ten-year "
            "consent decree capping annual commercial price growth at medical-CPI plus "
            "one percent, mandating minimum charity-care thresholds, and preserving the "
            "smaller hospital's labor-and-delivery unit. Antitrust scholars remain "
            "divided on this behavioral remedy: price caps are notoriously difficult to "
            "police once systems alter billing codes, yet blocking rescue mergers without "
            "public stabilization funding can leave underserved neighborhoods with no "
            "local emergency room at all. Creating an independent state health-cost "
            "commission with enforceable clawback penalties was proposed to verify "
            "compliance."
        ),
    },
    {
        "id": "c08_ai_hiring_audits",
        "topic": "Algorithmic hiring audits and labor standards",
        "text": (
            "Mid-sized logistics firms and retail chains increasingly rely on automated "
            "resume-screening software and video-interview scoring tools to sift through "
            "tens of thousands of hourly applications each quarter. Human-resources "
            "directors emphasize that rapid turnover and slim operating margins make "
            "manual review of every submission financially prohibitive, estimating that "
            "automated triage trims average time-to-hire from twenty-eight days to nine "
            "and saves roughly $340 per position filled. Yet job seekers with career "
            "gaps, non-standard regional accents, or disabilities report being silently "
            "rejected within minutes of applying, locked out of stable warehouse and "
            "customer-support jobs with no human explanation. Workforce economists warn "
            "that opaque automated blacklists can depress regional labor-force participation "
            "among older displaced workers trying to re-enter full-time employment.\n\n"
            "Following a high-profile municipal ordinance requiring annual third-party "
            "bias audits for automated employment decision tools, labor-department "
            "inspectors found inconsistent compliance across covered employers. Many "
            "vendors published vague summary tables aggregated across unrelated job "
            "categories, and municipal regulators admitted they had issued zero civil "
            "penalties in the first eighteen months due to ambiguous statutory definitions "
            "of what constitutes 'substantially assisting' a hiring decision. Civil-rights "
            "advocates and gig-worker collectives organized resume-testing clinics and "
            "compiled anonymized applicant logs showing stark disparities in callback "
            "rates for older workers and caregivers returning to the labor force. Several "
            "laid-off warehouse employees spoke at city council hearings urging clear "
            "appeals channels when software rejects qualified applicants.\n\n"
            "Employment attorneys, data scientists, and workforce boards are now pushing "
            "for clearer technical audit standards that require job-specific adverse-impact "
            "testing, plain-language candidate notice, and an accessible option to request "
            "human accommodation review without penalty. Small employers express concern "
            "that heavy compliance documentation could deter them from adopting legitimate "
            "skills-based matching tools that actually reduce subjective resume bias "
            "compared with rushed human screening. Standardizing accredited vendor-level "
            "audits—so small businesses do not pay repeatedly to test the same off-the-shelf "
            "software—offers a practical compromise between worker fairness and "
            "administrative feasibility."
        ),
    },
]

CODIFY_CATEGORIES_FWD: dict[str, str] = OrderedDict(
    [
        (
            "economic_pressure",
            "Specific text spans describing financial strain, rising costs, budget deficits, wage or margin pressures, insurance hikes, or livelihood risks.",
        ),
        (
            "institutional_action_or_failure",
            "Specific text spans describing decisions, programs, regulations, delays, oversight lapses, or formal interventions by governments, agencies, boards, or public institutions.",
        ),
        (
            "community_or_worker_response",
            "Specific text spans describing how residents, workers, families, unions, local associations, or civic groups react, organize, adapt, or share lived experiences.",
        ),
        (
            "proposed_remedy_or_tradeoff",
            "Specific text spans describing concrete policy solutions, compromises, pilot initiatives, or explicit trade-offs weighed to resolve the problem.",
        ),
    ]
)

CODIFY_CATEGORIES_REV: dict[str, str] = OrderedDict(
    reversed(list(CODIFY_CATEGORIES_FWD.items()))
)

# ---------------------------------------------------------------------------
# Instrumented LLMClient with strict call cap & cache telemetry
# ---------------------------------------------------------------------------


@dataclass
class CallRecord:
    call_index: int
    timestamp_utc: str
    task: str
    arm: str
    item_id: str
    is_warmup: bool
    model_requested: str
    model_returned: str
    prompt_chars: int
    prompt_tokens: int
    cached_tokens: int
    cache_miss_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    latency_s: float
    ok: bool
    error: str
    raw_usage: dict[str, Any]


def _extract_usage_fields(usage: Any) -> tuple[int, int, int, int, int, dict[str, Any]]:
    """Extract (prompt_tokens, cached_tokens, cache_miss_tokens, completion_tokens, reasoning_tokens, raw_dict)."""
    if usage is None:
        return 0, 0, 0, 0, 0, {}

    raw_dict: dict[str, Any] = {}
    if hasattr(usage, "model_dump"):
        try:
            raw_dict = usage.model_dump()
        except Exception:  # noqa: BLE001 - provider SDK usage shape varies
            raw_dict = {}

    prompt_tokens = int(
        getattr(usage, "prompt_tokens", 0) or raw_dict.get("prompt_tokens") or 0
    )
    completion_tokens = int(
        getattr(usage, "completion_tokens", 0) or raw_dict.get("completion_tokens") or 0
    )

    comp_details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = (
        getattr(comp_details, "reasoning_tokens", None)
        if comp_details is not None
        else None
    )
    if reasoning_tokens is None and isinstance(
        raw_dict.get("completion_tokens_details"), dict
    ):
        reasoning_tokens = raw_dict["completion_tokens_details"].get("reasoning_tokens")
    reasoning_tokens = int(reasoning_tokens or 0)

    extra = getattr(usage, "model_extra", None) or {}

    hit_direct = getattr(usage, "prompt_cache_hit_tokens", None)
    if hit_direct is None:
        hit_direct = extra.get(
            "prompt_cache_hit_tokens", raw_dict.get("prompt_cache_hit_tokens")
        )

    miss_direct = getattr(usage, "prompt_cache_miss_tokens", None)
    if miss_direct is None:
        miss_direct = extra.get(
            "prompt_cache_miss_tokens", raw_dict.get("prompt_cache_miss_tokens")
        )

    details = getattr(usage, "prompt_tokens_details", None)
    hit_details = (
        getattr(details, "cached_tokens", None) if details is not None else None
    )
    if hit_details is None and isinstance(raw_dict.get("prompt_tokens_details"), dict):
        hit_details = raw_dict["prompt_tokens_details"].get("cached_tokens")

    if hit_direct is not None:
        cached_tokens = int(hit_direct)
    elif hit_details is not None:
        cached_tokens = int(hit_details)
    else:
        cached_tokens = 0

    if miss_direct is not None:
        cache_miss_tokens = int(miss_direct)
    else:
        cache_miss_tokens = max(0, prompt_tokens - cached_tokens)

    return (
        prompt_tokens,
        cached_tokens,
        cache_miss_tokens,
        completion_tokens,
        reasoning_tokens,
        raw_dict,
    )


class BudgetExceededError(RuntimeError):
    """Raised when MAX_CALLS is reached."""


class InstrumentedChatClient(LLMClient):
    """OpenAI-compatible ChatCompletions client with hard call cap and cache telemetry."""

    def __init__(
        self,
        *,
        ledger_path: Path = LEDGER_PATH,
        max_calls: int = MAX_CALLS,
        disable_thinking: bool = True,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key
            or os.getenv("LEXIMETER_API_KEY")
            or os.getenv("OPENAI_API_KEY"),
            base_url=base_url
            or os.getenv("LEXIMETER_BASE_URL")
            or os.getenv("OPENAI_BASE_URL"),
        )
        self._ledger_path = ledger_path
        self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._max_calls = max_calls
        self._disable_thinking = disable_thinking
        self._lock = asyncio.Lock()
        self._total_calls = self._count_existing_calls()
        self.records: list[CallRecord] = []

    def _count_existing_calls(self) -> int:
        if not self._ledger_path.exists():
            return 0
        with self._ledger_path.open("r", encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())

    @property
    def total_calls(self) -> int:
        return self._total_calls

    async def complete(
        self,
        prompt: str,
        *,
        model: str,
        json_mode: bool = False,
        **generation_kwargs: Any,
    ) -> str:
        # Extract experiment metadata kwargs without forwarding them to OpenAI API.
        task_name = str(generation_kwargs.pop("_exp_task", "unknown"))
        arm_name = str(generation_kwargs.pop("_exp_arm", "unknown"))
        item_id = str(generation_kwargs.pop("_exp_item_id", ""))
        is_warmup = bool(generation_kwargs.pop("_exp_is_warmup", False))

        async with self._lock:
            if self._total_calls >= self._max_calls:
                raise BudgetExceededError(
                    f"Hard API call cap reached: {self._total_calls}/{self._max_calls}"
                )
            self._total_calls += 1
            call_idx = self._total_calls

        params: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        if self._disable_thinking:
            extra_body = dict(generation_kwargs.pop("extra_body", None) or {})
            extra_body.setdefault("thinking", {"type": "disabled"})
            params["extra_body"] = extra_body
        params.update(generation_kwargs)

        t0 = time.monotonic()
        ok = False
        err_msg = ""
        content = ""
        model_returned = model
        p_tok = c_hit = c_miss = comp_tok = reas_tok = 0
        raw_usage: dict[str, Any] = {}

        try:
            response = await self._client.chat.completions.create(**params)
            content = response.choices[0].message.content or ""
            model_returned = getattr(response, "model", None) or model
            p_tok, c_hit, c_miss, comp_tok, reas_tok, raw_usage = _extract_usage_fields(
                response.usage
            )
            ok = True
        except Exception as exc:
            err_msg = repr(exc)
            raise
        finally:
            latency_s = round(time.monotonic() - t0, 4)
            rec = CallRecord(
                call_index=call_idx,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                task=task_name,
                arm=arm_name,
                item_id=item_id,
                is_warmup=is_warmup,
                model_requested=model,
                model_returned=model_returned,
                prompt_chars=len(prompt),
                prompt_tokens=p_tok,
                cached_tokens=c_hit,
                cache_miss_tokens=c_miss,
                completion_tokens=comp_tok,
                reasoning_tokens=reas_tok,
                latency_s=latency_s,
                ok=ok,
                error=err_msg,
                raw_usage=raw_usage,
            )
            async with self._lock:
                self.records.append(rec)
                with self._ledger_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

        return content

    def estimate_tokens(self, prompt: str, *, model: str) -> int:
        raise NotImplementedError

    def estimate_cost(self, prompt: str, *, model: str) -> float:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Arm execution helpers (concurrent baseline vs 1-call-warmup prefix arms)
# ---------------------------------------------------------------------------


async def _call_once_with_1_retry(
    client: InstrumentedChatClient,
    prompt: str,
    *,
    model: str,
    task: str,
    arm: str,
    item_id: str,
    is_warmup: bool,
) -> str:
    last_exc: Exception | None = None
    for attempt in range(2):  # at most 1 retry so budget stays tight
        try:
            return await client.complete(
                prompt,
                model=model,
                json_mode=True,
                _exp_task=task,
                _exp_arm=arm,
                _exp_item_id=item_id,
                _exp_is_warmup=is_warmup,
            )
        except BudgetExceededError:
            raise
        except Exception as exc:  # noqa: BLE001 - transient provider errors
            last_exc = exc
            if attempt == 0:
                await asyncio.sleep(2.0)
    assert last_exc is not None
    raise last_exc


async def execute_arm_prompts(
    client: InstrumentedChatClient,
    *,
    task: str,
    arm: str,
    items: list[tuple[str, str]],  # [(item_id, prompt), ...]
    model: str,
    n_parallels: int,
    warmup_first: bool,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Run all prompts for one arm.

    If ``warmup_first`` is True, sends ``items[0]`` sequentially to populate
    DeepSeek's disk prefix cache, then executes ``items[1:]`` concurrently with
    ``n_parallels``.
    """
    arm_dir = RUNS_DIR / task / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    responses_path = arm_dir / "responses.json"
    meta_path = arm_dir / "arm_meta.json"

    if responses_path.exists() and meta_path.exists():
        print(f"[SKIP] {task}/{arm} already completed — loading saved responses.")
        return (
            json.loads(responses_path.read_text(encoding="utf-8")),
            json.loads(meta_path.read_text(encoding="utf-8")),
        )

    start_idx = len(client.records)
    t_arm_start = time.monotonic()
    responses: dict[str, str] = {}
    warmup_wall_s = 0.0
    batch_wall_s = 0.0

    if warmup_first and items:
        first_id, first_prompt = items[0]
        t_w0 = time.monotonic()
        responses[first_id] = await _call_once_with_1_retry(
            client,
            first_prompt,
            model=model,
            task=task,
            arm=arm,
            item_id=first_id,
            is_warmup=True,
        )
        warmup_wall_s = round(time.monotonic() - t_w0, 4)
        remaining = items[1:]
    else:
        remaining = items

    sem = asyncio.Semaphore(max(1, n_parallels))

    async def worker(iid: str, pr: str) -> None:
        async with sem:
            responses[iid] = await _call_once_with_1_retry(
                client,
                pr,
                model=model,
                task=task,
                arm=arm,
                item_id=iid,
                is_warmup=False,
            )

    t_b0 = time.monotonic()
    if remaining:
        await asyncio.gather(*(worker(iid, pr) for iid, pr in remaining))
    batch_wall_s = round(time.monotonic() - t_b0, 4)
    total_wall_s = round(time.monotonic() - t_arm_start, 4)

    arm_records = [asdict(r) for r in client.records[start_idx:]]
    meta = {
        "task": task,
        "arm": arm,
        "model": model,
        "n_items": len(items),
        "n_parallels": n_parallels,
        "warmup_first": warmup_first,
        "wall_time_s": total_wall_s,
        "warmup_wall_s": warmup_wall_s,
        "batch_wall_s": batch_wall_s,
        "calls": arm_records,
    }
    responses_path.write_text(
        json.dumps(responses, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Also save prompts (first 2 prompts for inspection without bloat)
    (arm_dir / "sample_prompts.json").write_text(
        json.dumps(
            [{"item_id": iid, "prompt": pr} for iid, pr in items[:2]],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return responses, meta


# ---------------------------------------------------------------------------
# Phase 1: 2-call sanity check (distinct rubric so it does NOT warm experiment cache)
# ---------------------------------------------------------------------------


async def run_sanity_check(client: InstrumentedChatClient, model: str) -> None:
    print(f"=== Running 2-call sanity check on model={model!r} ===")
    sanity_attrs = OrderedDict(
        [
            (
                "clarity",
                "How clearly and concisely the sentence conveys its main idea.",
            ),
            ("formality", "Degree of formal academic or official register."),
        ]
    )
    sample_text = (
        "Municipal transit ridership increased 6.4 percent year over year following "
        "signal-priority upgrades along three crosstown bus corridors."
    )
    baseline_tpl = load_prompt("ratings_prompt.jinja2")
    p_base = baseline_tpl.render(
        text=sample_text,
        attributes=sanity_attrs,
        scale=None,
        additional_instructions=None,
        modality="text",
    )
    p_pref = RATE_PREFIX_TEMPLATE.render(
        text=sample_text,
        attributes=sanity_attrs,
        scale=None,
        additional_instructions=None,
        modality="text",
    )

    out_dir = RUNS_DIR / "sanity"
    out_dir.mkdir(parents=True, exist_ok=True)

    r1 = await client.complete(
        p_base,
        model=model,
        json_mode=True,
        _exp_task="sanity",
        _exp_arm="sanity_baseline",
        _exp_item_id="s1",
    )
    r2 = await client.complete(
        p_pref,
        model=model,
        json_mode=True,
        _exp_task="sanity",
        _exp_arm="sanity_prefix",
        _exp_item_id="s2",
    )

    sanity_records = [asdict(r) for r in client.records[-2:]]
    payload = {
        "responses": {"sanity_baseline": r1, "sanity_prefix": r2},
        "records": sanity_records,
    }
    (out_dir / "sanity_result.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    for rec in sanity_records:
        print(
            f"  [{rec['arm']}] ok={rec['ok']} model_returned={rec['model_returned']} "
            f"prompt_tok={rec['prompt_tokens']} cached={rec['cached_tokens']} "
            f"miss={rec['cache_miss_tokens']} comp={rec['completion_tokens']} "
            f"latency={rec['latency_s']}s"
        )
        print(f"    raw_usage={rec['raw_usage']}")
    print(f"  baseline_response={r1.strip()}")
    print(f"  prefix_response  ={r2.strip()}")
    print(f"Total API calls used so far: {client.total_calls}/{MAX_CALLS}")


async def run_verify_cache(client: InstrumentedChatClient, model: str) -> None:
    """2-call check verifying thinking=disabled + live KV cache hit on matching prefix."""
    print(f"=== Running 2-call cache & non-thinking verify on model={model!r} ===")
    sanity_attrs = OrderedDict(
        [
            (
                "clarity",
                "How clearly and concisely the sentence conveys its main idea.",
            ),
            ("formality", "Degree of formal academic or official register."),
        ]
    )
    text_a = (
        "Municipal transit ridership increased 6.4 percent year over year following "
        "signal-priority upgrades along three crosstown bus corridors."
    )
    text_b = (
        "Regional freight rail volumes dipped 2.1 percent during the second quarter "
        "after spring flooding temporarily closed two river bridges."
    )
    p1 = RATE_PREFIX_TEMPLATE.render(
        text=text_a,
        attributes=sanity_attrs,
        scale=None,
        additional_instructions=None,
        modality="text",
    )
    p2 = RATE_PREFIX_TEMPLATE.render(
        text=text_b,
        attributes=sanity_attrs,
        scale=None,
        additional_instructions=None,
        modality="text",
    )

    r1 = await client.complete(
        p1,
        model=model,
        json_mode=True,
        _exp_task="sanity",
        _exp_arm="verify_nothink_warmup",
        _exp_item_id="v1",
        _exp_is_warmup=True,
    )
    r2 = await client.complete(
        p2,
        model=model,
        json_mode=True,
        _exp_task="sanity",
        _exp_arm="verify_nothink_hit",
        _exp_item_id="v2",
        _exp_is_warmup=False,
    )
    recs = [asdict(r) for r in client.records[-2:]]
    for rec in recs:
        print(
            f"  [{rec['arm']}] ok={rec['ok']} prompt_tok={rec['prompt_tokens']} "
            f"cached={rec['cached_tokens']} miss={rec['cache_miss_tokens']} "
            f"comp={rec['completion_tokens']} reasoning={rec['reasoning_tokens']} "
            f"latency={rec['latency_s']}s"
        )
    print(f"  r1={r1.strip()}")
    print(f"  r2={r2.strip()}")
    print(f"Total API calls used so far: {client.total_calls}/{MAX_CALLS}")


# ---------------------------------------------------------------------------
# Phase 2: Full 212-call experiment matrix (rate=100, classify=80, codify=32)
# ---------------------------------------------------------------------------


async def run_rate_experiment(client: InstrumentedChatClient, model: str) -> None:
    print("\n=== Task 1: RATE (5 arms x 20 paragraphs x 10 attributes = 100 calls) ===")
    baseline_tpl = load_prompt("ratings_prompt.jinja2")
    rate_parser = Rate(
        RateConfig(
            attributes=dict(RATE_ATTRIBUTES_FWD),
            save_dir=str(RUNS_DIR / "rate" / "_tmp"),
        )
    )

    arms_spec = [
        ("baseline_r1", baseline_tpl, RATE_ATTRIBUTES_FWD, False),
        ("baseline_r2", baseline_tpl, RATE_ATTRIBUTES_FWD, False),
        ("prefix_fwd_r1", RATE_PREFIX_TEMPLATE, RATE_ATTRIBUTES_FWD, True),
        ("prefix_rev_r1", RATE_PREFIX_TEMPLATE, RATE_ATTRIBUTES_REV, True),
        ("prefix_fwd_r2", RATE_PREFIX_TEMPLATE, RATE_ATTRIBUTES_FWD, True),
    ]
    attr_names = list(RATE_ATTRIBUTES_FWD.keys())

    for arm_name, tpl, attrs, warmup_first in arms_spec:
        items = [
            (
                p["id"],
                tpl.render(
                    text=p["text"],
                    attributes=attrs,
                    scale=None,
                    additional_instructions=None,
                    modality="text",
                ),
            )
            for p in PARAGRAPHS_20
        ]
        responses, meta = await execute_arm_prompts(
            client,
            task="rate",
            arm=arm_name,
            items=items,
            model=model,
            n_parallels=10,
            warmup_first=warmup_first,
        )
        # Parse using Leximeter's real Rate.parse_response
        rows = []
        for p in PARAGRAPHS_20:
            parsed = await rate_parser.parse_response(
                responses.get(p["id"], ""), attr_names
            )
            rows.append({"id": p["id"], "topic": p["topic"], **parsed})
        df = pd.DataFrame(rows)
        df.to_csv(RUNS_DIR / "rate" / arm_name / "parsed.csv", index=False)
        _print_arm_quick_summary("rate", arm_name, meta)


async def run_classify_experiment(client: InstrumentedChatClient, model: str) -> None:
    print("\n=== Task 2: CLASSIFY (4 arms x 20 paragraphs x 10 labels = 80 calls) ===")
    baseline_tpl = load_prompt("classification_prompt.jinja2")
    cls_parser = Classify(
        ClassifyConfig(
            labels=dict(CLASSIFY_LABELS_FWD),
            save_dir=str(RUNS_DIR / "classify" / "_tmp"),
        )
    )

    arms_spec = [
        ("baseline_r1", baseline_tpl, CLASSIFY_LABELS_FWD, False),
        ("baseline_r2", baseline_tpl, CLASSIFY_LABELS_FWD, False),
        ("prefix_fwd", CLASSIFY_PREFIX_TEMPLATE, CLASSIFY_LABELS_FWD, True),
        ("prefix_rev", CLASSIFY_PREFIX_TEMPLATE, CLASSIFY_LABELS_REV, True),
    ]
    label_names = list(CLASSIFY_LABELS_FWD.keys())

    for arm_name, tpl, labels, warmup_first in arms_spec:
        items = [
            (
                p["id"],
                tpl.render(
                    text=p["text"],
                    attributes=labels,
                    additional_instructions=None,
                    modality="text",
                ),
            )
            for p in PARAGRAPHS_20
        ]
        responses, meta = await execute_arm_prompts(
            client,
            task="classify",
            arm=arm_name,
            items=items,
            model=model,
            n_parallels=10,
            warmup_first=warmup_first,
        )
        # Parse using Leximeter's real Classify._parse
        rows = []
        for p in PARAGRAPHS_20:
            parsed = await cls_parser._parse(responses.get(p["id"], ""), label_names)
            rows.append({"id": p["id"], "topic": p["topic"], **parsed})
        df = pd.DataFrame(rows)
        df.to_csv(RUNS_DIR / "classify" / arm_name / "parsed.csv", index=False)
        _print_arm_quick_summary("classify", arm_name, meta)


async def run_codify_experiment(client: InstrumentedChatClient, model: str) -> None:
    print("\n=== Task 3: CODIFY (4 arms x 8 passages x 4 codes = 32 calls) ===")
    baseline_tpl = load_prompt("codify_prompt.jinja2")
    code_names = list(CODIFY_CATEGORIES_FWD.keys())

    arms_spec = [
        ("baseline_r1", baseline_tpl, dict(CODIFY_CATEGORIES_FWD), False),
        ("baseline_r2", baseline_tpl, dict(CODIFY_CATEGORIES_FWD), False),
        ("prefix_fwd", CODIFY_PREFIX_TEMPLATE, dict(CODIFY_CATEGORIES_FWD), True),
        ("prefix_rev", CODIFY_PREFIX_TEMPLATE, dict(CODIFY_CATEGORIES_REV), True),
    ]

    for arm_name, tpl, categories, warmup_first in arms_spec:
        items = [
            (
                p["id"],
                tpl.render(
                    text=p["text"],
                    categories=categories,
                    additional_instructions=None,
                    modality="text",
                ),
            )
            for p in CODIFY_PASSAGES_8
        ]
        responses, meta = await execute_arm_prompts(
            client,
            task="codify",
            arm=arm_name,
            items=items,
            model=model,
            n_parallels=8,
            warmup_first=warmup_first,
        )
        # Reuse Leximeter's real Codify parser & verbatim snippet locator!
        codifier = Codify(
            CodifyConfig(
                save_dir=str(RUNS_DIR / "codify" / arm_name),
                model=model,
                n_rounds=1,
            )
        )
        coded_rows: list[dict[str, Any]] = []
        raw_excerpt_counts: dict[str, int] = {c: 0 for c in code_names}
        for p in CODIFY_PASSAGES_8:
            raw_text = responses.get(p["id"], "")
            parsed_json = codifier.parse_json(raw_text) or {}
            cr = [
                ChunkResult(identifier=p["id"], chunk_text=p["text"], data=parsed_json)
            ]
            row_out: dict[str, Any] = {"id": p["id"], "topic": p["topic"]}
            for code in code_names:
                raw_list = parsed_json.get(code, [])
                if isinstance(raw_list, list):
                    raw_excerpt_counts[code] += sum(
                        1
                        for it in raw_list
                        if isinstance(it, dict) and it.get("beginning excerpt")
                    )
                snippets = codifier.consolidate_snippets(
                    p["text"], cr, code, debug_print=False
                )
                row_out[code] = snippets
            coded_rows.append(row_out)

        arm_dir = RUNS_DIR / "codify" / arm_name
        (arm_dir / "coded_snippets.json").write_text(
            json.dumps(
                {
                    "rows": coded_rows,
                    "hit_rate_stats": codifier.hit_rate_stats,
                    "raw_excerpt_counts": raw_excerpt_counts,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        _print_arm_quick_summary("codify", arm_name, meta)


def _print_arm_quick_summary(task: str, arm: str, meta: dict[str, Any]) -> None:
    calls = meta["calls"]
    p_tok = sum(c["prompt_tokens"] for c in calls)
    c_hit = sum(c["cached_tokens"] for c in calls)
    c_miss = sum(c["cache_miss_tokens"] for c in calls)
    comp = sum(c["completion_tokens"] for c in calls)
    hit_all = (100.0 * c_hit / p_tok) if p_tok else 0.0
    p_ex = sum(c["prompt_tokens"] for c in calls[1:])
    h_ex = sum(c["cached_tokens"] for c in calls[1:])
    hit_ex = (100.0 * h_ex / p_ex) if p_ex else 0.0
    cost = _calc_cost_usd(c_hit, c_miss, comp)
    print(
        f"  [{task}/{arm:13s}] calls={len(calls):2d} prompt={p_tok:6d} "
        f"cached={c_hit:6d} hit_all={hit_all:5.1f}% hit_ex_w={hit_ex:5.1f}% "
        f"comp={comp:5d} wall={meta['wall_time_s']:5.2f}s cost=${cost:.5f}"
    )


def _calc_cost_usd(
    cached_tokens: int,
    cache_miss_tokens: int,
    completion_tokens: int,
    pricing: dict[str, float] = PRICING_USD_PER_1M,
) -> float:
    return (
        cached_tokens * pricing["cache_hit"]
        + cache_miss_tokens * pricing["cache_miss"]
        + completion_tokens * pricing["output"]
    ) / 1_000_000.0


# ---------------------------------------------------------------------------
# Phase 3: Full statistical analysis
# ---------------------------------------------------------------------------


def _summarize_arm_efficiency(task: str, arm: str) -> dict[str, Any]:
    meta = json.loads(
        (RUNS_DIR / task / arm / "arm_meta.json").read_text(encoding="utf-8")
    )
    calls = meta["calls"]
    # Order calls by call_index so index 0 is genuinely the first (warmup) call!
    calls_sorted = sorted(calls, key=lambda c: c["call_index"])
    n_calls = len(calls_sorted)
    n_ok = sum(1 for c in calls_sorted if c["ok"])
    p_tok = sum(c["prompt_tokens"] for c in calls_sorted)
    c_hit = sum(c["cached_tokens"] for c in calls_sorted)
    c_miss = sum(c["cache_miss_tokens"] for c in calls_sorted)
    comp = sum(c["completion_tokens"] for c in calls_sorted)
    reas = sum(c.get("reasoning_tokens", 0) for c in calls_sorted)

    p_ex = sum(c["prompt_tokens"] for c in calls_sorted[1:])
    h_ex = sum(c["cached_tokens"] for c in calls_sorted[1:])
    m_ex = sum(c["cache_miss_tokens"] for c in calls_sorted[1:])

    latencies = [c["latency_s"] for c in calls_sorted]
    cost_usd = _calc_cost_usd(c_hit, c_miss, comp, PRICING_OFF_PEAK_USD_PER_1M)
    cost_uncached_usd = _calc_cost_usd(0, p_tok, comp, PRICING_OFF_PEAK_USD_PER_1M)
    cost_peak_usd = _calc_cost_usd(c_hit, c_miss, comp, PRICING_PEAK_USD_PER_1M)

    return {
        "task": task,
        "arm": arm,
        "warmup_first": meta["warmup_first"],
        "n_calls": n_calls,
        "n_ok": n_ok,
        "prompt_tokens": p_tok,
        "cached_tokens": c_hit,
        "cache_miss_tokens": c_miss,
        "completion_tokens": comp,
        "reasoning_tokens": reas,
        "cache_hit_rate_all": c_hit / p_tok if p_tok else 0.0,
        "cache_hit_rate_excl_warmup": h_ex / p_ex if p_ex else 0.0,
        "warmup_prompt_tokens": calls_sorted[0]["prompt_tokens"] if calls_sorted else 0,
        "warmup_cached_tokens": calls_sorted[0]["cached_tokens"] if calls_sorted else 0,
        "excl_warmup_prompt_tokens": p_ex,
        "excl_warmup_cached_tokens": h_ex,
        "excl_warmup_miss_tokens": m_ex,
        "wall_time_s": meta["wall_time_s"],
        "warmup_wall_s": meta.get("warmup_wall_s", 0.0),
        "batch_wall_s": meta.get("batch_wall_s", 0.0),
        "latency_mean_s": float(np.mean(latencies)) if latencies else 0.0,
        "latency_median_s": float(np.median(latencies)) if latencies else 0.0,
        "latency_p95_s": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "cost_usd": cost_usd,
        "cost_peak_usd": cost_peak_usd,
        "cost_if_uncached_usd": cost_uncached_usd,
        "input_cost_savings_pct": (
            100.0
            * (
                1.0
                - (
                    c_hit * PRICING_OFF_PEAK_USD_PER_1M["cache_hit"]
                    + c_miss * PRICING_OFF_PEAK_USD_PER_1M["cache_miss"]
                )
                / (p_tok * PRICING_OFF_PEAK_USD_PER_1M["cache_miss"])
            )
            if p_tok
            else 0.0
        ),
        "total_cost_savings_pct": (
            100.0 * (1.0 - cost_usd / cost_uncached_usd) if cost_uncached_usd else 0.0
        ),
    }


def _analyze_rate() -> dict[str, Any]:
    arms = [
        "baseline_r1",
        "baseline_r2",
        "prefix_fwd_r1",
        "prefix_rev_r1",
        "prefix_fwd_r2",
    ]
    attr_names = list(RATE_ATTRIBUTES_FWD.keys())
    dfs = {
        a: pd.read_csv(RUNS_DIR / "rate" / a / "parsed.csv").set_index("id")
        for a in arms
    }

    # Missing counts & descriptive distributions
    arm_stats: dict[str, Any] = {}
    for a in arms:
        vals = dfs[a][attr_names].to_numpy(dtype=float)
        arm_stats[a] = {
            "missing_cells": int(np.isnan(vals).sum()),
            "total_cells": int(vals.size),
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals)),
            "min": float(np.nanmin(vals)),
            "max": float(np.nanmax(vals)),
            "per_attr_mean": {c: float(dfs[a][c].mean()) for c in attr_names},
        }

    pairs = [
        ("baseline_noise", "baseline_r1", "baseline_r2"),
        ("prefix_noise", "prefix_fwd_r1", "prefix_fwd_r2"),
        ("layout_drift_fwd", "baseline_r1", "prefix_fwd_r1"),
        ("layout_drift_rev", "baseline_r1", "prefix_rev_r1"),
        ("order_flip_drift", "prefix_fwd_r1", "prefix_rev_r1"),
    ]

    comparisons: dict[str, Any] = {}
    buckets = {
        "early_1_3": attr_names[0:3],
        "middle_4_7": attr_names[3:7],
        "late_8_10": attr_names[7:10],
    }

    for label, a_name, b_name in pairs:
        da = dfs[a_name][attr_names]
        db = dfs[b_name][attr_names]
        va = da.to_numpy(dtype=float).ravel()
        vb = db.to_numpy(dtype=float).ravel()
        mask = ~np.isnan(va) & ~np.isnan(vb)
        xa, xb = va[mask], vb[mask]
        diff = xa - xb

        r_all, _r_p = stats.pearsonr(xa, xb)
        rho_all, _rho_p = stats.spearmanr(xa, xb)

        within_r = []
        within_rho = []
        per_attr: dict[str, Any] = {}
        for idx, col in enumerate(attr_names, start=1):
            ca = da[col].to_numpy(dtype=float)
            cb = db[col].to_numpy(dtype=float)
            m = ~np.isnan(ca) & ~np.isnan(cb)
            ca_m, cb_m = ca[m], cb[m]
            cd = ca_m - cb_m
            cr = (
                float(stats.pearsonr(ca_m, cb_m)[0])
                if np.std(ca_m) > 0 and np.std(cb_m) > 0
                else float("nan")
            )
            crho = (
                float(stats.spearmanr(ca_m, cb_m)[0])
                if np.std(ca_m) > 0 and np.std(cb_m) > 0
                else float("nan")
            )
            if not math.isnan(cr):
                within_r.append(cr)
            if not math.isnan(crho):
                within_rho.append(crho)
            per_attr[col] = {
                "fwd_pos": idx,
                "rev_pos": 11 - idx,
                "pearson_r": cr,
                "spearman_rho": crho,
                "mae": float(np.mean(np.abs(cd))),
                "rmse": float(np.sqrt(np.mean(cd**2))),
                "mean_signed_diff": float(np.mean(cd)),
                "std_diff": float(np.std(cd, ddof=1)),
            }

        # Position-bucket check (grouped by forward position 1-3, 4-7, 8-10)
        pos_buckets: dict[str, Any] = {}
        for bname, cols in buckets.items():
            ba = da[cols].to_numpy(dtype=float).ravel()
            bb = db[cols].to_numpy(dtype=float).ravel()
            bm = ~np.isnan(ba) & ~np.isnan(bb)
            bd = ba[bm] - bb[bm]
            t_stat, p_val = (
                stats.ttest_1samp(bd, 0.0)
                if len(bd) > 1 and np.std(bd, ddof=1) > 0
                else (0.0, 1.0)
            )
            pos_buckets[bname] = {
                "attributes": cols,
                "n_pairs": len(bd),
                "mean_signed_diff_a_minus_b": float(np.mean(bd)),
                "se": float(np.std(bd, ddof=1) / math.sqrt(len(bd)))
                if len(bd) > 1
                else 0.0,
                "mae": float(np.mean(np.abs(bd))),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
            }

        # Two-sample test: Early [1-3] diff vs Late [8-10] diff
        e_diff = (
            da[buckets["early_1_3"]].to_numpy(dtype=float)
            - db[buckets["early_1_3"]].to_numpy(dtype=float)
        ).ravel()
        l_diff = (
            da[buckets["late_8_10"]].to_numpy(dtype=float)
            - db[buckets["late_8_10"]].to_numpy(dtype=float)
        ).ravel()
        e_diff = e_diff[~np.isnan(e_diff)]
        l_diff = l_diff[~np.isnan(l_diff)]
        el_t, el_p = stats.ttest_ind(e_diff, l_diff, equal_var=False)

        comparisons[label] = {
            "arm_a": a_name,
            "arm_b": b_name,
            "n_pairs": int(mask.sum()),
            "pearson_r_pooled": float(r_all),
            "pearson_r_mean_within_attr": float(np.mean(within_r)),
            "spearman_rho_pooled": float(rho_all),
            "spearman_rho_mean_within_attr": float(np.mean(within_rho)),
            "mae": float(np.mean(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(diff**2))),
            "mean_signed_diff": float(np.mean(diff)),
            "std_diff": float(np.std(diff, ddof=1)),
            "early_minus_late_diff_shift": float(np.mean(e_diff) - np.mean(l_diff)),
            "early_vs_late_welch_t": float(el_t),
            "early_vs_late_welch_p": float(el_p),
            "position_buckets": pos_buckets,
            "per_attribute": per_attr,
        }

    return {"arm_stats": arm_stats, "comparisons": comparisons}


def _cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = float(np.mean(a == b))
    pa1 = float(np.mean(a == 1))
    pb1 = float(np.mean(b == 1))
    pe = pa1 * pb1 + (1.0 - pa1) * (1.0 - pb1)
    if abs(1.0 - pe) < 1e-12:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def _binary_f1(a: np.ndarray, b: np.ndarray) -> float:
    tp = int(np.sum((a == 1) & (b == 1)))
    fp = int(np.sum((a == 0) & (b == 1)))
    fn = int(np.sum((a == 1) & (b == 0)))
    denom = 2 * tp + fp + fn
    return (2.0 * tp / denom) if denom > 0 else 1.0


def _analyze_classify() -> dict[str, Any]:
    arms = ["baseline_r1", "baseline_r2", "prefix_fwd", "prefix_rev"]
    label_names = list(CLASSIFY_LABELS_FWD.keys())
    dfs = {
        a: pd.read_csv(RUNS_DIR / "classify" / a / "parsed.csv").set_index("id")
        for a in arms
    }

    def _to_int_matrix(df: pd.DataFrame) -> np.ndarray:
        out = np.full((len(df), len(label_names)), np.nan)
        for j, col in enumerate(label_names):
            for i, val in enumerate(df[col].tolist()):
                if pd.isna(val):
                    continue
                if isinstance(val, bool):
                    out[i, j] = 1.0 if val else 0.0
                else:
                    s = str(val).strip().lower()
                    if s in {"true", "1", "1.0"}:
                        out[i, j] = 1.0
                    elif s in {"false", "0", "0.0"}:
                        out[i, j] = 0.0
        return out

    mats = {a: _to_int_matrix(dfs[a]) for a in arms}
    arm_stats: dict[str, Any] = {}
    for a in arms:
        m = mats[a]
        arm_stats[a] = {
            "missing_cells": int(np.isnan(m).sum()),
            "overall_positive_rate": float(np.nanmean(m)),
            "per_label_positive_rate": {
                col: float(np.nanmean(m[:, j])) for j, col in enumerate(label_names)
            },
            "bucket_positive_rate": {
                "early_1_3": float(np.nanmean(m[:, 0:3])),
                "middle_4_7": float(np.nanmean(m[:, 3:7])),
                "late_8_10": float(np.nanmean(m[:, 7:10])),
            },
        }

    pairs = [
        ("baseline_noise", "baseline_r1", "baseline_r2"),
        ("layout_drift_fwd", "baseline_r1", "prefix_fwd"),
        ("layout_drift_rev", "baseline_r1", "prefix_rev"),
        ("order_flip_drift", "prefix_fwd", "prefix_rev"),
    ]
    buckets = {
        "early_1_3": slice(0, 3),
        "middle_4_7": slice(3, 7),
        "late_8_10": slice(7, 10),
    }

    comparisons: dict[str, Any] = {}
    for comp_name, a_name, b_name in pairs:
        ma, mb = mats[a_name], mats[b_name]
        va, vb = ma.ravel(), mb.ravel()
        valid = ~np.isnan(va) & ~np.isnan(vb)
        xa, xb = va[valid].astype(int), vb[valid].astype(int)

        per_label: dict[str, Any] = {}
        kappas = []
        f1s = []
        for j, col in enumerate(label_names):
            ca, cb = ma[:, j], mb[:, j]
            vm = ~np.isnan(ca) & ~np.isnan(cb)
            xa_c, xb_c = ca[vm].astype(int), cb[vm].astype(int)
            k = _cohen_kappa(xa_c, xb_c)
            f1 = _binary_f1(xa_c, xb_c)
            if not math.isnan(k):
                kappas.append(k)
            f1s.append(f1)
            per_label[col] = {
                "fwd_pos": j + 1,
                "rev_pos": 10 - j,
                "agreement_pct": 100.0 * float(np.mean(xa_c == xb_c)),
                "cohen_kappa": float(k),
                "f1": float(f1),
                "pos_rate_a": float(np.mean(xa_c)),
                "pos_rate_b": float(np.mean(xb_c)),
                "pos_rate_diff_a_minus_b": float(np.mean(xa_c) - np.mean(xb_c)),
            }

        pos_buckets: dict[str, Any] = {}
        for bname, sl in buckets.items():
            ba, bb = ma[:, sl].ravel(), mb[:, sl].ravel()
            bm = ~np.isnan(ba) & ~np.isnan(bb)
            ba_v, bb_v = ba[bm].astype(int), bb[bm].astype(int)
            pos_buckets[bname] = {
                "agreement_pct": 100.0 * float(np.mean(ba_v == bb_v)),
                "pos_rate_a": float(np.mean(ba_v)),
                "pos_rate_b": float(np.mean(bb_v)),
                "pos_rate_shift_a_minus_b": float(np.mean(ba_v) - np.mean(bb_v)),
            }

        comparisons[comp_name] = {
            "arm_a": a_name,
            "arm_b": b_name,
            "n_pairs": int(valid.sum()),
            "exact_agreement_pct": 100.0 * float(np.mean(xa == xb)),
            "cohen_kappa_pooled": float(_cohen_kappa(xa, xb)),
            "cohen_kappa_macro": float(np.mean(kappas)) if kappas else float("nan"),
            "f1_pooled": float(_binary_f1(xa, xb)),
            "f1_macro": float(np.mean(f1s)) if f1s else float("nan"),
            "pos_rate_a": float(np.mean(xa)),
            "pos_rate_b": float(np.mean(xb)),
            "pos_rate_diff_a_minus_b": float(np.mean(xa) - np.mean(xb)),
            "early_minus_late_pos_shift": (
                pos_buckets["early_1_3"]["pos_rate_shift_a_minus_b"]
                - pos_buckets["late_8_10"]["pos_rate_shift_a_minus_b"]
            ),
            "position_buckets": pos_buckets,
            "per_label": per_label,
        }

    return {"arm_stats": arm_stats, "comparisons": comparisons}


def _snippets_to_char_mask(text: str, snippets: list[str]) -> np.ndarray:
    mask = np.zeros(len(text), dtype=bool)
    text_low = text.lower()
    for snip in snippets:
        if not snip:
            continue
        idx = text.find(snip)
        if idx == -1:
            idx = text_low.find(snip.lower())
        if idx != -1:
            mask[idx : idx + len(snip)] = True
    return mask


def _analyze_codify() -> dict[str, Any]:
    arms = ["baseline_r1", "baseline_r2", "prefix_fwd", "prefix_rev"]
    code_names = list(CODIFY_CATEGORIES_FWD.keys())
    passage_by_id = {p["id"]: p["text"] for p in CODIFY_PASSAGES_8}

    arm_data = {
        a: json.loads(
            (RUNS_DIR / "codify" / a / "coded_snippets.json").read_text(
                encoding="utf-8"
            )
        )
        for a in arms
    }

    arm_stats: dict[str, Any] = {}
    masks_by_arm: dict[str, dict[tuple[str, str], np.ndarray]] = {}

    for a in arms:
        d = arm_data[a]
        rows = d["rows"]
        hr = d["hit_rate_stats"]
        total_proposed = sum(v.get("total", 0) for v in hr.values())
        total_located = sum(v.get("found", 0) for v in hr.values())
        total_strict = sum(v.get("strict_matches", 0) for v in hr.values())

        all_lens: list[int] = []
        per_code_counts = {c: 0 for c in code_names}
        per_code_chars = {c: 0 for c in code_names}
        arm_masks: dict[tuple[str, str], np.ndarray] = {}

        for row in rows:
            pid = row["id"]
            text = passage_by_id[pid]
            for c in code_names:
                snips = row.get(c, [])
                per_code_counts[c] += len(snips)
                for s in snips:
                    all_lens.append(len(s))
                    per_code_chars[c] += len(s)
                arm_masks[(pid, c)] = _snippets_to_char_mask(text, snips)

        masks_by_arm[a] = arm_masks
        total_snippets_kept = sum(per_code_counts.values())
        arm_stats[a] = {
            "total_raw_proposed": total_proposed,
            "total_located_dedup": total_located,
            "total_snippets_kept": total_snippets_kept,
            "unmatched_count": max(0, total_proposed - total_located),
            "locate_success_rate_pct": (100.0 * total_located / total_proposed)
            if total_proposed
            else 0.0,
            "strict_verbatim_rate_pct": (100.0 * total_strict / total_proposed)
            if total_proposed
            else 0.0,
            "mean_snippet_chars": float(np.mean(all_lens)) if all_lens else 0.0,
            "median_snippet_chars": float(np.median(all_lens)) if all_lens else 0.0,
            "per_code_counts": per_code_counts,
            "per_code_total_chars": per_code_chars,
        }

    pairs = [
        ("baseline_noise", "baseline_r1", "baseline_r2"),
        ("layout_drift_fwd", "baseline_r1", "prefix_fwd"),
        ("layout_drift_rev", "baseline_r1", "prefix_rev"),
        ("order_flip_drift", "prefix_fwd", "prefix_rev"),
    ]

    comparisons: dict[str, Any] = {}
    for comp_name, a_name, b_name in pairs:
        ma = masks_by_arm[a_name]
        mb = masks_by_arm[b_name]
        inter_total = 0
        union_total = 0
        sum_a_total = 0
        sum_b_total = 0

        pair_f1_all: list[float] = []
        pair_jacc_all: list[float] = []
        pair_f1_active: list[float] = []
        pair_jacc_active: list[float] = []
        per_code_micro: dict[str, dict[str, float]] = {}

        for c in code_names:
            c_inter = c_union = c_sa = c_sb = 0
            for p in CODIFY_PASSAGES_8:
                pid = p["id"]
                xa = ma[(pid, c)]
                xb = mb[(pid, c)]
                inter = int(np.sum(xa & xb))
                union = int(np.sum(xa | xb))
                sa = int(np.sum(xa))
                sb = int(np.sum(xb))

                inter_total += inter
                union_total += union
                sum_a_total += sa
                sum_b_total += sb
                c_inter += inter
                c_union += union
                c_sa += sa
                c_sb += sb

                if union == 0:
                    pair_f1_all.append(1.0)
                    pair_jacc_all.append(1.0)
                else:
                    f1_val = (2.0 * inter) / (sa + sb) if (sa + sb) > 0 else 1.0
                    j_val = inter / union
                    pair_f1_all.append(f1_val)
                    pair_jacc_all.append(j_val)
                    pair_f1_active.append(f1_val)
                    pair_jacc_active.append(j_val)

            per_code_micro[c] = {
                "char_f1": (2.0 * c_inter / (c_sa + c_sb))
                if (c_sa + c_sb) > 0
                else 1.0,
                "char_jaccard": (c_inter / c_union) if c_union > 0 else 1.0,
                "chars_a": c_sa,
                "chars_b": c_sb,
            }

        comparisons[comp_name] = {
            "arm_a": a_name,
            "arm_b": b_name,
            "micro_char_f1": (2.0 * inter_total / (sum_a_total + sum_b_total))
            if (sum_a_total + sum_b_total) > 0
            else 1.0,
            "micro_char_jaccard": (inter_total / union_total)
            if union_total > 0
            else 1.0,
            "macro_char_f1_active": float(np.mean(pair_f1_active))
            if pair_f1_active
            else 1.0,
            "macro_char_jaccard_active": float(np.mean(pair_jacc_active))
            if pair_jacc_active
            else 1.0,
            "macro_char_f1_all32": float(np.mean(pair_f1_all)),
            "macro_char_jaccard_all32": float(np.mean(pair_jacc_all)),
            "total_chars_a": sum_a_total,
            "total_chars_b": sum_b_total,
            "per_code_micro": per_code_micro,
        }

    return {"arm_stats": arm_stats, "comparisons": comparisons}


def run_analysis() -> dict[str, Any]:
    task_arms = [
        (
            "rate",
            [
                "baseline_r1",
                "baseline_r2",
                "prefix_fwd_r1",
                "prefix_rev_r1",
                "prefix_fwd_r2",
            ],
        ),
        ("classify", ["baseline_r1", "baseline_r2", "prefix_fwd", "prefix_rev"]),
        ("codify", ["baseline_r1", "baseline_r2", "prefix_fwd", "prefix_rev"]),
    ]

    efficiency_rows: list[dict[str, Any]] = []
    for task, arms in task_arms:
        for arm in arms:
            efficiency_rows.append(_summarize_arm_efficiency(task, arm))

    # Grand total from full call ledger (including sanity check!)
    ledger_records: list[dict[str, Any]] = []
    if LEDGER_PATH.exists():
        with LEDGER_PATH.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    ledger_records.append(json.loads(line))

    grand_calls = len(ledger_records)
    grand_ok = sum(1 for r in ledger_records if r["ok"])
    grand_prompt = sum(r["prompt_tokens"] for r in ledger_records)
    grand_cached = sum(r["cached_tokens"] for r in ledger_records)
    grand_miss = sum(r["cache_miss_tokens"] for r in ledger_records)
    grand_comp = sum(r["completion_tokens"] for r in ledger_records)
    grand_reas = sum(r.get("reasoning_tokens", 0) for r in ledger_records)
    grand_cost = _calc_cost_usd(
        grand_cached, grand_miss, grand_comp, PRICING_OFF_PEAK_USD_PER_1M
    )
    grand_cost_peak = _calc_cost_usd(
        grand_cached, grand_miss, grand_comp, PRICING_PEAK_USD_PER_1M
    )
    grand_cost_uncached = _calc_cost_usd(
        0, grand_prompt, grand_comp, PRICING_OFF_PEAK_USD_PER_1M
    )

    rate_res = _analyze_rate()
    classify_res = _analyze_classify()
    codify_res = _analyze_codify()

    summary = {
        "pricing_off_peak_usd_per_1m": PRICING_OFF_PEAK_USD_PER_1M,
        "pricing_peak_usd_per_1m": PRICING_PEAK_USD_PER_1M,
        "grand_totals_incl_sanity": {
            "total_calls": grand_calls,
            "ok_calls": grand_ok,
            "prompt_tokens": grand_prompt,
            "cached_tokens": grand_cached,
            "cache_miss_tokens": grand_miss,
            "completion_tokens": grand_comp,
            "reasoning_tokens": grand_reas,
            "actual_cost_off_peak_usd": grand_cost,
            "equivalent_cost_peak_usd": grand_cost_peak,
            "counterfactual_uncached_off_peak_usd": grand_cost_uncached,
        },
        "efficiency": efficiency_rows,
        "rate": rate_res,
        "classify": classify_res,
        "codify": codify_res,
    }

    out_path = RUNS_DIR / "analysis_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(efficiency_rows).to_csv(
        RUNS_DIR / "efficiency_summary.csv", index=False
    )

    print("\n================ GRAND TOTALS (INCL. SANITY) ================")
    print(json.dumps(summary["grand_totals_incl_sanity"], indent=2))
    print("\n================ EFFICIENCY SUMMARY ================")
    for row in efficiency_rows:
        print(
            f"{row['task']:8s} | {row['arm']:13s} | calls={row['n_calls']:2d} | "
            f"prompt={row['prompt_tokens']:6d} | cached={row['cached_tokens']:6d} | "
            f"hit_all={100 * row['cache_hit_rate_all']:5.1f}% | "
            f"hit_ex_w={100 * row['cache_hit_rate_excl_warmup']:5.1f}% | "
            f"comp={row['completion_tokens']:5d} | wall={row['wall_time_s']:5.2f}s | "
            f"cost=${row['cost_usd']:.5f}"
        )

    print("\n================ RATE COMPARISONS ================")
    for cname, c in rate_res["comparisons"].items():
        print(
            f"{cname:18s} ({c['arm_a']} vs {c['arm_b']}): "
            f"r={c['pearson_r_pooled']:.4f} (within={c['pearson_r_mean_within_attr']:.4f}) | "
            f"rho={c['spearman_rho_pooled']:.4f} | MAE={c['mae']:.2f} | RMSE={c['rmse']:.2f} | "
            f"mean_diff={c['mean_signed_diff']:+.2f} | "
            f"E[1-3]={c['position_buckets']['early_1_3']['mean_signed_diff_a_minus_b']:+.2f} "
            f"M[4-7]={c['position_buckets']['middle_4_7']['mean_signed_diff_a_minus_b']:+.2f} "
            f"L[8-10]={c['position_buckets']['late_8_10']['mean_signed_diff_a_minus_b']:+.2f} "
            f"(E-L p={c['early_vs_late_welch_p']:.3f})"
        )

    print("\n================ CLASSIFY COMPARISONS ================")
    for cname, c in classify_res["comparisons"].items():
        pb = c["position_buckets"]
        print(
            f"{cname:18s} ({c['arm_a']} vs {c['arm_b']}): "
            f"agree={c['exact_agreement_pct']:.1f}% | kappa={c['cohen_kappa_pooled']:.4f} | "
            f"F1={c['f1_pooled']:.4f} | pos_diff={c['pos_rate_diff_a_minus_b']:+.3f} | "
            f"E_shift={pb['early_1_3']['pos_rate_shift_a_minus_b']:+.3f} "
            f"M_shift={pb['middle_4_7']['pos_rate_shift_a_minus_b']:+.3f} "
            f"L_shift={pb['late_8_10']['pos_rate_shift_a_minus_b']:+.3f}"
        )

    print("\n================ CODIFY STATS & COMPARISONS ================")
    for aname, s in codify_res["arm_stats"].items():
        print(
            f"  arm={aname:12s} | proposed={s['total_raw_proposed']:3d} | "
            f"located={s['total_located_dedup']:3d} ({s['locate_success_rate_pct']:.1f}%) | "
            f"strict={s['strict_verbatim_rate_pct']:.1f}% | "
            f"mean_len={s['mean_snippet_chars']:.1f} chars"
        )
    for cname, c in codify_res["comparisons"].items():
        print(
            f"  {cname:18s} ({c['arm_a']} vs {c['arm_b']}): "
            f"micro_F1={c['micro_char_f1']:.4f} | micro_Jacc={c['micro_char_jaccard']:.4f} | "
            f"macro_F1_active={c['macro_char_f1_active']:.4f} | macro_Jacc_active={c['macro_char_jaccard_active']:.4f}"
        )

    print(f"\nSaved full analysis JSON to {out_path}")
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def async_main(phase: str) -> None:
    load_dotenv(WORKTREE_ROOT / ".env", override=False)
    model = os.getenv("LEXIMETER_MODEL") or "deepseek-chat"
    protocol = os.getenv("LEXIMETER_PROTOCOL")
    base_url = os.getenv("LEXIMETER_BASE_URL")
    print(
        f"Resolved config: protocol={protocol!r}, base_url={base_url!r}, model={model!r}"
    )

    if phase == "analyze":
        run_analysis()
        return

    client = InstrumentedChatClient()
    print(f"Budget status before run: {client.total_calls}/{MAX_CALLS} calls used.")

    if phase == "sanity":
        await run_sanity_check(client, model)
    elif phase == "verify-cache":
        await run_verify_cache(client, model)
    elif phase == "full":
        await run_rate_experiment(client, model)
        await run_classify_experiment(client, model)
        await run_codify_experiment(client, model)
        print(
            f"\nFull matrix complete! Total API calls used: {client.total_calls}/{MAX_CALLS}"
        )
        run_analysis()
    else:
        raise ValueError(f"Unknown phase: {phase}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leximeter KV-cache prompt layout A/B experiment"
    )
    parser.add_argument(
        "--phase",
        choices=["sanity", "verify-cache", "full", "analyze"],
        required=True,
        help="Execution phase: 'sanity' (2 calls), 'verify-cache' (2 calls), 'full' (212 calls + analysis), or 'analyze' (offline).",
    )
    args = parser.parse_args()
    asyncio.run(async_main(args.phase))


if __name__ == "__main__":
    main()
