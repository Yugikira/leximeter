# Leximeter

Leximeter is a text-only, provider-neutral LLM measurement library that turns qualitative text corpora into structured quantitative data.

## Language

### Core Units

**Item**:
A single unit of qualitative input — either a prose passage or a named entity — being measured or transformed.
_Avoid_: Entry, record, document, row

**Modality**:
The textual nature of an item, distinguishing prose passages from named entities.
_Avoid_: Media type, input type

**Task**:
A self-contained text measurement, cleaning, or generation workflow that turns qualitative inputs into structured outputs.
_Avoid_: Job, pipeline, step

### Measurement Constructs

**Attribute**:
A named natural-language construct, paired with a plain-language definition, along which items are rated, ranked, or extracted.
_Avoid_: Dimension, feature, criterion, metric

**Label**:
A named boolean predicate, paired with a definition, tested for presence or absence on each item.
_Avoid_: Tag, class

**Code**:
A qualitative rubric under which verbatim evidence is marked within an item.
_Avoid_: Category, tag, label, bucket

**Snippet**:
A verbatim substring located inside an item as evidence for a code.
_Avoid_: Excerpt, quote, span

**Bucket**:
A mutually exclusive taxonomy category produced by consolidating many raw candidate terms.
_Avoid_: Cluster, theme, group

### Evaluation and Iteration

**Run**:
An independent repeated model evaluation of the same item and construct, aggregated across runs for reliability.
_Avoid_: Trial, repetition, sample

**Round**:
A sequential stage in an iterative task whose inputs depend on earlier stages' outcomes.
_Avoid_: Iteration, cycle, pass, attempt

**Circle / Square**:
Neutral names for the two sides of a pairwise contrast, used to reduce positional or evaluative bias.
_Avoid_: Left/right, item A/B, control/treatment

### Debiasing and Synthetic Surveys

**Confounder**:
An observable textual signal stripped from an item to isolate how much a measurement reflects surface cues rather than the target construct.
_Avoid_: Removal signal, bias source

**Remaining Signal**:
The residual presence of a confounder still detectable in a stripped item, used in two-step debiasing to adjust for incomplete removal.
_Avoid_: Residual bias

**Persona**:
An expanded synthetic respondent profile derived from a seeded entity and surveyed in a synthetic poll.
_Avoid_: Profile, agent, respondent

### Providers

**Provider**:
An external or local service supplying one capability: language generation, structured decisions, text embeddings, or web search.
_Avoid_: Vendor, backend

**Decision Provider**:
A provider that evaluates typed questions (yes/no, single-choice, or ordinal score) against an item in a single forward pass, returning calibrated probabilities without generating prose.
_Avoid_: Classifier provider, classifier model, reranker

**Protocol**:
The wire API family spoken by a provider (such as Responses, Chat Completions, or System One), independent of which provider hosts the endpoint.
_Avoid_: Vendor API, provider type
