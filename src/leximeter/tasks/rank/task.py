"""Rank task: pairwise tournaments yielding relative z-scores.

Ported from GABRIEL's ``tasks/rank.py`` (non-recursive path).  Each round
builds pairs with an information-gain heuristic (neighbourhood of the
current rating ranking + high-SE items + a small random exploration quota),
adjudicates them with the LLM, then refits the Bradley–Terry model on the
**full** outcome history.  Per attribute the output carries the z-score
(``{attr}``), the centred BT estimate (``{attr}_raw``) and its standard
error (``{attr}_se``).  Every round is checkpointed to
``{base}_round{rnd}.csv`` (with ``Batch``/``Pair``/``IdA``/``IdB`` columns)
so interrupted runs resume and replay completed rounds; new items added on
resume are caught up on the already-completed rounds.

Not ported (see spec backlog): recursive mode, primer scores, and the
initial rating pass.
"""

from __future__ import annotations

import os
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ...client.decision import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionClient,
    DecisionQuestion,
    DecisionResult,
)
from ...client.resolve import resolve_decision_model
from ...templates import load_prompt
from ...utils.file_utils import save_dataframe_with_fallback
from ...utils.parsing import safest_json
from .. import _common
from ..base import Task
from .bt import bt_standard_errors, fit_bt

_VALID_CHOICE_OUTCOMES = frozenset({"circle", "square", "draw", "insufficient_signal"})


def _build_pair_state(
    entry_circle: str,
    entry_square: str,
    *,
    circle_first: bool,
    additional_instructions: str | None,
) -> dict[str, Any]:
    """Build the Circle/Square state dict, preserving ``circle_first`` key order."""
    state: dict[str, Any] = (
        {"circle": entry_circle, "square": entry_square}
        if circle_first
        else {"square": entry_square, "circle": entry_circle}
    )
    if additional_instructions:
        state["additional_instructions"] = additional_instructions
    return state


def _build_rank_choice(
    attr: str,
    definition: str,
    *,
    modality: str,
    circle_first: bool,
) -> ChoiceQuestion:
    """Build a 4-way ``ChoiceQuestion`` adjudicating Circle vs Square on ``attr``."""
    noun = "entity" if modality == "entity" else "entry"
    desc = str(definition).strip() if definition is not None else ""
    attr_clause = f"'{attr}' ({desc})" if desc else f"'{attr}'"
    circle_desc = (
        f"Circle {noun} shows stronger direct signal of {attr_clause} than Square."
    )
    square_desc = (
        f"Square {noun} shows stronger direct signal of {attr_clause} than Circle."
    )
    draw_desc = (
        f"Draw: Circle and Square show indistinguishable strength on {attr_clause}."
    )
    insufficient_desc = (
        "Insufficient signal: neither Circle nor Square shows direct signal "
        f"for {attr_clause}."
    )
    if circle_first:
        criteria: dict[str, str | None] = {
            "circle": circle_desc,
            "square": square_desc,
            "draw": draw_desc,
            "insufficient_signal": insufficient_desc,
        }
    else:
        criteria = {
            "square": square_desc,
            "circle": circle_desc,
            "draw": draw_desc,
            "insufficient_signal": insufficient_desc,
        }
    instructions = (
        f"Compare Circle and Square on {attr_clause}. "
        "Choose 'circle' if Circle shows stronger direct signal than Square, "
        "'square' if Square shows stronger direct signal than Circle, "
        "'draw' if their strength is indistinguishable, or "
        "'insufficient_signal' if neither shows direct signal."
    )
    return ChoiceQuestion(instructions=instructions, criteria=criteria)


def _resolve_choice_winner(ans: ChoiceAnswer) -> str | None:
    """Resolve a :class:`ChoiceAnswer` into one of the 4 rank outcomes or ``None``."""
    if ans.refused:
        return None
    if ans.probabilities:
        norm_probs: dict[str, float] = {}
        for opt, prob in ans.probabilities.items():
            key = str(opt).strip().lower()
            if key in _VALID_CHOICE_OUTCOMES:
                norm_probs[key] = float(prob)
        if norm_probs:
            max_prob = max(norm_probs.values())
            winners = [k for k, v in norm_probs.items() if v == max_prob]
            if len(winners) == 1:
                return winners[0]
            if "circle" in winners and "square" in winners:
                return "draw"
            if ans.choice is not None:
                choice_norm = str(ans.choice).strip().lower()
                if choice_norm in winners:
                    return choice_norm
            return "draw"
    if ans.choice is not None:
        choice_norm = str(ans.choice).strip().lower()
        if choice_norm in _VALID_CHOICE_OUTCOMES:
            return choice_norm
    return None


def _is_decision_result_payload(obj: dict[str, Any]) -> bool:
    """Return ``True`` when ``obj`` is a serialized :class:`DecisionResult`."""
    raw_answers = obj.get("answers")
    if not isinstance(raw_answers, dict) or "model" not in obj:
        return False
    if not raw_answers:
        return "input_tokens" in obj and "output_tokens" in obj
    return all(
        isinstance(item, dict) and item.get("type") in {"noul", "choice", "score"}
        for item in raw_answers.values()
    )


def _decision_result_to_winners(
    result: DecisionResult,
    batch_attrs: list[str],
) -> dict[str, str]:
    """Extract ``{attr: winner}`` from a :class:`DecisionResult`, skipping refusals."""
    winners: dict[str, str] = {}
    lower_answers = {str(k).strip().lower(): v for k, v in result.answers.items()}
    for attr in batch_attrs:
        ans = result.answers.get(attr)
        if ans is None:
            ans = lower_answers.get(str(attr).strip().lower())
        if not isinstance(ans, ChoiceAnswer):
            continue
        winner = _resolve_choice_winner(ans)
        if winner is not None:
            winners[attr] = winner
    return winners


async def _decode_round_outcomes(
    resp_raw: Any,
    batch_attrs: list[str],
) -> dict[str, Any]:
    """Decode one round CSV ``Response`` cell (DecisionResult JSON or LLM JSON)."""
    if isinstance(resp_raw, DecisionResult):
        return _decision_result_to_winners(resp_raw, batch_attrs)
    safe_obj = await _coerce_dict(resp_raw)
    if not safe_obj:
        return {}
    if _is_decision_result_payload(safe_obj):
        parsed = DecisionResult.from_dict(safe_obj)
        return _decision_result_to_winners(parsed, batch_attrs)
    return safe_obj


def _is_completed_round_csv(path: str) -> bool:
    """Return ``True`` when a round CSV carries completed round metadata."""
    try:
        header = pd.read_csv(path, nrows=1)
    except (OSError, ValueError):
        return False
    if {"Batch", "IdA", "IdB"}.issubset(header.columns):
        return True
    if "Identifier" in header.columns and not header.empty:
        return len(str(header["Identifier"].iloc[0]).split("|")) == 5
    return False


def _attach_round_metadata(
    resp_df: pd.DataFrame,
    meta_map: dict[str, tuple[int, int, str, str]],
) -> pd.DataFrame:
    """Attach ``Batch``/``Pair``/``IdA``/``IdB`` metadata while preserving existing rows."""
    idents = resp_df["Identifier"].astype(str)
    for col, idx, default in (
        ("Batch", 0, np.nan),
        ("Pair", 1, np.nan),
        ("IdA", 2, ""),
        ("IdB", 3, ""),
    ):
        if col in resp_df.columns:
            existing = resp_df[col].tolist()
            resp_df[col] = [
                meta_map[ident][idx] if ident in meta_map else prev
                for ident, prev in zip(idents, existing)
            ]
        else:
            resp_df[col] = [
                meta_map[ident][idx] if ident in meta_map else default
                for ident in idents
            ]
    return resp_df


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _hash_text_identifier(value: Any, *, strict: bool) -> str | None:
    """Stable identifier for one passage; ``None`` when the value is missing."""
    if _is_missing_scalar(value):
        return None
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="ignore")
    elif isinstance(value, str):
        text = value
    elif strict:
        return None
    else:
        text = str(value)
    text = text.strip()
    if not text:
        return None
    return _common.hash_identifier(text)


async def _coerce_dict(raw: Any) -> dict[str, Any]:
    """Coerce a raw completion into a JSON object (unwrap one level)."""
    obj = await safest_json(raw)
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        obj2 = await safest_json(obj)
        if isinstance(obj2, dict):
            return obj2
    if isinstance(obj, list) and obj:
        inner = await safest_json(obj[0])
        if isinstance(inner, dict):
            return inner
    return {}


def _record_outcomes(
    safe_obj: dict[str, Any],
    batch_attrs: list[str],
    id_a: str,
    id_b: str,
    history_pairs: dict[str, list[tuple[str, str]]],
) -> None:
    """Translate one adjudication into pairwise outcomes in ``history_pairs``.

    ``circle``/``left``/``text a`` wins mean entry A won; ``square``/``right``/
    ``text b`` mean B won; ``draw`` and ``insufficient signal`` count for both
    directions (a tie contributes to both items).
    """
    batch_attr_map = {str(k).strip().lower(): k for k in batch_attrs}
    for attr_raw, winner_raw in safe_obj.items():
        attr_key_l = str(attr_raw).strip().lower()
        if attr_key_l not in batch_attr_map:
            continue
        real_attr = batch_attr_map[attr_key_l]
        val = winner_raw
        if isinstance(val, dict) and "winner" in val:
            val = val.get("winner")
        if isinstance(val, str):
            v = val.strip().lower()
        else:
            v = ""
        if v.startswith(("cir", "c", "left", "text a")):
            history_pairs[real_attr].append((id_a, id_b))
        elif v.startswith(("squ", "b", "right", "text b")):
            history_pairs[real_attr].append((id_b, id_a))
        elif v.startswith(("draw", "insufficient")):
            history_pairs[real_attr].append((id_a, id_b))
            history_pairs[real_attr].append((id_b, id_a))
        else:
            continue


@dataclass
class RankConfig:
    attributes: dict[str, str]
    save_dir: str = "rank"
    file_name: str = "rankings"
    model: str = "gpt-5.4-mini"
    decision_model: str | None = None
    n_parallels: int = 50
    n_rounds: int = 5
    matches_per_round: int = 3
    learning_rate: float = 0.1
    additional_instructions: str | None = None
    modality: str = "text"
    n_attributes_per_run: int | None = None
    circle_first: bool | None = None

    def __post_init__(self) -> None:
        if self.additional_instructions is not None:
            cleaned = str(self.additional_instructions).strip()
            self.additional_instructions = cleaned or None


class Rank(Task):
    """Compare items pairwise and output centered Bradley–Terry scores."""

    def __init__(self, cfg: RankConfig, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        self._template = load_prompt("rankings_prompt.jinja2")
        # random state; intentionally unseeded (reproducibility of pair order
        # and circle-first assignment is not guaranteed, matching GABRIEL).
        self.rng = random.Random()
        self._last_se_agg: dict[str, float] | None = None

        # Internal pairing/BT hyperparameters (mirrors GABRIEL; deliberately
        # not part of the public config).
        self._EXPLORE_FRAC = 0.2  # fraction of random pairings per round
        self._CANDIDATE_NEIGHBORS = 20  # neighbourhood size for info-gain pairing
        self._HIGH_SE_FRAC = 0.25  # fraction of high-uncertainty items
        self._MAX_ITER = 1000  # maximum iterations for BT optimisation
        self._TOL = 1e-6  # convergence tolerance for BT
        # Ridge stabilises the Fisher-information inversion; too small a
        # value (e.g. 1e-9) yields inflated SEs for sparsely compared items.
        self._SE_RIDGE = 1e-5
        # Cap on candidate pairs per round so info-gain pairing stays
        # tractable for very large item sets.
        self._MAX_CANDIDATE_PAIRS_PER_ROUND = 200_000

    # ------------------------------------------------------------------
    # Bradley–Terry bookkeeping
    # ------------------------------------------------------------------
    def _refit_all(
        self,
        attr_keys: list[str],
        item_ids: list[str],
        history_pairs: dict[str, list[tuple[str, str]]],
        ratings: dict[str, dict[str, float]],
        se_store: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """Refit every attribute on its full outcome history.

        Updates ``ratings`` and ``se_store`` in place and returns the
        per-item SE averaged across attributes (the pairing signal for the
        next round); items without comparisons get SE 1.0.
        """
        se_agg_next: dict[str, float] = {i: 0.0 for i in item_ids}
        se_agg_counts: dict[str, int] = {i: 0 for i in item_ids}
        for attr in attr_keys:
            outcomes = history_pairs[attr]
            if len(outcomes) == 0:
                continue
            bt_scores, n_ij, p_ij = fit_bt(
                item_ids=item_ids,
                outcomes=outcomes,
                pseudo=self.cfg.learning_rate,
                max_iter=self._MAX_ITER,
                tol=self._TOL,
                return_info=True,
            )
            for i in item_ids:
                ratings[i][attr] = bt_scores[i]
            s_vec = np.array([bt_scores[i] for i in item_ids])
            se_vec = bt_standard_errors(
                s=s_vec,
                n_ij=n_ij,
                p_ij=p_ij,
                ridge=self._SE_RIDGE,
            )
            for i, se_val in zip(item_ids, se_vec):
                se_store[attr][i] = float(se_val)
                se_agg_next[i] += float(se_val)
                se_agg_counts[i] += 1
        for i in item_ids:
            if se_agg_counts[i] > 0:
                se_agg_next[i] /= se_agg_counts[i]
            else:
                se_agg_next[i] = 1.0
        return se_agg_next

    @staticmethod
    def _center_ratings(
        attr_keys: list[str],
        item_ids: list[str],
        ratings: dict[str, dict[str, float]],
    ) -> None:
        """Centre each attribute's ratings to zero mean (identifiability)."""
        for attr in attr_keys:
            vals = [ratings[i][attr] for i in item_ids]
            mean_val = float(np.mean(vals))
            for i in item_ids:
                ratings[i][attr] -= mean_val

    # ------------------------------------------------------------------
    # Pairing strategy
    # ------------------------------------------------------------------
    def _pairs_info_gain(
        self,
        item_ids: list[str],
        texts_by_id: dict[str, str],
        current_ratings: dict[str, float],
        se_agg: dict[str, float],
        mpr: int,
    ) -> list[tuple[tuple[str, str], tuple[str, str]]]:
        """Select pairs by maximising expected information gain.

        A bounded candidate set is built from the current rating ranking
        (nearby neighbours), the items with the largest aggregated standard
        errors (random opponents), plus a small random exploration quota;
        quotas fill any shortfall.  Each pair is scored by
        ``outcome_var * parameter_uncertainty * closeness`` and the top
        scoring pairs are selected greedily so that every item participates
        in exactly ``mpr`` matches.
        """
        n = len(item_ids)
        if n < 2:
            return []
        max_pairs = max(1, self._MAX_CANDIDATE_PAIRS_PER_ROUND)
        desired_neighbors = max_pairs // max(1, n)
        candidate_neighbors = max(
            mpr, min(self._CANDIDATE_NEIGHBORS, desired_neighbors)
        )

        def logistic_clip(x: float) -> float:
            if x > 50:
                return 1.0
            if x < -50:
                return 0.0
            return 1.0 / (1.0 + np.exp(-x))

        ids_sorted = sorted(item_ids, key=lambda i: current_ratings[i])
        idx_of = {i_id: k for k, i_id in enumerate(ids_sorted)}
        num_high_se = max(1, int(self._HIGH_SE_FRAC * n))
        high_se_ids = sorted(item_ids, key=lambda i: se_agg.get(i, 1.0), reverse=True)[
            :num_high_se
        ]
        candidate_pairs_set: set[tuple[str, str]] = set()
        for i_id in item_ids:
            pos = idx_of[i_id]
            lower = max(0, pos - candidate_neighbors)
            upper = min(n, pos + candidate_neighbors + 1)
            for j in ids_sorted[lower:upper]:
                if i_id == j:
                    continue
                candidate_pairs_set.add((i_id, j) if i_id < j else (j, i_id))
        for hs in high_se_ids:
            others = [x for x in item_ids if x != hs]
            k = min(candidate_neighbors, len(others))
            samp = self.rng.sample(others, k)
            for j in samp:
                candidate_pairs_set.add((hs, j) if hs < j else (j, hs))
        remaining_capacity = max_pairs - len(candidate_pairs_set)
        n_random_targets = int(self._EXPLORE_FRAC * n * mpr)
        if remaining_capacity > 0:
            n_random_targets = min(n_random_targets, remaining_capacity)
            for _ in range(n_random_targets):
                if n < 2:
                    break
                a, b = self.rng.sample(item_ids, 2)
                candidate_pairs_set.add((a, b) if a < b else (b, a))
        partners_count = {i: 0 for i in item_ids}
        for a, b in candidate_pairs_set:
            partners_count[a] += 1
            partners_count[b] += 1
        for i_id in item_ids:
            while partners_count[i_id] < mpr:
                potential = [x for x in item_ids if x != i_id]
                if not potential:
                    break
                j = self.rng.choice(potential)
                pair = (i_id, j) if i_id < j else (j, i_id)
                if pair not in candidate_pairs_set:
                    candidate_pairs_set.add(pair)
                    partners_count[i_id] += 1
                    partners_count[j] += 1
                else:
                    partners_count[i_id] += 1
                    partners_count[j] += 1
        scored_pairs: list[tuple[float, str, str]] = []
        for a, b in candidate_pairs_set:
            diff = current_ratings[a] - current_ratings[b]
            p = logistic_clip(diff)
            outcome_var = p * (1 - p)
            var_a = se_agg.get(a, 1.0) ** 2
            var_b = se_agg.get(b, 1.0) ** 2
            param_unc = var_a + var_b
            # Encourage comparisons between similarly-rated items (high
            # information gain) while still prioritising uncertain pairs.
            closeness = 1.0 / (1.0 + abs(diff))
            score = outcome_var * param_unc * closeness
            scored_pairs.append((score, a, b))
        scored_pairs.sort(key=lambda x: x[0], reverse=True)
        needed: dict[str, int] = {i: mpr for i in item_ids}
        pairs_selected: list[tuple[str, str]] = []
        pairs_seen: set[tuple[str, str]] = set()
        for score, a, b in scored_pairs:
            if needed[a] > 0 and needed[b] > 0:
                tup = (a, b) if a < b else (b, a)
                if tup in pairs_seen:
                    continue
                pairs_selected.append(tup)
                pairs_seen.add(tup)
                needed[a] -= 1
                needed[b] -= 1
        while any(cnt > 0 for cnt in needed.values()):
            ids_needing = [i for i, cnt in needed.items() if cnt > 0]
            if not ids_needing:
                break
            a = self.rng.choice(ids_needing)
            potential = [x for x in item_ids if x != a]
            if not potential:
                break
            b = self.rng.choice(potential)
            tup = (a, b) if a < b else (b, a)
            pairs_selected.append(tup)
            needed[a] -= 1
            needed[b] -= 1
        return [((a, texts_by_id[a]), (b, texts_by_id[b])) for a, b in pairs_selected]

    def _generate_pairs(
        self,
        item_ids: list[str],
        texts_by_id: dict[str, str],
        current_ratings: dict[str, float] | None,
        se_agg: dict[str, float] | None,
    ) -> list[tuple[tuple[str, str], tuple[str, str]]]:
        """Dispatch to the pairing strategy (always info-gain)."""
        mpr = max(1, self.cfg.matches_per_round)
        if current_ratings is None:
            current_ratings = {i: 0.0 for i in item_ids}
        if se_agg is None or len(se_agg) != len(item_ids):
            se_full = {i: 1.0 for i in item_ids}
        else:
            se_full = se_agg
        return self._pairs_info_gain(
            item_ids, texts_by_id, current_ratings, se_full, mpr
        )

    # ------------------------------------------------------------------
    # Shared adjudication helper (normal rounds + catch-up)
    # ------------------------------------------------------------------
    async def _adjudicate_pairs(
        self,
        *,
        pairs: list[tuple[str, str]],
        texts_by_id: dict[str, str],
        attr_batches: list[list[str]],
        rnd: int,
        id_prefix: str,
        round_path: str,
        reset_files: bool,
        decision_client: DecisionClient | None,
        effective_model: str,
        history_pairs: dict[str, list[tuple[str, str]]],
        **kwargs: Any,
    ) -> None:
        """Adjudicate ``pairs`` across ``attr_batches``, checkpoint, and record outcomes."""
        from ...engine.batch import batch_decide, batch_execute

        if not pairs or not attr_batches:
            return
        ids: list[str] = []
        meta_map: dict[str, tuple[int, int, str, str]] = {}
        if decision_client is not None:
            states: list[str | dict[str, Any]] = []
            question_sets: list[dict[str, DecisionQuestion]] = []
            for batch_idx, batch in enumerate(attr_batches):
                attr_def_map = (
                    {a: self.cfg.attributes[a] for a in batch}
                    if isinstance(self.cfg.attributes, dict)
                    else {a: "" for a in batch}
                )
                for pair_idx, (id_a, id_b) in enumerate(pairs):
                    raw_ident = f"{id_prefix}{rnd}|{batch_idx}|{pair_idx}|{id_a}|{id_b}"
                    hashed_ident = _common.hash_identifier(raw_ident)
                    circle_first_flag = (
                        self.cfg.circle_first
                        if self.cfg.circle_first is not None
                        else self.rng.random() < 0.5
                    )
                    states.append(
                        _build_pair_state(
                            texts_by_id[id_a],
                            texts_by_id[id_b],
                            circle_first=circle_first_flag,
                            additional_instructions=self.cfg.additional_instructions,
                        )
                    )
                    question_sets.append(
                        {
                            attr: _build_rank_choice(
                                attr,
                                attr_def_map[attr],
                                modality=self.cfg.modality,
                                circle_first=circle_first_flag,
                            )
                            for attr in batch
                        }
                    )
                    ids.append(hashed_ident)
                    meta_map[hashed_ident] = (batch_idx, pair_idx, id_a, id_b)
            resp_df = await batch_decide(
                states=states,
                question_sets=question_sets,
                identifiers=ids,
                client=decision_client,
                model=effective_model,
                n_parallels=self.cfg.n_parallels,
                save_path=round_path,
                reset_files=reset_files,
                **kwargs,
            )
        else:
            prompts: list[str] = []
            for batch_idx, batch in enumerate(attr_batches):
                attr_def_map = (
                    {a: self.cfg.attributes[a] for a in batch}
                    if isinstance(self.cfg.attributes, dict)
                    else {a: "" for a in batch}
                )
                for pair_idx, (id_a, id_b) in enumerate(pairs):
                    raw_ident = f"{id_prefix}{rnd}|{batch_idx}|{pair_idx}|{id_a}|{id_b}"
                    hashed_ident = _common.hash_identifier(raw_ident)
                    circle_first_flag = (
                        self.cfg.circle_first
                        if self.cfg.circle_first is not None
                        else self.rng.random() < 0.5
                    )
                    prompts.append(
                        self._template.render(
                            entry_circle=texts_by_id[id_a],
                            entry_square=texts_by_id[id_b],
                            attributes=attr_def_map,
                            additional_instructions=self.cfg.additional_instructions
                            or "",
                            modality=self.cfg.modality,
                            circle_first=circle_first_flag,
                        )
                    )
                    ids.append(hashed_ident)
                    meta_map[hashed_ident] = (batch_idx, pair_idx, id_a, id_b)
            resp_df = await batch_execute(
                prompts=prompts,
                identifiers=ids,
                client=self._get_client(),
                model=self.cfg.model,
                json_mode=self.cfg.modality != "audio",
                n_parallels=self.cfg.n_parallels,
                save_path=round_path,
                reset_files=reset_files,
                **kwargs,
            )

        resp_df = _attach_round_metadata(resp_df, meta_map)
        resp_df.to_csv(round_path, index=False)

        for ident, resp in zip(resp_df.Identifier, resp_df.Response):
            meta = meta_map.get(str(ident))
            if not meta:
                continue
            batch_idx, _, id_a, id_b = meta
            safe_obj = await _decode_round_outcomes(resp, attr_batches[batch_idx])
            if not safe_obj:
                continue
            _record_outcomes(
                safe_obj, attr_batches[batch_idx], id_a, id_b, history_pairs
            )

    # ------------------------------------------------------------------
    # Resume: catch up new items on already-completed rounds
    # ------------------------------------------------------------------
    async def _catch_up_existing_rounds(
        self,
        new_ids: list[str],
        round_indices: list[int],
        item_ids: list[str],
        texts_by_id: dict[str, str],
        attr_batches: list[list[str]],
        attr_keys: list[str],
        history_pairs: dict[str, list[tuple[str, str]]],
        ratings: dict[str, dict[str, float]],
        se_store: dict[str, dict[str, float]],
        base_name: str,
        _write_checkpoint: Callable[[], None],
        reset_files: bool,
        decision_client: DecisionClient | None,
        effective_model: str,
        **kwargs: Any,
    ) -> None:
        """Give ``new_ids`` the matches they missed in already-completed rounds."""
        if not new_ids:
            return
        for rnd in round_indices:
            round_path = os.path.join(self.cfg.save_dir, f"{base_name}_round{rnd}.csv")
            if not os.path.exists(round_path):
                continue
            try:
                df_round = pd.read_csv(round_path)
            except (OSError, ValueError):
                continue
            counts: dict[str, int] = {}
            if {"IdA", "IdB"}.issubset(df_round.columns):
                for a, b in zip(df_round["IdA"], df_round["IdB"]):
                    counts[str(a)] = counts.get(str(a), 0) + 1
                    counts[str(b)] = counts.get(str(b), 0) + 1
            else:
                for ident in df_round.get("Identifier", []):
                    parts = str(ident).split("|")
                    if len(parts) != 5:
                        continue
                    _, _, _, id_a, id_b = parts
                    counts[id_a] = counts.get(id_a, 0) + 1
                    counts[id_b] = counts.get(id_b, 0) + 1
            pairs_needed: list[tuple[str, str]] = []
            for nid in new_ids:
                needed = self.cfg.matches_per_round - counts.get(nid, 0)
                if needed <= 0:
                    continue
                opponents = [i for i in item_ids if i != nid]
                self.rng.shuffle(opponents)
                for opp in opponents[:needed]:
                    pairs_needed.append((nid, opp))
            if not pairs_needed:
                continue
            await self._adjudicate_pairs(
                pairs=pairs_needed,
                texts_by_id=texts_by_id,
                attr_batches=attr_batches,
                rnd=rnd,
                id_prefix="catchup|",
                round_path=round_path,
                reset_files=reset_files,
                decision_client=decision_client,
                effective_model=effective_model,
                history_pairs=history_pairs,
                **kwargs,
            )
            se_agg_next = self._refit_all(
                attr_keys, item_ids, history_pairs, ratings, se_store
            )
            self._last_se_agg = se_agg_next
            self._center_ratings(attr_keys, item_ids, ratings)
            _write_checkpoint()

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------
    async def run(
        self,
        df: pd.DataFrame,
        column_name: str,
        *,
        reset_files: bool = False,
        n_runs: int | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Execute the ranking procedure.

        Parameters
        ----------
        df:
            Input DataFrame containing the passages to be ranked.
        column_name:
            Name of the column in ``df`` that holds the text for each passage.
        reset_files:
            If ``True``, ignore saved results and recompute; otherwise the
            final output file is returned immediately when complete, or the
            run resumes from the last completed round.
        n_runs:
            Deprecated/ignored parameter provided for compatibility with
            :class:`Rate`; ``n_rounds`` controls the number of iterations.
        **kwargs:
            Additional keyword arguments forwarded to :func:`batch_execute`.

        Returns
        -------
        pandas.DataFrame
            One row per input passage.  For each attribute: ``"<attribute>"``
            (z-score), ``"<attribute>_raw"`` (centred Bradley–Terry estimate)
            and ``"<attribute>_se"`` (standard error).  Also written to
            ``save_dir``.
        """
        base_name = self._base_name()
        run_metadata = _common.load_run_metadata(
            self.cfg.save_dir, base_name, reset_files=reset_files
        )
        self.cfg.attributes = _common.load_persisted_attributes(
            save_dir=self.cfg.save_dir,
            incoming=self.cfg.attributes,
            reset_files=reset_files,
            task_name="Rank",
            item_name="attributes",
            legacy_filename=f"{base_name}_attrs.json",
        )

        final_path = os.path.join(self.cfg.save_dir, f"{base_name}_final.csv")
        if n_runs is not None:
            print(
                "Parameter 'n_runs' is ignored. Use 'n_rounds' to control the "
                f"number of iterations. Current n_rounds={self.cfg.n_rounds}."
            )

        df_proc = df.reset_index(drop=True).copy()
        strict_text_mode = self.cfg.modality in {"text", "entity", "web"}
        hashed_ids = df_proc[column_name].map(
            lambda x: _hash_text_identifier(x, strict=strict_text_mode)
        )
        valid_mask = hashed_ids.notna()
        dropped = int((~valid_mask).sum())
        if dropped > 0:
            total = len(df_proc)
            pct = (dropped / total * 100.0) if total else 0.0
            print(
                f"[Rank] Dropping {dropped}/{total} rows ({pct:.1f}%) with "
                f"malformed '{column_name}' values before ranking."
            )
        df_proc = df_proc.loc[valid_mask].copy().reset_index(drop=True)
        df_proc["_id"] = hashed_ids.loc[valid_mask].astype(str).reset_index(drop=True)

        if df_proc.empty:
            if isinstance(self.cfg.attributes, dict):
                attr_keys_empty = list(self.cfg.attributes.keys())
            else:
                attr_keys_empty = list(self.cfg.attributes)
            df_out = df_proc[[c for c in df.columns if c in df_proc.columns]].copy()
            for attr in attr_keys_empty:
                df_out[attr] = pd.Series(dtype="float64")
                df_out[f"{attr}_raw"] = pd.Series(dtype="float64")
                df_out[f"{attr}_se"] = pd.Series(dtype="float64")
            save_dataframe_with_fallback(df_out, final_path, index=False, label="Rank")
            return df_out

        # Infer progress from ``<base_name>_round<k>.csv`` files.  If the
        # final checkpoint exists for the last round and covers all items,
        # reuse it; otherwise resume from the next incomplete round.
        start_round = 0
        existing_rounds: list[int] = []
        if not reset_files:
            try:
                for fname in os.listdir(self.cfg.save_dir):
                    if fname.startswith(f"{base_name}_round") and fname.endswith(
                        ".csv"
                    ):
                        try:
                            idx_str = fname[len(base_name) + 6 : -4]
                            rnd_idx = int(idx_str)
                        except ValueError:
                            continue
                        if _is_completed_round_csv(
                            os.path.join(self.cfg.save_dir, fname)
                        ):
                            existing_rounds.append(rnd_idx)
            except OSError:
                existing_rounds = []
        if existing_rounds:
            last_completed = max(existing_rounds)
            if os.path.exists(final_path):
                try:
                    final_df = pd.read_csv(final_path)
                    identifier_col = column_name
                    if identifier_col not in final_df.columns:
                        raise ValueError(
                            "Existing ranking output is missing identifier "
                            f"column '{identifier_col}'."
                        )
                    final_ids = set(
                        final_df[identifier_col]
                        .map(lambda x: _hash_text_identifier(x, strict=True))
                        .dropna()
                        .astype(str)
                    )
                    if (
                        last_completed >= self.cfg.n_rounds - 1
                        and set(df_proc["_id"]) <= final_ids
                    ):
                        return final_df
                except (OSError, ValueError):
                    pass
            start_round = last_completed + 1

        texts = list(zip(df_proc["_id"], df_proc[column_name].astype(str)))
        texts_by_id = {i: t for i, t in texts}
        item_ids = [i for i, _ in texts]

        if isinstance(self.cfg.attributes, dict):
            attr_keys = list(self.cfg.attributes.keys())
        else:
            attr_keys = list(self.cfg.attributes)

        ratings: dict[str, dict[str, float]] = {
            i: {a: 0.0 for a in attr_keys} for i in item_ids
        }
        history_pairs: dict[str, list[tuple[str, str]]] = {a: [] for a in attr_keys}
        se_store: dict[str, dict[str, float]] = {
            a: {i: np.nan for i in item_ids} for a in attr_keys
        }

        decision_client = self._get_decision_client()
        effective_model = (
            resolve_decision_model(self.cfg.decision_model, fallback=self.cfg.model)
            or self.cfg.model
            if decision_client is not None
            else self.cfg.model
        )
        if self.cfg.n_attributes_per_run is not None and (
            not isinstance(self.cfg.n_attributes_per_run, int)
            or self.cfg.n_attributes_per_run < 1
        ):
            raise ValueError("n_attributes_per_run must be None or an integer >= 1")
        if decision_client is not None:
            provider_cap = max(
                1, int(getattr(decision_client, "max_questions_per_call", 64))
            )
            requested_batch_size: int | None = (
                min(self.cfg.n_attributes_per_run, provider_cap)
                if self.cfg.n_attributes_per_run is not None
                else provider_cap
            )
        else:
            requested_batch_size = self.cfg.n_attributes_per_run

        # Attribute batches, defined once so replay lines up with the cache.
        attr_items = [(attr, attr) for attr in attr_keys]
        attr_batch_items, effective_n_attributes_per_run = (
            _common.resolve_attribute_batches(
                attr_items,
                requested_n=requested_batch_size,
                metadata=run_metadata,
                reset_files=reset_files,
            )
        )
        attr_batches: list[list[str]] = [
            [name for name, _ in batch] for batch in attr_batch_items
        ]
        attr_count = len(attr_keys)
        if (
            effective_n_attributes_per_run is not None
            and attr_count > effective_n_attributes_per_run
        ):
            batches = (
                attr_count + effective_n_attributes_per_run - 1
            ) // effective_n_attributes_per_run
            print(
                f"[Rank] {attr_count} attributes provided. "
                f"n_attributes_per_run={effective_n_attributes_per_run}. "
                f"Splitting into {batches} prompt batches; set "
                "n_attributes_per_run=None to process them together."
            )
        _common.update_run_metadata(
            self.cfg.save_dir,
            base_name,
            model=effective_model,
            n_attributes_per_run=effective_n_attributes_per_run,
            attribute_batches=[[name for name, _ in b] for b in attr_batch_items],
        )

        # Write the current ratings/se/z-scores to the final CSV.  The z-score
        # normalises the centred BT scores across items for each attribute.
        def _write_checkpoint() -> None:
            zscores_local: dict[str, dict[str, float]] = {}
            for attr in attr_keys:
                vals = np.array([ratings[i][attr] for i in item_ids])
                mean = vals.mean()
                std = vals.std(ddof=0)
                if std == 0:
                    zscores_local[attr] = {i: 0.0 for i in item_ids}
                else:
                    zscores_local[attr] = {
                        i: float((ratings[i][attr] - mean) / std) for i in item_ids
                    }
            for attr in attr_keys:
                val_map = {i: ratings[i][attr] for i in item_ids}
                df_proc[f"{attr}_raw"] = df_proc["_id"].map(val_map)
                se_map = {i: se_store[attr].get(i, np.nan) for i in item_ids}
                df_proc[f"{attr}_se"] = df_proc["_id"].map(se_map)
                z_map = zscores_local.get(attr, {i: np.nan for i in item_ids})
                df_proc[attr] = df_proc["_id"].map(z_map)

            # Original user columns first, then per attribute: z-score, raw, se.
            original_cols = [c for c in df.columns]
            new_cols: list[str] = []
            for attr in attr_keys:
                new_cols.append(attr)
                new_cols.append(f"{attr}_raw")
                new_cols.append(f"{attr}_se")
            final_cols = original_cols + new_cols
            final_cols = [c for c in final_cols if c in df_proc.columns]
            df_out_local = df_proc[final_cols].copy()
            save_dataframe_with_fallback(
                df_out_local, final_path, index=False, label="Rank"
            )

        # Replay completed rounds to reconstruct ratings and uncertainties.
        if start_round > 0:
            for replay_rnd in range(start_round):
                round_path = os.path.join(
                    self.cfg.save_dir, f"{base_name}_round{replay_rnd}.csv"
                )
                if not os.path.exists(round_path):
                    break
                try:
                    df_round = pd.read_csv(round_path)
                    df_round["Response"] = df_round["Response"].apply(
                        lambda x: None if pd.isna(x) else x
                    )
                except (OSError, ValueError):
                    continue
                if {"Batch", "IdA", "IdB"}.issubset(df_round.columns):
                    for batch_idx_raw, id_a, id_b, resp_raw in zip(
                        df_round["Batch"],
                        df_round["IdA"],
                        df_round["IdB"],
                        df_round["Response"],
                    ):
                        if pd.isna(batch_idx_raw):
                            continue
                        try:
                            batch_idx = int(batch_idx_raw)
                        except (TypeError, ValueError):
                            continue
                        if batch_idx < 0 or batch_idx >= len(attr_batches):
                            continue
                        safe_obj = await _decode_round_outcomes(
                            resp_raw, attr_batches[batch_idx]
                        )
                        if not safe_obj:
                            continue
                        _record_outcomes(
                            safe_obj,
                            attr_batches[batch_idx],
                            str(id_a),
                            str(id_b),
                            history_pairs,
                        )
                else:
                    for ident, resp_raw in zip(
                        df_round["Identifier"], df_round["Response"]
                    ):
                        parts = str(ident).split("|")
                        if len(parts) != 5:
                            continue
                        _, batch_idx_str, _, id_a, id_b = parts
                        try:
                            batch_idx = int(batch_idx_str)
                        except (TypeError, ValueError):
                            continue
                        if batch_idx < 0 or batch_idx >= len(attr_batches):
                            continue
                        safe_obj = await _decode_round_outcomes(
                            resp_raw, attr_batches[batch_idx]
                        )
                        if not safe_obj:
                            continue
                        _record_outcomes(
                            safe_obj, attr_batches[batch_idx], id_a, id_b, history_pairs
                        )
                se_agg_next = self._refit_all(
                    attr_keys, item_ids, history_pairs, ratings, se_store
                )
                self._last_se_agg = se_agg_next
                self._center_ratings(attr_keys, item_ids, ratings)
                _write_checkpoint()

        # New items added on resume need to catch up on completed rounds.
        seen_ids: set[str] = set()
        for pair_list in history_pairs.values():
            for a, b in pair_list:
                seen_ids.add(a)
                seen_ids.add(b)
        new_ids = [i for i in item_ids if i not in seen_ids]
        await self._catch_up_existing_rounds(
            new_ids=new_ids,
            round_indices=list(range(start_round)),
            item_ids=item_ids,
            texts_by_id=texts_by_id,
            attr_batches=attr_batches,
            attr_keys=attr_keys,
            history_pairs=history_pairs,
            ratings=ratings,
            se_store=se_store,
            base_name=base_name,
            _write_checkpoint=_write_checkpoint,
            reset_files=reset_files,
            decision_client=decision_client,
            effective_model=effective_model,
            **kwargs,
        )

        # New rounds.
        for rnd in range(start_round, self.cfg.n_rounds):
            # aggregate current ratings across attributes for pairing
            current_agg = {
                i: float(np.mean(list(ratings[i].values()))) for i in item_ids
            }
            se_agg_local = self._last_se_agg
            use_current = rnd > 0 or start_round > 0
            se_source = (
                se_agg_local
                if (rnd > 0 or start_round > 0 or se_agg_local is not None)
                else None
            )
            pairs = self._generate_pairs(
                item_ids=item_ids,
                texts_by_id=texts_by_id,
                current_ratings=current_agg if use_current else None,
                se_agg=se_source,
            )
            if not pairs:
                break
            round_path = os.path.join(self.cfg.save_dir, f"{base_name}_round{rnd}.csv")
            await self._adjudicate_pairs(
                pairs=[(id_a, id_b) for (id_a, _), (id_b, _) in pairs],
                texts_by_id=texts_by_id,
                attr_batches=attr_batches,
                rnd=rnd,
                id_prefix="",
                round_path=round_path,
                reset_files=reset_files,
                decision_client=decision_client,
                effective_model=effective_model,
                history_pairs=history_pairs,
                **kwargs,
            )

            # refit on the full history, then centre and checkpoint
            se_agg_next = self._refit_all(
                attr_keys, item_ids, history_pairs, ratings, se_store
            )
            self._last_se_agg = se_agg_next
            self._center_ratings(attr_keys, item_ids, ratings)
            _write_checkpoint()

        return pd.read_csv(final_path)
