"""JSON parsing helpers for model output.

Ported from GABRIEL's ``utils/parsing.py`` (heuristic chain, ``{}``-on-failure
``safe_json``, LLM-repair ``safest_json``) and adapted to the provider-agnostic
architecture: ``safest_json`` / ``clean_json_df`` take an optional injected
``client`` instead of importing a global function.
"""

from __future__ import annotations

import ast
import json
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from ..client.base import LLMClient

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)

# model used when an LLM is required to reformat malformed JSON
JSON_LLM_MODEL = os.getenv("JSON_LLM_MODEL", "gpt-5.4-mini")


def _parse_json(txt: Any) -> Union[dict, list]:
    """Strict JSON parsing with common cleaning heuristics."""
    if isinstance(txt, dict):
        return txt

    if isinstance(txt, list):
        if len(txt) == 1:
            return _parse_json(txt[0])
        return txt

    if isinstance(txt, (bytes, bytearray)):
        txt = txt.decode(errors="ignore")

    if txt is None:
        raise ValueError("None provided")

    cleaned = str(txt).strip()

    if (cleaned.startswith('"') and cleaned.endswith('"')) or (
        cleaned.startswith("'") and cleaned.endswith("'")
    ):
        cleaned = cleaned[1:-1].strip()

    m = _JSON_FENCE_RE.search(cleaned)
    if m:
        cleaned = m.group(1).strip()

    for parser in (json.loads, ast.literal_eval):
        try:
            out = parser(cleaned)
            if isinstance(out, (dict, list)):
                return out
        except Exception:
            pass

    # attempt to strip `//` and `/* */` style comments before parsing
    try:
        no_line_comments = re.sub(r"(?<!:)//.*$", "", cleaned, flags=re.MULTILINE)
        no_comments = re.sub(r"/\*.*?\*/", "", no_line_comments, flags=re.S)
        out = json.loads(no_comments)
        if isinstance(out, (dict, list)):
            return out
    except Exception:
        pass

    brace = re.search(r"\{[\s\S]*\}", cleaned)
    if brace:
        try:
            out = json.loads(brace.group(0))
            if isinstance(out, (dict, list)):
                return out
        except Exception:
            pass

    bracket = re.search(r"\[[\s\S]*\]", cleaned)
    if bracket:
        candidate = bracket.group(0).strip()
        try:
            out = json.loads(candidate)
            if isinstance(out, (dict, list)):
                return out
        except Exception:
            pass

        m = re.fullmatch(r"\[\s*(['\"])(.*)\1\s*\]", candidate, re.S)
        if m:
            inner = m.group(2).strip()
            try:
                out = json.loads(inner)
                if isinstance(out, (dict, list)):
                    return out
            except Exception:
                inner_bracket = re.search(r"\[[\s\S]*\]", inner)
                if inner_bracket:
                    try:
                        out = json.loads(inner_bracket.group(0))
                        if isinstance(out, (dict, list)):
                            return out
                    except Exception:
                        pass

    start_brace = cleaned.find("{")
    end_brace = cleaned.rfind("}")
    if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
        try:
            out = json.loads(cleaned[start_brace : end_brace + 1])
            if isinstance(out, (dict, list)):
                return out
        except Exception:
            pass

    start_bracket = cleaned.find("[")
    end_bracket = cleaned.rfind("]")
    if start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket:
        try:
            out = json.loads(cleaned[start_bracket : end_bracket + 1])
            if isinstance(out, (dict, list)):
                return out
        except Exception:
            pass

    raise ValueError(f"Failed to parse JSON: {cleaned[:200]}")


def safe_json(txt: Any) -> Union[dict, list]:
    """Best-effort JSON parser returning ``{}`` on failure.

    Runs synchronously and never uses the LLM; it applies a number of local
    heuristics to coerce ``txt`` into a JSON object or list.
    """

    try:
        return _parse_json(txt)
    except Exception:
        return {}


def parse_json_with_status(txt: Any) -> Tuple[Optional[Union[dict, list]], bool]:
    """Parse JSON and report success separately from the parsed value."""

    try:
        return _parse_json(txt), True
    except Exception:
        return None, False


async def safest_json(
    txt: Any,
    *,
    model: Optional[str] = None,
    use_llm_fallback: bool = False,
    llm_timeout: Optional[float] = 60.0,
    client: Optional[LLMClient] = None,
) -> Union[dict, list, Any]:
    """Parse JSON with an optional LLM-based repair step.

    The function first attempts to parse ``txt`` locally using ``_parse_json``.
    When ``use_llm_fallback`` is ``False`` (the default) any parsing failure
    results in ``None``.  When ``use_llm_fallback`` is ``True`` an extra
    completion call asks the model to reformat the text as valid JSON (a pure
    formatting task).  ``client`` may be injected for tests; otherwise the
    environment-resolved client is used.  ``llm_timeout`` prevents the repair
    step from hanging indefinitely.
    """

    try:
        return _parse_json(txt)
    except Exception:
        if not use_llm_fallback:
            return None

        if model is None:
            model = JSON_LLM_MODEL

        if client is None:
            from ..client.resolve import resolve_client

            client = resolve_client()

        fixed = await client.complete(
            prompt=(
                "Please parse the following text **without changing any content** "
                "into valid JSON. This is a pure formatting task.\n\n" + str(txt)
            ),
            model=model,
            json_mode=True,
            timeout=llm_timeout,
        )
        if fixed:
            try:
                return _parse_json(fixed)
            except Exception:
                return None
        return None


async def clean_json_df(
    df: pd.DataFrame,
    columns: List[str],
    *,
    id_col: str,
    model: str = "gpt-5.4-mini",
    exclude_valid_json: bool = False,
    save_path: Optional[str] = None,
    client: Optional[LLMClient] = None,
    **engine_kwargs: Any,
) -> pd.DataFrame:
    """Ensure specified DataFrame columns contain valid JSON.

    Parameters
    ----------
    df:
        Input DataFrame whose columns may contain malformed JSON strings.
    columns:
        Names of columns to inspect and clean.
    id_col:
        Name of a **unique** column in ``df`` used to merge cleaned JSON
        responses back into the original DataFrame. A :class:`ValueError` is
        raised if the column is missing or contains duplicate values.
    model:
        Model name used to repair invalid JSON. Defaults to ``"gpt-5.4-mini"``.
    exclude_valid_json:
        When ``False`` (default), only entries that fail to parse are sent to
        the model. When ``True``, all entries are processed regardless of
        validity.
    save_path:
        Optional path for the raw-responses checkpoint CSV. When ``None``
        (default), a temporary file is used and deleted afterwards.
    client:
        Optional injected LLM client (tests); defaults to the
        environment-resolved client.
    **engine_kwargs:
        Forwarded to :func:`leximeter.engine.batch.batch_execute`.

    Returns
    -------
    DataFrame with new ``<column>_cleaned`` columns containing the cleaned
    JSON structures. Rows that were already valid retain their original value.
    """

    from ..engine.batch import batch_execute

    df = df.copy()

    if id_col not in df.columns:
        raise ValueError(f"Column '{id_col}' not found in DataFrame")
    if not df[id_col].is_unique:
        raise ValueError(f"Column '{id_col}' must contain unique values")

    prompts: List[str] = []
    identifiers: List[str] = []
    # ``mapping`` maps each identifier to its originating column and ``id_col``
    # value so responses can be merged back using a stable key.
    mapping: Dict[str, Tuple[str, Any]] = {}

    for col in columns:
        cleaned_col = f"{col}_cleaned"
        df[cleaned_col] = None
        col_idx = df.columns.get_loc(cleaned_col)
        for _, (id_val, val) in enumerate(zip(df[id_col], df[col])):
            valid = True
            try:
                _parse_json(val)
            except Exception:
                valid = False
            if exclude_valid_json or not valid:
                prompt = (
                    "Please parse the following text **without changing any content** "
                    "into valid JSON. This is a pure formatting task.\n\n" + str(val)
                )
                ident = f"{id_val}__{col}"
                prompts.append(prompt)
                identifiers.append(ident)
                mapping[ident] = (col, id_val)
            else:
                df.iat[row_pos, col_idx] = val

    if prompts:
        cleanup = False
        if save_path is None:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".csv")
            os.close(tmp_fd)
            os.remove(tmp_path)
            cleanup = True
        else:
            tmp_path = save_path
        if client is None:
            from ..client.resolve import resolve_client

            client = resolve_client()
        try:
            resp_df = await batch_execute(
                prompts=prompts,
                identifiers=identifiers,
                client=client,
                model=model,
                json_mode=True,
                save_path=tmp_path,
                reset_files=True,
                **engine_kwargs,
            )
        finally:
            if cleanup:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        for _, row in resp_df.iterrows():
            ident = str(row.get("Identifier", "")).strip()
            if ident not in mapping:
                continue
            col, id_val = mapping[ident]
            col_idx = df.columns.get_loc(f"{col}_cleaned")
            df.loc[df[id_col] == id_val, df.columns[col_idx]] = row["Response"]

    return df


__all__ = [
    "JSON_LLM_MODEL",
    "_parse_json",
    "safe_json",
    "parse_json_with_status",
    "safest_json",
    "clean_json_df",
]
