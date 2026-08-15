"""Checkpoint CSV read/write for resumable batch execution."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

COLUMNS = ["Identifier", "Response", "Successful", "Error Log", "Time Taken"]

_TRUE_VALUES = {"true", "1", "yes", "y", "completed", "succeeded", "success"}


def load_checkpoint(path: str) -> pd.DataFrame:
    """Load a checkpoint, keeping the last row per identifier."""
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    df = df.drop_duplicates(subset=["Identifier"], keep="last")
    return df.reset_index(drop=True)


def completed_identifiers(checkpoint: pd.DataFrame) -> set:
    """Return identifiers that completed successfully in ``checkpoint``."""
    if checkpoint.empty or "Successful" not in checkpoint.columns:
        return set()
    successful = checkpoint["Successful"].astype(str).str.strip().str.lower()
    mask = successful.isin(_TRUE_VALUES)
    return set(checkpoint.loc[mask, "Identifier"].astype(str))


def append_rows(path: str, rows: list[dict[str, Any]]) -> None:
    """Append ``rows`` to the checkpoint CSV, creating it if needed."""
    if not rows:
        return
    frame = pd.DataFrame(rows, columns=COLUMNS)
    header = not os.path.exists(path)
    frame.to_csv(path, mode="a", header=header, index=False)


def merge_with_checkpoint(checkpoint: pd.DataFrame, rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Return the checkpoint plus new rows as a single frame."""
    new_frame = pd.DataFrame(rows, columns=COLUMNS)
    if checkpoint.empty:
        return new_frame
    return pd.concat([checkpoint, new_frame], ignore_index=True)
