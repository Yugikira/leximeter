"""Load: prepare a folder of text files into a DataFrame."""

from __future__ import annotations

import os

import pandas as pd

_TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".tsv", ".log"}


def load(
    folder_path: str,
    *,
    save_name: str | None = None,
    modality: str = "text",
    reset_files: bool = False,
) -> pd.DataFrame:
    """Read every text file in ``folder_path`` into a DataFrame.

    Columns: ``path``, ``name``, ``text``.
    """
    if modality != "text":
        raise ValueError("leximeter is text-only; only modality='text' is supported")

    folder = os.path.expandvars(os.path.expanduser(folder_path))
    rows = []
    for entry in sorted(os.listdir(folder)):
        path = os.path.join(folder, entry)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(entry)[1].lower()
        if ext not in _TEXT_EXTENSIONS:
            continue
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except (OSError, ValueError):
            continue
        rows.append({"path": path, "name": entry, "text": content})

    out = pd.DataFrame(rows)
    if save_name:
        out_path = os.path.join(folder, save_name)
        if reset_files or not os.path.exists(out_path):
            try:
                out.to_csv(out_path, index=False)
            except OSError:
                pass
    return out
