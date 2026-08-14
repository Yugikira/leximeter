"""Public facade.

Each public function constructs a task config and delegates to a task.
Parameters are explicit; provider selection happens via ``client=`` /
``embedding_client=`` / ``search_tool=`` (or environment defaults).  Extra
keyword arguments are forwarded to the engine / model call (e.g.
``max_output_tokens``, ``temperature``, ``timeout``, ``max_retries``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from .client.base import LLMClient
from .client.embedding import EmbeddingClient
from .client.search import SearchTool
from .tasks.rate import Rate, RateConfig


async def rate(
    df: pd.DataFrame,
    column_name: str,
    *,
    attributes: Dict[str, str],
    save_dir: str,
    model: str = "gpt-5.4-mini",
    n_runs: int = 1,
    n_parallels: int = 50,
    n_attributes_per_run: Optional[int] = None,
    rating_scale: Optional[str] = None,
    additional_instructions: Optional[str] = None,
    modality: str = "text",
    file_name: str = "ratings.csv",
    reset_files: bool = False,
    client: Optional[LLMClient] = None,
    embedding_client: Optional[EmbeddingClient] = None,
    search_tool: Optional[SearchTool] = None,
    **engine_kwargs: Any,
) -> pd.DataFrame:
    """Score each item 0-100 on each natural-language attribute.

    Returns ``df`` with one column per attribute (the mean across ``n_runs``).
    """
    cfg = RateConfig(
        attributes=attributes,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_runs=n_runs,
        n_parallels=n_parallels,
        n_attributes_per_run=n_attributes_per_run,
        rating_scale=rating_scale,
        additional_instructions=additional_instructions,
        modality=modality,
    )
    task = Rate(
        cfg,
        client=client,
        embedding_client=embedding_client,
        search_tool=search_tool,
    )
    return await task.run(df, column_name, reset_files=reset_files, **engine_kwargs)
