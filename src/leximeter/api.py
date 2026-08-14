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
from .tasks.bucket import Bucket, BucketConfig
from .tasks.classify import Classify, ClassifyConfig
from .tasks.codify import Codify, CodifyConfig
from .tasks.compare import Compare, CompareConfig
from .tasks.deduplicate import Deduplicate, DeduplicateConfig
from .tasks.deidentify import Deidentify, DeidentifyConfig
from .tasks.extract import Extract, ExtractConfig
from .tasks.filter import Filter, FilterConfig
from .tasks.merge import Merge, MergeConfig
from .tasks.paraphrase import Paraphrase, ParaphraseConfig
from .tasks.rate import Rate, RateConfig
from .tasks.seed import Seed, SeedConfig
from .tasks.whatever import Whatever, WhateverConfig


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


async def classify(
    df: pd.DataFrame,
    column_name: str,
    *,
    labels: Dict[str, str],
    save_dir: str,
    model: str = "gpt-5.4-mini",
    n_runs: int = 1,
    n_parallels: int = 50,
    min_frequency: float = 0.6,
    n_attributes_per_run: Optional[int] = None,
    additional_instructions: Optional[str] = None,
    modality: str = "text",
    file_name: str = "classify_responses.csv",
    reset_files: bool = False,
    client: Optional[LLMClient] = None,
    embedding_client: Optional[EmbeddingClient] = None,
    search_tool: Optional[SearchTool] = None,
    **engine_kwargs: Any,
) -> pd.DataFrame:
    """Decide whether each label applies to each item.

    Returns ``df`` with one boolean column per label (True when the proportion
    of runs marking the label True is >= ``min_frequency``).
    """
    cfg = ClassifyConfig(
        labels=labels,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_runs=n_runs,
        n_parallels=n_parallels,
        min_frequency=min_frequency,
        n_attributes_per_run=n_attributes_per_run,
        additional_instructions=additional_instructions,
        modality=modality,
    )
    task = Classify(
        cfg,
        client=client,
        embedding_client=embedding_client,
        search_tool=search_tool,
    )
    return await task.run(df, column_name, reset_files=reset_files, **engine_kwargs)


async def extract(
    df: pd.DataFrame,
    column_name: str,
    *,
    attributes: Dict[str, str],
    save_dir: str,
    model: str = "gpt-5.4-mini",
    n_runs: int = 1,
    n_parallels: int = 50,
    n_attributes_per_run: Optional[int] = None,
    additional_instructions: Optional[str] = None,
    modality: str = "entity",
    types: Optional[Dict[str, Any]] = None,
    file_name: str = "extraction.csv",
    reset_files: bool = False,
    client: Optional[LLMClient] = None,
    embedding_client: Optional[EmbeddingClient] = None,
    search_tool: Optional[SearchTool] = None,
    **engine_kwargs: Any,
) -> pd.DataFrame:
    """Extract structured facts per item.

    Returns ``df`` with one column per requested attribute (unknown → ``NA``).
    """
    cfg = ExtractConfig(
        attributes=attributes,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_runs=n_runs,
        n_parallels=n_parallels,
        n_attributes_per_run=n_attributes_per_run,
        additional_instructions=additional_instructions,
        modality=modality,
        types=types,
    )
    task = Extract(
        cfg,
        client=client,
        embedding_client=embedding_client,
        search_tool=search_tool,
    )
    return await task.run(df, column_name, reset_files=reset_files, **engine_kwargs)


async def filter(
    df: pd.DataFrame,
    column_name: str,
    *,
    condition: str,
    save_dir: str,
    model: str = "gpt-5.4-nano",
    n_runs: int = 1,
    n_parallels: int = 50,
    entities_per_call: int = 150,
    shuffle: bool = True,
    random_seed: int = 42,
    threshold: float = 0.5,
    additional_instructions: Optional[str] = None,
    file_name: str = "filter_responses.csv",
    reset_files: bool = False,
    client: Optional[LLMClient] = None,
    embedding_client: Optional[EmbeddingClient] = None,
    search_tool: Optional[SearchTool] = None,
    **engine_kwargs: Any,
) -> pd.DataFrame:
    """Keep items whose natural-language ``condition`` holds.

    Returns ``df`` with a boolean ``keep`` column and a ``score`` column (the
    proportion of runs marking the item as meeting the condition).
    """
    cfg = FilterConfig(
        condition=condition,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_runs=n_runs,
        n_parallels=n_parallels,
        entities_per_call=entities_per_call,
        shuffle=shuffle,
        random_seed=random_seed,
        threshold=threshold,
        additional_instructions=additional_instructions,
    )
    task = Filter(
        cfg,
        client=client,
        embedding_client=embedding_client,
        search_tool=search_tool,
    )
    return await task.run(df, column_name, reset_files=reset_files, **engine_kwargs)


async def whatever(
    prompts: Optional[Any] = None,
    *,
    df: Optional[pd.DataFrame] = None,
    column_name: Optional[str] = None,
    identifiers: Optional[list] = None,
    identifier_column: Optional[str] = None,
    save_dir: str = "whatever",
    model: str = "gpt-5.4-mini",
    json_mode: bool = False,
    n_parallels: int = 50,
    file_name: str = "custom_prompt_responses.csv",
    reset_files: bool = False,
    client: Optional[LLMClient] = None,
    embedding_client: Optional[EmbeddingClient] = None,
    search_tool: Optional[SearchTool] = None,
    **engine_kwargs: Any,
) -> pd.DataFrame:
    """Run arbitrary prompts, reusing concurrency and checkpointing.

    Returns a DataFrame with ``Identifier`` and ``Response`` columns (plus
    ``Response JSON`` when ``json_mode=True``).
    """
    cfg = WhateverConfig(
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        json_mode=json_mode,
        n_parallels=n_parallels,
    )
    task = Whatever(
        cfg,
        client=client,
        embedding_client=embedding_client,
        search_tool=search_tool,
    )
    return await task.run(
        prompts,
        df=df,
        column_name=column_name,
        identifiers=identifiers,
        identifier_column=identifier_column,
        reset_files=reset_files,
        **engine_kwargs,
    )


async def compare(
    df: pd.DataFrame,
    circle_column_name: str,
    square_column_name: str,
    *,
    save_dir: str,
    differentiate: bool = True,
    additional_instructions: Optional[str] = None,
    model: str = "gpt-5.4-mini",
    n_parallels: int = 50,
    modality: str = "text",
    circle_first: Optional[bool] = None,
    file_name: str = "comparison_responses.csv",
    reset_files: bool = False,
    client: Optional[LLMClient] = None,
    embedding_client: Optional[EmbeddingClient] = None,
    search_tool: Optional[SearchTool] = None,
    **engine_kwargs: Any,
) -> pd.DataFrame:
    """Identify similarities/differences between paired entries.

    Returns a long DataFrame with ``circle``, ``square``, ``attribute`` and
    ``explanation`` columns (one row per term).
    """
    cfg = CompareConfig(
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        differentiate=differentiate,
        additional_instructions=additional_instructions,
        modality=modality,
        circle_first=circle_first,
    )
    task = Compare(
        cfg,
        client=client,
        embedding_client=embedding_client,
        search_tool=search_tool,
    )
    return await task.run(
        df, circle_column_name, square_column_name, reset_files=reset_files, **engine_kwargs
    )


async def paraphrase(
    df: pd.DataFrame,
    column_name: str,
    *,
    instructions: str,
    save_dir: str,
    model: str = "gpt-5.4-mini",
    n_parallels: int = 50,
    modality: str = "text",
    json_mode: bool = False,
    revised_column_name: Optional[str] = None,
    file_name: str = "paraphrase_responses.csv",
    reset_files: bool = False,
    client: Optional[LLMClient] = None,
    embedding_client: Optional[EmbeddingClient] = None,
    search_tool: Optional[SearchTool] = None,
    **engine_kwargs: Any,
) -> pd.DataFrame:
    """Rewrite each item per ``instructions``.

    Returns ``df`` with a new paraphrased column (``revised_column_name`` or
    ``<column_name>_paraphrased``).
    """
    cfg = ParaphraseConfig(
        instructions=instructions,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        modality=modality,
        json_mode=json_mode,
        revised_column_name=revised_column_name,
    )
    task = Paraphrase(
        cfg,
        client=client,
        embedding_client=embedding_client,
        search_tool=search_tool,
    )
    return await task.run(df, column_name, reset_files=reset_files, **engine_kwargs)


async def seed(
    instructions: str,
    *,
    save_dir: str,
    model: str = "gpt-5.5",
    n_parallels: int = 50,
    num_entities: int = 1000,
    entities_per_generation: int = 50,
    existing_entities_cap: int = 100,
    deduplicate: bool = False,
    deduplicate_sample_seed: int = 42,
    file_name: str = "seed_entities.csv",
    existing_entities: Optional[list] = None,
    reset_files: bool = False,
    client: Optional[LLMClient] = None,
    embedding_client: Optional[EmbeddingClient] = None,
    search_tool: Optional[SearchTool] = None,
    **engine_kwargs: Any,
) -> pd.DataFrame:
    """Generate ``num_entities`` distinct entities per ``instructions``.

    Returns a DataFrame with ``entity`` and ``entity_id`` columns.
    """
    cfg = SeedConfig(
        instructions=instructions,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        num_entities=num_entities,
        entities_per_generation=entities_per_generation,
        existing_entities_cap=existing_entities_cap,
        deduplicate=deduplicate,
        deduplicate_sample_seed=deduplicate_sample_seed,
    )
    task = Seed(
        cfg,
        client=client,
        embedding_client=embedding_client,
        search_tool=search_tool,
    )
    return await task.run(
        existing_entities=existing_entities, reset_files=reset_files, **engine_kwargs
    )


async def deidentify(
    df: pd.DataFrame,
    column_name: str,
    *,
    save_dir: str,
    grouping_column: Optional[str] = None,
    mapping_column: Optional[str] = None,
    model: str = "gpt-5.4-mini",
    n_parallels: int = 50,
    max_words_per_call: int = 7500,
    additional_instructions: Optional[str] = None,
    n_passes: int = 1,
    use_existing_mappings_only: bool = False,
    file_name: str = "deidentified.csv",
    reset_files: bool = False,
    client: Optional[LLMClient] = None,
    embedding_client: Optional[EmbeddingClient] = None,
    search_tool: Optional[SearchTool] = None,
    **engine_kwargs: Any,
) -> pd.DataFrame:
    """Replace PII with consistent fake PII.

    Returns ``df`` with ``deidentified_text`` and ``mapping`` columns.
    """
    cfg = DeidentifyConfig(
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        max_words_per_call=max_words_per_call,
        additional_instructions=additional_instructions,
        n_passes=n_passes,
        use_existing_mappings_only=use_existing_mappings_only,
    )
    task = Deidentify(
        cfg,
        client=client,
        embedding_client=embedding_client,
        search_tool=search_tool,
    )
    return await task.run(
        df,
        column_name,
        grouping_column=grouping_column,
        mapping_column=mapping_column,
        reset_files=reset_files,
        **engine_kwargs,
    )


async def merge(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    *,
    save_dir: str,
    on: Optional[str] = None,
    left_on: Optional[str] = None,
    right_on: Optional[str] = None,
    how: str = "left",
    model: str = "gpt-5.4-nano",
    n_parallels: int = 50,
    additional_instructions: Optional[str] = None,
    short_list_len: int = 16,
    long_list_len: int = 256,
    file_name: str = "merge_responses.csv",
    reset_files: bool = False,
    client: Optional[LLMClient] = None,
    embedding_client: Optional[EmbeddingClient] = None,
    search_tool: Optional[SearchTool] = None,
    **engine_kwargs: Any,
) -> pd.DataFrame:
    """Fuzzy-join two tables via LLM matching (pure-LLM path in v1).

    Returns ``df_left`` merged with ``df_right`` on the matched key.
    """
    cfg = MergeConfig(
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        additional_instructions=additional_instructions,
        short_list_len=short_list_len,
        long_list_len=long_list_len,
    )
    task = Merge(
        cfg,
        client=client,
        embedding_client=embedding_client,
        search_tool=search_tool,
    )
    return await task.run(
        df_left,
        df_right,
        on=on,
        left_on=left_on,
        right_on=right_on,
        how=how,
        reset_files=reset_files,
        **engine_kwargs,
    )


async def deduplicate(
    df: pd.DataFrame,
    column_name: str,
    *,
    save_dir: str,
    model: str = "gpt-5.4-mini",
    n_parallels: int = 50,
    additional_instructions: Optional[str] = None,
    group_size: int = 500,
    modality: str = "entity",
    file_name: str = "deduplicate_responses.csv",
    reset_files: bool = False,
    client: Optional[LLMClient] = None,
    embedding_client: Optional[EmbeddingClient] = None,
    search_tool: Optional[SearchTool] = None,
    **engine_kwargs: Any,
) -> pd.DataFrame:
    """Map conceptually-equivalent terms to one representative.

    Returns ``df`` with a ``mapped_<column_name>`` column.
    """
    cfg = DeduplicateConfig(
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        additional_instructions=additional_instructions,
        group_size=group_size,
        modality=modality,
    )
    task = Deduplicate(
        cfg,
        client=client,
        embedding_client=embedding_client,
        search_tool=search_tool,
    )
    return await task.run(df, column_name, reset_files=reset_files, **engine_kwargs)


async def bucket(
    df: pd.DataFrame,
    column_name: str,
    *,
    save_dir: str,
    bucket_count: int = 10,
    model: str = "gpt-5.4-mini",
    n_parallels: int = 50,
    additional_instructions: Optional[str] = None,
    n_terms_per_prompt: int = 250,
    file_name: str = "bucket_definitions.csv",
    reset_files: bool = False,
    client: Optional[LLMClient] = None,
    embedding_client: Optional[EmbeddingClient] = None,
    search_tool: Optional[SearchTool] = None,
    **engine_kwargs: Any,
) -> pd.DataFrame:
    """Propose ``bucket_count`` buckets that group the input terms.

    Returns a DataFrame with ``bucket`` and ``definition`` columns.
    """
    cfg = BucketConfig(
        bucket_count=bucket_count,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        additional_instructions=additional_instructions,
        n_terms_per_prompt=n_terms_per_prompt,
    )
    task = Bucket(
        cfg,
        client=client,
        embedding_client=embedding_client,
        search_tool=search_tool,
    )
    return await task.run(df, column_name, reset_files=reset_files, **engine_kwargs)


async def codify(
    df: pd.DataFrame,
    column_name: str,
    *,
    save_dir: str,
    categories: Optional[Dict[str, str]] = None,
    additional_instructions: str = "",
    model: str = "gpt-5.4-mini",
    n_parallels: int = 50,
    modality: str = "text",
    file_name: str = "coding_results.csv",
    reset_files: bool = False,
    client: Optional[LLMClient] = None,
    embedding_client: Optional[EmbeddingClient] = None,
    search_tool: Optional[SearchTool] = None,
    **engine_kwargs: Any,
) -> pd.DataFrame:
    """Mark snippets matching each qualitative code.

    Returns ``df`` with a ``coded_passages`` column (parsed JSON per item).
    """
    cfg = CodifyConfig(
        categories=categories,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        additional_instructions=additional_instructions,
        modality=modality,
    )
    task = Codify(
        cfg,
        client=client,
        embedding_client=embedding_client,
        search_tool=search_tool,
    )
    return await task.run(
        df,
        column_name,
        categories=categories,
        additional_instructions=additional_instructions,
        reset_files=reset_files,
        **engine_kwargs,
    )
