"""Public facade.

Each public function constructs a task config and delegates to a task.
Parameters are explicit; provider selection happens via ``client=`` /
``embedding_client=`` / ``search_tool=`` (or environment defaults).  Extra
keyword arguments are forwarded to the engine / model call (e.g.
``max_output_tokens``, ``temperature``, ``timeout``).

Complex tasks (``discover`` / ``ideate`` / ``poll`` / ``debias``) accept a
task-level ``max_retries``: when the whole run raises, it is retried from
its on-disk checkpoints (``reset_files`` forced to ``False``) instead of
re-running completed stages.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from .client.base import LLMClient
from .client.decision import DecisionClient
from .client.embedding import EmbeddingClient
from .client.search import SearchTool
from .engine.retry import run_task_with_retry
from .tasks.bucket import Bucket, BucketConfig
from .tasks.classify import Classify, ClassifyConfig
from .tasks.codify import Codify, CodifyConfig
from .tasks.compare import Compare, CompareConfig
from .tasks.debias import Debias, DebiasConfig, DebiasResult
from .tasks.deduplicate import Deduplicate, DeduplicateConfig
from .tasks.deidentify import Deidentify, DeidentifyConfig
from .tasks.discover import Discover, DiscoverConfig
from .tasks.extract import Extract, ExtractConfig
from .tasks.filter import Filter, FilterConfig
from .tasks.ideate import Ideate, IdeateConfig
from .tasks.merge import Merge, MergeConfig
from .tasks.paraphrase import Paraphrase, ParaphraseConfig
from .tasks.poll import Poll, PollConfig
from .tasks.rank import Rank, RankConfig
from .tasks.rate import Rate, RateConfig
from .tasks.seed import Seed, SeedConfig
from .tasks.whatever import Whatever, WhateverConfig


async def rate(
    df: pd.DataFrame,
    column_name: str,
    *,
    attributes: dict[str, str],
    save_dir: str,
    model: str = "gpt-5.4-mini",
    n_runs: int = 1,
    n_parallels: int = 50,
    n_attributes_per_run: int | None = None,
    rating_scale: str | None = None,
    additional_instructions: str | None = None,
    modality: str = "text",
    file_name: str = "ratings.csv",
    reset_files: bool = False,
    client: LLMClient | None = None,
    embedding_client: EmbeddingClient | None = None,
    search_tool: SearchTool | None = None,
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
    column_name: str | None = None,
    *,
    labels: dict[str, str],
    save_dir: str,
    model: str = "gpt-5.4-mini",
    decision_model: str | None = None,
    n_runs: int = 1,
    n_parallels: int = 50,
    min_frequency: float = 0.6,
    n_attributes_per_run: int | None = None,
    additional_instructions: str | None = None,
    modality: str = "text",
    differentiate: bool = False,
    circle_column_name: str | None = None,
    square_column_name: str | None = None,
    circle_first: bool | None = None,
    include_probabilities: bool = False,
    file_name: str = "classify_responses.csv",
    reset_files: bool = False,
    client: LLMClient | None = None,
    embedding_client: EmbeddingClient | None = None,
    search_tool: SearchTool | None = None,
    decision_client: DecisionClient | None = None,
    **engine_kwargs: Any,
) -> pd.DataFrame:
    """Decide whether each label applies to each item.

    Returns ``df`` with one boolean column per label (True when the proportion
    of runs — or calibrated probability from a Decision Provider — is
    ``>= min_frequency``) plus ``predicted_classes`` (and ``{label}_prob``
    columns when ``include_probabilities=True`` with a Decision Provider).
    """
    cfg = ClassifyConfig(
        labels=labels,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        decision_model=decision_model,
        n_runs=n_runs,
        n_parallels=n_parallels,
        min_frequency=min_frequency,
        n_attributes_per_run=n_attributes_per_run,
        additional_instructions=additional_instructions,
        modality=modality,
        differentiate=differentiate,
        circle_first=circle_first,
        include_probabilities=include_probabilities,
    )
    task = Classify(
        cfg,
        client=client,
        embedding_client=embedding_client,
        search_tool=search_tool,
        decision_client=decision_client,
    )
    return await task.run(
        df,
        column_name,
        circle_column_name=circle_column_name,
        square_column_name=square_column_name,
        reset_files=reset_files,
        **engine_kwargs,
    )


async def extract(
    df: pd.DataFrame,
    column_name: str,
    *,
    attributes: dict[str, str],
    save_dir: str,
    model: str = "gpt-5.4-mini",
    n_runs: int = 1,
    n_parallels: int = 50,
    n_attributes_per_run: int | None = None,
    additional_instructions: str | None = None,
    modality: str = "entity",
    types: dict[str, Any] | None = None,
    file_name: str = "extraction.csv",
    reset_files: bool = False,
    client: LLMClient | None = None,
    embedding_client: EmbeddingClient | None = None,
    search_tool: SearchTool | None = None,
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
    decision_model: str | None = None,
    n_runs: int = 1,
    n_parallels: int = 50,
    entities_per_call: int = 150,
    shuffle: bool = True,
    random_seed: int = 42,
    threshold: float = 0.5,
    additional_instructions: str | None = None,
    file_name: str = "filter_responses.csv",
    reset_files: bool = False,
    client: LLMClient | None = None,
    embedding_client: EmbeddingClient | None = None,
    search_tool: SearchTool | None = None,
    decision_client: DecisionClient | None = None,
    **engine_kwargs: Any,
) -> pd.DataFrame:
    """Keep items whose natural-language ``condition`` holds.

    Returns ``df`` with a boolean ``keep`` column and a ``score`` column (the
    proportion of runs marking the item as meeting the condition, or the
    calibrated probability from a Decision Provider).
    """
    cfg = FilterConfig(
        condition=condition,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        decision_model=decision_model,
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
        decision_client=decision_client,
    )
    return await task.run(df, column_name, reset_files=reset_files, **engine_kwargs)


async def whatever(
    prompts: Any | None = None,
    *,
    df: pd.DataFrame | None = None,
    column_name: str | None = None,
    identifiers: list | None = None,
    identifier_column: str | None = None,
    save_dir: str = "whatever",
    model: str = "gpt-5.4-mini",
    json_mode: bool = False,
    web_search: bool | None = None,
    web_search_filters: dict[str, Any] | None = None,
    search_context_size: str = "medium",
    n_parallels: int = 50,
    file_name: str = "custom_prompt_responses.csv",
    reset_files: bool = False,
    client: LLMClient | None = None,
    embedding_client: EmbeddingClient | None = None,
    search_tool: SearchTool | None = None,
    **engine_kwargs: Any,
) -> pd.DataFrame:
    """Run arbitrary prompts, reusing concurrency and checkpointing.

    When ``web_search`` is enabled (or ``web_search_filters`` is provided), each
    prompt is first expanded with retrieved context from ``search_tool`` (or the
    configured ``LEXIMETER_SEARCH_PROVIDER``) before being sent to the model.

    Returns a DataFrame with ``Identifier`` and ``Response`` columns (plus
    ``Response JSON`` when ``json_mode=True``).
    """
    cfg = WhateverConfig(
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        json_mode=json_mode,
        web_search=web_search,
        web_search_filters=web_search_filters,
        search_context_size=search_context_size,
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
    additional_instructions: str | None = None,
    model: str = "gpt-5.4-mini",
    n_parallels: int = 50,
    modality: str = "text",
    circle_first: bool | None = None,
    file_name: str = "comparison_responses.csv",
    reset_files: bool = False,
    client: LLMClient | None = None,
    embedding_client: EmbeddingClient | None = None,
    search_tool: SearchTool | None = None,
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
        df,
        circle_column_name,
        square_column_name,
        reset_files=reset_files,
        **engine_kwargs,
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
    revised_column_name: str | None = None,
    file_name: str = "paraphrase_responses.csv",
    n_rounds: int = 1,
    n_runs: int = 1,
    later_round_run_multiplier: int = 5,
    use_modified_source: bool = False,
    validation_attribute: dict[str, str] | None = None,
    reset_files: bool = False,
    client: LLMClient | None = None,
    embedding_client: EmbeddingClient | None = None,
    search_tool: SearchTool | None = None,
    **engine_kwargs: Any,
) -> pd.DataFrame:
    """Rewrite each item per ``instructions``.

    Returns ``df`` with new paraphrased columns (``revised_column_name`` or
    ``<column_name>_revised``; suffixed ``_1.._n`` when ``n_runs > 1``) plus
    a ``<col>_approved`` column per output.  With ``n_rounds > 1`` a
    recursive validation loop re-generates failed candidates (classifier
    compares original vs. modified text) until they pass or the round
    budget runs out (unapproved candidates then get a random fallback).
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
        n_rounds=n_rounds,
        n_runs=n_runs,
        later_round_run_multiplier=later_round_run_multiplier,
        use_modified_source=use_modified_source,
        validation_attribute=validation_attribute,
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
    existing_entities: list | None = None,
    reset_files: bool = False,
    max_retries: int = 0,
    client: LLMClient | None = None,
    embedding_client: EmbeddingClient | None = None,
    search_tool: SearchTool | None = None,
    **engine_kwargs: Any,
) -> pd.DataFrame:
    """Generate ``num_entities`` distinct entities per ``instructions``.

    ``max_retries`` retries the whole generation run when it raises (the
    SDK's own per-request timeout+retry remains the only inner retry layer,
    per the worksummary_0816 exit-protection decision).  A completed seed
    is cached to ``seed_entities.csv`` + ``seed_state.json``: re-running
    with the same inputs makes zero provider calls and reloads the file.
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
    return await run_task_with_retry(
        task,
        max_retries,
        existing_entities=existing_entities,
        reset_files=reset_files,
        **engine_kwargs,
    )


async def deidentify(
    df: pd.DataFrame,
    column_name: str,
    *,
    save_dir: str,
    grouping_column: str | None = None,
    mapping_column: str | None = None,
    model: str = "gpt-5.4-mini",
    n_parallels: int = 50,
    max_words_per_call: int = 7500,
    additional_instructions: str | None = None,
    n_passes: int = 1,
    use_existing_mappings_only: bool = False,
    file_name: str = "deidentified.csv",
    reset_files: bool = False,
    client: LLMClient | None = None,
    embedding_client: EmbeddingClient | None = None,
    search_tool: SearchTool | None = None,
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
    on: str | None = None,
    left_on: str | None = None,
    right_on: str | None = None,
    how: str = "left",
    model: str = "gpt-5.4-nano",
    n_parallels: int = 50,
    n_runs: int = 1,
    additional_instructions: str | None = None,
    short_list_len: int = 16,
    long_list_len: int = 256,
    use_embeddings: bool = True,
    max_attempts: int = 4,
    short_list_multiplier: float = 0.5,
    auto_match_threshold: float = 0.75,
    use_best_auto_match: bool = False,
    candidate_scan_chunks: int = 5,
    verbose: bool = True,
    file_name: str = "merge_responses.csv",
    reset_files: bool = False,
    client: LLMClient | None = None,
    embedding_client: EmbeddingClient | None = None,
    search_tool: SearchTool | None = None,
    **engine_kwargs: Any,
) -> pd.DataFrame:
    """Fuzzy-join two tables via LLM matching.

    Short-side keys are matched against the long side in ``max_attempts``
    rounds (short-list size shrinks 16 → 8 → 4 → 2); remaining keys are
    clustered with kmeans2 (embedding mode) or alphabetically, and each
    cluster is matched against the best candidate window.  Embedding
    auto-matches above ``auto_match_threshold`` are applied directly when
    unambiguous.  Group plans are checkpointed per attempt
    (``merge_groups_attempt{n}.json``) and progress appended to
    ``merge_progress.csv``.

    Returns ``df_left`` merged with ``df_right`` on the matched key.
    """
    cfg = MergeConfig(
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        n_runs=n_runs,
        additional_instructions=additional_instructions,
        short_list_len=short_list_len,
        long_list_len=long_list_len,
        use_embeddings=use_embeddings,
        max_attempts=max_attempts,
        short_list_multiplier=short_list_multiplier,
        auto_match_threshold=auto_match_threshold,
        use_best_auto_match=use_best_auto_match,
        candidate_scan_chunks=candidate_scan_chunks,
        verbose=verbose,
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
    additional_instructions: str | None = None,
    group_size: int = 500,
    modality: str = "entity",
    use_embeddings: bool = True,
    file_name: str = "deduplicate_responses.csv",
    reset_files: bool = False,
    client: LLMClient | None = None,
    embedding_client: EmbeddingClient | None = None,
    search_tool: SearchTool | None = None,
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
        use_embeddings=use_embeddings,
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
    additional_instructions: str | None = None,
    differentiate: bool = False,
    n_terms_per_prompt: int = 250,
    repeat_bucketing: int = 5,
    repeat_voting: int = 25,
    next_round_frac: float = 0.25,
    top_k_per_round: int = 1,
    raw_term_definitions: bool = True,
    file_name: str = "bucket_definitions.csv",
    reset_files: bool = False,
    client: LLMClient | None = None,
    embedding_client: EmbeddingClient | None = None,
    search_tool: SearchTool | None = None,
    **engine_kwargs: Any,
) -> pd.DataFrame:
    """Propose ``bucket_count`` buckets that group the input terms.

    Three-stage voting pipeline (generate → reduce → finalize), fully
    checkpointed to ``bucket_generation.csv`` / ``vote_reduce*.csv`` /
    ``vote_final*.csv`` / ``bucket_state.json``.  Returns a DataFrame with
    ``bucket`` and ``definition`` columns.
    """
    cfg = BucketConfig(
        bucket_count=bucket_count,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        additional_instructions=additional_instructions,
        differentiate=differentiate,
        n_terms_per_prompt=n_terms_per_prompt,
        repeat_bucketing=repeat_bucketing,
        repeat_voting=repeat_voting,
        next_round_frac=next_round_frac,
        top_k_per_round=top_k_per_round,
        raw_term_definitions=raw_term_definitions,
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
    categories: dict[str, str] | None = None,
    additional_instructions: str = "",
    model: str = "gpt-5.4-mini",
    n_parallels: int = 50,
    modality: str = "text",
    file_name: str = "coding_results.csv",
    n_rounds: int = 2,
    max_words_per_call: int = 1000,
    max_categories_per_call: int = 8,
    debug_print: bool = False,
    completion_classifier_instructions: str | None = None,
    reset_files: bool = False,
    client: LLMClient | None = None,
    embedding_client: EmbeddingClient | None = None,
    search_tool: SearchTool | None = None,
    **engine_kwargs: Any,
) -> pd.DataFrame:
    """Mark snippets matching each qualitative code.

    Static mode (``categories`` provided) returns one column per category
    holding the list of verbatim snippets found in each row; dynamic mode
    (``categories=None`` + ``additional_instructions``) returns a single
    ``coded_passages`` column mapping discovered labels to snippet lists.
    When ``n_rounds > 1`` a completion loop re-codes remaining text until
    nothing new is found.  The final frame is always written to
    ``coded_passages.csv``.
    """
    cfg = CodifyConfig(
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        modality=modality,
        n_rounds=n_rounds,
        max_words_per_call=max_words_per_call,
        max_categories_per_call=max_categories_per_call,
        debug_print=debug_print,
        completion_classifier_instructions=completion_classifier_instructions,
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


async def rank(
    df: pd.DataFrame,
    column_name: str,
    *,
    attributes: dict[str, str],
    save_dir: str,
    model: str = "gpt-5.4-mini",
    n_rounds: int = 5,
    matches_per_round: int = 3,
    learning_rate: float = 0.1,
    n_parallels: int = 50,
    additional_instructions: str | None = None,
    modality: str = "text",
    n_attributes_per_run: int | None = None,
    circle_first: bool | None = None,
    file_name: str = "rankings",
    reset_files: bool = False,
    client: LLMClient | None = None,
    embedding_client: EmbeddingClient | None = None,
    search_tool: SearchTool | None = None,
    **engine_kwargs: Any,
) -> pd.DataFrame:
    """Run pairwise tournaments and fit Bradley–Terry ratings per attribute.

    Returns ``df`` with, per attribute, a z-score column ``{attr}``, the
    centred Bradley–Terry estimate ``{attr}_raw`` and its standard error
    ``{attr}_se``.  Each round is checkpointed to ``{stem}_round{r}.csv``
    (Batch/Pair/IdA/IdB columns); interrupted runs resume and replay them.
    """
    cfg = RankConfig(
        attributes=attributes,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_rounds=n_rounds,
        matches_per_round=matches_per_round,
        learning_rate=learning_rate,
        n_parallels=n_parallels,
        additional_instructions=additional_instructions,
        modality=modality,
        n_attributes_per_run=n_attributes_per_run,
        circle_first=circle_first,
    )
    task = Rank(
        cfg,
        client=client,
        embedding_client=embedding_client,
        search_tool=search_tool,
    )
    return await task.run(df, column_name, reset_files=reset_files, **engine_kwargs)


async def discover(
    df: pd.DataFrame,
    *,
    column_name: str | None = None,
    circle_column_name: str | None = None,
    square_column_name: str | None = None,
    save_dir: str,
    bucket_count: int = 10,
    model: str = "gpt-5.4-mini",
    n_parallels: int = 50,
    n_runs: int = 1,
    min_frequency: float = 0.6,
    additional_instructions: str | None = None,
    max_words_per_call: int = 1000,
    max_categories_per_call: int = 8,
    modality: str = "text",
    n_terms_per_prompt: int = 250,
    repeat_bucketing: int = 5,
    repeat_voting: int = 25,
    next_round_frac: float = 0.25,
    top_k_per_round: int = 1,
    raw_term_definitions: bool = True,
    reset_files: bool = False,
    max_retries: int = 0,
    client: LLMClient | None = None,
    embedding_client: EmbeddingClient | None = None,
    search_tool: SearchTool | None = None,
    **engine_kwargs: Any,
) -> dict[str, Any]:
    """Contrast two columns (or mine one) and bucket distinguishing features.

    Pipeline: candidate discovery (dynamic :func:`codify` for a single
    column, :func:`compare` for a circle/square pair) → :func:`bucket`
    taxonomy → :func:`classify` back onto the data.  For pairs the result
    includes a per-label ``summary`` DataFrame with ``actual_pct``,
    ``inverted_pct`` and ``net_pct`` (actual − inverted, percentage
    points).  Results are persisted to ``discover_results_snapshot.json``
    and ``discover_results_export.zip``.

    ``max_retries`` retries the whole pipeline from its checkpoints when a
    run raises; a completed run with unchanged inputs is reloaded from the
    snapshot with zero provider calls.
    """
    cfg = DiscoverConfig(
        save_dir=save_dir,
        model=model,
        n_parallels=n_parallels,
        n_runs=n_runs,
        min_frequency=min_frequency,
        bucket_count=bucket_count,
        additional_instructions=additional_instructions,
        max_words_per_call=max_words_per_call,
        max_categories_per_call=max_categories_per_call,
        modality=modality,
        n_terms_per_prompt=n_terms_per_prompt,
        repeat_bucketing=repeat_bucketing,
        repeat_voting=repeat_voting,
        next_round_frac=next_round_frac,
        top_k_per_round=top_k_per_round,
        raw_term_definitions=raw_term_definitions,
    )
    task = Discover(
        cfg,
        client=client,
        embedding_client=embedding_client,
        search_tool=search_tool,
    )
    return await run_task_with_retry(
        task,
        max_retries,
        df,
        column_name=column_name,
        circle_column_name=circle_column_name,
        square_column_name=square_column_name,
        reset_files=reset_files,
        **engine_kwargs,
    )


async def ideate(
    topic: str,
    *,
    save_dir: str,
    n_ideas: int = 1000,
    scientific_theory: bool = True,
    model: str = "gpt-5.4-mini",
    n_parallels: int = 50,
    additional_instructions: str | None = None,
    file_name: str = "ideation.csv",
    reset_files: bool = False,
    max_retries: int = 0,
    client: LLMClient | None = None,
    embedding_client: EmbeddingClient | None = None,
    search_tool: SearchTool | None = None,
    **engine_kwargs: Any,
) -> pd.DataFrame:
    """Generate ``n_ideas`` reports on a topic and parse them into fields.

    ``max_retries`` retries the whole run from its checkpoints when it
    raises; a completed run with unchanged inputs is reloaded from
    ``ideation.csv`` with zero provider calls.
    """
    cfg = IdeateConfig(
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        n_ideas=n_ideas,
        scientific_theory=scientific_theory,
        additional_instructions=additional_instructions,
    )
    task = Ideate(
        cfg,
        client=client,
        embedding_client=embedding_client,
        search_tool=search_tool,
    )
    return await run_task_with_retry(
        task,
        max_retries,
        topic,
        additional_instructions=additional_instructions,
        reset_files=reset_files,
        **engine_kwargs,
    )


async def poll(
    df: pd.DataFrame | None = None,
    column_name: str | None = None,
    *,
    population_description: str | None = None,
    questions: Any | None = None,
    save_dir: str,
    seed_model: str = "gpt-5.5",
    persona_model: str = "gpt-5.5",
    poll_model: str = "gpt-5.5",
    n_parallels: int = 50,
    num_personas: int = 1000,
    entities_per_generation: int = 50,
    n_questions_per_run: int = 8,
    file_name: str = "poll_results.csv",
    reset_files: bool = False,
    max_retries: int = 0,
    client: LLMClient | None = None,
    embedding_client: EmbeddingClient | None = None,
    search_tool: SearchTool | None = None,
    **engine_kwargs: Any,
) -> pd.DataFrame:
    """Seed a population, expand personas, and survey them.

    ``max_retries`` retries the whole run from its checkpoints when it
    raises (e.g. a seed-stage provider failure); a completed run with
    unchanged inputs is reloaded from ``poll_results.csv`` with zero
    provider calls.
    """
    cfg = PollConfig(
        population_description=population_description,
        questions=list(questions) if questions else [],
        save_dir=save_dir,
        file_name=file_name,
        seed_model=seed_model,
        persona_model=persona_model,
        poll_model=poll_model,
        n_parallels=n_parallels,
        num_personas=num_personas,
        entities_per_generation=entities_per_generation,
        n_questions_per_run=n_questions_per_run,
    )
    task = Poll(
        cfg,
        client=client,
        embedding_client=embedding_client,
        search_tool=search_tool,
    )
    return await run_task_with_retry(
        task,
        max_retries,
        df=df,
        column_name=column_name,
        population_description=population_description,
        questions=questions,
        reset_files=reset_files,
        **engine_kwargs,
    )


async def debias(
    df: pd.DataFrame,
    column_name: str,
    *,
    save_dir: str = "debias",
    mode: Literal["rate", "classify", "extract", "rank"] = "rate",
    measurement_attribute: str | None = None,
    removal_attribute: str | None = None,
    attributes: dict[str, str],
    signal_dictionary: dict[str, str],
    removal_method: Literal["codify", "paraphrase"] = "codify",
    remaining_signal: bool = True,
    remaining_signal_attribute: str | None = None,
    remaining_signal_description: str | None = None,
    run_name: str | None = None,
    strip_percentages: list[int] | None = None,
    categories_to_strip: list[str] | None = None,
    model: str = "gpt-5.4-mini",
    n_parallels: int = 50,
    measurement_kwargs: dict[str, Any] | None = None,
    removal_kwargs: dict[str, Any] | None = None,
    robust_regression: bool = True,
    random_seed: int = 12345,
    verbose: bool = True,
    reset_files: bool = False,
    max_retries: int = 0,
    client: LLMClient | None = None,
    embedding_client: EmbeddingClient | None = None,
    search_tool: SearchTool | None = None,
    **engine_kwargs: Any,
) -> DebiasResult:
    """Measure a signal, strip a confounder, re-measure, and regress out bias.

    ``max_retries`` retries the whole pipeline from its checkpoints when a
    run raises; a completed run with unchanged inputs is rebuilt from
    ``debias_results.csv`` + ``debias_metadata.json`` with zero provider
    calls.

    The pipeline measures ``measurement_attribute`` on the original text,
    removes the confounding signal (``removal_method``: ``"codify"`` strips
    snippets deterministically across ``strip_percentages``; ``"paraphrase"``
    rewrites), re-measures the variants, optionally measures the remaining
    signal, and computes simple-difference and two-step debiased columns
    ``"{attr} (raw - stripped)"`` / ``"{attr} (debiased)"``.  Regression
    diagnostics use robust HC3 standard errors (HC1 fallback) via
    :mod:`leximeter.utils.regression`.

    Returns a :class:`DebiasResult` with ``results``, ``metadata`` and
    ``regression``; results are saved under ``save_dir/run_name`` as
    ``debias_results.csv`` + ``debias_metadata.json``.
    """
    cfg = DebiasConfig(
        save_dir=save_dir,
        mode=mode,
        measurement_attribute=measurement_attribute,
        removal_attribute=removal_attribute,
        attributes=attributes,
        signal_dictionary=signal_dictionary,
        removal_method=removal_method,
        remaining_signal=remaining_signal,
        remaining_signal_attribute=remaining_signal_attribute,
        remaining_signal_description=remaining_signal_description,
        run_name=run_name,
        strip_percentages=strip_percentages,
        categories_to_strip=categories_to_strip,
        model=model,
        n_parallels=n_parallels,
        measurement_kwargs=measurement_kwargs or {},
        removal_kwargs=removal_kwargs or {},
        robust_regression=robust_regression,
        random_seed=random_seed,
        verbose=verbose,
    )
    task = Debias(
        cfg,
        client=client,
        embedding_client=embedding_client,
        search_tool=search_tool,
    )
    # Debias.run takes no engine kwargs; forwarding only the explicit args
    # keeps the retry wrapper generic while preserving existing behaviour.
    return await run_task_with_retry(
        task, max_retries, df, column_name, reset_files=reset_files
    )
