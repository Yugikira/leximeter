"""Business tasks.  Simple tasks are single modules; complex tasks are packages."""

from .base import AttributeTask, Task
from .bucket import Bucket, BucketConfig
from .classify import Classify, ClassifyConfig
from .codify import Codify, CodifyConfig
from .compare import Compare, CompareConfig
from .deduplicate import Deduplicate, DeduplicateConfig
from .deidentify import Deidentify, DeidentifyConfig
from .extract import Extract, ExtractConfig
from .filter import Filter, FilterConfig
from .merge import Merge, MergeConfig
from .paraphrase import Paraphrase, ParaphraseConfig
from .rank import Rank, RankConfig
from .rate import Rate, RateConfig
from .seed import Seed, SeedConfig
from .whatever import Whatever, WhateverConfig

__all__ = [
    "Task",
    "AttributeTask",
    "Rank",
    "RankConfig",
    "Rate",
    "RateConfig",
    "Classify",
    "ClassifyConfig",
    "Extract",
    "ExtractConfig",
    "Filter",
    "FilterConfig",
    "Compare",
    "CompareConfig",
    "Paraphrase",
    "ParaphraseConfig",
    "Deidentify",
    "DeidentifyConfig",
    "Seed",
    "SeedConfig",
    "Merge",
    "MergeConfig",
    "Deduplicate",
    "DeduplicateConfig",
    "Bucket",
    "BucketConfig",
    "Codify",
    "CodifyConfig",
    "Whatever",
    "WhateverConfig",
]
