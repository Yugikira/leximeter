"""Business tasks.  Simple tasks are single modules; complex tasks are packages."""

from .base import AttributeTask, Task
from .bucket import Bucket, BucketConfig
from .classify import Classify, ClassifyConfig
from .codify import Codify, CodifyConfig
from .compare import Compare, CompareConfig
from .debias import Debias, DebiasConfig
from .deduplicate import Deduplicate, DeduplicateConfig
from .deidentify import Deidentify, DeidentifyConfig
from .discover import Discover, DiscoverConfig
from .extract import Extract, ExtractConfig
from .filter import Filter, FilterConfig
from .ideate import Ideate, IdeateConfig
from .merge import Merge, MergeConfig
from .paraphrase import Paraphrase, ParaphraseConfig
from .poll import Poll, PollConfig
from .rank import Rank, RankConfig
from .rate import Rate, RateConfig
from .seed import Seed, SeedConfig
from .whatever import Whatever, WhateverConfig

__all__ = [
    "AttributeTask",
    "Bucket",
    "BucketConfig",
    "Classify",
    "ClassifyConfig",
    "Codify",
    "CodifyConfig",
    "Compare",
    "CompareConfig",
    "Debias",
    "DebiasConfig",
    "Deduplicate",
    "DeduplicateConfig",
    "Deidentify",
    "DeidentifyConfig",
    "Discover",
    "DiscoverConfig",
    "Extract",
    "ExtractConfig",
    "Filter",
    "FilterConfig",
    "Ideate",
    "IdeateConfig",
    "Merge",
    "MergeConfig",
    "Paraphrase",
    "ParaphraseConfig",
    "Poll",
    "PollConfig",
    "Rank",
    "RankConfig",
    "Rate",
    "RateConfig",
    "Seed",
    "SeedConfig",
    "Task",
    "Whatever",
    "WhateverConfig",
]
