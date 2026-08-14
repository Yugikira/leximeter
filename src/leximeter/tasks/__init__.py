"""Business tasks.  Simple tasks are single modules; complex tasks are packages."""

from .base import AttributeTask, Task
from .classify import Classify, ClassifyConfig
from .compare import Compare, CompareConfig
from .extract import Extract, ExtractConfig
from .filter import Filter, FilterConfig
from .paraphrase import Paraphrase, ParaphraseConfig
from .rate import Rate, RateConfig
from .whatever import Whatever, WhateverConfig

__all__ = [
    "Task",
    "AttributeTask",
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
    "Whatever",
    "WhateverConfig",
]
