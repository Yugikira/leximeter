"""Debias task: measure, strip the confounder, re-measure, regress."""

from .task import Debias, DebiasConfig

__all__ = ["Debias", "DebiasConfig"]
