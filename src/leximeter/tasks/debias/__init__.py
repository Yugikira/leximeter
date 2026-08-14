"""Debias task: measure, strip the confounder, re-measure, regress."""

from .task import Debias, DebiasConfig, DebiasResult

__all__ = ["Debias", "DebiasConfig", "DebiasResult"]
