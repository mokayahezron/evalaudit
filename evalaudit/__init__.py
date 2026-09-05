"""evalaudit: statistical validity checks for AI evaluations."""

__version__ = "0.1.0"

from ._types import ComparisonResult, ScoreCI
from .compare import compare_independent, compare_paired
from .scores import score_ci

__all__ = [
    "score_ci",
    "compare_paired",
    "compare_independent",
    "ScoreCI",
    "ComparisonResult",
]
