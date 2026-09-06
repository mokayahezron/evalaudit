"""evalaudit: statistical validity checks for AI evaluations."""

__version__ = "0.1.0"

from ._types import AgreementResult, ComparisonResult, KappaResult, ScoreCI
from .agreement import cohens_kappa, fleiss_kappa, rater_agreement
from .compare import compare_independent, compare_paired
from .scores import score_ci

__all__ = [
    "score_ci",
    "compare_paired",
    "compare_independent",
    "rater_agreement",
    "cohens_kappa",
    "fleiss_kappa",
    "ScoreCI",
    "ComparisonResult",
    "AgreementResult",
    "KappaResult",
]
