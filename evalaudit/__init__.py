"""evalaudit: statistical validity checks for AI evaluations."""

__version__ = "0.1.0"

from ._types import (
    AgreementResult,
    AuditConfig,
    AuditReport,
    ComparisonResult,
    Finding,
    JudgeValidation,
    KappaResult,
    LengthBias,
    PositionBias,
    PowerResult,
    ScoreCI,
    SkippedCheck,
)
from .agreement import cohens_kappa, fleiss_kappa, rater_agreement
from .compare import compare_independent, compare_paired
from .judge import judge_validation, length_bias, position_bias
from .power import detectable_effect, min_sample_size
from .scores import score_ci
from .audit import audit

__all__ = [
    "score_ci",
    "compare_paired",
    "compare_independent",
    "rater_agreement",
    "cohens_kappa",
    "fleiss_kappa",
    "judge_validation",
    "position_bias",
    "length_bias",
    "detectable_effect",
    "min_sample_size",
    "audit",
    "ScoreCI",
    "ComparisonResult",
    "AgreementResult",
    "KappaResult",
    "JudgeValidation",
    "PositionBias",
    "LengthBias",
    "PowerResult",
    "AuditConfig",
    "AuditReport",
    "Finding",
    "SkippedCheck",
]
