"""evalaudit: statistical validity checks for AI evaluations."""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("evalaudit")
except PackageNotFoundError:
    # Running from a source tree with nothing installed. There is no
    # version to report and there is no literal to fall back on, since a
    # literal here is what put two different releases on the same number.
    __version__ = "unknown"

from ._types import (
    AgreementResult,
    AuditConfig,
    AuditReport,
    BTResult,
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
from .pairwise import bradley_terry, to_elo
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
    "bradley_terry",
    "to_elo",
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
    "BTResult",
    "JudgeValidation",
    "PositionBias",
    "LengthBias",
    "PowerResult",
    "AuditConfig",
    "AuditReport",
    "Finding",
    "SkippedCheck",
]
