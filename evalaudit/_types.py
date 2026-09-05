"""Return types for evalaudit.

Every public function returns one of these. Each carries the numbers and a
``summary()`` that states, in plain English, what the numbers do and do not
support. The summary is the point of the library: scipy already computes
McNemar, but nothing tells a product manager their result does not hold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


@dataclass(frozen=True)
class ScoreCI:
    """Confidence interval on a single eval score."""

    estimate: float
    ci_low: float
    ci_high: float
    n: int
    method: str
    binary: bool
    confidence: float = 0.95

    @property
    def margin(self) -> float:
        """Half-width of the interval."""
        return (self.ci_high - self.ci_low) / 2

    @property
    def width(self) -> float:
        return self.ci_high - self.ci_low

    def summary(self) -> str:
        conf = f"{self.confidence * 100:.0f}%"
        if self.binary:
            head = (
                f"{_pct(self.estimate)} pass rate "
                f"({conf} CI: {_pct(self.ci_low)}-{_pct(self.ci_high)}, n={self.n})."
            )
            span = f"{self.width * 100:.0f} points"
        else:
            head = (
                f"Mean score {self.estimate:.3f} "
                f"({conf} CI: {self.ci_low:.3f}-{self.ci_high:.3f}, n={self.n})."
            )
            span = f"{self.width:.3f}"

        tail = (
            f" The interval spans {span}; treat differences smaller than that "
            f"as unresolved."
        )
        if self.n < 30:
            tail += (
                f" With only {self.n} observations this estimate is weak "
                f"regardless of the point value."
            )
        return head + tail

    def __str__(self) -> str:  # pragma: no cover
        return self.summary()


@dataclass(frozen=True)
class ComparisonResult:
    """Result of comparing two systems.

    Returned by both entry points. ``paired`` says which one produced it,
    since that is the difference that matters for reading the interval.
    """

    difference: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int
    method: str
    binary: bool
    paired: bool
    confidence: float = 0.95
    n_discordant: Optional[int] = None

    @property
    def width(self) -> float:
        return self.ci_high - self.ci_low

    @property
    def crosses_zero(self) -> bool:
        return self.ci_low <= 0.0 <= self.ci_high

    def summary(self) -> str:
        conf = f"{self.confidence * 100:.0f}%"
        pairing = "paired" if self.paired else "independent"

        if self.binary:
            head = (
                f"Difference {_pct(self.difference)} "
                f"({conf} CI: {_pct(self.ci_low)} to {_pct(self.ci_high)}, "
                f"{pairing}, n={self.n})."
            )
        else:
            head = (
                f"Difference {self.difference:.3f} "
                f"({conf} CI: {self.ci_low:.3f} to {self.ci_high:.3f}, "
                f"{pairing}, n={self.n})."
            )

        if self.crosses_zero:
            verdict = (
                " The interval crosses zero, so the data cannot confirm "
                "that either system is better."
            )
        else:
            direction = "A" if self.difference > 0 else "B"
            verdict = f" System {direction} is higher by this measure."

        if self.n_discordant == 0:
            # An empty discordant table. The p-value is a convention here,
            # not a result, and the summary has to say so.
            verdict = (
                " No items changed between systems. The test has an empty "
                "table, so the p-value cannot support any claim in either "
                "direction."
            )
        elif self.n_discordant is not None:
            verdict += (
                f" {self.n_discordant} of {self.n} items changed between systems."
            )

        return head + verdict

    def __str__(self) -> str:  # pragma: no cover
        return self.summary()
