"""Return types for evalaudit.

Every public function returns one of these. Each carries the numbers and a
``summary()`` that states, in plain English, what the numbers do and do not
support. The summary is the point of the library: scipy already computes
McNemar, but nothing tells a product manager their result does not hold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


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


@dataclass(frozen=True)
class KappaResult:
    """Cohen's or Fleiss' kappa.

    Here because people ask for kappa by name. Krippendorff's alpha handles
    everything these two do and several things they do not, so
    ``rater_agreement`` is the one to reach for.
    """

    kappa: float
    p_observed: float
    p_expected: float
    n_items: int
    n_raters: int
    n_categories: int
    method: str

    def summary(self) -> str:
        name = "Cohen's" if self.method == "cohen" else "Fleiss'"
        if not (self.kappa == self.kappa):  # NaN
            return (
                f"{name} kappa is undefined. Every rating fell in one "
                f"category, so agreement by chance is already total and "
                f"there is nothing left for kappa to measure."
            )

        head = (
            f"{name} kappa {self.kappa:.3f} "
            f"({self.n_items} items, {self.n_raters} raters, "
            f"{self.n_categories} categories). "
            f"Raters agreed on {_pct(self.p_observed)} of ratings, "
            f"and {_pct(self.p_expected)} was expected by chance."
        )
        tail = (
            " Kappa moves with how often each category is used, so the same "
            "raters score lower on a lopsided scale than a balanced one. "
            "Krippendorff's alpha does not have that problem and handles "
            "missing data, so prefer rater_agreement for anything you report."
        )
        return head + tail

    def __str__(self) -> str:  # pragma: no cover
        return self.summary()


# Below this share of usable bootstrap resamples the interval is refused
# rather than computed from the survivors. Undefined resamples are the
# unanimous ones, so dropping them shaves the top off the distribution and
# pulls the upper bound down, hardest exactly when agreement is high.
_MIN_USABLE_SHARE = 0.90

# Krippendorff's own thresholds. Above 0.800 is reliable, 0.667 to 0.800
# supports tentative conclusions, below 0.667 supports none.
_ALPHA_RELIABLE = 0.800
_ALPHA_FLOOR = 0.667


@dataclass(frozen=True)
class AgreementResult:
    """Krippendorff's alpha, plus the two tables that say where it came from.

    ``item_disagreement`` ranks items by how much the graders differed on
    them, normalised so an item graded ten times is not punished against one
    graded twice. ``rater_dropout`` recomputes alpha with each rater removed.

    Both tables invite a reader to act on the top row, so ``summary()``
    refuses to name a rater unless dropping them shifts alpha by more than
    the sampling error on alpha.

    ``alpha`` is NaN when the data cannot carry an estimate: when fewer than
    two items were graded twice, or when every rating that could be compared
    was identical. Neither is perfect agreement and neither gets a number.
    """

    alpha: float
    ci_low: float
    ci_high: float
    n_items: int
    n_raters: int
    n_overlapping_items: int
    level: str
    item_disagreement: pd.DataFrame
    rater_dropout: pd.DataFrame
    confidence: float = 0.95
    n_boot: int = 0
    n_boot_usable: int = 0

    @property
    def has_interval(self) -> bool:
        return self.ci_low == self.ci_low and self.ci_high == self.ci_high

    @property
    def width(self) -> float:
        return self.ci_high - self.ci_low

    @property
    def dropout_is_distinguishable(self) -> bool:
        """True when some rater's removal shifts alpha by more than noise.

        The comparison is the leave-one-out shift against the sampling error
        on alpha, taken as half the interval width. Comparing the shifted
        alpha to ``ci_high`` instead looks equivalent and is not: when alpha
        sits near zero the interval runs far below it and barely above, so
        ``ci_high`` is a low bar and pure noise clears it.

        Without an interval there is nothing to measure the differences
        against, so the answer is False rather than a guess.
        """
        if not self.has_interval or self.rater_dropout.empty:
            return False
        best = self.rater_dropout["delta"].max()
        return bool(best == best and best > self.width / 2)

    @property
    def top_dropout_rater(self) -> Optional[str]:
        """The rater worth investigating, or None when the data cannot say.

        The dropout table always has a first row. This returns None unless
        that first row is doing more than sorting noise.
        """
        if not self.dropout_is_distinguishable:
            return None
        return self.rater_dropout.iloc[0]["rater_id"]

    def summary(self) -> str:
        if self.n_overlapping_items == 0:
            return (
                "Agreement was never measurable. No item was graded by more "
                "than one rater, so there is nothing to compare and "
                "Krippendorff's alpha does not exist for this data. Grade a "
                "sample of items twice and run this again."
            )

        carried = (
            f"{self.n_overlapping_items} of {self.n_items} items "
            f"graded more than once"
        )

        if self.alpha != self.alpha:  # NaN
            return self._undefined(carried) + self._dropout_sentence()

        if self.has_interval:
            head = (
                f"Krippendorff's alpha {self.alpha:.3f} "
                f"({self.confidence * 100:.0f}% CI: {self.ci_low:.3f} to "
                f"{self.ci_high:.3f}, {self.level}, {carried})."
            )
        else:
            head = (
                f"Krippendorff's alpha {self.alpha:.3f} "
                f"({self.level}, {carried})." + self._refusal_sentence()
            )

        return head + self._verdict() + self._dropout_sentence()

    @property
    def _no_variance(self) -> bool:
        """Every pairable rating was the same value.

        vs_chance divides by expected disagreement, so it is NaN across the
        board exactly when there was none.
        """
        table = self.item_disagreement
        return bool(not table.empty and table["vs_chance"].isna().all())

    def _undefined(self, carried: str) -> str:
        """Why there is no alpha. Two causes, and they are opposite news.

        Checked in the same order as the dropout notes: agreement first,
        because a rubric everyone applied identically is a finding, and a
        thin overlap is a scheduling problem.
        """
        if self._no_variance:
            return (
                f"Krippendorff's alpha is undefined ({self.level}, "
                f"{carried}). Every rating that could be compared was "
                f"identical, so there is no disagreement to divide by. That "
                f"is not perfect agreement, it is a scale nobody varied."
            )
        return (
            f"Krippendorff's alpha is undefined ({self.level}, {carried}). "
            f"One item cannot carry a reliability estimate, so there is no "
            f"number to report and no interval around it. The arithmetic "
            f"does return a value at this size, which is why it is withheld "
            f"rather than shown. Grade a larger sample twice."
        )

    def _refusal_sentence(self) -> str:
        """Why there is no interval, when there is still an alpha."""
        if self.n_boot == 0:
            return " An interval was not requested."
        dropped = self.n_boot - self.n_boot_usable
        return (
            f" No interval: {dropped} of {self.n_boot} resamples came back "
            f"undefined, above the {(1 - _MIN_USABLE_SHARE) * 100:.0f}% this "
            f"reports through. Those resamples are the unanimous ones, so "
            f"percentiles of the rest would understate the upper bound."
        )

    def _verdict(self) -> str:
        if self.alpha >= _ALPHA_RELIABLE:
            return (
                f" That is at or above {_ALPHA_RELIABLE:.3f}, the "
                f"conventional bar for treating coded data as reliable."
            )
        if self.alpha >= _ALPHA_FLOOR:
            return (
                f" That sits between {_ALPHA_FLOOR:.3f} and "
                f"{_ALPHA_RELIABLE:.3f}, which supports tentative "
                f"conclusions and no firm ones."
            )
        return (
            f" That is below {_ALPHA_FLOOR:.3f}, the conventional floor for "
            f"drawing any conclusion from coded data. Fix the rubric before "
            f"reading anything into the scores it produced."
        )

    def _dropout_sentence(self) -> str:
        if self.rater_dropout.empty:
            return ""

        if self.dropout_is_distinguishable:
            row = self.rater_dropout.iloc[0]
            return (
                f" Dropping {row['rater_id']} raises alpha to "
                f"{row['alpha_without']:.3f}, a shift of {row['delta']:.3f} "
                f"against a sampling error of {self.width / 2:.3f}. They "
                f"graded {int(row['n_ratings'])} items, so weigh that shift "
                f"against how much of the grading it rests on."
            )

        if self.rater_dropout["alpha_without"].isna().all():
            return (
                " Every leave-one-out alpha is undefined, so the data cannot "
                "distinguish the raters."
            )

        if not self.has_interval:
            return (
                " Without an interval on alpha there is nothing to measure "
                "the leave-one-out differences against, so the data cannot "
                "distinguish the raters."
            )

        return (
            f" The largest leave-one-out shift is "
            f"{self.rater_dropout['delta'].max():.3f}, inside the sampling "
            f"error of {self.width / 2:.3f} on alpha, so the data cannot "
            f"distinguish the raters. Do not read the top of the dropout "
            f"table as an outlier."
        )

    def __str__(self) -> str:  # pragma: no cover
        return self.summary()
