"""Return types for evalaudit.

Every public function returns one of these. Each carries the numbers and a
``summary()`` that states, in plain English, what the numbers do and do not
support. The summary is the point of the library: scipy already computes
McNemar, but nothing tells a product manager their result does not hold.
"""

from __future__ import annotations

import html as _html
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _plural(word: str, n: int) -> str:
    return word if n == 1 else word + "s"


def _escape(text: str) -> str:
    """HTML escape for a text node.

    System names, slice labels and rater ids come out of the client's
    spreadsheet and land in the report. They are data, not markup.
    """
    return _html.escape(str(text), quote=False)


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
                f"identical, so there is no disagreement to divide by. This "
                f"comes from a scale nobody varied. It is not perfect "
                f"agreement."
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


# --------------------------------------------------------------------------
# pairwise
# --------------------------------------------------------------------------

# How the two tie policies read in a summary. Davidson's model is the third
# entry a reader might expect and it is deliberately absent. See
# evalaudit.pairwise for why there is a hook and no implementation.
_TIE_WORDS = {
    "split": "ties split",
    "drop": "ties dropped",
}


def _join(names) -> str:
    """Names in a sentence, with the Oxford comma."""
    names = list(names)
    if len(names) == 1:
        return str(names[0])
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(str(n) for n in names[:-1]) + f", and {names[-1]}"


@dataclass(frozen=True)
class BTResult:
    """Bradley-Terry ratings, and how few of the models they actually order.

    ``ratings`` carries one row per model with its rating, an interval and
    the number of comparisons behind it. ``win_matrix`` is the modelled
    probability that the row model beats the column model. ``pairs`` has one
    row per pair of models, higher-rated model first, with the gap between
    their ratings, a percentile interval on that gap, and whether the pair
    is separable. ``separable_pairs`` is the separable rows of ``pairs``.

    A pair is separable when the interval on its gap excludes zero.
    ``n_separable`` against ``n_pairs`` is the number worth reporting. A six
    model leaderboard has fifteen pairs, and published boards routinely rank
    all six off data that orders a few of them.

    A pair that does not separate is one the data has not established an
    order for, in either direction. The data does not show those two models
    level. More comparisons could separate them, and nothing here says how
    many it would take.

    The ratings are differences and nothing else. One model is pinned at zero
    to make the fit identifiable, and which one is a free choice, so the level
    of a single rating carries no information. Only gaps do. The rating
    intervals are on each rating measured against the average of the field,
    then shifted onto the anchor named in ``reference``. Anchoring them on
    the reference instead would hand the reference an interval of width
    zero. The gap intervals need no anchor. A constant added to both ratings
    cancels in the gap, so no choice of reference moves them or changes
    ``n_separable``.

    Before 0.3.0 a pair was called separable when its two rating intervals
    did not overlap. That test undercounts. Two rating intervals can overlap
    while the gap between them is well measured, because both ratings carry
    the error of the field average they are measured against, and that error
    cancels in the gap.

    ``resample`` records what one bootstrap draw picked up, ``"items"`` or
    ``"comparisons"``.

    ``rating`` is NaN for every model when the fit does not exist. That is
    two situations and they are not the same. ``connected`` is False when the
    models fall into groups that never met, so there is no common scale.
    ``undefeated`` or ``winless`` is non-empty when they all met and some set
    of models never lost outside itself, which sends its rating to infinity.
    """

    ratings: pd.DataFrame
    win_matrix: pd.DataFrame
    separable_pairs: pd.DataFrame
    reference: str
    ties: str
    n_models: int
    n_comparisons: int
    n_ties: int
    n_dropped: int
    n_items: int
    n_pairs: int
    connected: bool
    comparable_groups: tuple
    undefeated: tuple
    winless: tuple
    confidence: float = 0.95
    n_boot: int = 0
    n_boot_usable: int = 0
    elo_scale: Optional[float] = None
    elo_base: Optional[float] = None
    resample: str = "items"
    pairs: Optional[pd.DataFrame] = None

    @property
    def has_fit(self) -> bool:
        """True when the maximum exists and is unique.

        Ford's condition. Every model has to be reachable from every other
        through a chain of wins, which is stronger than everyone having
        played. A model that never lost has no finite rating.
        """
        return self.connected and not self.undefeated

    @property
    def has_interval(self) -> bool:
        if self.ratings.empty:
            return False
        return bool(self.ratings["ci_low"].notna().all())

    @property
    def is_elo(self) -> bool:
        return self.elo_scale is not None

    @property
    def n_separable(self) -> int:
        return int(len(self.separable_pairs))

    @property
    def units(self) -> str:
        return "Elo" if self.is_elo else "log-odds"

    def summary(self) -> str:
        if not self.connected:
            return self._disconnected()
        if not self.has_fit:
            return self._no_maximum()
        return (
            self._head()
            + self._separability()
            + self._resample_sentence()
            + self._reference_sentence()
        )

    # -- the two refusals ---------------------------------------------------

    def _disconnected(self) -> str:
        groups = "; ".join(_join(g) for g in self.comparable_groups)
        return (
            f"No ratings. The {self.n_models} models fall into "
            f"{len(self.comparable_groups)} groups that never met: {groups}. "
            f"Bradley-Terry puts models on one scale by chaining comparisons "
            f"between them, and there is no chain from one group to the "
            f"next, so nothing here can rank them against each other. Run "
            f"comparisons across the groups, or rate each group on its own "
            f"and report them as separate boards."
        )

    def _no_maximum(self) -> str:
        head = (
            f"No ratings. Every model was compared, and the fit still does "
            f"not exist. "
        )
        if len(self.undefeated) == 1:
            body = (
                f"{self.undefeated[0]} never lost a comparison, so the "
                f"likelihood keeps rising as its rating goes up and there is "
                f"no maximum to report."
            )
        elif len(self.winless) == 1:
            body = (
                f"{self.winless[0]} never won a comparison, so the "
                f"likelihood keeps rising as its rating goes down and there "
                f"is no maximum to report."
            )
        else:
            body = (
                f"No model outside {_join(self.undefeated)} ever beat a "
                f"model inside it, so that group's ratings run away from the "
                f"rest and there is no maximum to report."
            )
        return head + body + (
            " A rating needs every model reachable from every other through "
            "a chain of wins, which a clean sweep breaks. Report the record "
            "directly, or add comparisons that close the chain."
        )

    # -- the ordinary report ------------------------------------------------

    def _head(self) -> str:
        scale = (
            f"Elo ratings (scale {self.elo_scale:g}, base {self.elo_base:g})"
            if self.is_elo
            else "Bradley-Terry ratings"
        )
        ties = _TIE_WORDS.get(self.ties, self.ties)
        if self.ties == "drop" and self.n_dropped:
            ties = f"{ties}, {self.n_dropped} of them"
        elif self.n_ties:
            ties = f"{ties}, {self.n_ties} of them"
        return (
            f"{scale} for {self.n_models} models from "
            f"{self.n_comparisons} {_plural('comparison', self.n_comparisons)} "
            f"({ties})."
        )

    def _separability(self) -> str:
        if not self.has_interval:
            return self._no_interval_sentence()

        conf = f"{self.confidence * 100:.0f}%"
        head = (
            f" {self.n_separable} of {self.n_pairs} pairs "
            f"{'is' if self.n_pairs == 1 else 'are'} separable at {conf}, "
            f"meaning the interval on the gap between the two ratings "
            f"excludes zero."
        )
        if self.n_separable == 0:
            return head + (
                " This data does not establish an order for any pair. A "
                "ranking built on it puts the models in a line this data has "
                "not established. That does not mean the models are level. "
                "More comparisons could separate them."
            )
        rest = self.n_pairs - self.n_separable
        if rest == 0:
            return head + " Every pair is ordered by this data."
        which = "that pair" if rest == 1 else "those pairs"
        return head + (
            f" For the other {rest} {_plural('pair', rest)} the interval "
            f"includes zero, so this data does not establish an order for "
            f"{which} in either direction. That does not mean the models are "
            f"level. More comparisons could separate them. A leaderboard that "
            f"puts those models in a line is showing an order this data has "
            f"not established."
        )

    def _resample_sentence(self) -> str:
        """Said only when comparisons were resampled over shared items.

        With no item ids at all there is nothing to say which comparisons
        share a prompt, so nothing is said.
        """
        if (
            not self.has_interval
            or self.resample != "comparisons"
            or not 0 < self.n_items < self.n_comparisons
        ):
            return ""
        return (
            f" The intervals resample single comparisons, and these "
            f"{self.n_comparisons} comparisons share {self.n_items} "
            f"{_plural('item', self.n_items)}. Judgements of the same item "
            f"tend to move together, so intervals built this way can run "
            f"narrow. The default resamples whole items."
        )

    def _no_interval_sentence(self) -> str:
        if self.n_boot == 0:
            return (
                " No interval was requested, so no pair can be called "
                "separable. Separability is the finding here, so run this "
                "again with resamples."
            )
        dropped = self.n_boot - self.n_boot_usable
        return (
            f" No interval: {dropped} of {self.n_boot} resamples came back "
            f"undefined, above the {(1 - _MIN_USABLE_SHARE) * 100:.0f}% this "
            f"reports through. A resample that misses a comparison can leave "
            f"a model unbeaten inside it, and those resamples are the ones "
            f"with the widest ratings, so percentiles of the rest would "
            f"understate the spread. Without an interval no pair can be "
            f"called separable."
        )

    def _reference_sentence(self) -> str:
        anchor = f"{self.elo_base:g}" if self.is_elo else "0"
        return (
            f" Ratings are anchored on {self.reference} at {anchor}. Only "
            f"differences between models mean anything, and adding the same "
            f"amount to every rating would change nothing about the fit or "
            f"about which pairs separate."
        )

    def __str__(self) -> str:  # pragma: no cover
        return self.summary()


# Slices thinner than this get a hedge in the summary even when they clear
# the interval. Same count ScoreCI uses to call an estimate weak.
_THIN_SLICE = 30

# Why a slice has no alpha. Checked in this order, which is the opposite of
# the order rater_dropout uses. There, a rater whose removal leaves everyone
# agreeing is a finding worth stating over the item count. Here a slice with
# one item is too small to measure whatever else is true of it.
_NOTE_SLICE_ONE_ITEM = "one item, too few to measure agreement"
_NOTE_SLICE_NO_VARIANCE = (
    "human and judge used one label throughout, so alpha has no denominator"
)


@dataclass(frozen=True)
class JudgeValidation:
    """How well an LLM judge's labels line up with human labels.

    ``agreement`` is Krippendorff's alpha between the two, so it is the same
    statistic ``rater_agreement`` reports and reads the same way. A judge is
    a rater. ``accuracy`` is the plain share of items where the two labels
    matched, reported beside it because people ask for it and because the
    gap between the two numbers is informative on its own. On a lopsided
    label set a judge that always guesses the common label scores high
    accuracy and no agreement at all.

    ``by_slice`` is the part worth reading. Judges track humans on the easy
    cases and come apart where the decision is close, so a good headline
    number routinely hides the failure. The table is sorted ascending, worst
    first, and ``summary()`` refuses to name the top row unless its
    agreement falls below the interval on the overall figure.
    """

    agreement: float
    ci_low: float
    ci_high: float
    accuracy: float
    n_items: int
    n_dropped: int
    level: str
    by_slice: pd.DataFrame
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
    def slice_is_distinguishable(self) -> bool:
        """True when the worst slice clears the overall interval outright.

        The noise guard rater_dropout uses, in the form this statistic
        takes. Cut a homogeneous population into five slices and one of them
        is last by sampling noise alone, so a table that always ranks
        somebody first needs a rule about when the first row means anything.

        The rule is that the slice's own interval has to sit entirely below
        the interval on the overall figure. The slice's agreement being
        below the overall lower bound is necessary and is not enough on its
        own, because a slice holds a fraction of the items and its sampling
        error is correspondingly wider. Cut three hundred homogeneous items
        into five slices of sixty and the lowest one drops under the overall
        lower bound most of the time, so that test alone names a slice on
        noise, which is the thing it exists to prevent.

        One slice is the whole dataset under another name, so it never
        qualifies. Without intervals there is nothing to place the slices
        against and the answer is False rather than a guess.
        """
        if not self.has_interval or len(self.by_slice) < 2:
            return False
        worst = self.by_slice.iloc[0]
        return bool(
            np.isfinite(worst["agreement"])
            and np.isfinite(worst["ci_high"])
            and worst["ci_high"] < self.ci_low
        )

    @property
    def worst_slice(self):
        """The slice worth investigating, or None when the data cannot say."""
        if not self.slice_is_distinguishable:
            return None
        return self.by_slice.iloc[0]["slice"]

    def summary(self) -> str:
        return self._head() + self._verdict() + self._slice_sentence()

    def _head(self) -> str:
        conf = f"{self.confidence * 100:.0f}%"
        accuracy = f" Plain accuracy is {_pct(self.accuracy)}."
        dropped = ""
        if self.n_dropped:
            dropped = (
                f" {self.n_dropped} items were set aside because one side "
                f"had no label."
            )

        if self.agreement != self.agreement:  # NaN
            return (
                f"Judge and human agreement is undefined ({self.level}, "
                f"{self.n_items} items). Every label that could be compared "
                f"was identical, so there is no disagreement to divide by. "
                f"This comes from a rubric with one label in it. It is not "
                f"perfect agreement." + accuracy + dropped
            )

        if self.has_interval:
            head = (
                f"Judge and human agree at alpha {self.agreement:.3f} "
                f"({conf} CI: {self.ci_low:.3f} to {self.ci_high:.3f}, "
                f"{self.level}, {self.n_items} items)."
            )
        else:
            head = (
                f"Judge and human agree at alpha {self.agreement:.3f} "
                f"({self.level}, {self.n_items} items). No interval was "
                f"computed, so nothing here is placed against sampling error."
            )
        return head + accuracy + dropped

    def _verdict(self) -> str:
        if self.agreement != self.agreement:
            return ""
        if self.agreement >= _ALPHA_RELIABLE:
            return (
                f" That is at or above {_ALPHA_RELIABLE:.3f}, the "
                f"conventional bar for treating coded data as reliable."
            )
        if self.agreement >= _ALPHA_FLOOR:
            return (
                f" That sits between {_ALPHA_FLOOR:.3f} and "
                f"{_ALPHA_RELIABLE:.3f}, which supports tentative "
                f"conclusions and no firm ones."
            )
        return (
            f" That is below {_ALPHA_FLOOR:.3f}, the conventional floor for "
            f"drawing any conclusion from coded data. The judge is not a "
            f"stand-in for the humans at this level."
        )

    def _slice_sentence(self) -> str:
        if self.by_slice.empty:
            return ""

        if len(self.by_slice) < 2:
            return (
                " One slice is the whole dataset under another name, so "
                "there is nothing to compare it against."
            )

        if self.slice_is_distinguishable:
            row = self.by_slice.iloc[0]
            count = int(row["n_items"])
            sentence = (
                f" Worst slice {str(row['slice'])!r} agrees at "
                f"{row['agreement']:.3f} on {count} items, and its interval "
                f"tops out at {row['ci_high']:.3f}, clear of the "
                f"{self.ci_low:.3f} lower bound on the overall figure. The "
                f"headline number is carried by the rest of the data."
            )
            if count < _THIN_SLICE:
                sentence += (
                    f" That slice holds only {count} items, so treat the gap "
                    f"as provisional and grade more of them before acting."
                )
            return sentence

        if not self.has_interval:
            return (
                " Without an interval on the overall figure there is nothing "
                "to place the slices against, so the data cannot single out "
                "a slice. Do not read the top of the table as a failure."
            )

        row = self.by_slice.iloc[0]
        if not np.isfinite(row["agreement"]) or not np.isfinite(row["ci_high"]):
            return (
                " No slice carries both an agreement figure and an interval, "
                "so the data cannot single out a slice. Do not read the top "
                "of the table as a failure."
            )
        return (
            f" The lowest slice {str(row['slice'])!r} runs up to "
            f"{row['ci_high']:.3f} against a lower bound of "
            f"{self.ci_low:.3f} on the overall figure, so the two overlap "
            f"and the data cannot single out a slice. Something is always "
            f"last. Do not read the top of the table as a failure."
        )

    def __str__(self) -> str:  # pragma: no cover
        return self.summary()


@dataclass(frozen=True)
class PositionBias:
    """Whether the judge favours whichever output it saw first.

    Two designs answer this question and they answer it with different
    arithmetic, so ``design`` says which one the data turned out to be.

    ``"randomised"`` means each pair was judged once with the order shuffled
    beforehand. Then the position-A win rate is tested against a half.

    ``"both_orders"`` means the same pair was judged twice, once each way.
    Then the headline is the consistency rate, the share of pairs where the
    judge named the same output both times, and the direction of the pairs
    it flipped on says whether the flipping was position or noise.
    """

    design: str
    estimate: float
    ci_low: float
    ci_high: float
    p_value: float
    position_a_rate: float
    n_a_wins: int
    n_decisive: int
    consistency_rate: float
    n_pairs: int
    n_pairs_scored: int
    n_judgements: int
    n_both_orders: int
    n_ties: int
    confidence: float = 0.95

    @property
    def has_position_effect(self) -> bool:
        """True when the interval on the position-A rate clears a half.

        In the both-orders design this reads the direction of the flips, not
        the consistency rate, because an inconsistent judge and a
        position-biased one are different problems.
        """
        if self.design == "both_orders":
            if self.n_decisive == 0 or self.p_value != self.p_value:
                return False
            return bool(self.p_value < 1 - self.confidence)
        if not (self.ci_low == self.ci_low):
            return False
        return not (self.ci_low <= 0.5 <= self.ci_high)

    def summary(self) -> str:
        conf = f"{self.confidence * 100:.0f}%"
        if self.design == "both_orders":
            return self._both_orders(conf) + self._ties()
        return self._randomised(conf) + self._ties()

    def _randomised(self, conf: str) -> str:
        head = (
            f"Each pair was judged once, so this reports the position-A win "
            f"rate. Position A won {_pct(self.position_a_rate)} of "
            f"{self.n_decisive} judgements ({conf} CI: {_pct(self.ci_low)} "
            f"to {_pct(self.ci_high)}, exact binomial p={self.p_value:.4f})."
        )

        if self.has_position_effect:
            side = "first" if self.position_a_rate > 0.5 else "second"
            verdict = (
                f" The interval clears 50%, so the judge favours whichever "
                f"output it sees {side}."
            )
        else:
            verdict = (
                " The interval covers 50%, so the data cannot show that "
                "position moved the judge."
            )

        caveat = (
            " This assumes presentation order was randomised, which the data "
            "cannot confirm. If the same system sat in position A each time, "
            "the same number appears when that system is simply better."
        )

        stray = ""
        if self.n_both_orders:
            stray = (
                f" {self.n_both_orders} pairs were also run in the reverse "
                f"order, too few to change which analysis applies."
            )
        return head + verdict + caveat + stray

    def _both_orders(self, conf: str) -> str:
        head = (
            f"{self.n_both_orders} of {self.n_pairs} pairs were run in both "
            f"orders, so this reports the consistency rate. The judge named "
            f"the same output under both orderings on "
            f"{_pct(self.consistency_rate)} of {self.n_pairs_scored} pairs "
            f"({conf} CI: {_pct(self.ci_low)} to {_pct(self.ci_high)})."
        )

        if self.n_decisive == 0:
            return head + (
                " It never flipped, so there is no direction to test and "
                "nothing here points at position."
            )

        direction = (
            f" Of the {self.n_decisive} pairs it flipped on, {self.n_a_wins} "
            f"went to whichever output was shown first "
            f"({_pct(self.position_a_rate)}, exact binomial "
            f"p={self.p_value:.4f})."
        )

        if self.has_position_effect:
            side = "first" if self.position_a_rate > 0.5 else "second"
            verdict = (
                f" The flips have a direction, so this is position bias "
                f"rather than an unsteady judge. It reaches for whatever it "
                f"sees {side}."
            )
        else:
            verdict = (
                " The flips split evenly across the two positions, so this "
                "is an unsteady judge rather than a position-biased one. "
                "Inconsistency is its own problem and does not become "
                "position bias without a direction."
            )
        return head + direction + verdict

    def _ties(self) -> str:
        if not self.n_ties:
            return ""
        return (
            f" {self.n_ties} judgements were ties and are left out of the "
            f"rates above."
        )

    def __str__(self) -> str:  # pragma: no cover
        return self.summary()


@dataclass(frozen=True)
class LengthBias:
    """Two logistic fits on the same length difference, and they answer
    different questions.

    The first regresses judge preference on the length difference between
    the two answers. A positive coefficient says the judge goes for the
    longer answer, and on its own that is not bias, because longer answers
    may be better.

    The second regresses judge-human disagreement on the same length
    difference, oriented by what the humans picked, so a positive
    coefficient says the judge breaks with the humans on the pairs where the
    humans went short. Holding the human verdict fixed removes the part of
    the length-quality link the human labels capture. It does not remove the
    rest. A binary human label is a coarse measure of quality and the length
    difference still carries quality information the label missed, so on a
    corpus where length tracks quality closely this coefficient stays
    positive for a judge with no length preference at all. It is the sharper
    of the two and it is not a clean separation.
    """

    coefficient: float
    ci_low: float
    ci_high: float
    p_value: float
    sd_difference: float
    note: str
    disagreement_coefficient: float
    disagreement_ci_low: float
    disagreement_ci_high: float
    disagreement_p_value: float
    disagreement_sd_difference: float
    disagreement_note: str
    n: int
    n_disagreements: int
    longer_rate: float
    has_human: bool
    confidence: float = 0.95

    @property
    def odds_ratio_per_sd(self) -> float:
        """What one standard deviation of extra length does to the odds.

        The raw coefficient is a log-odds per character, which nobody can
        read. This puts it on a step the data actually contains.
        """
        return float(np.exp(self.coefficient * self.sd_difference))

    @property
    def disagreement_odds_ratio_per_sd(self) -> float:
        return float(
            np.exp(self.disagreement_coefficient * self.disagreement_sd_difference)
        )

    def summary(self) -> str:
        return self._preference() + self._disagreement()

    def _preference(self) -> str:
        conf = f"{self.confidence * 100:.0f}%"
        if self.coefficient != self.coefficient:  # NaN
            return (
                f"No judge preference model: {self.note} ({self.n} pairs). "
                f"The judge picked the longer answer on "
                f"{_pct(self.longer_rate)} of pairs."
            )

        head = (
            f"Judge preference against length: {self.sd_difference:.0f} "
            f"characters of extra length multiplies the odds the judge picks "
            f"that answer by {self.odds_ratio_per_sd:.2f} ({conf} CI: "
            f"{np.exp(self.ci_low * self.sd_difference):.2f} to "
            f"{np.exp(self.ci_high * self.sd_difference):.2f}, "
            f"p={self.p_value:.4f}, {self.n} pairs). It picked the longer "
            f"answer on {_pct(self.longer_rate)} of pairs."
        )

        if self.ci_low <= 0 <= self.ci_high:
            return head + (
                " The interval covers no effect, so the data cannot show "
                "that length moved the judge."
            )
        if self.coefficient > 0:
            return head + (
                " A positive coefficient here is not bias on its own, because "
                "longer answers may simply be better."
            )
        # The two signs do not read the same way, so they do not get the
        # same sentence. The excuse that saves a positive coefficient is
        # that length carries quality, and it does not run in reverse.
        return head + (
            " A negative coefficient is harder to explain away than a "
            "positive one. Length can track quality, so a preference for "
            "long answers may be reading real content. Brevity rarely "
            "tracks quality in the same way, so a preference for short "
            "answers usually points at the judge."
        )

    def _disagreement(self) -> str:
        conf = f"{self.confidence * 100:.0f}%"
        if not self.has_human:
            return (
                " Without human labels there is no second model, so this "
                "cannot separate a length preference from longer answers "
                "being better. Supply human_preferences to get the sharper "
                "of the two."
            )

        if self.disagreement_coefficient != self.disagreement_coefficient:
            return (
                f" No disagreement model: {self.disagreement_note} "
                f"({self.n_disagreements} disagreements)."
            )

        head = (
            f" Judge-human disagreement against the same length difference, "
            f"oriented by what the humans picked: "
            f"{self.disagreement_sd_difference:.0f} characters multiplies "
            f"the odds the judge breaks with them by "
            f"{self.disagreement_odds_ratio_per_sd:.2f} ({conf} CI: "
            f"{np.exp(self.disagreement_ci_low * self.disagreement_sd_difference):.2f}"
            f" to "
            f"{np.exp(self.disagreement_ci_high * self.disagreement_sd_difference):.2f}"
            f", p={self.disagreement_p_value:.4f}, "
            f"{self.n_disagreements} disagreements)."
        )

        if self.disagreement_ci_low <= 0 <= self.disagreement_ci_high:
            verdict = (
                " That interval covers no effect, so the judge does not "
                "depart from the humans in the direction of length."
            )
        elif self.disagreement_coefficient > 0:
            verdict = (
                " The judge departs from the humans in the direction of "
                "length."
            )
        else:
            # The predictor is the length the humans passed over minus the
            # one they picked, so a negative coefficient is the judge
            # breaking with them on the pairs where the humans went long.
            verdict = (
                " The judge departs from the humans in the direction of "
                "brevity."
            )

        caveat = (
            " Holding the human verdict fixed strips the part of the "
            "length-quality link the human labels capture, and it does not "
            "remove the rest, since a binary label is a coarse measure of "
            "quality. Read this as an indication rather than as proof."
        )
        return head + verdict + caveat

    def __str__(self) -> str:  # pragma: no cover
        return self.summary()


def _points(x: float) -> str:
    """A difference as points of pass rate, trimmed of trailing zeros."""
    return f"{round(x * 100, 1):g}"


def _rate(x: float) -> str:
    """A rate as a percentage, trimmed of trailing zeros."""
    return f"{round(x * 100, 1):g}%"


@dataclass(frozen=True)
class PowerResult:
    """What an eval of a given size could find, or the size a target needs.

    One relation solved in two directions. ``solved_for`` says which, either
    ``"difference"`` from :func:`evalaudit.detectable_effect` or ``"n"``
    from :func:`evalaudit.min_sample_size`.

    ``n`` is the number of paired comparisons in a paired design and the
    total number of items across both groups in an independent one, which
    is the convention ``ComparisonResult`` already uses.

    ``attainable`` is False when no difference reaches the requested power
    at this size. For a paired design ``difference`` is then the figure the
    arithmetic returns even though it sits above the discordance rate,
    which is the ceiling a paired difference cannot pass. For an
    independent design there is no such figure and ``difference`` is NaN.

    ``discordance_assumed`` is True when no discordance rate was supplied
    and the conservative default stood in. The summary says so every time.
    Nothing here assumes that rate quietly.
    """

    difference: float
    n: int
    baseline: float
    power: float
    alpha: float
    paired: bool
    solved_for: str
    attainable: bool
    discordance_rate: Optional[float] = None
    discordance_assumed: bool = False
    n_discordant: Optional[int] = None
    n_per_group: Optional[float] = None
    reference_difference: Optional[float] = None
    n_for_reference: Optional[int] = None

    @property
    def has_difference(self) -> bool:
        return bool(np.isfinite(self.difference))

    def summary(self) -> str:
        if self.solved_for == "n":
            return self._required_size()
        return self._reach()

    # ------------------------------------------------------------------
    # min_sample_size
    # ------------------------------------------------------------------

    def _required_size(self) -> str:
        target = (
            f"To detect a {_points(self.difference)} point difference at "
            f"{_rate(self.power)} power and a {_rate(self.alpha)} "
            f"significance level"
        )
        if self.paired:
            head = f"{target} you need roughly {self.n:,} paired comparisons."
            return (
                head
                + self._discordant_sentence()
                + self._assumed_sentence()
                + (
                    f" The baseline pass rate does not enter a paired binary "
                    f"calculation. Two systems can both pass "
                    f"{_rate(self.baseline)} of items and disagree on none of "
                    f"them or on all of them, and it is the disagreement that "
                    f"sets the number above."
                )
            )

        head = (
            f"{target} from a {_rate(self.baseline)} baseline you need "
            f"roughly {self.n:,} items, {self.n_per_group:,g} in each group."
        )
        return head + self._pairing_sentence()

    # ------------------------------------------------------------------
    # detectable_effect
    # ------------------------------------------------------------------

    def _reach(self) -> str:
        if self.paired:
            head = f"This eval ran {self.n:,} paired comparisons."
        else:
            head = (
                f"This eval ran {self.n:,} items, {self.n_per_group:,g} in "
                f"each group."
            )
        if not self.attainable:
            return head + self._out_of_reach() + self._discordant_sentence()
        return (
            head
            + self._smallest_finding()
            + self._discordant_sentence()
            + self._reference_sentence()
            + self._assumed_sentence()
            + (self._floor_sentence() if self.paired else self._pairing_sentence())
        )

    def _smallest_finding(self) -> str:
        against = "" if self.paired else f" against a {_rate(self.baseline)} baseline"
        return (
            f" At {_rate(self.power)} power and a {_rate(self.alpha)} "
            f"significance level{against} the smallest difference it could "
            f"have found is {_points(self.difference)} points, and anything "
            f"smaller was out of reach before the first item was graded."
        )

    def _out_of_reach(self) -> str:
        if self.paired:
            if self.has_difference:
                needed = (
                    f" Reaching {_rate(self.power)} power here would take a "
                    f"difference of {_points(self.difference)} points."
                )
            else:
                needed = (
                    f" No difference reaches {_rate(self.power)} power at "
                    f"this size."
                )
            return needed + (
                f" A difference can never exceed the discordance rate, which "
                f"here is {_rate(self.discordance_rate)}, so no difference at "
                f"all was detectable. This eval could not have found "
                f"anything, whatever the two systems really do."
            )
        return (
            f" A {_rate(self.baseline)} baseline leaves at most a "
            f"{_points(1.0 - self.baseline)} point difference before the pass "
            f"rate hits 100%, and this many items cannot find even that at "
            f"{_rate(self.power)} power. So no difference at all was "
            f"detectable, and this eval could not have found anything."
        )

    # ------------------------------------------------------------------
    # Shared clauses
    # ------------------------------------------------------------------

    def _discordant_sentence(self) -> str:
        if not self.paired or self.n_discordant is None:
            return ""
        return (
            f" At a {_rate(self.discordance_rate)} discordance rate that is "
            f"about {self.n_discordant:,} discordant pairs, and McNemar reads "
            f"only those. The rest of the items agree across both systems and "
            f"carry no information about which one is better."
        )

    def _reference_sentence(self) -> str:
        if self.n_for_reference is None or self.reference_difference is None:
            return ""
        unit = "comparisons" if self.paired else "items"
        return (
            f" To find a {_points(self.reference_difference)} point "
            f"difference you would have needed roughly "
            f"{self.n_for_reference:,} {unit}, and this eval ran {self.n:,}."
        )

    def _assumed_sentence(self) -> str:
        if not self.discordance_assumed:
            return ""
        return (
            f" The discordance rate was not supplied, so "
            f"{_rate(self.discordance_rate)} was assumed. That is a "
            f"conservative stand-in and it is not a measurement. Supply the "
            f"rate your own eval produced, since a lower rate needs fewer "
            f"pairs and a higher one needs more."
        )

    def _floor_sentence(self) -> str:
        return (
            " This is the standard normal approximation for McNemar, and it "
            "runs a little ahead of the continuity corrected and exact forms "
            "evalaudit actually uses. Read the number as a floor on what the "
            "eval could have found. The true reach is slightly worse."
        )

    def _pairing_sentence(self) -> str:
        return (
            " Running both systems on the same items would cut this sharply. "
            "Pairing removes the item to item difficulty that an independent "
            "comparison has to absorb, and most evals can pair."
        )

    def __str__(self) -> str:  # pragma: no cover
        return self.summary()


# --------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------

# The three words a finding can carry, in the order a report is read. They
# are not degrees of one thing. Critical means the conclusion does not hold,
# warning means it holds and something weakens it, info is context.
_SEVERITIES = ("critical", "warning", "info")

# The line for judge-human agreement. It is the same number as _ALPHA_FLOOR
# and it is not the same kind of number. See AuditConfig.
_JUDGE_THRESHOLD = _ALPHA_FLOOR


@dataclass(frozen=True)
class AuditConfig:
    """What the audit is being asked to check, and what it may assume.

    Everything here has a default that suits a two system pass rate eval, so
    a first run needs none of it. The three worth setting are ``claim``,
    which puts the sentence being audited at the top of the report,
    ``effect_of_interest``, which is the difference the decision actually
    turns on, and ``claims_direction``, which says whether anyone is
    asserting that one system beat the other. A margin whose interval
    crosses zero is critical when a direction is being claimed and a warning
    when it is not, because in the second case nobody is leaning on it.

    ``paired`` is left as None so the design is read off the data. Two
    systems scored on the same number of items are taken as paired, which is
    how most evals are built. Set it when that guess would be wrong.

    ``discordance_rate`` is for paired binary designs only. When both score
    sets are 0/1 the rate is measured from the data and this is ignored,
    since a measured rate always beats a supplied one.

    ``n_boot`` sizes the agreement and judge bootstraps, the two that
    resample. The score and comparison intervals keep their own module
    defaults, which are higher because those resamples are cheap.

    Thresholds
    ----------
    The two default to the same number and they earned it differently, so
    they are separate settings and should be set separately.

    ``agreement_threshold`` is 0.667, which is Krippendorff's own published
    line: above 0.800 is reliable, 0.667 to 0.800 supports tentative
    conclusions, below 0.667 supports none. ``rater_agreement`` already
    reads its verdict off those same cutoffs, so the audit is repeating the
    module rather than inventing a rule.

    ``judge_threshold`` is 0.667 because it was borrowed from there. It is a
    convention imported from content analysis, where it describes when two
    humans coding text agree well enough to pool their work. Nobody has
    validated it as the point where an LLM judge becomes safe to rank
    systems with, and this library is not claiming they have. Set it from
    what the decision can tolerate. A judge picking which of two models
    ships needs to track humans far more closely than one triaging a queue
    for human review, and neither number is 0.667 for any reason beyond
    habit.
    """

    claim: Optional[str] = None
    claims_direction: bool = True
    effect_of_interest: Optional[float] = None
    confidence: float = 0.95
    level: str = "nominal"
    agreement_threshold: float = _ALPHA_FLOOR
    judge_threshold: float = _JUDGE_THRESHOLD
    power: float = 0.8
    alpha: float = 0.05
    paired: Optional[bool] = None
    discordance_rate: Optional[float] = None
    n_boot: int = 1000
    seed: Optional[int] = None


@dataclass(frozen=True)
class Finding:
    """One thing the audit found, and what it means for the claim.

    ``severity`` is one of the three words in ``_SEVERITIES``. ``check``
    names the check that produced this, which is what the report sorts on
    inside a severity. ``result`` is the object the underlying module
    returned, kept so a reader can go back to the numbers, and ``detail``
    carries that object's own ``summary()`` word for word, framed by a line
    on what it means for the claim and a line on what to do about it.
    """

    check: str
    severity: str
    title: str
    detail: str
    result: object = None

    def summary(self) -> str:
        return f"[{self.severity}] {self.title}. {self.detail}"

    def __str__(self) -> str:  # pragma: no cover
        return self.summary()


@dataclass(frozen=True)
class SkippedCheck:
    """A check that could not run, and the data that would have let it.

    Silence about a check reads as a pass. Every check the audit knows how
    to run either produces a finding or turns up here saying what was
    missing, so a thin report is visibly thin.
    """

    check: str
    title: str
    reason: str

    def summary(self) -> str:
        return f"{self.title}: {self.reason}"

    def __str__(self) -> str:  # pragma: no cover
        return self.summary()


@dataclass(frozen=True)
class AuditReport:
    """Everything the audit found, ranked by what changes the decision.

    ``findings`` is sorted by severity first, so the report opens with what
    undoes the conclusion, and by check order inside a severity. A reader
    who stops after the first entry has read the most important one.

    ``not_run`` is the other half of the report. A check with no data behind
    it is listed there with the reason, never left out.

    A report of nothing but info is a result, not an empty file. It says the
    eval holds up on everything that could be tested, and it still carries
    every number, because those numbers are what makes it worth sending.
    """

    findings: tuple
    not_run: tuple
    config: AuditConfig

    @property
    def severity_counts(self) -> dict:
        """How many findings of each severity, zeros included.

        The zeros are the point. A report with no criticals says so rather
        than leaving the reader to count.
        """
        counts = {severity: 0 for severity in _SEVERITIES}
        for finding in self.findings:
            counts[finding.severity] += 1
        return counts

    @property
    def worst_severity(self) -> Optional[str]:
        """The severity of the top finding, or None on an empty report."""
        if not self.findings:
            return None
        return self.findings[0].severity

    def summary(self) -> str:
        counts = self.severity_counts
        parts = []
        if self.config.claim:
            parts.append(f"Claim under audit: {self.config.claim}.")
        parts.append(self._verdict(counts))
        parts.append(self._coverage())
        return " ".join(parts)

    def _verdict(self, counts: dict) -> str:
        criticals = counts["critical"]
        warnings = counts["warning"]
        if criticals:
            return (
                f"{criticals} critical {_plural('finding', criticals)}, "
                f"{warnings} {_plural('warning', warnings)} and "
                f"{counts['info']} for context. On the critical "
                f"{_plural('finding', criticals)} the conclusion this data "
                f"is being asked to support does not hold as stated."
            )
        if warnings:
            return (
                f"No critical findings. {warnings} "
                f"{_plural('warning', warnings)} and {counts['info']} for "
                f"context. The result stands and the design weakens it, so "
                f"report it with what the {_plural('warning', warnings)} "
                f"say attached."
            )
        ran = len(self.findings)
        return (
            f"No critical findings and no warnings. All {ran} "
            f"{_plural('check', ran)} that ran came back clean, so the eval "
            f"holds up on everything this report could test."
        )

    def _coverage(self) -> str:
        if not self.not_run:
            return "Every check ran."
        missing = len(self.not_run)
        tail = "It is" if missing == 1 else "They are"
        return (
            f"{missing} {_plural('check', missing)} could not run for lack "
            f"of data. {tail} listed at the end."
        )

    def to_markdown(self) -> str:
        lines = ["# Eval audit", "", self.summary(), "", "## Findings", ""]
        for position, finding in enumerate(self.findings, start=1):
            lines += [
                f"### {position}. [{finding.severity}] {finding.title}",
                "",
                finding.detail,
                "",
            ]
        if self.not_run:
            lines += ["## Checks that could not run", ""]
            lines += [f"- **{s.title}**: {s.reason}" for s in self.not_run]
            lines.append("")
        return "\n".join(lines)

    def to_html(self) -> str:
        parts = [
            '<section class="evalaudit-report">',
            "<h1>Eval audit</h1>",
            f'<p class="audit-summary">{_escape(self.summary())}</p>',
            '<ol class="audit-findings">',
        ]
        for finding in self.findings:
            parts += [
                f'<li class="audit-finding audit-{finding.severity}">',
                f"<h2>{_escape(finding.title)}</h2>",
                f'<p class="audit-severity">{finding.severity}</p>',
                f"<p>{_escape(finding.detail)}</p>",
                "</li>",
            ]
        parts.append("</ol>")
        if self.not_run:
            parts += [
                "<h2>Checks that could not run</h2>",
                '<ul class="audit-not-run">',
            ]
            parts += [
                f"<li><strong>{_escape(s.title)}</strong>: "
                f"{_escape(s.reason)}</li>"
                for s in self.not_run
            ]
            parts.append("</ul>")
        parts.append("</section>")
        return "\n".join(parts)

    def __str__(self) -> str:  # pragma: no cover
        return self.summary()
