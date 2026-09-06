"""Do your graders agree with each other?

The question underneath every human-graded eval, and the one most reports
skip. A model comparison built on ratings the graders themselves do not
reproduce is measuring the rubric, not the model.

Krippendorff's alpha is the measure here because real grading is never fully
crossed. Graders miss items, join late, quit halfway. Alpha handles that.
Cohen's kappa needs exactly two raters who both graded everything, and
Fleiss' kappa needs a fixed number of raters per item. Both are here because
people ask for them by name.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ._types import _MIN_USABLE_SHARE, AgreementResult, KappaResult

__all__ = ["rater_agreement", "cohens_kappa", "fleiss_kappa"]

_LEVELS = {"nominal", "ordinal", "interval"}
_COLUMNS = ("item_id", "rater_id", "rating")

# The four reasons a leave-one-out alpha comes back undefined. They call for
# different responses, so they get different sentences. "No item left with
# two or more ratings" means the design fell apart without that rater. "The
# remaining raters agreed everywhere" means it did not, and they agreed. One
# item left means it held together and there is still not enough to measure.
_NOTE_SINGLE_RATER = "only one rater left, nothing to compare"
_NOTE_NO_OVERLAP = "no item left with two or more ratings"
_NOTE_NO_VARIANCE = "remaining raters agreed everywhere, so alpha has no denominator"
_NOTE_ONE_ITEM = "only one item left with two or more ratings"

# Ceiling on the working array the bootstrap allocates, in floats. The
# resample loop runs in blocks sized to fit under it. Blocks are not
# resamples: every resample inside a block is still computed in one pass.
_BLOCK_BUDGET = 4_000_000


def rater_agreement(
    ratings: pd.DataFrame,
    level: str = "nominal",
    confidence: float = 0.95,
    bootstrap_ci: bool = True,
    n_boot: int = 1000,
    seed: Optional[int] = None,
) -> AgreementResult:
    """Krippendorff's alpha on long-format ratings.

    Parameters
    ----------
    ratings
        A frame with ``item_id``, ``rater_id`` and ``rating``. One row per
        rating given. Items nobody graded twice are kept in ``n_items`` and
        excluded from alpha, since they carry no information about
        agreement. Rows with a missing rating are dropped.
    level
        ``"nominal"`` for unordered labels, where any two different ratings
        disagree equally. ``"ordinal"`` for ranked categories, where the
        distance between two ratings depends on how many ratings fell
        between them. ``"interval"`` for scales where the numbers mean
        something, and 1 against 5 is four times the disagreement of 1
        against 2. Pick the one that matches the rubric. A 1-5 quality scale
        is usually ordinal, and calling it nominal throws away the fact that
        adjacent grades are nearly the same judgement.
    confidence
        Nominal coverage of the bootstrap interval, default 0.95.
    bootstrap_ci
        Set False to skip the interval. Then ``ci_low`` and ``ci_high`` are
        NaN and nothing in the result will name an outlier rater, since
        there is no sampling error to judge one against.
    n_boot
        Bootstrap resamples. The interval resamples the items graded more
        than once, not every item in the frame. Items graded once contribute
        nothing to alpha and nothing to its sampling error, so resampling
        them would let the double-grading rate, a scheduling decision, leak
        into a statistic about raters.
    seed
        Seeds the bootstrap. Set it in anything you publish.

    Returns
    -------
    AgreementResult

    Notes
    -----
    Alpha comes back NaN in two cases and neither is perfect agreement.
    Fewer than two items graded twice cannot carry a reliability estimate,
    though the arithmetic will still return a value at that size, so the
    value is withheld. And when every rating that could be compared was
    identical there is no disagreement to divide by.
    """
    if level not in _LEVELS:
        raise ValueError(f"level must be one of {sorted(_LEVELS)}, got {level!r}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    data = _clean(ratings)
    n_items = int(data["item_id"].nunique())
    n_raters = int(data["rater_id"].nunique())

    units = _Units.build(data, level)
    alpha = units.alpha()

    ci_low, ci_high, drawn, usable = _bootstrap(
        units, confidence, bootstrap_ci, n_boot, seed
    )

    return AgreementResult(
        alpha=alpha,
        ci_low=ci_low,
        ci_high=ci_high,
        n_items=n_items,
        n_raters=n_raters,
        n_overlapping_items=units.n_units,
        level=level,
        item_disagreement=units.item_table(),
        rater_dropout=_dropout_table(data, level, alpha),
        confidence=confidence,
        n_boot=drawn,
        n_boot_usable=usable,
    )


# --------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------

def _clean(ratings: pd.DataFrame) -> pd.DataFrame:
    """Validate, copy, and drop rows with no rating."""
    if not isinstance(ratings, pd.DataFrame):
        raise ValueError("ratings must be a pandas DataFrame")

    missing = [c for c in _COLUMNS if c not in ratings.columns]
    if missing:
        raise ValueError(
            f"ratings is missing required column(s): {', '.join(missing)}. "
            f"Expected {', '.join(_COLUMNS)}."
        )

    data = ratings.loc[:, list(_COLUMNS)].copy()
    data = data[data["rating"].notna()]
    if data.empty:
        raise ValueError("ratings must not be empty")

    duplicated = data.duplicated(subset=["item_id", "rater_id"])
    if duplicated.any():
        first = data[duplicated].iloc[0]
        raise ValueError(
            f"ratings has duplicate rater/item pairs, starting with "
            f"item {first['item_id']!r} rated twice by {first['rater_id']!r}. "
            f"Two rows for one rater on one item double-count that opinion "
            f"and push alpha up. Aggregate them first."
        )
    return data


def _coded_values(data: pd.DataFrame, level: str):
    """Ratings as an ordered value domain plus per-row codes.

    Ordinal and interval need numbers, because both distances read the
    values as positions. Nominal does not, so labels are allowed there.
    """
    raw = data["rating"].to_numpy()
    try:
        numeric = raw.astype(float)
    except (TypeError, ValueError):
        if level != "nominal":
            raise ValueError(
                f"level={level!r} needs numeric ratings, because the distance "
                f"between two ratings depends on their values. Map the labels "
                f"to numbers, or use level='nominal'."
            ) from None
        codes, domain = pd.factorize(raw, sort=False)
        return np.arange(len(domain), dtype=float), np.asarray(codes)

    values = np.unique(numeric)
    return values, np.searchsorted(values, numeric)


# --------------------------------------------------------------------------
# The coincidence matrix, held as per-unit value counts
#
# Alpha only ever reads a unit through its value counts, so that is what gets
# stored. The coincidence matrix of a resample is the sum of the per-unit
# ones, which makes the whole bootstrap a matrix product against a
# multiplicity matrix rather than a loop.
# --------------------------------------------------------------------------

class _Units:
    """Value counts for the items graded more than once."""

    def __init__(self, item_ids, counts, values, level):
        self.item_ids = item_ids           # (U,)
        self.counts = counts               # (U, V) ratings per value
        self.values = values               # (V,) the value domain, ascending
        self.level = level
        self.sizes = counts.sum(axis=1)    # (U,) ratings per item

    @classmethod
    def build(cls, data: pd.DataFrame, level: str) -> "_Units":
        per_item = data.groupby("item_id", sort=False)["rating"].transform("size")
        pairable = data[per_item >= 2]

        values, codes = _coded_values(pairable, level)
        if pairable.empty:
            return cls(np.array([]), np.zeros((0, len(values))), values, level)

        item_codes, item_ids = pd.factorize(pairable["item_id"], sort=False)
        counts = np.zeros((len(item_ids), len(values)))
        np.add.at(counts, (item_codes, codes), 1.0)
        return cls(np.asarray(item_ids), counts, values, level)

    @property
    def n_units(self) -> int:
        return len(self.item_ids)

    # -- the two halves of alpha ------------------------------------------

    def numerators(self, distances=None) -> np.ndarray:
        """Per-unit observed disagreement, the coincidence contribution.

        ``sum over ordered pairs of squared distance, divided by (m - 1)``.
        Nominal and interval have closed forms that skip the distance
        matrix, which matters when the ratings are continuous and the value
        domain is nearly as large as the data.
        """
        counts, sizes = self.counts, self.sizes
        if self.level == "nominal":
            return (sizes ** 2 - (counts ** 2).sum(axis=1)) / (sizes - 1)
        if self.level == "interval":
            s1 = counts @ self.values
            s2 = counts @ self.values ** 2
            return 2 * (sizes * s2 - s1 ** 2) / (sizes - 1)
        return np.einsum("uc,ck,uk->u", counts, distances, counts) / (sizes - 1)

    def expected(self, marginals, distances=None) -> float:
        """The chance-disagreement denominator, before dividing by n(n-1)."""
        n = marginals.sum()
        if self.level == "nominal":
            return float(n ** 2 - (marginals ** 2).sum())
        if self.level == "interval":
            s1 = marginals @ self.values
            s2 = marginals @ self.values ** 2
            return float(2 * (n * s2 - s1 ** 2))
        return float(marginals @ distances @ marginals)

    @property
    def is_degenerate(self) -> bool:
        """True when every pairable rating is the same value.

        Alpha has no denominator then. That is not perfect agreement, it is
        a scale nobody varied, and the two call for different responses.
        """
        if self.n_units == 0:
            return False
        marginals = self.counts.sum(axis=0)
        distances = _ordinal_distances(marginals) if self.level == "ordinal" else None
        return self.expected(marginals, distances) == 0

    def alpha(self) -> float:
        """Alpha, or NaN when the data cannot carry one.

        Two items is the floor. One item gives arithmetic that looks like a
        reliability estimate and is not one, and a reader skimming a report
        will quote whatever number is in front of them.
        """
        if self.n_units < 2:
            return float("nan")
        marginals = self.counts.sum(axis=0)
        distances = _ordinal_distances(marginals) if self.level == "ordinal" else None
        n = marginals.sum()
        expected = self.expected(marginals, distances)
        if expected == 0:
            return float("nan")
        observed = self.numerators(distances).sum()
        return float(1.0 - (n - 1) * observed / expected)

    # -- the item table ----------------------------------------------------

    def item_table(self) -> pd.DataFrame:
        """Each item's disagreement, averaged over its ordered pairs.

        Dividing the coincidence contribution by the grading count turns it
        into the mean squared distance between two randomly picked ratings
        of that item. An item graded ten times and one graded twice with the
        same spread land on the same number.
        """
        columns = ["item_id", "n_ratings", "disagreement", "vs_chance"]
        if self.n_units == 0:
            return pd.DataFrame(columns=columns)

        marginals = self.counts.sum(axis=0)
        distances = _ordinal_distances(marginals) if self.level == "ordinal" else None
        n = marginals.sum()
        expected = self.expected(marginals, distances)

        disagreement = self.numerators(distances) / self.sizes
        chance = expected / (n * (n - 1)) if expected > 0 else float("nan")

        table = pd.DataFrame({
            "item_id": self.item_ids,
            "n_ratings": self.sizes.astype(int),
            "disagreement": disagreement,
            "vs_chance": disagreement / chance,
        })
        return (
            table.sort_values("disagreement", ascending=False, kind="mergesort")
            .reset_index(drop=True)
        )


def _ordinal_distances(marginals: np.ndarray) -> np.ndarray:
    """Krippendorff's ordinal metric, for one set of marginals or many.

        d(c, k) = (sum of marginals from c to k, minus half of each end)^2

    The distance depends on how the ratings were distributed, so it has to
    be rebuilt for every resample rather than computed once.
    """
    marg = np.atleast_2d(marginals)
    cumulative = np.cumsum(marg, axis=1)
    spread = (
        cumulative[:, None, :]
        - cumulative[:, :, None]
        + (marg[:, :, None] - marg[:, None, :]) / 2
    )
    out = spread ** 2
    return out[0] if marginals.ndim == 1 else out


# --------------------------------------------------------------------------
# The bootstrap
# --------------------------------------------------------------------------

def _bootstrap(units, confidence, wanted, n_boot, seed):
    """Percentile interval over resampled items.

    Returns the bounds, how many resamples were asked for, and how many came
    back defined. A resample is undefined when every rating in it is the
    same value, which is the unanimous case, so dropping those biases the
    upper bound downward. Past a fixed share of them the interval is refused
    rather than reported.
    """
    if not wanted:
        return float("nan"), float("nan"), 0, 0
    if n_boot < 1:
        raise ValueError(f"n_boot must be at least 1, got {n_boot}")

    if units.n_units < 2:
        # One item resampled is the same item every time. Any width drawn
        # around it would be a property of the arithmetic, not the data.
        return float("nan"), float("nan"), n_boot, 0

    alphas = _resample_alphas(units, n_boot, seed)
    good = alphas[np.isfinite(alphas)]
    usable = int(good.size)

    if usable < _MIN_USABLE_SHARE * n_boot:
        return float("nan"), float("nan"), n_boot, usable

    tail = (1 - confidence) / 2
    lo = float(np.percentile(good, 100 * tail))
    hi = float(np.percentile(good, 100 * (1 - tail)))
    return lo, hi, n_boot, usable


def _resample_alphas(units, n_boot, seed) -> np.ndarray:
    """Alpha for every resample, computed in blocks and never one at a time.

    Each resample is a multiplicity vector over items. The coincidence
    matrix is additive over items, so a block of resamples is one matrix
    product against the per-item quantities.
    """
    rng = np.random.default_rng(seed)
    n_units, n_values = units.counts.shape
    ordinal = units.level == "ordinal"

    fixed_numerators = None if ordinal else units.numerators()
    if ordinal:
        # outer products per unit, so a block's coincidence matrix is one dot
        outer = (
            units.counts[:, :, None] * units.counts[:, None, :]
            / (units.sizes - 1)[:, None, None]
        ).reshape(n_units, n_values * n_values)

    per_resample = n_values * n_values if ordinal else n_values
    block = int(max(1, min(n_boot, _BLOCK_BUDGET // max(per_resample, 1))))

    out = np.empty(n_boot)
    for start in range(0, n_boot, block):
        size = min(block, n_boot - start)
        idx = rng.integers(0, n_units, size=(size, n_units))
        picks = _multiplicities(idx, n_units)

        marginals = picks @ units.counts          # (size, V)
        totals = marginals.sum(axis=1)            # (size,)

        if ordinal:
            distances = _ordinal_distances(marginals)
            coincidence = picks @ outer
            observed = (coincidence * distances.reshape(size, -1)).sum(axis=1)
            expected = np.einsum("bc,bck,bk->b", marginals, distances, marginals)
        else:
            observed = picks @ fixed_numerators
            expected = _expected_batch(units, marginals, totals)

        with np.errstate(divide="ignore", invalid="ignore"):
            alphas = 1.0 - (totals - 1) * observed / expected
        out[start:start + size] = np.where(expected > 0, alphas, np.nan)

    return out


def _multiplicities(idx: np.ndarray, n_units: int) -> np.ndarray:
    """How many times each item was drawn, per resample.

    One bincount over the flattened index matrix. Building this with
    np.add.at, or worse a loop, is where a vectorised bootstrap quietly
    stops being one.
    """
    n_boot = idx.shape[0]
    offset = idx + np.arange(n_boot)[:, None] * n_units
    flat = np.bincount(offset.ravel(), minlength=n_boot * n_units)
    return flat.reshape(n_boot, n_units).astype(float)


def _expected_batch(units, marginals, totals) -> np.ndarray:
    if units.level == "nominal":
        return totals ** 2 - (marginals ** 2).sum(axis=1)
    s1 = marginals @ units.values
    s2 = marginals @ units.values ** 2
    return 2 * (totals * s2 - s1 ** 2)


# --------------------------------------------------------------------------
# Leave one rater out
# --------------------------------------------------------------------------

def _dropout_table(data: pd.DataFrame, level: str, alpha: float) -> pd.DataFrame:
    """Alpha recomputed without each rater, and why it is missing when it is.

    A NaN here has three possible causes and they call for opposite
    responses, so each gets its own note. Losing the overlap means the
    design depended on that rater. Losing the variance means the graders who
    remain agreed on everything, which is a finding rather than a fault.
    """
    columns = ["rater_id", "n_ratings", "alpha_without", "delta", "note"]
    raters = list(dict.fromkeys(data["rater_id"]))
    if len(raters) < 2:
        return pd.DataFrame(columns=columns)

    rows = []
    for rater in raters:
        kept = data[data["rater_id"] != rater]
        n_ratings = int((data["rater_id"] == rater).sum())

        if kept["rater_id"].nunique() < 2:
            note, without = _NOTE_SINGLE_RATER, float("nan")
        else:
            units = _Units.build(kept, level)
            without = units.alpha()
            if units.n_units == 0:
                note = _NOTE_NO_OVERLAP
            elif units.is_degenerate:
                # checked before the item count, because "they agreed on
                # everything" is the more specific finding and the one worth
                # acting on. The count is why there is no number; the
                # agreement is what happened.
                note = _NOTE_NO_VARIANCE
            elif units.n_units < 2:
                note = _NOTE_ONE_ITEM
            else:
                note = ""

        rows.append({
            "rater_id": rater,
            "n_ratings": n_ratings,
            "alpha_without": without,
            "delta": without - alpha,
            "note": note,
        })

    table = pd.DataFrame(rows, columns=columns)
    return (
        table.sort_values(
            "delta", ascending=False, na_position="last", kind="mergesort"
        )
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------
# The two kappas, for people who expect them
# --------------------------------------------------------------------------

def cohens_kappa(r1: Sequence, r2: Sequence) -> KappaResult:
    """Cohen's kappa between two raters who both graded every item.

    Parameters
    ----------
    r1, r2
        Aligned ratings, one entry per item.

    Returns
    -------
    KappaResult

    Notes
    -----
    Kappa moves with how often each category gets used, so two raters score
    lower on a lopsided scale than on a balanced one at the same accuracy.
    Alpha does not have that problem. This is here for compatibility.
    """
    a = np.asarray(list(r1))
    b = np.asarray(list(r2))
    if a.size == 0:
        raise ValueError("r1 and r2 must not be empty")
    if a.shape != b.shape:
        raise ValueError(
            f"r1 and r2 must be the same length, got {a.size} and {b.size}"
        )

    categories = np.unique(np.concatenate([a, b]))
    table = np.zeros((len(categories), len(categories)))
    np.add.at(
        table,
        (np.searchsorted(categories, a), np.searchsorted(categories, b)),
        1.0,
    )

    n = table.sum()
    observed = np.trace(table) / n
    expected = float(table.sum(axis=1) @ table.sum(axis=0)) / (n * n)
    kappa = float("nan") if expected == 1 else (observed - expected) / (1 - expected)

    return KappaResult(
        kappa=float(kappa),
        p_observed=float(observed),
        p_expected=float(expected),
        n_items=int(a.size),
        n_raters=2,
        n_categories=int(len(categories)),
        method="cohen",
    )


def fleiss_kappa(ratings: pd.DataFrame) -> KappaResult:
    """Fleiss' kappa on long-format ratings with a fixed rater count.

    Parameters
    ----------
    ratings
        Same shape as ``rater_agreement`` takes. Every item must have the
        same number of ratings, which is what Fleiss is defined on.

    Returns
    -------
    KappaResult
    """
    data = _clean(ratings)

    per_item = data.groupby("item_id", sort=False)["rating"].size()
    if per_item.nunique() != 1:
        raise ValueError(
            f"Fleiss' kappa needs the same number of ratings on every item, "
            f"got between {per_item.min()} and {per_item.max()}. Ragged "
            f"grading is the normal case and rater_agreement handles it."
        )

    n_raters = int(per_item.iloc[0])
    if n_raters < 2:
        raise ValueError("Fleiss' kappa needs at least two ratings per item")

    categories, codes = _coded_values(data, "nominal")
    item_codes, item_ids = pd.factorize(data["item_id"], sort=False)
    table = np.zeros((len(item_ids), len(categories)))
    np.add.at(table, (item_codes, codes), 1.0)

    n_items = len(item_ids)
    proportions = table.sum(axis=0) / (n_items * n_raters)
    agreement = ((table ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))

    observed = float(agreement.mean())
    expected = float((proportions ** 2).sum())
    kappa = float("nan") if expected == 1 else (observed - expected) / (1 - expected)

    return KappaResult(
        kappa=float(kappa),
        p_observed=observed,
        p_expected=expected,
        n_items=int(n_items),
        n_raters=int(data["rater_id"].nunique()),
        n_categories=int(len(categories)),
        method="fleiss",
    )
