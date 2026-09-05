"""Comparing two systems.

The mistake this module exists to stop: running an unpaired test on paired
data. Most evals score both systems on the same items, and that pairing
removes the item-to-item difficulty that otherwise swamps the difference.
Ignore it and the interval comes out several times too wide.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy import optimize as _optimize
from scipy import stats as _stats

from ._types import ComparisonResult
from .scores import _is_binary

__all__ = ["compare_paired", "compare_independent"]

_PAIRED_METHODS = {"auto", "mcnemar", "bootstrap", "wilcoxon"}
_INDEPENDENT_METHODS = {"auto", "bootstrap", "t", "score"}

# Below this many discordant pairs the chi-square approximation is not
# trustworthy and McNemar falls back to the exact binomial test.
_EXACT_BELOW = 25

# Stands in for an infinite score at the edges of the range, where the
# variance under the hypothesis goes to zero. A finite value keeps the root
# finder happy and the sign is all that matters there.
_EDGE_SCORE = 1e6

# Above this many items the signed-rank test drops the exact distribution
# for the normal approximation. This is scipy's own cutoff, and the interval
# follows the test across it so the two always agree.
_SIGNED_RANK_EXACT_MAX = 50

# The signed-rank interval reports a midpoint of two differences. Below this
# many distinct non-zero differences those midpoints number three or fewer,
# and an interval picked off three values cannot line up with the p-value
# beside it. Binary scores are the common way to get here, though any pair
# of systems that only ever differ by one amount does it too.
_MIN_DISTINCT_SHIFTS = 3


def compare_paired(
    a: Sequence[float],
    b: Sequence[float],
    method: str = "auto",
    confidence: float = 0.95,
    n_boot: int = 10000,
    seed: Optional[int] = None,
) -> ComparisonResult:
    """Compare two systems scored on the same items.

    Parameters
    ----------
    a, b
        Scores for system A and system B, aligned item by item. Same length.
    method
        ``"auto"`` picks McNemar for binary data and the paired bootstrap for
        continuous data.
        ``"mcnemar"`` is the test for paired binary outcomes. It looks only at
        items where the two systems disagree, since items they both pass or
        both fail say nothing about which system is better.
        ``"bootstrap"`` resamples the per-item differences.
        ``"wilcoxon"`` is the signed-rank test, reported with the
        Hodges-Lehmann shift and its distribution-free interval. It needs
        the two systems to differ by at least three distinct amounts, since
        below that the interval it would return cannot agree with its own
        p-value. Binary scores never clear that bar. Use McNemar for those.
    confidence
        Nominal coverage, default 0.95.
    n_boot
        Bootstrap resamples, used by ``"bootstrap"`` only.
    seed
        Seeds the bootstrap so results are reproducible.

    Returns
    -------
    ComparisonResult

    Examples
    --------
    >>> compare_paired([1] * 40 + [0] * 10, [1] * 35 + [0] * 15).summary()
    'Difference 10.0% ...'
    """
    if method not in _PAIRED_METHODS:
        raise ValueError(f"method must be one of {_PAIRED_METHODS}, got {method!r}")

    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError(
            f"a and b must be the same length for paired data, "
            f"got {x.size} and {y.size}"
        )
    if x.size == 0:
        raise ValueError("a and b must not be empty")

    n = int(x.size)
    binary = _is_binary(x) and _is_binary(y)

    if method == "auto":
        method = "mcnemar" if binary else "bootstrap"

    if method == "mcnemar":
        if not binary:
            raise ValueError("mcnemar needs binary 0/1 scores")
        return _mcnemar(x, y, n, confidence)
    if method == "bootstrap":
        return _paired_bootstrap(x, y, n, binary, confidence, n_boot, seed)
    shifts = x - y
    n_distinct = int(np.unique(shifts[shifts != 0]).size)
    if 0 < n_distinct < _MIN_DISTINCT_SHIFTS:
        raise ValueError(
            f"wilcoxon needs at least {_MIN_DISTINCT_SHIFTS} distinct "
            f"non-zero differences between the two systems and got "
            f"{n_distinct}. The interval is built from midpoints of those "
            f"differences, so this few leaves it on a three-value lattice, "
            f"where it contradicts its own p-value on roughly 41% of "
            f"samples. Binary 0/1 scores always land here, and so does any "
            f"scale the two systems only use two levels of. Use "
            f"method='mcnemar' for 0/1 data, or method='bootstrap' otherwise"
        )
    return _wilcoxon(x, y, n, binary, confidence)


def compare_independent(
    a: Sequence[float],
    b: Sequence[float],
    method: str = "auto",
    confidence: float = 0.95,
    n_boot: int = 10000,
    seed: Optional[int] = None,
) -> ComparisonResult:
    """Compare two systems scored on separate samples of items.

    Use this only when the two systems really were run on different items.
    If they were run on the same items, use :func:`compare_paired`. That is
    the same numbers analysed properly, and it gives a much tighter interval.

    Parameters
    ----------
    a, b
        Scores for system A and system B. Lengths may differ.
    method
        ``"auto"`` picks the score method for binary data and the bootstrap
        for continuous data.
        ``"score"`` is the score interval and score test for a difference of
        two independent proportions. The interval is the test inverted, so
        the two never disagree about zero.
        ``"bootstrap"`` resamples each group separately.
        ``"t"`` is Welch's t interval, which does not assume equal variance.
    confidence
        Nominal coverage, default 0.95.
    n_boot
        Bootstrap resamples, used by ``"bootstrap"`` only.
    seed
        Seeds the bootstrap so results are reproducible.

    Returns
    -------
    ComparisonResult
        With ``paired=False``. ``n`` is the total number of items across both
        groups.
    """
    if method not in _INDEPENDENT_METHODS:
        raise ValueError(
            f"method must be one of {_INDEPENDENT_METHODS}, got {method!r}"
        )

    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size == 0 or y.size == 0:
        raise ValueError("a and b must not be empty")

    binary = _is_binary(x) and _is_binary(y)
    if method == "auto":
        method = "score" if binary else "bootstrap"

    n_total = int(x.size + y.size)
    difference = float(np.mean(x) - np.mean(y))

    if method == "score":
        if not binary:
            raise ValueError("score needs binary 0/1 scores")
        ci_low, ci_high = _score_diff_ci(x, y, confidence)
        p_value = _two_proportion_p(x, y)
    elif method == "bootstrap":
        rng = np.random.default_rng(seed)
        idx_a = rng.integers(0, x.size, size=(n_boot, x.size))
        idx_b = rng.integers(0, y.size, size=(n_boot, y.size))
        boot = x[idx_a].mean(axis=1) - y[idx_b].mean(axis=1)
        ci_low, ci_high = _percentile_ci(boot, confidence)
        p_value = _bootstrap_p(boot)
    else:  # "t"
        ci_low, ci_high, p_value = _welch(x, y, confidence)

    return ComparisonResult(
        difference=difference,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        n=n_total,
        method=method,
        binary=binary,
        paired=False,
        confidence=confidence,
    )


# --------------------------------------------------------------------------
# Paired methods
# --------------------------------------------------------------------------

def _mcnemar(
    x: np.ndarray, y: np.ndarray, n: int, confidence: float
) -> ComparisonResult:
    """McNemar's test on the discordant pairs.

    Exact binomial below 25 discordant pairs, chi-square with the continuity
    correction at 25 and above. The threshold is a convention and the two
    answers sit close together near it, so the choice changes little. It is
    pinned by a test because an off-by-one here is invisible in the output.
    """
    a_hit = x.astype(bool)
    b_hit = y.astype(bool)
    n_a_only = int(np.sum(a_hit & ~b_hit))
    n_b_only = int(np.sum(~a_hit & b_hit))
    n_disc = n_a_only + n_b_only

    if n_disc == 0:
        # Nothing to test. The p-value is 1 by convention, and summary()
        # says plainly that the table was empty.
        p_value = 1.0
    elif n_disc < _EXACT_BELOW:
        smaller = min(n_a_only, n_b_only)
        p_value = float(min(2 * _stats.binom.cdf(smaller, n_disc, 0.5), 1.0))
    else:
        stat = (abs(n_a_only - n_b_only) - 1.0) ** 2 / n_disc
        p_value = float(_stats.chi2.sf(stat, 1))

    difference = (n_a_only - n_b_only) / n
    ci_low, ci_high = _paired_proportion_ci(n_a_only, n_b_only, n, confidence)

    return ComparisonResult(
        difference=difference,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        n=n,
        method="mcnemar",
        binary=True,
        paired=True,
        confidence=confidence,
        n_discordant=n_disc,
    )


def _tango_score(n_a_only: int, n_b_only: int, n: int, delta: float) -> float:
    """Tango's score statistic for the hypothesis that the difference is delta.

    The nuisance parameter is the rate of B-only items, and it is replaced by
    its maximum likelihood estimate under the hypothesis, which is the root
    of a quadratic. At delta=0 the whole thing collapses to

        (b - c) / sqrt(b + c)

    which is McNemar's statistic without the continuity correction. That is
    the property that keeps the interval and the test telling the same story.
    """
    b, c = n_a_only, n_b_only
    quad_a = 2.0 * n
    quad_b = -b - c + (2.0 * n - b + c) * delta
    quad_c = -c * delta * (1.0 - delta)
    disc = max(quad_b * quad_b - 4.0 * quad_a * quad_c, 0.0)
    q = (-quad_b + np.sqrt(disc)) / (2.0 * quad_a)

    num = b - c - n * delta
    den = n * (2.0 * q + delta * (1.0 - delta))
    if den <= 0:
        # Only at the edges of the range, where delta is barely possible.
        if num == 0:
            return 0.0
        return _EDGE_SCORE if num > 0 else -_EDGE_SCORE
    return num / np.sqrt(den)


def _paired_proportion_ci(
    n_a_only: int, n_b_only: int, n: int, confidence: float
) -> tuple[float, float]:
    """Tango's score interval on a paired difference in proportions.

    The bounds are the two values of the difference the data would just
    barely reject, found by solving the score statistic for +/- z. Tango
    (1998), "Equivalence test and confidence interval for the difference in
    proportions for the paired-sample design".

    The Wald interval is the obvious alternative and it is worse in two
    ways. It under-covers when the pass rates run near 1, which is where
    eval numbers usually sit. And with no discordant pairs its standard
    error is zero, so it returns a single point and claims two systems are
    identical on the strength of an empty table. The score interval stays
    wide there, which is the honest answer.
    """
    z = _stats.norm.ppf(1 - (1 - confidence) / 2)
    d_hat = (n_a_only - n_b_only) / n
    edge = 1.0 - 1e-9

    if d_hat <= -edge:
        lo = -1.0
    else:
        def below(delta: float) -> float:
            return _tango_score(n_a_only, n_b_only, n, delta) - z

        lo = (
            float(_optimize.brentq(below, -edge, d_hat))
            if below(-edge) > 0
            else -1.0
        )

    if d_hat >= edge:
        hi = 1.0
    else:
        def above(delta: float) -> float:
            return _tango_score(n_a_only, n_b_only, n, delta) + z

        hi = (
            float(_optimize.brentq(above, d_hat, edge))
            if above(edge) < 0
            else 1.0
        )

    return lo, hi


def _paired_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    n: int,
    binary: bool,
    confidence: float,
    n_boot: int,
    seed: Optional[int],
) -> ComparisonResult:
    """Percentile bootstrap of the mean per-item difference.

    Resample items, not systems. Each resample takes the pair (a_i, b_i)
    together, which is what keeps the shared item difficulty out of the
    interval.
    """
    d = x - y
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    ci_low, ci_high = _percentile_ci(boot, confidence)

    return ComparisonResult(
        difference=float(np.mean(d)),
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=_bootstrap_p(boot),
        n=n,
        method="bootstrap",
        binary=binary,
        paired=True,
        confidence=confidence,
    )


def _wilcoxon(
    x: np.ndarray, y: np.ndarray, n: int, binary: bool, confidence: float
) -> ComparisonResult:
    """Wilcoxon signed-rank test with the Hodges-Lehmann shift.

    The effect size is the median of the Walsh averages of the per-item
    differences, and the interval comes from the signed-rank distribution.
    Building the Walsh averages costs n^2 memory. That is fine for eval sets
    and would not be for millions of rows.

    The test and the interval always come from the same distribution. Both
    are exact for 50 items or fewer with no zero differences and no two
    differences the same size, and both fall back to the normal
    approximation otherwise, tie correction included. Letting scipy pick the
    test on its own while the interval used the approximation would put a
    p-value and an interval from two different distributions on the same
    line.

    Data whose non-zero differences take fewer than three distinct values
    never reaches here. compare_paired turns it away, and this is why. Two
    distinct differences, say plus and minus one, give Walsh averages of
    only minus one, zero and one, so the interval can land nowhere else.
    Measured over 600 binary samples it reported a bound of exactly zero
    while the p-value rejected on 41% of them. No choice of variance fixes
    it, since the limit is the resolution of the estimate rather than its
    spread. Binary scores are the common case and McNemar answers the same
    question on them properly.
    """
    d = x - y
    if np.all(d == 0):
        # Every item tied, so nothing is left to rank and nothing was
        # analysed.
        return ComparisonResult(
            difference=0.0,
            ci_low=0.0,
            ci_high=0.0,
            p_value=1.0,
            n=0,
            method="wilcoxon",
            binary=binary,
            paired=True,
            confidence=confidence,
        )

    # Items the two systems tied on carry no sign, and the signed-rank test
    # drops them. The estimate and the interval drop them too, so all three
    # describe the same set of items.
    d_signed = d[d != 0]
    exact = _signed_rank_is_exact(d_signed)
    p_value = float(
        _stats.wilcoxon(d_signed, method="exact" if exact else "approx").pvalue
    )
    walsh = _walsh_averages(d_signed)
    difference = float(np.median(walsh))
    ci_low, ci_high = _signed_rank_ci(
        walsh,
        d_signed.size,
        confidence,
        exact,
        0.0 if exact else _tie_correction(d_signed),
    )

    return ComparisonResult(
        difference=difference,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        # The items the two systems tied on are not in any of these numbers,
        # so they are not in the count either.
        n=int(d_signed.size),
        method="wilcoxon",
        binary=binary,
        paired=True,
        confidence=confidence,
    )


def _walsh_averages(d: np.ndarray) -> np.ndarray:
    """Sorted pairwise averages (d_i + d_j) / 2 for i <= j."""
    i, j = np.triu_indices(len(d))
    return np.sort((d[i] + d[j]) / 2.0)


def _signed_rank_is_exact(d_signed: np.ndarray) -> bool:
    """Whether the exact signed-rank distribution applies to these numbers.

    It needs every difference distinct in size, since ties break the ranking
    the distribution is built on. Zeros are already gone by this point. The
    size cap is scipy's own cutoff for the exact test.
    """
    if d_signed.size > _SIGNED_RANK_EXACT_MAX:
        return False
    magnitudes = np.abs(d_signed)
    return bool(np.unique(magnitudes).size == magnitudes.size)


def _signed_rank_counts(n: int) -> np.ndarray:
    """How many sign patterns give each possible rank sum.

    Under the null every one of the 2^n sign patterns is equally likely, so
    counting subsets of {1, ..., n} by their sum gives the whole null
    distribution. Counts stay whole numbers below 2^53 for n up to 50, so
    float64 holds them exactly.
    """
    counts = np.zeros(n * (n + 1) // 2 + 1)
    counts[0] = 1.0
    for rank in range(1, n + 1):
        counts[rank:] += counts[:-rank].copy()
    return counts


def _tie_correction(d_signed: np.ndarray) -> float:
    """The sum of t^3 - t over groups of differences with the same size.

    Tied magnitudes share an average rank, which cuts the spread of the
    signed-rank statistic. This is the quantity scipy subtracts from the
    variance to account for it, and groups of one contribute nothing.
    """
    _, sizes = np.unique(np.abs(d_signed), return_counts=True)
    return float(np.sum(sizes.astype(float) ** 3 - sizes))


def _signed_rank_trim(
    n: int, confidence: float, exact: bool, tie_correction: float = 0.0
) -> int:
    """How many Walsh averages to drop from each end of the sorted list.

    The interval is the set of shifts the signed-rank test would not reject,
    so the number trimmed is the test's critical value. Taking it from the
    exact distribution keeps the interval and the p-value on the same
    footing, which is the whole point of doing it this way.
    """
    if not exact:
        # Same variance scipy uses, tie correction included. Without the
        # correction the interval would be drawn from a wider distribution
        # than the p-value beside it, which on binary data, where every
        # difference is the same size, is most of the distribution.
        # scipy's approximate p-value carries no continuity correction, so
        # neither does this.
        z = _stats.norm.ppf(1 - (1 - confidence) / 2)
        spread = n * (n + 1) * (2 * n + 1) - tie_correction / 2.0
        sd = np.sqrt(max(spread, 0.0) / 24.0)
        return int(np.floor(n * (n + 1) / 4.0 - z * sd))

    counts = _signed_rank_counts(n)
    cdf = np.cumsum(counts) / counts.sum()
    eligible = np.nonzero(cdf <= (1 - confidence) / 2)[0]
    if eligible.size == 0:
        # Below about six items the distribution is too coarse for this
        # confidence level to rule anything out. Trim nothing and return the
        # widest interval the data offer, which is what R does in the same
        # corner. The interval can then sit clear of zero while the test
        # declines to reject, the same discreteness that shows up in McNemar
        # at tiny discordant counts.
        return 0
    return int(eligible[-1])


def _signed_rank_ci(
    walsh: np.ndarray,
    n: int,
    confidence: float,
    exact: bool,
    tie_correction: float = 0.0,
) -> tuple[float, float]:
    """Distribution-free interval on the Hodges-Lehmann shift."""
    m = len(walsh)
    trim = _signed_rank_trim(n, confidence, exact, tie_correction)
    k = int(np.clip(trim, 0, (m - 1) // 2))
    return float(walsh[k]), float(walsh[m - 1 - k])


# --------------------------------------------------------------------------
# Independent methods
# --------------------------------------------------------------------------

def _constrained_rates(
    p1: float, n1: int, p2: float, n2: int, delta: float
) -> tuple[float, float]:
    """Maximum likelihood pass rates given that the difference is delta.

    The two rates are tied together by the hypothesis, so only one is free,
    and it is the root of a cubic. Farrington and Manning (1990) give the
    closed form used here. At delta=0 the root is the pooled pass rate,
    which is what makes the interval and the z test the same procedure.
    """
    t = n2 / n1
    a = 1.0 + t
    b = -(1.0 + t + p1 + t * p2 + delta * (t + 2.0))
    c = delta * delta + delta * (2.0 * p1 + t + 1.0) + p1 + t * p2
    d = -p1 * delta * (1.0 + delta)

    v = (b / a / 3.0) ** 3 - b * c / (6.0 * a * a) + d / a / 2.0
    s = np.sqrt(max((b / a / 3.0) ** 2 - c / a / 3.0, 0.0))
    if s == 0:
        q1 = -b / (3.0 * a)
    else:
        u = s if v > 0 else -s
        w = (np.pi + np.arccos(np.clip(v / u**3, -1.0, 1.0))) / 3.0
        q1 = 2.0 * u * np.cos(w) - b / a / 3.0
    return q1, q1 - delta


def _score_diff_stat(
    p1: float, n1: int, p2: float, n2: int, delta: float
) -> float:
    """Score statistic for the hypothesis that the difference is delta."""
    num = p1 - p2 - delta
    if num == 0:
        return 0.0
    q1, q2 = _constrained_rates(p1, n1, p2, n2, delta)
    var = q1 * (1.0 - q1) / n1 + q2 * (1.0 - q2) / n2
    if var <= 0:
        return _EDGE_SCORE if num > 0 else -_EDGE_SCORE
    return num / np.sqrt(var)


def _score_diff_ci(
    x: np.ndarray, y: np.ndarray, confidence: float
) -> tuple[float, float]:
    """Score interval for a difference of independent proportions.

    Mee (1984), with the constrained estimates solved as in Farrington and
    Manning (1990). Same construction as the paired case: the bounds are the
    two differences the data would just barely reject, so the interval and
    the p-value are one procedure rather than two.

    Miettinen and Nurminen (1985) give the same interval with the variance
    multiplied by N / (N - 1). That factor is left out here on purpose. The
    p-value reported beside this interval is the pooled two-proportion z
    test, whose variance carries no such factor, so adding it would pull the
    bounds slightly wider than the test and break the agreement that is the
    reason for using a score interval at all. Measured over 9213 tables the
    correction produced 4 disagreements where this version produces none,
    and it bought no coverage in exchange: both sit within 0.0005 of each
    other across five simulated settings, with Miettinen-Nurminen intervals
    about 0.3 to 1 percent wider. Anyone wanting that variant should use the
    factor in both places, not one.

    Newcombe's square-and-add interval is the usual alternative and it is a
    fine interval. It came out of this package because it is built from two
    one-sample Wilson intervals, so nothing ties it to the two-sample test
    reported beside it, and the two can disagree about zero.
    """
    z = _stats.norm.ppf(1 - (1 - confidence) / 2)
    n1, n2 = int(x.size), int(y.size)
    p1, p2 = float(np.mean(x)), float(np.mean(y))
    d_hat = p1 - p2
    edge = 1.0 - 1e-9

    if d_hat <= -edge:
        lo = -1.0
    else:
        def below(delta: float) -> float:
            return _score_diff_stat(p1, n1, p2, n2, delta) - z

        lo = (
            float(_optimize.brentq(below, -edge, d_hat))
            if below(-edge) > 0
            else -1.0
        )

    if d_hat >= edge:
        hi = 1.0
    else:
        def above(delta: float) -> float:
            return _score_diff_stat(p1, n1, p2, n2, delta) + z

        hi = (
            float(_optimize.brentq(above, d_hat, edge))
            if above(edge) < 0
            else 1.0
        )

    return lo, hi


def _two_proportion_p(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sided score test on two independent proportions.

    The pooled variance is the maximum likelihood estimate under the null,
    which makes this the score test at a difference of zero and the exact
    test that :func:`_score_diff_ci` inverts.
    """
    n1, n2 = int(x.size), int(y.size)
    p1, p2 = float(np.mean(x)), float(np.mean(y))
    pooled = (np.sum(x) + np.sum(y)) / (n1 + n2)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (p1 - p2) / se
    return float(2 * _stats.norm.sf(abs(z)))


def _welch(
    x: np.ndarray, y: np.ndarray, confidence: float
) -> tuple[float, float, float]:
    """Welch's t interval and p-value for two independent means."""
    n1, n2 = int(x.size), int(y.size)
    if n1 < 2 or n2 < 2:
        raise ValueError(
            f"the t method needs at least 2 items per group to estimate a "
            f"variance, got {n1} and {n2}. Use method='bootstrap' if one "
            f"group really has a single item"
        )
    res = _stats.ttest_ind(x, y, equal_var=False)
    v1 = float(np.var(x, ddof=1)) / n1
    v2 = float(np.var(y, ddof=1)) / n2
    se = np.sqrt(v1 + v2)
    df = (v1 + v2) ** 2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1))
    t_crit = _stats.t.ppf(1 - (1 - confidence) / 2, df=df)
    diff = float(np.mean(x) - np.mean(y))
    return diff - t_crit * se, diff + t_crit * se, float(res.pvalue)


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def _percentile_ci(boot: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = 1 - confidence
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return lo, hi


def _bootstrap_p(boot: np.ndarray) -> float:
    """Two-sided bootstrap p-value.

    The share of resamples falling on the far side of zero, doubled. It
    cannot resolve past 1 / n_boot, so it is floored there rather than
    reported as zero.
    """
    n_boot = len(boot)
    tail = min(float(np.mean(boot <= 0)), float(np.mean(boot >= 0)))
    return float(min(max(2 * tail, 1.0 / n_boot), 1.0))
