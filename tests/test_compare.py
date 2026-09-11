"""Tests for evalaudit.compare.

Written before the implementation. Make these pass.

McNemar tests are validated against statsmodels as an independent
implementation. The paired bootstrap has a coverage test in the same
style as test_scores.py.
"""

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

from evalaudit import ComparisonResult
from evalaudit.compare import (
    compare_paired,
    compare_independent,
    _signed_rank_trim,
    _tie_correction,
)


# --------------------------------------------------------------------------
# McNemar, binary paired data
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "a, b",
    [
        # large discordant count, chi-square path
        ([1]*60 + [0]*20 + [1]*10 + [0]*10, [1]*60 + [1]*20 + [0]*10 + [0]*10),
        # another large-count case
        ([1]*40 + [0]*30 + [1]*15 + [0]*15, [1]*40 + [1]*30 + [0]*15 + [0]*15),
    ],
    ids=["large-1", "large-2"],
)
def test_mcnemar_matches_statsmodels_large(a, b):
    """Chi-square with continuity correction when discordant >= 25."""
    r = compare_paired(a, b, method="mcnemar")

    a_arr, b_arr = np.array(a, dtype=bool), np.array(b, dtype=bool)
    table = np.array([
        [(a_arr & b_arr).sum(), (a_arr & ~b_arr).sum()],
        [(~a_arr & b_arr).sum(), (~a_arr & ~b_arr).sum()],
    ])
    sm = sm_mcnemar(table, exact=False, correction=True)

    assert r.p_value == pytest.approx(sm.pvalue, abs=1e-6)
    assert r.binary is True
    assert r.paired is True
    assert r.method == "mcnemar"


@pytest.mark.parametrize(
    "a, b",
    [
        # small discordant count, exact binomial path
        ([1]*8 + [0]*1 + [1]*1 + [0]*10, [1]*8 + [1]*1 + [0]*1 + [0]*10),
        # very few discordant pairs
        ([1]*5 + [0]*2 + [1]*3 + [0]*10, [1]*5 + [1]*2 + [0]*3 + [0]*10),
        # edge: all discordant in one direction
        ([1]*3 + [0]*0 + [1]*5 + [0]*12, [1]*3 + [1]*0 + [0]*5 + [0]*12),
    ],
    ids=["small-1", "small-2", "one-direction"],
)
def test_mcnemar_matches_statsmodels_exact(a, b):
    """Exact binomial test when discordant < 25."""
    r = compare_paired(a, b, method="mcnemar")

    a_arr, b_arr = np.array(a, dtype=bool), np.array(b, dtype=bool)
    table = np.array([
        [(a_arr & b_arr).sum(), (a_arr & ~b_arr).sum()],
        [(~a_arr & b_arr).sum(), (~a_arr & ~b_arr).sum()],
    ])
    sm = sm_mcnemar(table, exact=True)

    assert r.p_value == pytest.approx(sm.pvalue, abs=1e-6)
    assert r.n_discordant is not None
    assert r.n_discordant < 25


def test_mcnemar_discordant_count():
    """n_discordant should equal b + c from the contingency table."""
    a = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0]
    b = [1, 0, 1, 1, 0, 0, 1, 1, 1, 0]
    r = compare_paired(a, b, method="mcnemar")
    a_arr, b_arr = np.array(a, dtype=bool), np.array(b, dtype=bool)
    expected_disc = int((a_arr & ~b_arr).sum() + (~a_arr & b_arr).sum())
    assert r.n_discordant == expected_disc


def test_mcnemar_identical_systems():
    """Two identical score vectors should have p near 1 and difference 0."""
    scores = [1, 0, 1, 1, 0, 0, 1, 1, 0, 1]
    r = compare_paired(scores, scores, method="mcnemar")
    assert r.difference == pytest.approx(0.0)
    assert r.p_value == pytest.approx(1.0)
    assert r.n_discordant == 0


def test_auto_picks_mcnemar_for_binary():
    a = [1, 0, 1, 1, 0, 0, 1, 0]
    b = [0, 0, 1, 1, 1, 0, 1, 0]
    r = compare_paired(a, b)
    assert r.method == "mcnemar"
    assert r.binary is True


# --------------------------------------------------------------------------
# Paired bootstrap, continuous data
# --------------------------------------------------------------------------

def test_auto_picks_bootstrap_for_continuous():
    rng = np.random.default_rng(0)
    a = rng.normal(3.0, 1.0, 50)
    b = rng.normal(2.8, 1.0, 50)
    r = compare_paired(a, b, seed=1)
    assert r.method == "bootstrap"
    assert r.binary is False


def test_paired_bootstrap_deterministic():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 60)
    b = rng.normal(0.3, 1, 60)
    r1 = compare_paired(a, b, method="bootstrap", seed=42, n_boot=2000)
    r2 = compare_paired(a, b, method="bootstrap", seed=42, n_boot=2000)
    assert r1.ci_low == r2.ci_low
    assert r1.ci_high == r2.ci_high


def test_paired_bootstrap_difference_is_mean_diff():
    rng = np.random.default_rng(2)
    a = rng.normal(5.0, 1.0, 80)
    b = rng.normal(4.5, 1.0, 80)
    r = compare_paired(a, b, method="bootstrap", seed=7)
    assert r.difference == pytest.approx(float(np.mean(a) - np.mean(b)))
    assert r.ci_low < r.difference < r.ci_high


def test_paired_bootstrap_coverage():
    """The 95% bootstrap CI on the paired difference should cover the true
    difference about 95% of the time."""
    rng = np.random.default_rng(20)
    true_diff = 0.5
    covered = 0
    trials = 400
    for _ in range(trials):
        base = rng.normal(0, 1, 80)
        a = base + rng.normal(0, 0.3, 80)
        b = base + rng.normal(-true_diff, 0.3, 80)
        r = compare_paired(
            a, b, method="bootstrap", n_boot=1000,
            seed=int(rng.integers(1_000_000)),
        )
        if r.ci_low <= true_diff <= r.ci_high:
            covered += 1
    rate = covered / trials
    assert 0.91 <= rate <= 0.99, f"coverage was {rate:.3f}"


def test_paired_bootstrap_interval_narrows():
    rng = np.random.default_rng(3)
    a_small = rng.normal(0, 1, 40)
    b_small = rng.normal(0.2, 1, 40)
    a_large = rng.normal(0, 1, 4000)
    b_large = rng.normal(0.2, 1, 4000)
    r_small = compare_paired(a_small, b_small, seed=1)
    r_large = compare_paired(a_large, b_large, seed=1)
    assert r_large.width < r_small.width


# --------------------------------------------------------------------------
# Wilcoxon signed-rank
# --------------------------------------------------------------------------

def test_wilcoxon_runs():
    rng = np.random.default_rng(5)
    # Both systems see the same 50 items, so they share the item difficulty
    # term. Drawing a and b independently would leave nothing for a paired
    # test to exploit and the shift would not clear significance.
    difficulty = rng.normal(3.0, 1.0, 50)
    a = difficulty + rng.normal(0.0, 0.3, 50)
    b = difficulty + rng.normal(-0.5, 0.3, 50)
    r = compare_paired(a, b, method="wilcoxon")
    assert r.method == "wilcoxon"
    assert r.p_value < 0.05
    assert r.ci_low < r.difference < r.ci_high


# --------------------------------------------------------------------------
# Summary verdicts
# --------------------------------------------------------------------------

def test_summary_states_crosses_zero():
    a = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    b = [0, 1, 0, 1, 1, 0, 1, 0, 1, 0]
    r = compare_paired(a, b, method="mcnemar")
    assert "includes zero" in r.summary()


def test_summary_states_direction_when_clear():
    rng = np.random.default_rng(10)
    a = rng.normal(5.0, 0.5, 200)
    b = rng.normal(3.0, 0.5, 200)
    r = compare_paired(a, b, method="bootstrap", seed=1)
    s = r.summary()
    assert "includes zero" not in s
    assert "System A" in s


def test_summary_reports_confidence_level():
    rng = np.random.default_rng(11)
    a = rng.normal(0, 1, 60)
    b = rng.normal(0, 1, 60)
    r90 = compare_paired(a, b, method="bootstrap", confidence=0.90, seed=1)
    r99 = compare_paired(a, b, method="bootstrap", confidence=0.99, seed=1)
    assert "90%" in r90.summary()
    assert "99%" in r99.summary()


def test_summary_reports_discordant_for_binary():
    a = [1, 1, 0, 0, 1, 0, 1, 0, 1, 1]
    b = [1, 0, 1, 0, 1, 0, 0, 1, 1, 1]
    r = compare_paired(a, b, method="mcnemar")
    # The whole sentence, both counts included. "changed between systems"
    # alone is also the tail of "No items changed between systems", the
    # empty-table sentence, so it passes when the summary says the opposite.
    # The leading space stops "14 of 10" from matching.
    assert " 4 of 10 items changed between systems." in r.summary()


# --------------------------------------------------------------------------
# Independent samples
# --------------------------------------------------------------------------

def test_independent_continuous():
    rng = np.random.default_rng(6)
    a = rng.normal(3.0, 1.0, 60)
    b = rng.normal(2.5, 1.0, 50)
    r = compare_independent(a, b, seed=1)
    assert r.paired is False
    assert r.difference == pytest.approx(float(np.mean(a) - np.mean(b)))


def test_independent_binary():
    a = [1]*30 + [0]*20
    b = [1]*20 + [0]*30
    r = compare_independent(a, b)
    assert r.paired is False
    assert r.binary is True


def test_independent_summary_says_independent():
    rng = np.random.default_rng(7)
    a = rng.normal(0, 1, 40)
    b = rng.normal(0, 1, 30)
    r = compare_independent(a, b, seed=1)
    assert "independent" in r.summary()


# --------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------

def test_paired_rejects_empty():
    with pytest.raises(ValueError):
        compare_paired([], [])


def test_paired_rejects_length_mismatch():
    with pytest.raises(ValueError):
        compare_paired([1, 0, 1], [1, 0])


def test_independent_rejects_empty():
    with pytest.raises(ValueError):
        compare_independent([], [1, 0])


def test_rejects_unknown_method():
    with pytest.raises(ValueError):
        compare_paired([1, 0], [0, 1], method="nonsense")


# --------------------------------------------------------------------------
# Pairing has to actually be used
# --------------------------------------------------------------------------

def test_pairing_narrows_interval_on_correlated_data():
    """Paired analysis of correlated data must beat the independent one.

    Both calls see the same two vectors. The paired interval cancels the
    shared per-item difficulty, so it should come out several times
    narrower. If the two widths land close together, compare_paired is
    throwing the pairing away and computing an independent interval, which
    is the failure that passes every other test in this file.
    """
    rng = np.random.default_rng(31)
    n = 200
    difficulty = rng.normal(0.0, 1.0, n)          # shared by both systems
    a = difficulty + rng.normal(0.0, 0.05, n)
    b = difficulty + rng.normal(-0.2, 0.05, n)

    corr = float(np.corrcoef(a, b)[0, 1])
    assert corr > 0.95, f"test data is not correlated enough, r={corr:.3f}"

    r_paired = compare_paired(a, b, method="bootstrap", seed=1, n_boot=4000)
    r_indep = compare_independent(a, b, method="bootstrap", seed=1, n_boot=4000)

    # Same point estimate, so any difference in width comes from the pairing.
    assert r_paired.difference == pytest.approx(r_indep.difference, abs=1e-9)
    assert r_paired.paired is True
    assert r_indep.paired is False
    assert r_paired.width < 0.25 * r_indep.width, (
        f"paired width {r_paired.width:.4f} against independent "
        f"{r_indep.width:.4f}; the pairing is being ignored"
    )


# --------------------------------------------------------------------------
# The exact/chi-square switch at 25 discordant pairs
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "n_a_only, n_b_only, expect_exact",
    [
        (17, 7, True),    # 24 discordant, still exact
        (18, 7, False),   # 25 discordant, switches to chi-square
    ],
    ids=["24-discordant-exact", "25-discordant-chisquare"],
)
def test_mcnemar_threshold_is_at_25(n_a_only, n_b_only, expect_exact):
    """24 discordant pairs takes the exact path, 25 takes chi-square.

    Both paths return a p-value, so an off-by-one here runs the wrong test
    and reports the answer with no complaint. Pin the boundary from both
    sides by checking the p-value against each statsmodels path.
    """
    n_concordant = 20
    a = [1] * n_a_only + [0] * n_b_only + [1] * n_concordant + [0] * n_concordant
    b = [0] * n_a_only + [1] * n_b_only + [1] * n_concordant + [0] * n_concordant

    r = compare_paired(a, b, method="mcnemar")
    assert r.n_discordant == n_a_only + n_b_only

    a_arr, b_arr = np.array(a, dtype=bool), np.array(b, dtype=bool)
    table = np.array([
        [(a_arr & b_arr).sum(), (a_arr & ~b_arr).sum()],
        [(~a_arr & b_arr).sum(), (~a_arr & ~b_arr).sum()],
    ])
    p_exact = sm_mcnemar(table, exact=True).pvalue
    p_chi2 = sm_mcnemar(table, exact=False, correction=True).pvalue

    # Guard: the two paths have to disagree here, or this test proves nothing.
    # These splits keep them about 3-5% apart in relative terms.
    assert abs(p_exact - p_chi2) / p_chi2 > 0.01

    if expect_exact:
        assert r.p_value == pytest.approx(p_exact, rel=1e-8)
        assert r.p_value != pytest.approx(p_chi2, rel=1e-3)
    else:
        assert r.p_value == pytest.approx(p_chi2, rel=1e-8)
        assert r.p_value != pytest.approx(p_exact, rel=1e-3)


# --------------------------------------------------------------------------
# Nothing changed between the two systems
# --------------------------------------------------------------------------

def test_zero_discordant_pairs_is_reported_honestly():
    """Two systems that agree on every item give McNemar an empty table.

    There is no division to do and no evidence in either direction. The
    call must return rather than raise, and the summary must say the table
    was empty instead of dressing a conventional p-value up as a result.
    """
    a = [1, 1, 0, 0, 1, 0, 1, 1, 0, 0]
    r = compare_paired(a, list(a), method="mcnemar")

    assert r.n_discordant == 0
    assert r.difference == pytest.approx(0.0)
    assert not np.isnan(r.p_value)

    s = r.summary().lower()
    assert "no items changed" in s
    assert "cannot support any claim" in s
    # It must not read as a verdict for either system.
    assert "system a is higher" not in s
    assert "system b is higher" not in s


# --------------------------------------------------------------------------
# The binary paired interval itself
# --------------------------------------------------------------------------

def _paired_binary_table(n_a_only, n_b_only, n):
    """Build two score vectors with the given discordant counts.

    The concordant items are split evenly between both-pass and both-fail,
    which McNemar ignores anyway.
    """
    n_concordant = n - n_a_only - n_b_only
    n_both = n_concordant // 2
    n_neither = n_concordant - n_both
    a = [1] * n_a_only + [0] * n_b_only + [1] * n_both + [0] * n_neither
    b = [0] * n_a_only + [1] * n_b_only + [1] * n_both + [0] * n_neither
    return a, b


@pytest.mark.parametrize(
    "n, p_a, p_b, rho",
    [
        (100, 0.75, 0.60, 0.6),
        (100, 0.95, 0.90, 0.8),   # high pass rates, few discordant pairs
    ],
    ids=["moderate-rates", "high-rates"],
)
def test_mcnemar_coverage(n, p_a, p_b, rho):
    """The 95% interval on the paired difference in proportions should cover
    the true difference about 95% of the time.

    Each item gets a shared draw with probability rho and an independent one
    otherwise, so the two systems agree on most items while the marginal
    pass rates stay exactly p_a and p_b. That keeps the true difference
    known while the data stay correlated the way real paired evals are.

    The second setting is the one that strains the interval. Pass rates near
    1 leave few discordant pairs, and that is where a normal approximation
    on the difference starts to under-cover.
    """
    rng = np.random.default_rng(21)
    true_diff = p_a - p_b
    trials = 400
    covered = 0
    for _ in range(trials):
        shared = rng.random(n)
        u_a = np.where(rng.random(n) < rho, shared, rng.random(n))
        u_b = np.where(rng.random(n) < rho, shared, rng.random(n))
        a = (u_a < p_a).astype(int)
        b = (u_b < p_b).astype(int)
        r = compare_paired(a, b, method="mcnemar")
        if r.ci_low <= true_diff <= r.ci_high:
            covered += 1
    rate = covered / trials
    assert 0.91 <= rate <= 0.99, f"coverage was {rate:.3f}"


@pytest.mark.parametrize(
    "n_a_only, n_b_only, n, significant",
    [
        (30, 5, 100, True),
        (5, 30, 100, True),     # same table the other way round
        (20, 2, 80, True),
        (40, 10, 200, True),
        (13, 1, 60, True),      # exact path
        (12, 2, 60, True),      # exact path
        (10, 8, 50, False),
        (15, 13, 100, False),
        (25, 20, 120, False),
        (22, 11, 90, False),    # p = 0.082, the interval clears zero by 0.003
        (5, 5, 40, False),
        (1, 0, 40, False),
        (0, 0, 30, False),      # empty table
    ],
)
def test_interval_agrees_with_test(n_a_only, n_b_only, n, significant):
    """The interval and the p-value are two views of the same evidence.

    A p-value under 0.05 has to come with an interval that clears zero, and
    a p-value over 0.05 with one that contains it. Disagreement means one of
    the two is wrong, and the interval is the number people quote.

    These tables sit clear of the boundary on purpose. The reported p-value
    is the exact binomial or the continuity-corrected chi-square, and both
    are conservative next to the score statistic the interval inverts, so
    the two can genuinely differ for a reported p between about 0.05 and
    0.08. See test_significant_p_always_clears_zero for the half of the
    claim that holds everywhere.
    """
    a, b = _paired_binary_table(n_a_only, n_b_only, n)
    r = compare_paired(a, b, method="mcnemar")

    assert (r.p_value < 0.05) is significant, f"p was {r.p_value:.4f}"
    if significant:
        assert not r.crosses_zero, (
            f"p={r.p_value:.4f} rejects but the interval "
            f"({r.ci_low:.4f}, {r.ci_high:.4f}) contains zero"
        )
    else:
        assert r.crosses_zero, (
            f"p={r.p_value:.4f} does not reject but the interval "
            f"({r.ci_low:.4f}, {r.ci_high:.4f}) excludes zero"
        )


def test_significant_p_always_clears_zero():
    """Sweep the tables: a significant p must never carry an interval that
    contains zero.

    This is the direction that holds for every table, since the reported
    p-value is conservative against the statistic the interval inverts. If
    it ever fails, the interval and the test have come apart.
    """
    for n in (30, 60, 100):
        for n_a_only in range(0, 16):
            for n_b_only in range(0, 16):
                if n_a_only + n_b_only > n:
                    continue
                a, b = _paired_binary_table(n_a_only, n_b_only, n)
                r = compare_paired(a, b, method="mcnemar")
                if r.p_value < 0.05:
                    assert not r.crosses_zero, (
                        f"n={n} b={n_a_only} c={n_b_only}: p={r.p_value:.4f} "
                        f"rejects but the interval keeps zero "
                        f"({r.ci_low:.4f}, {r.ci_high:.4f})"
                    )


# --------------------------------------------------------------------------
# The independent binary path
# --------------------------------------------------------------------------

def test_independent_interval_and_test_never_disagree():
    """On independent binary data the interval is the test inverted.

    The pooled variance in the two-proportion z test is the maximum
    likelihood estimate under a difference of zero, which makes that test
    the score test at zero and the interval its exact inversion. So unlike
    the paired case, where the exact binomial is conservative against the
    statistic the interval inverts, this holds in both directions for every
    table.
    """
    for n_a, n_b in [(30, 30), (40, 25)]:
        for k_a in range(n_a + 1):
            for k_b in range(n_b + 1):
                a = [1] * k_a + [0] * (n_a - k_a)
                b = [1] * k_b + [0] * (n_b - k_b)
                r = compare_independent(a, b)
                assert r.method == "score"
                assert (r.p_value < 0.05) is (not r.crosses_zero), (
                    f"{k_a}/{n_a} against {k_b}/{n_b}: p={r.p_value:.5f} "
                    f"but the interval is ({r.ci_low:.5f}, {r.ci_high:.5f})"
                )


# --------------------------------------------------------------------------
# Degenerate group sizes
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "a, b",
    [
        ([1.5], [1.0, 2.0, 3.0]),
        ([1.0, 2.0, 3.0], [1.5]),
        ([1.5], [2.5]),
    ],
    ids=["first-group-of-one", "second-group-of-one", "both-groups-of-one"],
)
def test_independent_t_rejects_single_item_group(a, b):
    """Welch's t needs a variance, and one item does not have one.

    Without the guard this divides by n - 1 and dies with a bare
    ZeroDivisionError from inside the library.
    """
    with pytest.raises(ValueError, match="at least 2 items per group"):
        compare_independent(a, b, method="t")


def test_independent_bootstrap_allows_single_item_group():
    """The bootstrap has no such problem, and the error message says so."""
    r = compare_independent([1.5], [1.0, 2.0, 3.0], method="bootstrap", seed=1)
    assert r.n == 4
    assert r.difference == pytest.approx(1.5 - 2.0)


# --------------------------------------------------------------------------
# Welch's t, continuous independent samples
#
# The interval is checked against statsmodels CompareMeans, which shares no
# code with compare_independent. The p-value is checked against
# scipy.stats.ttest_ind with equal_var=False, and against statsmodels as
# well. The implementation calls that same scipy function, so the scipy check
# alone pins the wiring and says nothing about the arithmetic.
#
# Every fixture has unequal variances and unequal group sizes. On equal
# variances with equal sizes Welch and Student agree, so a fixture like that
# passes whichever of the two the code computes.
# --------------------------------------------------------------------------

# Each case is (items in A, spread of A, items in B, spread of B).
WELCH_CASES = {
    # The small group is the tight one. Student's pooled variance is carried
    # by the wide group, and its interval comes out about twice as wide.
    "small-tight-vs-large-wide": (12, 0.05, 70, 0.30),
    # The small group is the wide one. Student's interval comes out well
    # under half as wide.
    "small-wide-vs-large-tight": (12, 0.30, 70, 0.05),
}


def welch_samples(case):
    n_a, sd_a, n_b, sd_b = WELCH_CASES[case]
    rng = np.random.default_rng(3)
    return rng.normal(0.62, sd_a, n_a), rng.normal(0.55, sd_b, n_b)


def statsmodels_means(a, b):
    return CompareMeans(DescrStatsW(a), DescrStatsW(b))


@pytest.mark.parametrize("case", sorted(WELCH_CASES))
def test_welch_fixtures_can_tell_welch_from_student(case):
    """The fixtures, checked for the property they exist to have.

    If the Welch and Student intervals came out close, an implementation
    computing Student's would pass every Welch test below.
    """
    a, b = welch_samples(case)
    means = statsmodels_means(a, b)
    welch_lo, welch_hi = means.tconfint_diff(alpha=0.05, usevar="unequal")
    student_lo, student_hi = means.tconfint_diff(alpha=0.05, usevar="pooled")
    ratio = (welch_hi - welch_lo) / (student_hi - student_lo)
    assert not 0.67 < ratio < 1.5, (
        f"Welch and Student widths are within a factor of {ratio:.2f}"
    )


@pytest.mark.parametrize("case", sorted(WELCH_CASES))
def test_welch_interval_matches_statsmodels(case):
    a, b = welch_samples(case)
    r = compare_independent(a, b, method="t")
    lo, hi = statsmodels_means(a, b).tconfint_diff(alpha=0.05, usevar="unequal")
    assert r.method == "t"
    assert r.paired is False
    assert r.binary is False
    assert r.n == a.size + b.size
    assert r.difference == pytest.approx(a.mean() - b.mean(), abs=1e-12)
    assert r.ci_low == pytest.approx(lo, abs=1e-10)
    assert r.ci_high == pytest.approx(hi, abs=1e-10)


@pytest.mark.parametrize("case", sorted(WELCH_CASES))
def test_welch_p_value_matches_scipy_and_statsmodels(case):
    a, b = welch_samples(case)
    r = compare_independent(a, b, method="t")
    assert r.p_value == pytest.approx(
        stats.ttest_ind(a, b, equal_var=False).pvalue, abs=1e-12
    )
    assert r.p_value == pytest.approx(
        statsmodels_means(a, b).ttest_ind(usevar="unequal")[1], abs=1e-10
    )


@pytest.mark.parametrize("confidence", [0.50, 0.80, 0.90, 0.99])
def test_welch_interval_follows_the_confidence_level(confidence):
    """None of these levels is 0.95, so a critical value fixed at the
    default fails every case."""
    a, b = welch_samples("small-tight-vs-large-wide")
    r = compare_independent(a, b, method="t", confidence=confidence)
    lo, hi = statsmodels_means(a, b).tconfint_diff(
        alpha=1 - confidence, usevar="unequal"
    )
    assert r.confidence == confidence
    assert r.ci_low == pytest.approx(lo, abs=1e-10)
    assert r.ci_high == pytest.approx(hi, abs=1e-10)


def test_welch_at_two_items_per_group():
    """The smallest groups the single-item guard lets through.

    The spreads differ by a factor of twenty, so the Welch-Satterthwaite
    degrees of freedom sit near 1 where Student would use 2.
    """
    a = np.array([0.40, 0.44])
    b = np.array([0.10, 0.90])
    r = compare_independent(a, b, method="t")
    lo, hi = statsmodels_means(a, b).tconfint_diff(alpha=0.05, usevar="unequal")
    assert r.ci_low == pytest.approx(lo, abs=1e-10)
    assert r.ci_high == pytest.approx(hi, abs=1e-10)
    assert r.p_value == pytest.approx(
        stats.ttest_ind(a, b, equal_var=False).pvalue, abs=1e-12
    )


# --------------------------------------------------------------------------
# Public surface
# --------------------------------------------------------------------------

def test_comparisons_return_comparison_result():
    """Both entry points return the same public type.

    It is named for what it is rather than for the paired case, since
    compare_independent returns one too.
    """
    paired = compare_paired([1, 0, 1, 1], [0, 0, 1, 1])
    independent = compare_independent([1, 0, 1, 1], [0, 0, 1])
    assert isinstance(paired, ComparisonResult)
    assert isinstance(independent, ComparisonResult)
    assert paired.paired is True
    assert independent.paired is False


# --------------------------------------------------------------------------
# Wilcoxon, exact distribution
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "n, critical",
    [
        (6, 0), (7, 2), (8, 3), (9, 5), (10, 8), (11, 10), (12, 13),
        (13, 17), (14, 21), (15, 25), (16, 29), (17, 34), (18, 40),
        (20, 52), (25, 89),
    ],
)
def test_signed_rank_trim_matches_published_table(n, critical):
    """The number of Walsh averages trimmed is the test's critical value.

    These are the published two-sided 5% critical values for the signed-rank
    statistic, the same ones printed in the back of a statistics textbook.
    They are an outside check on the distribution built by
    _signed_rank_counts, in the same spirit as checking McNemar against
    statsmodels.
    """
    assert _signed_rank_trim(n, 0.95, exact=True) == critical


def test_wilcoxon_uses_the_exact_p_value_when_it_applies():
    """Below the size cap, with no ties, the p-value is the exact one."""
    rng = np.random.default_rng(4)
    d = rng.normal(0.4, 1.0, 30)
    r = compare_paired(d, np.zeros(30), method="wilcoxon")
    assert r.p_value == pytest.approx(
        stats.wilcoxon(d, method="exact").pvalue, rel=1e-12
    )


def test_wilcoxon_falls_back_together_past_the_size_cap():
    """Over the cap both the test and the interval use the approximation.

    The point is that they move together. A p-value from the exact
    distribution next to an interval from the normal approximation would be
    two different tests printed on one line.
    """
    rng = np.random.default_rng(5)
    d = rng.normal(0.3, 1.0, 80)
    r = compare_paired(d, np.zeros(80), method="wilcoxon")
    assert r.p_value == pytest.approx(
        stats.wilcoxon(d, method="approx").pvalue, rel=1e-12
    )


def test_wilcoxon_interval_and_test_agree():
    """The interval clears zero exactly when the signed-rank test rejects.

    Both now come from the same distribution, so this holds sample by
    sample rather than on average.
    """
    rng = np.random.default_rng(3)
    for _ in range(250):
        n = int(rng.integers(6, 51))
        d = rng.normal(rng.uniform(-0.9, 0.9), 1.0, n)
        r = compare_paired(d, np.zeros(n), method="wilcoxon")
        assert (r.p_value < 0.05) is (not r.crosses_zero), (
            f"n={n}: p={r.p_value:.5f} but the interval is "
            f"({r.ci_low:.4f}, {r.ci_high:.4f})"
        )


def test_wilcoxon_drops_tied_items_everywhere_or_nowhere():
    """Items the two systems tied on leave the test, the estimate and the
    interval together.

    The signed-rank test drops zero differences, so if the Walsh averages
    kept them the interval would be built on a different set of items than
    the p-value beside it.
    """
    d = np.array([0.0, 0.0, 1.0, -2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    r = compare_paired(d, np.zeros(len(d)), method="wilcoxon")
    nonzero = d[d != 0]
    assert r.p_value == pytest.approx(
        stats.wilcoxon(nonzero, method="exact").pvalue, rel=1e-12
    )
    assert (r.p_value < 0.05) is (not r.crosses_zero)


def test_wilcoxon_coverage():
    """The exact interval should cover the true shift about 95% of the time."""
    rng = np.random.default_rng(11)
    true_shift = 0.5
    trials = 400
    covered = 0
    for _ in range(trials):
        d = rng.normal(true_shift, 1.0, 20)
        r = compare_paired(d, np.zeros(20), method="wilcoxon")
        if r.ci_low <= true_shift <= r.ci_high:
            covered += 1
    rate = covered / trials
    assert 0.91 <= rate <= 0.99, f"coverage was {rate:.3f}"


# --------------------------------------------------------------------------
# Wilcoxon on binary data, where every difference is the same size
# --------------------------------------------------------------------------

def test_wilcoxon_uses_the_tie_corrected_variance():
    """Rubric scores where every item that moved, moved by the same amount.

    The differing items all differ by 2, so their ranks are one large tie
    and the variance correction is at its biggest. Without it the interval
    would be drawn from a wider distribution than the p-value printed
    beside it. Binary data would show the same structure and is the reason
    the correction exists, but compare_paired refuses that combination now,
    so the case is built from a 1 to 5 scale instead.
    """
    a = np.array([5.0] * 25 + [1.0] * 11 + [4.0] * 4 + [3.0] * 20)
    b = np.array([3.0] * 25 + [3.0] * 11 + [3.0] * 4 + [3.0] * 20)
    d = a - b
    # Three distinct shifts clears the guard, and 36 of the 40 differing
    # items still share one magnitude, so the tie group stays large.
    assert set(np.unique(d)) == {-2.0, 0.0, 1.0, 2.0}

    r = compare_paired(a, b, method="wilcoxon")
    differing = d[d != 0]

    # The p-value is scipy's approximation on the items that differ.
    assert r.p_value == pytest.approx(
        stats.wilcoxon(differing, method="approx").pvalue, rel=1e-12
    )

    # Two tie groups, 36 differences of size 2 and 4 of size 1.
    n = differing.size
    assert n == 40
    correction = float(36**3 - 36) + float(4**3 - 4)
    assert _tie_correction(differing) == correction

    z = stats.norm.ppf(0.975)
    spread = n * (n + 1) * (2 * n + 1) - correction / 2
    expected = int(np.floor(n * (n + 1) / 4 - z * np.sqrt(spread / 24)))
    assert _signed_rank_trim(n, 0.95, False, correction) == expected

    # Leaving the correction out would trim from a different distribution.
    assert _signed_rank_trim(n, 0.95, False) != expected


def test_wilcoxon_refuses_binary_data():
    """Binary scores and the signed-rank interval do not go together.

    Every difference is -1, 0 or 1, so every Walsh average is one of five
    values and the interval can only land on that lattice. It reported a
    bound of exactly zero while the p-value rejected on 41% of measured
    samples. Returning a number that argues with the number beside it is
    worse than refusing, so compare_paired refuses and names the method
    that handles this data properly.
    """
    a = np.array([1.0] * 25 + [0.0] * 11 + [1.0] * 20 + [0.0] * 14)
    b = np.array([0.0] * 25 + [1.0] * 11 + [1.0] * 20 + [0.0] * 14)

    with pytest.raises(ValueError, match="mcnemar"):
        compare_paired(a, b, method="wilcoxon")

    # The message has to name the cause, not just refuse.
    with pytest.raises(ValueError, match="distinct non-zero differences"):
        compare_paired(a, b, method="wilcoxon")
    with pytest.raises(ValueError, match="three-value lattice"):
        compare_paired(a, b, method="wilcoxon")

    # The route the caller is pointed at works on the same data.
    assert compare_paired(a, b, method="mcnemar").method == "mcnemar"
    assert compare_paired(a, b).method == "mcnemar"


def test_wilcoxon_refuses_scores_that_only_look_binary():
    """A rubric using two of its levels is the 0/1 problem in disguise.

    Scores of 3 and 5 differ by plus or minus 2, which gives Walsh averages
    of -2, 0 and 2. That is the same three-value lattice 0/1 data produces,
    so the guard has to catch it on the shape of the differences rather
    than on the scores being 0 and 1.
    """
    a = [3.0, 5.0, 3.0, 5.0, 5.0, 3.0, 5.0, 5.0]
    b = [5.0, 5.0, 3.0, 3.0, 5.0, 5.0, 3.0, 3.0]
    assert set(np.unique(np.asarray(a) - np.asarray(b))) == {-2.0, 0.0, 2.0}

    with pytest.raises(ValueError, match="distinct"):
        compare_paired(a, b, method="wilcoxon")

    # McNemar is not the way out here, since this is not 0/1 data.
    with pytest.raises(ValueError, match="binary 0/1"):
        compare_paired(a, b, method="mcnemar")
    assert compare_paired(a, b, method="bootstrap", seed=1).method == "bootstrap"


def test_wilcoxon_reports_the_analysed_item_count():
    """Tied items leave the test, so they leave the reported n as well.

    Counting them would overstate how much evidence the interval rests on.
    """
    a = [5.0, 4, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    b = [3.0, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    differing = int(np.sum(np.asarray(a) != np.asarray(b)))
    assert differing == 3

    r = compare_paired(a, b, method="wilcoxon")
    assert r.n == differing
    assert f"n={differing}" in r.summary()


def test_wilcoxon_counts_nothing_when_every_item_ties():
    """No item carried a signed difference, so the analysed count is zero."""
    r = compare_paired([2.5, 4.0, 3.5], [2.5, 4.0, 3.5], method="wilcoxon")
    assert r.n == 0
    assert r.p_value == pytest.approx(1.0)
    assert r.difference == pytest.approx(0.0)
