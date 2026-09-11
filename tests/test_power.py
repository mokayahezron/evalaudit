"""Tests for evalaudit.power.

Written before the implementation. Make these pass.

Three kinds of check run here.

External. The independent two-proportion arithmetic is checked against
statsmodels and against the two sample size figures every textbook prints
for it, 97 per group for 0.40 against 0.60 and 388 per group for 0.50
against 0.60. The paired arithmetic is checked against Connor's published
formula written in its odds ratio parameterisation, which is a different
algebraic route to the same number, and against the one sample binomial
formula it collapses to when every pair is discordant.

Simulation. Data is generated at the reported detectable difference and run
through the tests evalaudit actually ships, compare_paired and
compare_independent. The rejection rate has to come out near the power that
was asked for. This is the check that catches a formula that is internally
tidy and wrong.

Inverse consistency. The two entry points solve one relation in opposite
directions, so composing them has to return where it started, and one item
short of the answer has to fall short.
"""

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.proportion import (
    power_proportions_2indep,
    samplesize_proportions_2indep_onetail,
)

from evalaudit import PowerResult, detectable_effect, min_sample_size
from evalaudit.power import (
    _DEFAULT_DISCORDANCE,
    _REFERENCE_DIFFERENCE,
    _power_independent,
    _power_paired,
)


def _z(alpha, power):
    return stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)


# --------------------------------------------------------------------------
# Independent binary, against statsmodels and published tables
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "baseline, mde, expected_per_group",
    [
        # The two figures every sample size table prints for the two
        # proportion test at 5% and 80%.
        (0.40, 0.20, 97),
        (0.50, 0.10, 388),
    ],
    ids=["40-vs-60", "50-vs-60"],
)
def test_independent_matches_published_tables(baseline, mde, expected_per_group):
    r = min_sample_size(mde, baseline=baseline, paired=False)
    assert r.n_per_group == expected_per_group
    assert r.n == 2 * expected_per_group


@pytest.mark.parametrize(
    "baseline, mde",
    [(0.5, 0.1), (0.4, 0.2), (0.7, 0.1), (0.9, 0.05), (0.3, 0.05), (0.5, 0.02)],
)
@pytest.mark.parametrize("power", [0.8, 0.9])
def test_independent_sample_size_matches_statsmodels(baseline, mde, power):
    """statsmodels solves the same pooled normal approximation."""
    r = min_sample_size(mde, baseline=baseline, power=power, paired=False)
    expected = samplesize_proportions_2indep_onetail(
        mde, baseline, power, alternative="two-sided"
    )
    assert r.n_per_group == int(np.ceil(expected))


@pytest.mark.parametrize(
    "baseline, mde, n_per_group",
    [(0.5, 0.1, 388), (0.4, 0.2, 97), (0.7, 0.1, 294), (0.9, 0.05, 435)],
)
def test_independent_power_matches_statsmodels(baseline, mde, n_per_group):
    """The power function agrees with statsmodels to the far tail.

    statsmodels adds the probability of rejecting on the wrong side, which
    the closed form sample size formula leaves out. That term is under 1e-5
    at any setting a real eval runs, and leaving it out is what makes the
    two entry points exact inverses of each other.
    """
    ours = _power_independent(n_per_group, baseline, mde, 0.05)
    theirs = power_proportions_2indep(
        mde, baseline, n_per_group, return_results=False
    )
    assert ours == pytest.approx(theirs, abs=1e-5)
    assert ours <= theirs


@pytest.mark.parametrize("baseline", [0.3, 0.5, 0.7, 0.9])
@pytest.mark.parametrize("n", [200, 776, 2000])
def test_independent_detectable_effect_hits_requested_power(baseline, n):
    """The returned difference is the one that gives exactly the power asked."""
    r = detectable_effect(n, baseline=baseline, paired=False)
    if not r.attainable:
        pytest.skip("no difference reaches this power at this size")
    achieved = power_proportions_2indep(
        r.difference, baseline, n / 2, return_results=False
    )
    assert achieved == pytest.approx(0.8, abs=1e-5)


def test_independent_reports_both_group_size_and_total():
    r = min_sample_size(0.1, paired=False)
    assert r.n == 2 * r.n_per_group
    assert r.n_discordant is None
    assert r.discordance_rate is None
    assert r.paired is False


def _solve_independent_mde(n_per_group, baseline, power=0.8, alpha=0.05):
    """Bisection on the power function, written out here on purpose.

    The implementation is free to use whatever solver it likes. This is a
    second route to the same root, so a broken solver shows up as a
    disagreement rather than as two matching wrong answers.
    """
    lo, hi = 1e-12, (1.0 - baseline) * (1 - 1e-9)
    for _ in range(200):
        mid = (lo + hi) / 2
        if _power_independent(n_per_group, baseline, mid, alpha) < power:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def test_independent_detectable_effect_splits_n_evenly():
    """n is the total across both groups, so 776 items means 388 each."""
    total = detectable_effect(776, baseline=0.5, paired=False)
    assert total.difference == pytest.approx(
        _solve_independent_mde(388, 0.5), rel=1e-9
    )


# --------------------------------------------------------------------------
# Paired binary, against Connor's published formula
# --------------------------------------------------------------------------

def _connor_odds_ratio_form(discordance_rate, odds_ratio, alpha=0.05, power=0.8):
    """Connor (1987) written in terms of the discordant odds ratio.

    n = [z(1+psi) + z_beta sqrt((1+psi)^2 - (psi-1)^2 pi)]^2 / [(psi-1)^2 pi]

    The implementation works in the difference parameterisation. The two are
    the same formula after algebra, so agreement here checks the algebra.
    """
    za, zb = _z(alpha, power)
    psi = odds_ratio
    num = za * (psi + 1) + zb * np.sqrt(
        (psi + 1) ** 2 - (psi - 1) ** 2 * discordance_rate
    )
    return num**2 / ((psi - 1) ** 2 * discordance_rate)


@pytest.mark.parametrize("rate", [0.05, 0.1, 0.3, 0.5])
@pytest.mark.parametrize("mde", [0.01, 0.02, 0.05])
@pytest.mark.parametrize("power", [0.8, 0.9])
def test_paired_sample_size_matches_connor(rate, mde, power):
    if mde > rate:
        pytest.skip("a difference cannot exceed the discordance rate")
    if mde == rate:
        pytest.skip(
            "the odds ratio parameterisation cannot reach this point. The "
            "c cell is empty there, so psi is b over zero and the check has "
            "nothing to compute. The implementation handles it, and "
            "test_a_difference_equal_to_the_discordance_rate_is_allowed "
            "covers it"
        )
    r = min_sample_size(mde, power=power, discordance_rate=rate)
    b = (rate + mde) / 2
    c = (rate - mde) / 2
    expected = _connor_odds_ratio_form(rate, b / c, power=power)
    assert r.n == int(np.ceil(expected))


def test_paired_collapses_to_one_sample_binomial_when_all_pairs_discordant():
    """With every pair discordant McNemar is a sign test on n items.

    The sample size then has to be the one sample binomial figure for
    testing a half, n = [z + z_beta sqrt(1 - d^2)]^2 / d^2.
    """
    za, zb = _z(0.05, 0.8)
    for mde in (0.05, 0.1, 0.2):
        expected = (za + zb * np.sqrt(1 - mde**2)) ** 2 / mde**2
        r = min_sample_size(mde, discordance_rate=1.0)
        assert r.n == int(np.ceil(expected))


@pytest.mark.parametrize("n", [220, 500, 1000, 5000])
@pytest.mark.parametrize("rate", [0.05, 0.1, 0.3, 0.6])
def test_paired_detectable_effect_hits_requested_power(n, rate):
    r = detectable_effect(n, discordance_rate=rate)
    achieved = _power_paired(n, rate, r.difference, 0.05)
    assert achieved == pytest.approx(0.8, abs=1e-9)


def test_paired_power_function_is_the_normal_approximation():
    """Pinned so the power function cannot drift away from the formula."""
    za, _ = _z(0.05, 0.8)
    n, rate, d = 500, 0.3, 0.06
    expected = stats.norm.cdf(
        (np.sqrt(n) * d - za * np.sqrt(rate)) / np.sqrt(rate - d * d)
    )
    assert _power_paired(n, rate, d, 0.05) == pytest.approx(expected, rel=1e-12)


def test_paired_ignores_baseline():
    """Two systems at any pass rate need the same number of pairs.

    This is the point of the module. Paired binary power runs on the
    discordance rate, and the baseline does not enter it.
    """
    sizes = {
        min_sample_size(0.05, baseline=b, discordance_rate=0.2).n
        for b in (0.1, 0.3, 0.5, 0.75, 0.95)
    }
    assert len(sizes) == 1

    effects = {
        round(detectable_effect(400, baseline=b, discordance_rate=0.2).difference, 12)
        for b in (0.1, 0.3, 0.5, 0.75, 0.95)
    }
    assert len(effects) == 1


# --------------------------------------------------------------------------
# Simulation against the tests evalaudit actually ships
# --------------------------------------------------------------------------

# 200,000 draws puts the standard error on a rejection rate near 0.8 at
# 0.00092, which is what lets the conservatism band below name a number
# instead of gesturing at one. At the 4,000 draws this file first used the
# standard error is 0.0065, seven times wider than the effect being
# measured, and no honest band fits inside that.
_SIM_REPS = 200_000
_SIM_SEED = 11


def _mcnemar_p_values(n_a_only, n_b_only):
    """compare_paired's decision rule, vectorised over many tables.

    Exact binomial below 25 discordant pairs, continuity corrected
    chi-square at 25 and above, p=1 on an empty table. Held to the shipped
    function by test_vectorised_mcnemar_reproduces_compare_paired, which is
    what licenses using it in place of 200,000 calls.
    """
    discordant = n_a_only + n_b_only
    p = np.ones(len(discordant))
    big = discordant >= 25
    stat = (np.abs(n_a_only - n_b_only) - 1.0) ** 2 / np.maximum(discordant, 1)
    p[big] = stats.chi2.sf(stat[big], 1)
    small = (discordant > 0) & ~big
    p[small] = np.minimum(
        2 * stats.binom.cdf(
            np.minimum(n_a_only, n_b_only)[small], discordant[small], 0.5
        ),
        1.0,
    )
    return p


def _score_p_values(a_passes, b_passes, n_per_group):
    """compare_independent's score test, vectorised over many samples."""
    pooled = (a_passes + b_passes) / (2 * n_per_group)
    se = np.sqrt(pooled * (1 - pooled) * (2 / n_per_group))
    p = np.ones(len(a_passes))
    usable = se > 0
    diff = (a_passes - b_passes)[usable] / n_per_group
    p[usable] = 2 * stats.norm.sf(np.abs(diff / se[usable]))
    return p


def _draw_paired_tables(n, rate, difference, reps, seed):
    """Discordant cell counts for reps evals of n pairs each.

    Each item lands in one of four cells. Two are discordant and carry the
    difference, and the concordant mass is split evenly because the split
    cannot reach McNemar's statistic.
    """
    rng = np.random.default_rng(seed)
    b = (rate + difference) / 2
    c = (rate - difference) / 2
    rest = (1 - rate) / 2
    counts = rng.multinomial(n, [b, c, rest, rest], size=reps)
    return counts[:, 0], counts[:, 1]


def _paired_rejection_rate(n, rate, difference, reps=_SIM_REPS, seed=_SIM_SEED):
    n_a_only, n_b_only = _draw_paired_tables(n, rate, difference, reps, seed)
    return float(np.mean(_mcnemar_p_values(n_a_only, n_b_only) < 0.05))


def _independent_rejection_rate(
    n_per_group, baseline, difference, reps=_SIM_REPS, seed=_SIM_SEED
):
    rng = np.random.default_rng(seed)
    a_passes = rng.binomial(n_per_group, baseline + difference, reps)
    b_passes = rng.binomial(n_per_group, baseline, reps)
    return float(np.mean(_score_p_values(a_passes, b_passes, n_per_group) < 0.05))


def test_vectorised_mcnemar_reproduces_compare_paired():
    """The fast rule and the shipped function have to be the same test."""
    from evalaudit import compare_paired

    n, rate, difference = 800, 0.3, 0.054
    rng = np.random.default_rng(3)
    b = (rate + difference) / 2
    c = (rate - difference) / 2
    rest = (1 - rate) / 2
    counts = rng.multinomial(n, [b, c, rest, rest], size=1500)

    mine = _mcnemar_p_values(counts[:, 0], counts[:, 1])
    for row, fast in zip(counts, mine):
        n_b, n_c, n_both, n_neither = row
        a_scores = np.concatenate(
            [np.ones(n_b), np.zeros(n_c), np.ones(n_both), np.zeros(n_neither)]
        )
        b_scores = np.concatenate(
            [np.zeros(n_b), np.ones(n_c), np.ones(n_both), np.zeros(n_neither)]
        )
        shipped = compare_paired(a_scores, b_scores, method="mcnemar").p_value
        assert fast == pytest.approx(shipped, abs=1e-12)


def test_vectorised_score_test_reproduces_compare_independent():
    from evalaudit import compare_independent

    n_per_group = 400
    rng = np.random.default_rng(4)
    a_passes = rng.binomial(n_per_group, 0.6, 1500)
    b_passes = rng.binomial(n_per_group, 0.5, 1500)

    mine = _score_p_values(a_passes, b_passes, n_per_group)
    for n_a, n_b, fast in zip(a_passes, b_passes, mine):
        a_scores = np.concatenate([np.ones(n_a), np.zeros(n_per_group - n_a)])
        b_scores = np.concatenate([np.ones(n_b), np.zeros(n_per_group - n_b)])
        shipped = compare_independent(a_scores, b_scores, method="score").p_value
        assert fast == pytest.approx(shipped, abs=1e-12)


@pytest.mark.parametrize("n, rate", [(800, 0.3), (1500, 0.2)])
def test_paired_simulated_power_is_near_nominal(n, rate):
    r = detectable_effect(n, discordance_rate=rate)
    achieved = _paired_rejection_rate(n, rate, r.difference)
    assert achieved == pytest.approx(0.8, abs=0.04)


@pytest.mark.parametrize("n, rate", [(800, 0.3), (1500, 0.2)])
def test_paired_nominal_power_runs_ahead_of_the_shipped_test(n, rate):
    """The normal approximation promises more than McNemar delivers, by this much.

    evalaudit runs the continuity corrected chi-square above 25 discordant
    pairs and the exact binomial below it, and both are conservative. So
    real power at the reported difference lands under the power that was
    asked for, and the reported difference is a floor on what the eval
    could have found rather than a promise.

    The gap is named rather than left open. Asserting only that power falls
    short passes on an implementation that is wildly under powered, which
    turns a known and documented shortfall into cover for a real bug.

    The band is 0.010 to 0.025 against a nominal 0.800. Measured over
    200,000 draws the gap is 0.0160 at n=800 with a 0.3 rate and 0.0163 at
    n=1500 with a 0.2 rate, with a standard error of 0.00092. So the lower
    edge sits about 6.5 standard errors below the measured gap and the
    upper edge about 9.5 above it. The band excludes zero, so an
    implementation whose nominal power actually matches the shipped test
    fails here, and it excludes a gap past 2.5 points, so an implementation
    that quietly under powers fails too.
    """
    r = detectable_effect(n, discordance_rate=rate)
    achieved = _paired_rejection_rate(n, rate, r.difference)
    gap = 0.8 - achieved
    assert 0.010 <= gap <= 0.025


@pytest.mark.parametrize("n_per_group, baseline", [(400, 0.5), (1000, 0.7)])
def test_independent_simulated_power_is_near_nominal(n_per_group, baseline):
    """No continuity correction in the score test, so no systematic gap.

    Measured over 200,000 draws power comes out at 0.8015 and 0.7989 with a
    standard error of 0.0009, so the 0.01 tolerance is eleven standard
    errors wide and still an order of magnitude tighter than the paired
    case, where the correction costs real power.
    """
    r = detectable_effect(2 * n_per_group, baseline=baseline, paired=False)
    achieved = _independent_rejection_rate(n_per_group, baseline, r.difference)
    assert achieved == pytest.approx(0.8, abs=0.01)


# --------------------------------------------------------------------------
# The discordance rate, and saying when it was assumed
# --------------------------------------------------------------------------

def test_default_discordance_rate_is_three_tenths():
    assert _DEFAULT_DISCORDANCE == 0.3


def test_missing_rate_falls_back_to_the_default_and_flags_it():
    r = detectable_effect(500)
    assert r.discordance_rate == _DEFAULT_DISCORDANCE
    assert r.discordance_assumed is True
    assert r.difference == pytest.approx(
        detectable_effect(500, discordance_rate=0.3).difference
    )


def test_supplied_rate_is_not_flagged_as_assumed():
    r = detectable_effect(500, discordance_rate=0.12)
    assert r.discordance_rate == 0.12
    assert r.discordance_assumed is False
    assert "assum" not in r.summary().lower()


@pytest.mark.parametrize(
    "result",
    [
        detectable_effect(500),
        min_sample_size(0.05),
    ],
    ids=["detectable_effect", "min_sample_size"],
)
def test_assumed_rate_is_stated_in_the_summary(result):
    """Never assume the rate silently. The output has to own the assumption."""
    text = result.summary().lower()
    assert "assum" in text
    assert "30%" in text or "0.3" in text
    assert "not measured" in text or "was not supplied" in text


def test_default_rate_is_the_conservative_direction():
    """A lower discordance rate is easier to work with, so 0.3 is pessimistic.

    Systems that agree on most items make McNemar sharp, because the few
    pairs that differ carry the whole signal. The usual eval rate sits well
    under 0.3, so defaulting there overstates the sample needed and
    understates the reach, which is the safe way to be wrong in an audit.
    """
    assumed = detectable_effect(500)
    measured = detectable_effect(500, discordance_rate=0.05)
    assert assumed.difference > measured.difference

    # 0.02 rather than 0.05, because a five point difference cannot occur at
    # all when only 5% of pairs are discordant.
    assumed_n = min_sample_size(0.02)
    measured_n = min_sample_size(0.02, discordance_rate=0.05)
    assert assumed_n.n > measured_n.n


def test_detectable_difference_grows_with_the_discordance_rate():
    """The counterintuitive direction, pinned.

    More disagreement between the two systems means more noise in the
    per item comparison, not more signal.
    """
    rates = [0.02, 0.05, 0.1, 0.3, 0.6]
    got = [detectable_effect(2000, discordance_rate=r).difference for r in rates]
    assert got == sorted(got)


def test_sample_size_grows_with_the_discordance_rate():
    rates = [0.05, 0.1, 0.3, 0.6]
    got = [min_sample_size(0.02, discordance_rate=r).n for r in rates]
    assert got == sorted(got)


def test_discordant_pairs_are_the_effective_sample():
    """500 items at a 2.4% discordance rate is a 12 item study."""
    r = detectable_effect(500, discordance_rate=12 / 500)
    assert r.n == 500
    assert r.n_discordant == 12
    text = r.summary()
    assert "12" in text
    assert "discordant" in text.lower()


def test_sample_size_reports_the_discordant_pairs_it_implies():
    r = min_sample_size(0.05, discordance_rate=0.3)
    assert r.n_discordant == int(round(r.n * 0.3))
    assert "discordant" in r.summary().lower()


def test_discordant_pairs_round_rather_than_truncate():
    """300 items at 3.3% is 9.9 pairs, and that is 10, not 9.

    Truncating would understate the effective sample every time the product
    is not whole, which is most of the time, and it would understate it in
    the direction that flatters the eval.
    """
    assert detectable_effect(300, discordance_rate=0.033).n_discordant == 10
    assert min_sample_size(0.01, discordance_rate=0.033).n_discordant == int(
        round(min_sample_size(0.01, discordance_rate=0.033).n * 0.033)
    )


def test_discordance_rate_is_rejected_for_independent_designs():
    with pytest.raises(ValueError, match="discordance"):
        detectable_effect(500, paired=False, discordance_rate=0.3)
    with pytest.raises(ValueError, match="discordance"):
        min_sample_size(0.05, paired=False, discordance_rate=0.3)


def test_independent_results_carry_no_discordance_fields():
    r = detectable_effect(500, paired=False)
    assert r.discordance_rate is None
    assert r.discordance_assumed is False
    assert r.n_discordant is None


# --------------------------------------------------------------------------
# The two entry points are one relation solved in opposite directions
# --------------------------------------------------------------------------

@pytest.mark.parametrize("rate", [0.05, 0.1, 0.3, 0.5])
@pytest.mark.parametrize("mde", [0.01, 0.02, 0.05])
def test_paired_roundtrip(rate, mde):
    if mde > rate:
        pytest.skip("a difference cannot exceed the discordance rate")
    n = min_sample_size(mde, discordance_rate=rate).n
    assert detectable_effect(n, discordance_rate=rate).difference <= mde
    assert detectable_effect(n - 1, discordance_rate=rate).difference > mde


@pytest.mark.parametrize("baseline", [0.3, 0.5, 0.7])
@pytest.mark.parametrize("mde", [0.02, 0.05, 0.1])
def test_independent_roundtrip(baseline, mde):
    per_group = min_sample_size(mde, baseline=baseline, paired=False).n_per_group
    got = detectable_effect(2 * per_group, baseline=baseline, paired=False)
    assert got.difference <= mde
    short = detectable_effect(2 * (per_group - 1), baseline=baseline, paired=False)
    assert short.difference > mde


# Rounding tolerance for the inverse sweeps below. Composing the two entry
# points has to land on the same n, and the only slack allowed is one item
# upward.
#
# min_sample_size takes a ceiling. detectable_effect(n) returns the
# difference whose power at n is exactly the target, so solving that
# difference back for a sample size lands on n itself, give or take
# floating point noise of order 1e-10. Noise a hair below n still ceils to
# n. Noise a hair above ceils to n+1. Nothing else is reachable, since
# landing on n-1 would take an error of a whole item, roughly ten orders of
# magnitude past what the arithmetic produces.
#
# The band is asymmetric on purpose. abs(back - n) <= 1 would also pass an
# implementation that systematically returns a sample size one item short
# of what the requested power needs, which is the one direction that
# matters. The sweeps assert the direction as well as the size, and
# separately assert that the returned n really does reach the requested
# power.
_INVERSE_SLACK = 1

# Slack on that power check. Composing the two functions runs the requested
# power through a normal quantile and back through a normal cdf, which
# lands on the target to within an ulp or two, so demanding an exact float
# comparison fails on arithmetic rather than on anything real. One item of
# sample size moves power by around 1e-3 at these sizes, six orders of
# magnitude above this tolerance, so a size that genuinely falls short by an
# item still fails the check.
_POWER_SLACK = 1e-9


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1])
@pytest.mark.parametrize("power", [0.7, 0.8, 0.9, 0.95])
def test_paired_functions_are_exact_inverses(alpha, power):
    for rate in (0.05, 0.1, 0.2, 0.3, 0.5, 0.8):
        for n in (120, 350, 900, 2500, 7000):
            forward = detectable_effect(
                n, power=power, alpha=alpha, discordance_rate=rate
            )
            if not forward.attainable:
                continue
            back = min_sample_size(
                forward.difference, power=power, alpha=alpha,
                discordance_rate=rate,
            )
            assert n <= back.n <= n + _INVERSE_SLACK, (
                f"rate={rate} n={n} difference={forward.difference!r} "
                f"came back as {back.n}"
            )
            assert (
                _power_paired(back.n, rate, forward.difference, alpha)
                >= power - _POWER_SLACK
            )


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1])
@pytest.mark.parametrize("power", [0.7, 0.8, 0.9, 0.95])
def test_independent_functions_are_exact_inverses(alpha, power):
    """Compared per group, which sidesteps the parity of the total.

    min_sample_size splits evenly, so the totals it can return are all even
    and an odd total could never come back unchanged. The per group figure
    is the quantity the arithmetic actually solves for.
    """
    for baseline in (0.2, 0.3, 0.5, 0.7, 0.9):
        for per_group in (60, 200, 800, 3000):
            forward = detectable_effect(
                2 * per_group, baseline=baseline, power=power, alpha=alpha,
                paired=False,
            )
            if not forward.attainable:
                continue
            back = min_sample_size(
                forward.difference, baseline=baseline, power=power,
                alpha=alpha, paired=False,
            )
            assert per_group <= back.n_per_group <= per_group + _INVERSE_SLACK, (
                f"baseline={baseline} per_group={per_group} "
                f"difference={forward.difference!r} came back as "
                f"{back.n_per_group}"
            )
            assert (
                _power_independent(
                    back.n_per_group, baseline, forward.difference, alpha
                )
                >= power - _POWER_SLACK
            )


def test_inverse_sweep_mostly_lands_exactly_on_n():
    """The slack is floating point, not a systematic offset.

    If the composition were biased the tolerance above would be hiding it,
    so this pins that the exact hit is the common case rather than the rare
    one.
    """
    exact = 0
    total = 0
    for rate in (0.05, 0.1, 0.2, 0.3, 0.5, 0.8):
        for n in (120, 350, 900, 2500, 7000):
            forward = detectable_effect(n, discordance_rate=rate)
            if not forward.attainable:
                continue
            total += 1
            exact += min_sample_size(
                forward.difference, discordance_rate=rate
            ).n == n
    assert total > 20
    assert exact / total > 0.5


@pytest.mark.parametrize("power", [0.7, 0.8, 0.9, 0.95])
def test_roundtrip_holds_at_other_power_levels(power):
    n = min_sample_size(0.05, power=power, discordance_rate=0.2).n
    back = detectable_effect(n, power=power, discordance_rate=0.2)
    assert back.difference <= 0.05
    assert back.power == power


# --------------------------------------------------------------------------
# Monotonicity
# --------------------------------------------------------------------------

def test_more_items_detect_smaller_differences():
    got = [detectable_effect(n, discordance_rate=0.3).difference
           for n in (100, 200, 500, 1000, 5000)]
    assert got == sorted(got, reverse=True)


def test_smaller_differences_need_more_items():
    got = [min_sample_size(d, discordance_rate=0.3).n
           for d in (0.1, 0.05, 0.02, 0.01)]
    assert got == sorted(got)


def test_sample_size_scales_with_the_inverse_square_of_the_difference():
    """Halving the difference roughly quadruples the sample."""
    small = min_sample_size(0.01, discordance_rate=0.3).n
    large = min_sample_size(0.02, discordance_rate=0.3).n
    assert small / large == pytest.approx(4.0, rel=0.02)


def test_more_power_costs_more_items():
    got = [min_sample_size(0.05, power=p, discordance_rate=0.3).n
           for p in (0.7, 0.8, 0.9, 0.95)]
    assert got == sorted(got)
    assert len(set(got)) == 4


def test_a_stricter_alpha_costs_more_items():
    got = [min_sample_size(0.05, alpha=a, discordance_rate=0.3).n
           for a in (0.1, 0.05, 0.01)]
    assert got == sorted(got)


def test_more_power_needs_a_larger_difference():
    got = [detectable_effect(500, power=p, discordance_rate=0.3).difference
           for p in (0.7, 0.8, 0.9, 0.95)]
    assert got == sorted(got)


# --------------------------------------------------------------------------
# Differences that were never reachable
# --------------------------------------------------------------------------

def test_paired_difference_above_the_discordance_rate_is_unattainable():
    """A difference can never exceed the discordance rate.

    220 pairs at a 2.4% discordance rate would need a difference of about
    2.9 points to reach 80% power, and 2.4 points is the arithmetic ceiling.
    So no difference of any size reaches 80% power at this size.
    """
    r = detectable_effect(220, discordance_rate=0.024)
    assert r.difference > 0.024
    assert r.attainable is False
    text = r.summary().lower()
    assert "no difference" in text or "nothing" in text


def test_attainable_when_the_difference_fits_under_the_rate():
    r = detectable_effect(5000, discordance_rate=0.024)
    assert r.difference < 0.024
    assert r.attainable is True


def test_independent_unattainable_against_a_ceiling():
    """A 95% baseline leaves 5 points of headroom, and 40 items cannot use it."""
    r = detectable_effect(40, baseline=0.95, paired=False)
    assert r.attainable is False
    text = r.summary().lower()
    assert "no difference" in text or "nothing" in text


def test_unattainable_results_still_summarise():
    for r in (
        detectable_effect(220, discordance_rate=0.024),
        detectable_effect(40, baseline=0.95, paired=False),
    ):
        assert isinstance(r.summary(), str)
        assert len(r.summary()) > 40


# The two sentences the audit prints beside a margin the eval did find. The
# old ones said a smaller difference "was out of reach before the first item
# was graded", and that an eval short of the requested power "could not have
# found anything". Both turn "below the requested power" into "impossible".
# A difference below the reach can still reach significance, less often.
STILL_REACHABLE = (
    "A smaller difference could still reach significance here, with a chance "
    "below 80%."
)
NULL_SAYS_LITTLE = "A null result from this eval says little about the systems."


def test_a_difference_below_the_reach_is_not_called_unreachable():
    r = detectable_effect(400, discordance_rate=0.16)
    assert r.attainable
    text = r.summary()
    assert "out of reach" not in text
    assert STILL_REACHABLE in text


def test_an_eval_short_of_the_power_is_not_said_to_find_nothing():
    for r in (
        detectable_effect(220, discordance_rate=0.024),
        detectable_effect(40, baseline=0.95, paired=False),
    ):
        assert r.attainable is False
        text = r.summary()
        assert "could not have found anything" not in text
        assert "no difference at all was detectable" not in text
        assert NULL_SAYS_LITTLE in text


def test_the_docstring_does_not_call_a_smaller_difference_unfindable():
    doc = " ".join(detectable_effect.__doc__.split())
    assert "could not have found anything smaller" not in doc
    assert "No smaller difference reaches the requested power" in doc


def test_min_sample_size_refuses_a_difference_above_the_discordance_rate():
    with pytest.raises(ValueError, match="discordance"):
        min_sample_size(0.4, discordance_rate=0.3)


def test_a_difference_equal_to_the_discordance_rate_is_allowed():
    """Every discordant pair falling the same way is a real table.

    The b cell holds the whole rate and the c cell is empty, which is the
    most lopsided outcome the design can produce rather than an impossible
    one. The variance under the alternative is rate minus rate squared,
    which is positive, so the arithmetic goes straight through it.
    """
    r = min_sample_size(0.3, discordance_rate=0.3)
    assert r.n > 0
    assert r.difference == 0.3
    assert _power_paired(r.n, 0.3, 0.3, 0.05) >= 0.8 - _POWER_SLACK


def test_the_boundary_convention_is_the_same_in_all_three_places():
    """detectable_effect, min_sample_size and the reference price agree.

    A difference exactly equal to the discordance rate is reachable, so
    min_sample_size prices it, the reference sentence quotes it, and
    detectable_effect calls a difference that size attainable. Splitting the
    convention across the three would let the summary quote a size that
    min_sample_size refuses, which is the same defect the reference price
    already had.
    """
    rate = _REFERENCE_DIFFERENCE

    priced = min_sample_size(rate, discordance_rate=rate)
    assert priced.n > 0

    quoted = detectable_effect(2000, discordance_rate=rate)
    assert quoted.n_for_reference == priced.n
    assert f"{priced.n:,}" in quoted.summary()

    # And a difference above the rate stays refused in all three.
    assert min_sample_size(rate, discordance_rate=rate * 2).n > 0
    with pytest.raises(ValueError, match="discordance"):
        min_sample_size(rate, discordance_rate=rate / 2)
    assert detectable_effect(2000, discordance_rate=rate / 2).n_for_reference is None


def test_min_sample_size_refuses_a_difference_past_a_full_pass_rate():
    with pytest.raises(ValueError, match="baseline"):
        min_sample_size(0.2, baseline=0.9, paired=False)


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_alpha_must_be_a_proportion(bad):
    with pytest.raises(ValueError, match="alpha"):
        detectable_effect(500, alpha=bad)
    with pytest.raises(ValueError, match="alpha"):
        min_sample_size(0.05, alpha=bad)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_power_must_be_a_proportion(bad):
    with pytest.raises(ValueError, match="power"):
        detectable_effect(500, power=bad)
    with pytest.raises(ValueError, match="power"):
        min_sample_size(0.05, power=bad)


def test_power_below_alpha_is_refused():
    """A two sided test rejects at the alpha rate when nothing is there.

    Asking for power under alpha asks for less than chance, which no sample
    size answers.
    """
    with pytest.raises(ValueError, match="power"):
        min_sample_size(0.05, power=0.04, alpha=0.05)
    with pytest.raises(ValueError, match="power"):
        detectable_effect(500, power=0.05, alpha=0.05)


@pytest.mark.parametrize("bad", [0, -1, 1])
def test_n_must_leave_something_to_estimate(bad):
    with pytest.raises(ValueError, match="n"):
        detectable_effect(bad)


@pytest.mark.parametrize("bad", [0.0, -0.05, 1.0, 1.2])
def test_mde_must_be_a_positive_difference(bad):
    with pytest.raises(ValueError, match="mde"):
        min_sample_size(bad, paired=False)


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_baseline_must_be_a_pass_rate(bad):
    with pytest.raises(ValueError, match="baseline"):
        min_sample_size(0.05, baseline=bad, paired=False)
    with pytest.raises(ValueError, match="baseline"):
        detectable_effect(500, baseline=bad, paired=False)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_discordance_rate_must_be_a_proportion(bad):
    with pytest.raises(ValueError, match="discordance"):
        detectable_effect(500, discordance_rate=bad)
    with pytest.raises(ValueError, match="discordance"):
        min_sample_size(0.01, discordance_rate=bad)


def test_a_rate_of_one_is_allowed():
    """Every pair discordant is a real design, and it is the sign test."""
    assert detectable_effect(500, discordance_rate=1.0).difference > 0


# --------------------------------------------------------------------------
# The result object and what it says
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "result",
    [
        detectable_effect(220),
        detectable_effect(220, paired=False),
        min_sample_size(0.05),
        min_sample_size(0.05, paired=False),
    ],
    ids=["eff-paired", "eff-indep", "n-paired", "n-indep"],
)
def test_result_is_a_frozen_dataclass_with_a_summary(result):
    assert isinstance(result, PowerResult)
    with pytest.raises(Exception):
        result.difference = 0.5
    assert isinstance(result.summary(), str)
    assert str(result) == result.summary()


@pytest.mark.parametrize(
    "result",
    [
        detectable_effect(220),
        detectable_effect(220, discordance_rate=0.024),
        detectable_effect(220, paired=False),
        min_sample_size(0.05),
        min_sample_size(0.05, paired=False),
    ],
)
def test_summaries_follow_the_house_style(result):
    text = result.summary()
    assert "—" not in text
    assert text.endswith(".")
    assert text[0].isupper()


def test_solved_for_says_which_direction_was_run():
    assert detectable_effect(220).solved_for == "difference"
    assert min_sample_size(0.05).solved_for == "n"


def test_detectable_effect_summary_names_what_the_eval_ran():
    """The shape the audit needs. The answer against the size that ran."""
    r = detectable_effect(220, discordance_rate=0.3)
    text = r.summary()
    assert "220" in text
    assert "10.3" in text  # the detectable difference in points


def test_detectable_effect_summary_prices_a_five_point_difference():
    """To detect 5 points you need N, and this eval ran n."""
    r = detectable_effect(220, discordance_rate=0.3)
    assert r.reference_difference == _REFERENCE_DIFFERENCE == 0.05
    assert r.n_for_reference == min_sample_size(0.05, discordance_rate=0.3).n
    text = r.summary()
    assert f"{r.n_for_reference:,}" in text
    assert "220" in text


def test_reference_price_is_dropped_when_it_cannot_be_computed():
    """A 5 point difference needs at least 5% discordance to be possible."""
    r = detectable_effect(500, discordance_rate=0.024)
    assert r.n_for_reference is None
    assert "5 point" not in r.summary()


def test_reference_price_is_quoted_at_the_boundary():
    """At exactly 5% discordance a 5 point difference is still reachable.

    The line the summary draws has to be the line min_sample_size draws, and
    that line is above the rate rather than at it. See
    test_the_boundary_convention_is_the_same_in_all_three_places.
    """
    r = detectable_effect(2000, discordance_rate=_REFERENCE_DIFFERENCE)
    assert r.n_for_reference is not None
    assert "5 point" in r.summary()


def test_min_sample_size_summary_states_the_requirement():
    r = min_sample_size(0.05, discordance_rate=0.3)
    text = r.summary()
    assert f"{r.n:,}" in text
    assert "80% power" in text
    assert "5 point" in text


def test_paired_summary_says_the_baseline_does_not_enter():
    text = min_sample_size(0.05, baseline=0.8, discordance_rate=0.3).summary().lower()
    assert "baseline" in text


def test_paired_summary_warns_the_number_is_a_floor():
    """Nominal power runs ahead of the shipped test, so say so.

    The word is asserted rather than any of a set of near synonyms.
    test_paired_nominal_power_runs_ahead_of_the_shipped_test measures a real
    shortfall of about 1.6 points of power, and a reader who is not told the
    number is a floor will read it as a promise.
    """
    text = detectable_effect(500, discordance_rate=0.3).summary().lower()
    assert "floor" in text
    assert "approximation" in text


def test_independent_summary_points_at_pairing():
    """Running both systems on the same items is nearly always available."""
    text = min_sample_size(0.05, paired=False).summary().lower()
    assert "pair" in text


def test_summary_reports_the_power_and_alpha_it_used():
    text = min_sample_size(0.05, power=0.9, alpha=0.01).summary()
    assert "90% power" in text
    assert "1%" in text


def test_result_carries_back_every_input():
    r = detectable_effect(333, baseline=0.72, power=0.85, alpha=0.02,
                          discordance_rate=0.4)
    assert (r.n, r.baseline, r.power, r.alpha) == (333, 0.72, 0.85, 0.02)
    assert r.paired is True
    assert r.discordance_rate == 0.4
