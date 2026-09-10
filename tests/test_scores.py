"""Tests for evalaudit.scores.

Written before the implementation. Make these pass.

The fixtures in test_wilson_known_values are exact Wilson score intervals
computed independently; if your implementation returns these, it is correct.
"""

import numpy as np
import pytest
from scipy import stats

from evalaudit.scores import score_ci


# --------------------------------------------------------------------------
# Wilson interval, binary data
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "k, n, expected_low, expected_high",
    [
        (42, 50, 0.714858, 0.916626),
        (9, 10, 0.595850, 0.982124),
        (0, 20, 0.000000, 0.161125),
        (20, 20, 0.838875, 1.000000),
        (58, 100, 0.482065, 0.672016),
    ],
)
def test_wilson_known_values(k, n, expected_low, expected_high):
    scores = [1] * k + [0] * (n - k)
    r = score_ci(scores, method="wilson")
    assert r.ci_low == pytest.approx(expected_low, abs=1e-5)
    assert r.ci_high == pytest.approx(expected_high, abs=1e-5)
    assert r.estimate == pytest.approx(k / n)
    assert r.n == n


def test_wilson_never_leaves_unit_interval():
    """The normal approximation goes below 0 here. Wilson must not."""
    r = score_ci([0] * 30 + [1], method="wilson")
    assert r.ci_low >= 0.0
    assert r.ci_high <= 1.0


def test_auto_picks_wilson_for_binary():
    r = score_ci([1, 0, 1, 1, 0, 1, 1, 1, 0, 1])
    assert r.method == "wilson"
    assert r.binary is True


def test_auto_picks_bootstrap_for_continuous():
    r = score_ci([0.2, 0.5, 0.9, 0.4, 0.7, 0.3], seed=0)
    assert r.method == "bootstrap"
    assert r.binary is False


# --------------------------------------------------------------------------
# Bootstrap, continuous data
# --------------------------------------------------------------------------

def test_bootstrap_is_deterministic_with_seed():
    xs = np.random.default_rng(1).normal(3.0, 1.0, 60)
    a = score_ci(xs, method="bootstrap", seed=42, n_boot=2000)
    b = score_ci(xs, method="bootstrap", seed=42, n_boot=2000)
    assert a.ci_low == b.ci_low
    assert a.ci_high == b.ci_high


def test_bootstrap_brackets_the_mean():
    xs = np.random.default_rng(2).normal(4.2, 0.8, 200)
    r = score_ci(xs, method="bootstrap", seed=7)
    assert r.ci_low < r.estimate < r.ci_high
    assert r.estimate == pytest.approx(float(np.mean(xs)))


def test_interval_narrows_with_more_data():
    rng = np.random.default_rng(3)
    small = score_ci(rng.normal(0, 1, 40), seed=1)
    large = score_ci(rng.normal(0, 1, 4000), seed=1)
    assert large.width < small.width


# --------------------------------------------------------------------------
# Coverage. This is the credibility argument for the whole package.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("true_p, n", [(0.5, 50), (0.9, 50), (0.3, 200)])
def test_wilson_coverage(true_p, n):
    """Across 1000 simulated datasets the 95% interval should cover the
    true rate about 95% of the time. Wilson under-covers slightly at the
    extremes, so the band is deliberately generous."""
    rng = np.random.default_rng(11)
    covered = 0
    trials = 1000
    for _ in range(trials):
        sample = rng.binomial(1, true_p, n)
        r = score_ci(sample, method="wilson")
        if r.ci_low <= true_p <= r.ci_high:
            covered += 1
    rate = covered / trials
    assert 0.92 <= rate <= 0.98, f"coverage was {rate:.3f}"


def test_bootstrap_coverage():
    rng = np.random.default_rng(12)
    true_mean = 3.0
    covered = 0
    trials = 400
    for _ in range(trials):
        sample = rng.normal(true_mean, 1.0, 80)
        r = score_ci(sample, method="bootstrap", n_boot=1000, seed=int(rng.integers(1e6)))
        if r.ci_low <= true_mean <= r.ci_high:
            covered += 1
    rate = covered / trials
    assert 0.91 <= rate <= 0.99, f"coverage was {rate:.3f}"


# --------------------------------------------------------------------------
# Student-t interval, continuous data
#
# The reference is scipy.stats.t.interval. It is handed the mean and scipy's
# own standard error, both computed here from the data, so nothing in the
# check is taken from the result under test.
# --------------------------------------------------------------------------

def scipy_t_interval(scores, confidence):
    x = np.asarray(scores, dtype=float)
    return stats.t.interval(
        confidence, len(x) - 1, loc=x.mean(), scale=stats.sem(x)
    )


@pytest.mark.parametrize("n", [3, 8, 40, 400])
def test_t_interval_matches_scipy(n):
    xs = np.random.default_rng(n).normal(0.6, 0.15, n)
    r = score_ci(xs, method="t")
    lo, hi = scipy_t_interval(xs, 0.95)
    assert r.method == "t"
    assert r.binary is False
    assert r.n == n
    assert r.estimate == pytest.approx(xs.mean(), abs=1e-12)
    assert r.ci_low == pytest.approx(lo, abs=1e-12)
    assert r.ci_high == pytest.approx(hi, abs=1e-12)


@pytest.mark.parametrize("confidence", [0.50, 0.80, 0.90, 0.99])
def test_t_interval_follows_the_confidence_level(confidence):
    """None of these levels is 0.95, so a critical value fixed at the
    default fails every case."""
    xs = np.random.default_rng(12).normal(0.6, 0.15, 25)
    r = score_ci(xs, method="t", confidence=confidence)
    lo, hi = scipy_t_interval(xs, confidence)
    assert r.confidence == confidence
    assert r.ci_low == pytest.approx(lo, abs=1e-12)
    assert r.ci_high == pytest.approx(hi, abs=1e-12)


def test_t_interval_at_two_items():
    """Two items is the smallest sample with a variance.

    One degree of freedom puts the critical value near 12.7, where the normal
    one is 1.96. Counting n degrees of freedom instead of n - 1, or reading z
    for t, moves the bounds furthest here.
    """
    xs = [0.40, 0.70]
    r = score_ci(xs, method="t")
    lo, hi = scipy_t_interval(xs, 0.95)
    assert r.n == 2
    assert r.estimate == pytest.approx(0.55, abs=1e-12)
    assert r.ci_low == pytest.approx(lo, abs=1e-12)
    assert r.ci_high == pytest.approx(hi, abs=1e-12)


# --------------------------------------------------------------------------
# The summary is the product
# --------------------------------------------------------------------------

def test_summary_mentions_estimate_and_bounds():
    r = score_ci([1] * 42 + [0] * 8, method="wilson")
    s = r.summary()
    assert "84.0%" in s
    assert "71.5%" in s or "71.4%" in s
    assert "n=50" in s


def test_summary_warns_on_small_n():
    r = score_ci([1, 1, 0, 1, 1])
    assert "5 observations" in r.summary()


# The continuous branch of the summary, asserted as whole sentences. The tail
# "The interval spans ..." is shared with the binary branch, so a fragment of
# it passes on either. "Mean score" occurs once in the package, at the
# continuous head, and only a whole sentence pins the head together with the
# numbers printed in it.
#
# The numbers are what scipy.stats.t.interval gives for this data. None of
# them sits within a ten-thousandth of a rounding boundary at three places,
# so a harmless change to how the mean is summed cannot flip a printed digit.
# Thirty items and twenty-nine sit either side of the small-sample warning.

def continuous_scores(n):
    return [0.35 + 0.02 * ((i * 3) % 13) for i in range(n)]


def test_continuous_summary_is_the_whole_sentence():
    r = score_ci(continuous_scores(30), method="t")
    assert r.summary() == (
        "Mean score 0.466 (95% CI: 0.438-0.494, n=30). The interval spans "
        "0.057; treat differences smaller than that as unresolved."
    )


def test_continuous_summary_below_thirty_items_carries_the_warning():
    r = score_ci(continuous_scores(29), method="t")
    assert r.summary() == (
        "Mean score 0.464 (95% CI: 0.435-0.493, n=29). The interval spans "
        "0.058; treat differences smaller than that as unresolved. With only "
        "29 observations this estimate is weak regardless of the point value."
    )


# --------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------

def test_rejects_empty():
    with pytest.raises(ValueError):
        score_ci([])


def test_rejects_unknown_method():
    with pytest.raises(ValueError):
        score_ci([1, 0, 1], method="nonsense")


def test_accepts_numpy_list_and_series():
    import pandas as pd
    base = [1, 0, 1, 1, 0]
    a = score_ci(base, method="wilson")
    b = score_ci(np.array(base), method="wilson")
    c = score_ci(pd.Series(base), method="wilson")
    assert a.ci_low == pytest.approx(b.ci_low) == pytest.approx(c.ci_low)


# --------------------------------------------------------------------------
# Non-default confidence levels
# --------------------------------------------------------------------------

def test_confidence_level_affects_width():
    """A hardcoded z=1.96 would pass every 0.95 test. This catches that."""
    data = [1] * 42 + [0] * 8

    r90 = score_ci(data, method="wilson", confidence=0.90)
    r95 = score_ci(data, method="wilson", confidence=0.95)
    r99 = score_ci(data, method="wilson", confidence=0.99)

    assert r90.width < r95.width < r99.width

    assert "90%" in r90.summary()
    assert "95%" in r95.summary()
    assert "99%" in r99.summary()

    xs = np.random.default_rng(0).normal(5.0, 1.0, 80)
    b90 = score_ci(xs, method="bootstrap", confidence=0.90, seed=1)
    b95 = score_ci(xs, method="bootstrap", confidence=0.95, seed=1)
    b99 = score_ci(xs, method="bootstrap", confidence=0.99, seed=1)

    assert b90.width < b95.width < b99.width
