"""Tests for evalaudit.scores.

Written before the implementation. Make these pass.

The fixtures in test_wilson_known_values are exact Wilson score intervals
computed independently; if your implementation returns these, it is correct.
"""

import numpy as np
import pytest

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
