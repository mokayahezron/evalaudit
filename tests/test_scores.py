"""Tests for evalaudit.scores.

Written before the implementation. Make these pass.

The fixtures in test_wilson_known_values are exact Wilson score intervals
computed independently; if your implementation returns these, it is correct.
"""

import math
import re
import warnings

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


# --------------------------------------------------------------------------
# What the Wilson interval refuses
#
# Wilson is an interval on a pass rate. Asked for on scores that are not 0/1
# it used to count passes by truncating the sum of the scores, and print that
# interval beside a mean it had nothing to do with.
# --------------------------------------------------------------------------

W_REFUSES_NOT_BINARY = (
    "the wilson method is an interval on a pass rate, and these scores are "
    "not all 0 or 1"
)


def wilson_by_hand(k, n, confidence=0.95):
    """The Wilson score interval for k passes in n, written out here."""
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return centre - half, centre + half


def test_wilson_refuses_scores_whose_mean_fell_outside_its_own_interval():
    """Ten scores of 0.99, so the mean is 0.990.

    Before the refusal the pass count came from truncating the sum, 9.9, to
    9. The interval for 9 passes in 10 runs from 0.596 to 0.982, and the
    summary printed the mean of 0.990 beside it, outside its own interval.
    That is the one input shape where the defect produced a sentence that
    contradicts itself, so it is pinned by name.
    """
    low, high = wilson_by_hand(9, 10)
    assert not low <= 0.99 <= high, (
        "the fixture no longer shows the mean outside the truncated interval"
    )
    with pytest.raises(ValueError, match=re.escape(W_REFUSES_NOT_BINARY)) as caught:
        score_ci([0.99] * 10, method="wilson")
    assert "method='bootstrap'" in str(caught.value)
    assert "0.99" in str(caught.value)


@pytest.mark.parametrize(
    "scores",
    [[0.5, 0.7], [0.2, 0.4, 0.6, 0.8], [1, 0, 1, 2], [0, 1, 0.5]],
    ids=["two-halves", "a-spread", "a-two-in-a-pass-rate", "one-half-among-passes"],
)
def test_wilson_refuses_scores_that_are_not_zero_or_one(scores):
    """The same refusal where the mean happens to land inside the interval.
    It landed inside by luck, and luck is why the defect went unnoticed."""
    with pytest.raises(ValueError, match=re.escape(W_REFUSES_NOT_BINARY)):
        score_ci(scores, method="wilson")


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
# What the t interval refuses
#
# Two inputs used to come back as numbers nobody should print. On a pass rate
# of 49 out of 50 the t interval ran to 102.0%. On a single score it returned
# NaN bounds behind two RuntimeWarnings. Both are refused now, before any
# arithmetic runs, and the helper turns a stray RuntimeWarning into a failure
# so a refusal that arrives after the NaN does not pass.
#
# The three refusals are different findings, so each message has a clause
# that occurs in one branch only, copied out by hand rather than imported.
# --------------------------------------------------------------------------

T_REFUSES_PASS_RATE = "A t interval on a pass rate can run past 0% or 100%"
T_REFUSES_NO_SPREAD = "so the t interval would have a width of zero"
T_REFUSES_ONE_SCORE = (
    "the t method needs at least 2 items to estimate a variance, got 1"
)


def assert_t_refuses(scores, clause):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match=re.escape(clause)) as caught:
            score_ci(scores, method="t")
    return str(caught.value)


def test_t_refuses_a_pass_rate():
    """49 of 50. Mean plus the t margin is 1.020 here, a 102.0% pass rate."""
    message = assert_t_refuses([1] * 49 + [0], T_REFUSES_PASS_RATE)
    assert "method='wilson'" in message


@pytest.mark.parametrize(
    "scores", [[1] * 20, [0] * 20], ids=["all-pass", "all-fail"]
)
def test_t_refuses_a_pass_rate_with_no_spread(scores):
    """All passes or all fails is still 0/1 data. ``auto`` already reads it
    that way and picks Wilson, so the t method does too.

    The t interval has width zero here, which claims twenty items fixed the
    rate exactly. Wilson on the same data is a real interval, more than ten
    points wide.
    """
    message = assert_t_refuses(scores, T_REFUSES_NO_SPREAD)
    assert "method='wilson'" in message
    assert T_REFUSES_PASS_RATE not in message
    assert score_ci(scores, method="wilson").width > 0.10


@pytest.mark.parametrize("value", [0.7, 3.5, -2.0])
def test_t_refuses_a_single_score(value):
    """One score has no spread to measure. Before the refusal this came back
    as NaN bounds, printed as nan% in the summary."""
    message = assert_t_refuses([value], T_REFUSES_ONE_SCORE)
    assert T_REFUSES_PASS_RATE not in message
    assert T_REFUSES_NO_SPREAD not in message


def test_a_single_pass_is_refused_as_a_pass_rate():
    """[1.0] is one score and it is 0/1 data. It gets the 0/1 refusal, which
    names Wilson, because Wilson works at one item. The one-score refusal
    names nothing that does."""
    message = assert_t_refuses([1.0], T_REFUSES_NO_SPREAD)
    assert T_REFUSES_ONE_SCORE not in message
    r = score_ci([1.0], method="wilson")
    assert 0.0 < r.ci_low < r.ci_high == 1.0


T_REFUSES_NO_VARIATION = "the t method needs scores that vary"


@pytest.mark.parametrize(
    "scores",
    [[3.0] * 10, [0.5] * 2, [-1.25] * 30],
    ids=["ten-threes", "two-halves", "thirty-negatives"],
)
def test_t_refuses_continuous_scores_with_no_spread(scores):
    """Every score is the same and none is 0 or 1. The t interval came back
    with width zero, which claims the observations fixed the mean exactly."""
    message = assert_t_refuses(scores, T_REFUSES_NO_VARIATION)
    assert T_REFUSES_PASS_RATE not in message
    assert T_REFUSES_NO_SPREAD not in message
    assert T_REFUSES_ONE_SCORE not in message


def test_one_continuous_score_keeps_the_one_score_refusal():
    """[3.0] has no spread and is also a single score. The single-score check
    runs first, so its message is the one that fires. Putting the no-spread
    check ahead of it would tell a caller with one score to look for a
    constant scorer."""
    message = assert_t_refuses([3.0], T_REFUSES_ONE_SCORE)
    assert T_REFUSES_NO_VARIATION not in message


def test_t_still_runs_on_continuous_scores_that_include_zero_and_one():
    """The refusal is for 0/1 data. A continuous scale that happens to touch
    both ends is not that, and the t interval matches scipy on it."""
    xs = [0.0, 1.0, 0.5, 0.25, 0.75, 0.6]
    r = score_ci(xs, method="t")
    lo, hi = scipy_t_interval(xs, 0.95)
    assert r.binary is False
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
        "0.057. To compare this score with another, read the interval on the "
        "difference, which compare_paired and compare_independent report. Two "
        "score intervals that overlap do not show the systems are level."
    )


def test_continuous_summary_below_thirty_items_carries_the_warning():
    r = score_ci(continuous_scores(29), method="t")
    assert r.summary() == (
        "Mean score 0.464 (95% CI: 0.435-0.493, n=29). The interval spans "
        "0.058. To compare this score with another, read the interval on the "
        "difference, which compare_paired and compare_independent report. Two "
        "score intervals that overlap do not show the systems are level. With "
        "only 29 observations this estimate is weak regardless of the point "
        "value."
    )


def test_the_summary_does_not_read_a_difference_off_one_interval():
    """A paired comparison resolves differences far smaller than either
    score's own interval, so the width of one interval says nothing about
    which differences are resolved. The old sentence told the reader to
    treat every difference under the width as unresolved."""
    for r in (score_ci([1] * 58 + [0] * 42), score_ci(continuous_scores(30))):
        text = r.summary()
        assert "treat differences smaller" not in text
        assert "unresolved" not in text
        assert "read the interval on the difference" in text


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
