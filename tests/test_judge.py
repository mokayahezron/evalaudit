"""Tests for evalaudit.judge.

Written before the implementation. Make these pass.

Three external references hold the numbers down. Judge-human agreement goes
against the same two references agreement.py uses, the ``krippendorff``
package on PyPI and the slow pair-counting implementation in
test_agreement.py, because a judge is a rater and the statistic is the same
one. The logistic regressions go against statsmodels. The exact binomial
test goes against a sum of binomial coefficients written out in pure Python,
which is the one reference scipy cannot have influenced.

The last section mutates the slice path and the two bias paths and names,
for each mutation, the test that catches it.

The human-baseline section at the end adds a fourth reference, the MT-Bench
figures published in analysis/mt-bench.md.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import krippendorff as kref
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint

from evalaudit import JudgeValidation, LengthBias, PositionBias
from evalaudit.judge import judge_validation, length_bias, position_bias

# The slow reference and the wide-to-long helper, borrowed rather than
# copied. If the control drifts, both test files find out together.
from tests.test_agreement import slow_alpha, to_long


LEVELS = ["nominal", "ordinal", "interval"]

# The phrase summary() opens the slice sentence with. Tests assert on this
# rather than on a bare slice name, because "hard" turns up inside plenty of
# sentences a summary legitimately contains.
NAMES_A_SLICE = "Worst slice "

# What the summary says instead when the worst slice sits inside the overall
# interval. Spelled out here rather than imported, so a test cannot pass by
# agreeing with whatever the implementation happens to say.
REFUSES_A_SLICE = "cannot show that the judge does worse on any one slice"

# The three sentences that carry REFUSES_A_SLICE, told apart. The phrase above
# is in all of them, so on its own it says a slice was refused and not why.
# The three refusals are different findings and a test that cannot separate
# them cannot tell a working rule from one that reports the wrong reason.
# Each constant runs through the shared tail and occurs once in the package,
# in one branch of JudgeValidation._slice_sentence, so pointing an assertion
# at one keeps the old check and adds the branch.
REFUSES_FOR_NO_INTERVAL = (
    "Without an interval on the overall figure there is nothing to place the "
    "slices against, so the data cannot show that the judge does worse on "
    "any one slice."
)
REFUSES_FOR_NO_FIGURE = (
    "No slice carries both an agreement figure and an interval, so the data "
    "cannot show that the judge does worse on any one slice."
)
REFUSES_FOR_OVERLAP = (
    "on the overall figure. The two overlap, so the data cannot show that the "
    "judge does worse on any one slice. That does not mean it does equally "
    "well on all of them."
)

# The clauses that tell the two position_bias verdicts apart, copied out by
# hand rather than imported.
#
# Asserting `"first" in summary` does not do this. Both verdicts print "went
# to whichever output was shown first" one sentence earlier, so that test
# holds whichever way has_position_effect goes, and a judge that took the
# first-shown output on every flip could be described as splitting evenly
# without anything failing. These are the words that differ. If the report
# is reworded, these fail and the new wording gets read.
CALLS_IT_POSITION_BIAS = (
    "The flips have a direction, so this is position bias rather than an "
    "unsteady judge. It reaches for whatever it sees first."
)
# The flips lean no clearer than chance. This used to say the flips "split
# evenly" and called the judge unsteady rather than position-biased, and it
# said so at 7 flips of 9. Nine flips cannot rule a lean out.
CANNOT_SHOW_A_DIRECTION = (
    "That share is not clear of 50% at this many flips, so the data cannot "
    "show that the flips have a direction. That does not mean the judge is "
    "free of position bias."
)
# The randomised branch's own pair. Same reasoning.
RANDOMISED_CLEARS_A_HALF = (
    "The interval clears 50%, so the judge favours whichever output it sees "
    "first."
)
RANDOMISED_INCLUDES_A_HALF = (
    "The interval includes 50%, so the data cannot show that position moved "
    "the judge. That does not mean the judge ignores position."
)

# The per-slice notes, same reasoning as the dropout notes in
# test_agreement.py. A reader is supposed to learn which case happened.
NOTE_SLICE_ONE_ITEM = "one item, too few to measure agreement"
NOTE_SLICE_NO_VARIANCE = (
    "human and judge used one label throughout, so alpha has no denominator"
)

# The notes a logistic fit returns instead of a coefficient.
NOTE_ONE_OUTCOME = "the outcome never varied, so there is nothing to model"
NOTE_NO_LENGTH_VARIATION = (
    "every pair had the same length difference, so the predictor is constant"
)
NOTE_SEPARATED = (
    "length difference splits the outcome perfectly, so the coefficient is unbounded"
)

# Slices thinner than this get a hedge in the summary even when they clear
# the interval. Same number ScoreCI uses to call an estimate weak.
THIN_SLICE = 30


# --------------------------------------------------------------------------
# Helpers, judge validation
# --------------------------------------------------------------------------

def judge_and_human(n, disagree_rate, rng, labels=(0, 1, 2)):
    """Human labels, and judge labels that depart from them at a set rate.

    A departure redraws the label rather than forcing a different one, so
    the effective disagreement sits a little below ``disagree_rate``. That
    is the honest version. A judge that is wrong picks something, not
    necessarily something else.
    """
    labels = np.asarray(labels)
    human = rng.choice(labels, size=n)
    judge = human.copy()
    flip = rng.random(n) < disagree_rate
    judge[flip] = rng.choice(labels, size=int(flip.sum()))
    return human, judge


def sliced_data(spec, rng, labels=(0, 1, 2)):
    """Build human, judge and slice arrays from (name, n, disagree_rate)."""
    humans, judges, slices = [], [], []
    for name, n, rate in spec:
        h, j = judge_and_human(n, rate, rng, labels)
        humans.append(h)
        judges.append(j)
        slices.extend([name] * n)
    return (
        np.concatenate(humans),
        np.concatenate(judges),
        np.asarray(slices, dtype=object),
    )


def as_long(human, judge):
    """The same two label series as a long ratings frame.

    Judge validation is a two-rater agreement problem wearing different
    words, and this is the bridge that lets the agreement references check
    it without knowing that.
    """
    stacked = np.vstack(
        [np.asarray(human, dtype=float), np.asarray(judge, dtype=float)]
    )
    return to_long(stacked, rater_names=["human", "judge"])


def reference_alpha(human, judge, level):
    """Alpha from the krippendorff package, on the same two series."""
    stacked = np.vstack(
        [np.asarray(human, dtype=float), np.asarray(judge, dtype=float)]
    )
    return kref.alpha(reliability_data=stacked, level_of_measurement=level)


# --------------------------------------------------------------------------
# Helpers, position bias
# --------------------------------------------------------------------------

COMPARISON_COLUMNS = ["pair_id", "option_a", "option_b", "winner"]


def randomised_frame(n_pairs, a_win_rate, rng):
    """One judgement per pair, presentation order already randomised."""
    rows = []
    for i in range(n_pairs):
        x, y = f"p{i}x", f"p{i}y"
        winner = x if rng.random() < a_win_rate else y
        rows.append((f"p{i}", x, y, winner))
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def both_orders_frame(n_pairs, consistency, rng, first_share=1.0):
    """Every pair judged twice, once in each order.

    ``consistency`` is the share of pairs where the judge names the same
    output both times. ``first_share`` says, of the pairs where it flips,
    how often it went to whichever output was shown first. At 1.0 every flip
    is a position-A flip, which is position bias in its pure form. At 0.5
    the flips are noise.
    """
    rows = []
    for i in range(n_pairs):
        x, y = f"p{i}x", f"p{i}y"
        if rng.random() < consistency:
            pick = x if rng.random() < 0.5 else y
            w1 = w2 = pick
        elif rng.random() < first_share:
            w1, w2 = x, y      # picked position A in both presentations
        else:
            w1, w2 = y, x      # picked position B in both presentations
        rows.append((f"p{i}", x, y, w1))
        rows.append((f"p{i}", y, x, w2))
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def exact_two_sided_p(k, n):
    """The exact binomial p-value against 0.5, from binomial coefficients.

    At p=0.5 the distribution is symmetric, so the two-sided p-value is
    twice the smaller tail, capped at one. Written with math.comb so the
    check does not run back through scipy, which is what the implementation
    uses.
    """
    if n == 0:
        return float("nan")
    small = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(small + 1)) / 2 ** n
    return min(1.0, 2 * tail)


# --------------------------------------------------------------------------
# Helpers, length bias
# --------------------------------------------------------------------------

def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))


def quality_pairs(n, rng, length_per_quality=120.0, judge_length_weight=0.0,
                  judge_noise=0.6, length_noise=60.0):
    """Pairs where quality may drive length, and the judge may chase length.

    Human preference follows quality. The judge follows quality too, plus
    ``judge_length_weight`` times the length difference.

    Two knobs, and the pair of them is what the two models get tested
    against. ``length_per_quality`` is how much of the length difference is
    real quality, so at a high value longer answers genuinely are better.
    ``judge_length_weight`` is the judge's own pull toward length, which is
    the bias. Set the first high and the second to zero and any correlation
    between judge preference and length is honest. Set the first to zero and
    length carries no quality at all, which is the corpus where the second
    model is exact.
    """
    q_a = rng.normal(0, 1, n)
    q_b = rng.normal(0, 1, n)
    len_a = 400 + length_per_quality * q_a + rng.normal(0, length_noise, n)
    len_b = 400 + length_per_quality * q_b + rng.normal(0, length_noise, n)

    human = (q_a > q_b).astype(int)

    utility = (q_a - q_b) / judge_noise + judge_length_weight * (len_a - len_b)
    judge = (rng.random(n) < logistic(utility)).astype(int)

    return human, judge, np.column_stack([len_a, len_b])


def human_oriented_difference(human, lengths):
    """Length the humans passed over, minus the length they picked.

    The predictor in the second model. Positive means the humans went for
    the shorter answer on that pair, so a positive coefficient means the
    judge breaks with them exactly where they did.
    """
    human = np.asarray(human)
    picked = np.where(human == 1, lengths[:, 0], lengths[:, 1])
    passed = np.where(human == 1, lengths[:, 1], lengths[:, 0])
    return passed - picked


def statsmodels_logit(x, y):
    """Slope, its 95% interval and its p-value, from statsmodels."""
    design = sm.add_constant(np.asarray(x, dtype=float))
    fit = sm.Logit(np.asarray(y, dtype=float), design).fit(disp=0)
    lo, hi = fit.conf_int(alpha=0.05)[1]
    return float(fit.params[1]), float(lo), float(hi), float(fit.pvalues[1])


# --------------------------------------------------------------------------
# Guard assertions
#
# Named, because the mutation section at the bottom reuses them. A mutation
# is only caught if one of these fails on it, and each mutation test says
# which one does.
# --------------------------------------------------------------------------

def assert_slices_ascending(table):
    """Agreement rises down the table, and undefined slices sit at the end.

    Ascending is the whole point of the ordering, because a reader looks at
    the top row. Undefined last, because a slice where human and judge used
    one label throughout is not a failure and must not be shown as the worst.
    """
    values = table["agreement"].to_numpy(dtype=float)
    measured = values[~np.isnan(values)]
    assert list(measured) == sorted(measured), "slices are not in ascending order"
    seen_undefined = False
    for value in values:
        if np.isnan(value):
            seen_undefined = True
        else:
            assert not seen_undefined, (
                "an undefined slice sorted above a measured one"
            )


def assert_slice_naming_rule(result):
    """A slice is named only when its own interval clears the overall one.

    The noise guard rater_dropout uses, in the form this statistic takes.
    Both halves matter: naming when the rule says so, and staying quiet when
    it does not.

    The slice sitting below the overall lower bound is necessary and not
    sufficient. A slice holds a fraction of the items, so its sampling error
    is wider, and its point estimate drops under a lower bound computed on
    the whole dataset by chance alone. Its interval has to clear as well.
    """
    text = result.summary()
    if result.worst_slice is None:
        assert NAMES_A_SLICE not in text
        # Asserted rather than guarded. Under two slices the rule has nothing
        # to choose between and neither branch of it means anything, so
        # arriving here with such a result is a broken fixture rather than a
        # case to pass over quietly.
        assert len(result.by_slice) > 1, (
            "the naming rule was asserted on a result with fewer than two "
            "slices, where the rule does not apply"
        )
        assert REFUSES_A_SLICE in text
        # Which refusal, not just that one happened. Every branch below
        # asserts, so no shape of data reaches the end of this having
        # checked nothing.
        row = result.by_slice.iloc[0]
        if not result.has_interval:
            assert REFUSES_FOR_NO_INTERVAL in text
        elif not (
            np.isfinite(row["agreement"]) and np.isfinite(row["ci_high"])
        ):
            assert REFUSES_FOR_NO_FIGURE in text
        else:
            assert REFUSES_FOR_OVERLAP in text
            # Read off the table rather than off slice_is_distinguishable,
            # which is the property under test and cannot vouch for itself.
            assert not row["ci_high"] < result.ci_low, (
                "refused a slice whose interval clears the overall one"
            )
        return

    assert NAMES_A_SLICE in text
    row = result.by_slice.iloc[0]
    assert result.worst_slice == row["slice"]
    assert result.has_interval, "named a slice with no interval to judge it against"
    assert row["agreement"] < result.ci_low, (
        "named a slice whose agreement sits inside the overall interval"
    )
    assert row["ci_high"] < result.ci_low, (
        "named a slice whose interval overlaps the overall interval"
    )
    assert REFUSES_A_SLICE not in text


def assert_slice_counts_are_per_slice(table, slices):
    """Each row carries its own item count, not the count for the whole run."""
    counted = pd.Series(list(slices)).value_counts()
    assert not table.empty
    for _, row in table.iterrows():
        assert int(row["n_items"]) == int(counted[row["slice"]]), (
            f"slice {row['slice']!r} reports {row['n_items']} items, "
            f"not the {counted[row['slice']]} it holds"
        )


def assert_slice_agreement_is_chance_corrected(table, human, judge, slices, level):
    """Each slice's number is alpha on that slice, not its raw match rate."""
    slices = np.asarray(list(slices), dtype=object)
    human = np.asarray(human)
    judge = np.asarray(judge)
    for _, row in table.iterrows():
        mask = slices == row["slice"]
        expected = slow_alpha(as_long(human[mask], judge[mask]), level)
        got = row["agreement"]
        if np.isnan(expected):
            assert np.isnan(got)
        else:
            assert got == pytest.approx(expected, abs=1e-9)


def assert_consistency_counts_outputs(result, frame):
    """Consistency counts pairs where the same output won under both orders.

    Not the pairs where the same position won, which is the opposite
    quantity on a pair the judge flipped.
    """
    same_output = 0
    total = 0
    for _, rows in frame.groupby("pair_id", sort=False):
        if len(rows) != 2:
            continue
        total += 1
        same_output += int(rows["winner"].iloc[0] == rows["winner"].iloc[1])
    assert result.consistency_rate == pytest.approx(same_output / total, abs=1e-12)


# --------------------------------------------------------------------------
# judge_validation, the headline number
# --------------------------------------------------------------------------

@pytest.mark.parametrize("level", LEVELS)
def test_agreement_matches_the_reference_package(level):
    """A judge is a rater. The number has to be the one the references give."""
    rng = np.random.default_rng(0)
    human, judge = judge_and_human(200, 0.25, rng, labels=(1, 2, 3, 4, 5))

    r = judge_validation(human, judge, level=level, n_boot=200, seed=1)

    assert r.agreement == pytest.approx(
        reference_alpha(human, judge, level), abs=1e-9
    )


@pytest.mark.parametrize("level", LEVELS)
def test_agreement_matches_the_slow_reference(level):
    """And the pair-counting implementation, which shares none of the
    vectorised shortcuts."""
    rng = np.random.default_rng(2)
    human, judge = judge_and_human(120, 0.4, rng, labels=(1, 2, 3, 4, 5))

    r = judge_validation(human, judge, level=level, n_boot=200, seed=1)

    assert r.agreement == pytest.approx(
        slow_alpha(as_long(human, judge), level), abs=1e-9
    )


def test_accuracy_is_the_plain_match_rate():
    """Accuracy is there because people ask for it, and it is not alpha.

    On a lopsided label set a judge that always guesses the common label
    scores high accuracy and no agreement at all. Both numbers are reported
    so the gap between them is visible.
    """
    human = [1] * 90 + [0] * 10
    judge = [1] * 100

    r = judge_validation(human, judge, n_boot=200, seed=1)

    assert r.accuracy == pytest.approx(0.90)
    assert r.agreement <= 0.0


def test_agreement_and_accuracy_are_both_reported():
    rng = np.random.default_rng(3)
    human, judge = judge_and_human(150, 0.3, rng)

    r = judge_validation(human, judge, n_boot=200, seed=1)

    assert r.accuracy == pytest.approx(float(np.mean(human == judge)))
    assert r.agreement < r.accuracy
    text = r.summary()
    assert f"{r.accuracy * 100:.1f}%" in text


def test_perfect_agreement_is_one():
    rng = np.random.default_rng(4)
    human = rng.choice([1, 2, 3], size=60)

    r = judge_validation(human, human.copy(), n_boot=200, seed=1)

    assert r.agreement == pytest.approx(1.0)
    assert r.accuracy == pytest.approx(1.0)


def test_levels_are_not_interchangeable():
    """A 1-5 rubric read as unordered labels throws away the fact that 4 and
    5 are nearly the same judgement, and the number moves."""
    rng = np.random.default_rng(5)
    human, judge = judge_and_human(200, 0.5, rng, labels=(1, 2, 3, 4, 5))

    values = {
        level: judge_validation(human, judge, level=level, n_boot=100, seed=1).agreement
        for level in LEVELS
    }

    assert values["nominal"] != pytest.approx(values["ordinal"], abs=1e-6)
    assert values["ordinal"] != pytest.approx(values["interval"], abs=1e-6)
    assert values["ordinal"] > values["nominal"]


def test_result_records_the_level_it_used():
    rng = np.random.default_rng(6)
    human, judge = judge_and_human(40, 0.2, rng)
    r = judge_validation(human, judge, level="ordinal", n_boot=50, seed=1)
    assert r.level == "ordinal"
    assert "ordinal" in r.summary()


def test_n_items_is_the_number_compared():
    rng = np.random.default_rng(7)
    human, judge = judge_and_human(85, 0.2, rng)
    r = judge_validation(human, judge, n_boot=50, seed=1)
    assert r.n_items == 85
    assert "85" in r.summary()


# --------------------------------------------------------------------------
# The interval on the headline number
# --------------------------------------------------------------------------

def test_bootstrap_is_deterministic_under_a_seed():
    rng = np.random.default_rng(8)
    human, judge = judge_and_human(100, 0.3, rng)

    one = judge_validation(human, judge, n_boot=300, seed=11)
    two = judge_validation(human, judge, n_boot=300, seed=11)

    assert one.ci_low == two.ci_low
    assert one.ci_high == two.ci_high


def test_bootstrap_brackets_the_estimate():
    rng = np.random.default_rng(9)
    human, judge = judge_and_human(200, 0.3, rng)

    r = judge_validation(human, judge, n_boot=500, seed=12)

    assert r.ci_low <= r.agreement <= r.ci_high
    assert r.has_interval


def test_interval_narrows_with_more_items():
    rng = np.random.default_rng(10)
    small_h, small_j = judge_and_human(40, 0.3, rng)
    big_h, big_j = judge_and_human(800, 0.3, rng)

    small = judge_validation(small_h, small_j, n_boot=400, seed=13)
    big = judge_validation(big_h, big_j, n_boot=400, seed=13)

    assert big.width < small.width


def test_bootstrap_respects_the_confidence_level():
    rng = np.random.default_rng(11)
    human, judge = judge_and_human(200, 0.3, rng)

    narrow = judge_validation(human, judge, confidence=0.80, n_boot=500, seed=14)
    wide = judge_validation(human, judge, confidence=0.99, n_boot=500, seed=14)

    assert narrow.width < wide.width
    assert narrow.confidence == 0.80
    assert "80%" in narrow.summary()


def test_bootstrap_can_be_switched_off_with_zero_resamples():
    rng = np.random.default_rng(12)
    human, judge = judge_and_human(60, 0.3, rng)

    r = judge_validation(human, judge, n_boot=0)

    assert not r.has_interval
    assert np.isnan(r.ci_low) and np.isnan(r.ci_high)
    assert r.worst_slice is None


def test_bootstrap_coverage():
    """The 95% interval should cover the population figure about 95% of the
    time when items are drawn from a fixed pool.

    Same construction as the coverage test in test_agreement.py, and the same
    slack on the bounds. Alpha's sampling distribution is skewed and the
    percentile bootstrap does not fully correct for that.
    """
    rng = np.random.default_rng(15)
    pool = 6000
    human = rng.choice([1, 2, 3, 4, 5], size=pool)
    noisy = rng.random(pool) < 0.3
    judge = np.where(noisy, rng.choice([1, 2, 3, 4, 5], size=pool), human)

    population = reference_alpha(human, judge, "nominal")

    trials, n, covered = 200, 150, 0
    for _ in range(trials):
        take = rng.choice(pool, size=n, replace=False)
        r = judge_validation(
            human[take], judge[take], n_boot=400,
            seed=int(rng.integers(1_000_000)),
        )
        if r.ci_low <= population <= r.ci_high:
            covered += 1

    rate = covered / trials
    assert 0.88 <= rate <= 0.99, f"coverage was {rate:.3f}"


# --------------------------------------------------------------------------
# The by-slice table, which is the point of this function
# --------------------------------------------------------------------------

def test_by_slice_has_the_columns_the_docstring_promises():
    rng = np.random.default_rng(16)
    human, judge, slices = sliced_data(
        [("easy", 60, 0.05), ("hard", 60, 0.5)], rng
    )

    r = judge_validation(human, judge, slices=slices, n_boot=200, seed=1)

    assert list(r.by_slice.columns) == [
        "slice", "n_items", "agreement", "ci_low", "ci_high", "accuracy", "note"
    ]


def test_by_slice_is_sorted_ascending_by_agreement():
    """The worst slice is the finding, so it goes first."""
    rng = np.random.default_rng(17)
    human, judge, slices = sliced_data(
        [("easy", 120, 0.02), ("medium", 120, 0.25), ("hard", 120, 0.7)], rng
    )

    r = judge_validation(human, judge, slices=slices, n_boot=200, seed=1)

    assert_slices_ascending(r.by_slice)
    assert r.by_slice.iloc[0]["slice"] == "hard"


def test_by_slice_agreement_is_alpha_on_the_slice():
    rng = np.random.default_rng(18)
    human, judge, slices = sliced_data(
        [("easy", 90, 0.05), ("hard", 90, 0.6)], rng, labels=(1, 2, 3, 4, 5)
    )

    r = judge_validation(human, judge, slices=slices, n_boot=200, seed=1)

    assert_slice_agreement_is_chance_corrected(
        r.by_slice, human, judge, slices, "nominal"
    )


@pytest.mark.parametrize("level", LEVELS)
def test_by_slice_uses_the_level_it_was_given(level):
    rng = np.random.default_rng(19)
    human, judge, slices = sliced_data(
        [("easy", 80, 0.1), ("hard", 80, 0.6)], rng, labels=(1, 2, 3, 4, 5)
    )

    r = judge_validation(human, judge, slices=slices, level=level,
                         n_boot=100, seed=1)

    assert_slice_agreement_is_chance_corrected(
        r.by_slice, human, judge, slices, level
    )


def test_by_slice_reports_the_item_count_per_slice():
    """A gap on 12 items and a gap on 1200 read differently, and the reader
    cannot tell them apart without the count."""
    rng = np.random.default_rng(20)
    human, judge, slices = sliced_data(
        [("easy", 200, 0.05), ("hard", 12, 0.6), ("medium", 45, 0.3)], rng
    )

    r = judge_validation(human, judge, slices=slices, n_boot=200, seed=1)

    assert_slice_counts_are_per_slice(r.by_slice, slices)


def test_by_slice_accuracy_is_the_match_rate_within_the_slice():
    rng = np.random.default_rng(21)
    human, judge, slices = sliced_data(
        [("easy", 100, 0.05), ("hard", 100, 0.6)], rng
    )

    r = judge_validation(human, judge, slices=slices, n_boot=200, seed=1)

    for _, row in r.by_slice.iterrows():
        mask = slices == row["slice"]
        assert row["accuracy"] == pytest.approx(
            float(np.mean(human[mask] == judge[mask]))
        )


def test_by_slice_covers_every_slice_once():
    rng = np.random.default_rng(22)
    human, judge, slices = sliced_data(
        [("a", 40, 0.1), ("b", 40, 0.3), ("c", 40, 0.5)], rng
    )

    r = judge_validation(human, judge, slices=slices, n_boot=100, seed=1)

    assert sorted(r.by_slice["slice"]) == ["a", "b", "c"]


def test_no_slices_means_an_empty_table_and_no_slice_sentence():
    rng = np.random.default_rng(23)
    human, judge = judge_and_human(80, 0.3, rng)

    r = judge_validation(human, judge, n_boot=200, seed=1)

    assert r.by_slice.empty
    assert list(r.by_slice.columns) == [
        "slice", "n_items", "agreement", "ci_low", "ci_high", "accuracy", "note"
    ]
    assert r.worst_slice is None
    assert NAMES_A_SLICE not in r.summary()
    assert REFUSES_A_SLICE not in r.summary()


def test_a_single_slice_is_not_a_breakdown():
    """One slice is the whole dataset under another name. There is nothing to
    compare it against, so nobody gets named."""
    rng = np.random.default_rng(24)
    human, judge = judge_and_human(100, 0.6, rng)

    r = judge_validation(
        human, judge, slices=["all"] * 100, n_boot=300, seed=1
    )

    assert len(r.by_slice) == 1
    assert r.worst_slice is None
    assert NAMES_A_SLICE not in r.summary()


def test_a_one_item_slice_is_undefined_and_says_why():
    rng = np.random.default_rng(25)
    human, judge, slices = sliced_data(
        [("big", 100, 0.2), ("tiny", 1, 0.0)], rng
    )

    r = judge_validation(human, judge, slices=slices, n_boot=200, seed=1)

    row = r.by_slice[r.by_slice["slice"] == "tiny"].iloc[0]
    assert np.isnan(row["agreement"])
    assert row["note"] == NOTE_SLICE_ONE_ITEM
    assert int(row["n_items"]) == 1


def test_a_slice_where_nobody_varied_is_undefined_not_perfect():
    """Human and judge both said the same single label all the way through.

    Alpha has no denominator there. That is not perfect agreement and it is
    certainly not the worst slice, so it sorts last and carries a note.
    """
    rng = np.random.default_rng(26)
    human, judge, slices = sliced_data([("mixed", 100, 0.3)], rng)
    human = np.concatenate([human, np.ones(30, dtype=int)])
    judge = np.concatenate([judge, np.ones(30, dtype=int)])
    slices = np.concatenate([slices, np.array(["flat"] * 30, dtype=object)])

    r = judge_validation(human, judge, slices=slices, n_boot=200, seed=1)

    row = r.by_slice[r.by_slice["slice"] == "flat"].iloc[0]
    assert np.isnan(row["agreement"])
    assert row["accuracy"] == pytest.approx(1.0)
    assert row["note"] == NOTE_SLICE_NO_VARIANCE
    assert r.by_slice.iloc[-1]["slice"] == "flat"
    assert_slices_ascending(r.by_slice)


def test_the_two_undefined_slice_notes_are_distinct():
    assert NOTE_SLICE_ONE_ITEM != NOTE_SLICE_NO_VARIANCE


def test_note_is_empty_when_the_slice_carries_a_number():
    rng = np.random.default_rng(27)
    human, judge, slices = sliced_data(
        [("easy", 60, 0.1), ("hard", 60, 0.6)], rng
    )

    r = judge_validation(human, judge, slices=slices, n_boot=200, seed=1)

    assert (r.by_slice["note"] == "").all()


# --------------------------------------------------------------------------
# The finding, and refusing to report it when it is noise
# --------------------------------------------------------------------------

def test_a_good_headline_can_hide_a_failed_slice():
    """The reason this function exists.

    Most items are easy and the judge tracks humans on them. The close calls
    are a minority and the judge is close to useless there. The overall
    number stays respectable and the slice table does not.
    """
    rng = np.random.default_rng(28)
    human, judge, slices = sliced_data(
        [("clear", 700, 0.02), ("borderline", 120, 0.85)], rng
    )

    r = judge_validation(human, judge, slices=slices, n_boot=500, seed=29)

    assert r.agreement > 0.6
    worst = r.by_slice.iloc[0]
    assert worst["slice"] == "borderline"
    assert worst["agreement"] < 0.3
    assert r.worst_slice == "borderline"
    assert NAMES_A_SLICE in r.summary()
    assert "borderline" in r.summary()
    assert_slice_naming_rule(r)


def test_summary_names_the_worst_slice_with_its_item_count():
    rng = np.random.default_rng(30)
    human, judge, slices = sliced_data(
        [("clear", 600, 0.02), ("borderline", 90, 0.9)], rng
    )

    r = judge_validation(human, judge, slices=slices, n_boot=500, seed=31)

    text = r.summary()
    assert r.worst_slice == "borderline"
    assert "90 items" in text


def test_summary_refuses_a_slice_cut_from_one_population():
    """Slices cut from one homogeneous population differ by sampling noise
    alone. Something is always last, and naming it invites a reader to go
    rewrite a rubric that was never the problem.

    This fixture is also the one that rules out the simpler guard. Three
    hundred items at a single disagreement rate, cut into five slices of
    sixty, and the lowest slice lands below the overall lower bound. Its own
    interval overlaps by a wide margin, which is what stops it being named.
    """
    rng = np.random.default_rng(32)
    human, judge = judge_and_human(300, 0.3, rng)
    slices = np.array(
        [f"s{i % 5}" for i in range(300)], dtype=object
    )

    r = judge_validation(human, judge, slices=slices, n_boot=600, seed=33)

    worst = r.by_slice.iloc[0]
    assert worst["agreement"] < r.ci_low, (
        "the fixture no longer reaches past the overall lower bound, so it "
        "no longer rules out the point-estimate guard"
    )
    assert worst["ci_high"] > r.ci_low
    assert r.worst_slice is None
    assert NAMES_A_SLICE not in r.summary()
    assert REFUSES_FOR_OVERLAP in r.summary()
    assert_slice_naming_rule(r)


def test_the_slice_rule_needs_the_slice_interval_to_clear_as_well():
    """The rule, asserted directly, on data spread across both sides of it.

    Two alternatives are ruled out here. Naming any slice below the overall
    figure names one on every dataset ever passed in, since something is
    always last. Naming any slice below the overall lower bound is the rule
    that looks right, and the case above shows it firing on a population
    with no structure in it at all.
    """
    cases = [
        [("clear", 600, 0.02), ("borderline", 120, 0.9)],
        [("a", 150, 0.3), ("b", 150, 0.35)],
        [("a", 40, 0.2), ("b", 40, 0.45)],
        [("a", 200, 0.25), ("b", 200, 0.3), ("c", 200, 0.32)],
    ]
    named_any = False
    first_rule_checks = 0

    for i, spec in enumerate(cases):
        rng = np.random.default_rng(100 + i)
        human, judge, slices = sliced_data(spec, rng)
        r = judge_validation(human, judge, slices=slices, n_boot=600, seed=40 + i)

        worst = r.by_slice.iloc[0]
        expected = bool(
            r.has_interval
            and len(r.by_slice) > 1
            and np.isfinite(worst["agreement"])
            and np.isfinite(worst["ci_high"])
            and worst["ci_high"] < r.ci_low
        )
        assert r.slice_is_distinguishable is expected
        assert (r.worst_slice is not None) is expected
        assert_slice_naming_rule(r)

        named_any |= expected
        # Asserted rather than guarded. A worst slice with no agreement
        # figure says nothing about either rule, so a NaN here is a fixture
        # that stopped testing what it was built to test.
        assert np.isfinite(worst["agreement"]), (
            f"case {i} carries no agreement on its worst slice, so it cannot "
            f"say which rule is in use"
        )
        # The first rule that must not be in use instead. The count is
        # asserted below, so a run where nothing was refused cannot pass as
        # a check on it.
        if not expected:
            first_rule_checks += 1
            assert worst["agreement"] < r.agreement

    assert named_any and first_rule_checks, (
        "the cases did not exercise both branches"
    )


def test_every_slice_carries_its_own_interval():
    rng = np.random.default_rng(33)
    human, judge, slices = sliced_data(
        [("easy", 120, 0.05), ("hard", 120, 0.6)], rng
    )

    r = judge_validation(human, judge, slices=slices, n_boot=400, seed=34)

    for _, row in r.by_slice.iterrows():
        assert row["ci_low"] <= row["agreement"] <= row["ci_high"]


def test_slice_intervals_are_wider_than_the_overall_one():
    """The reason the slice interval has to be in the rule at all."""
    rng = np.random.default_rng(36)
    human, judge = judge_and_human(400, 0.3, rng)
    slices = np.array([f"s{i % 4}" for i in range(400)], dtype=object)

    r = judge_validation(human, judge, slices=slices, n_boot=500, seed=37)

    widths = r.by_slice["ci_high"] - r.by_slice["ci_low"]
    assert (widths > r.width).all()


def test_slices_carry_no_interval_when_the_bootstrap_is_off():
    rng = np.random.default_rng(38)
    human, judge, slices = sliced_data(
        [("easy", 60, 0.05), ("hard", 60, 0.6)], rng
    )

    r = judge_validation(human, judge, slices=slices, n_boot=0)

    assert r.by_slice["ci_low"].isna().all()
    assert r.by_slice["ci_high"].isna().all()
    assert r.worst_slice is None


def test_summary_will_not_name_a_slice_without_an_interval():
    """No interval means nothing to place the slices against."""
    rng = np.random.default_rng(34)
    human, judge, slices = sliced_data(
        [("clear", 400, 0.02), ("borderline", 80, 0.9)], rng
    )

    r = judge_validation(human, judge, slices=slices, n_boot=0)

    assert not r.has_interval
    assert r.worst_slice is None
    assert not r.slice_is_distinguishable
    assert NAMES_A_SLICE not in r.summary()
    assert REFUSES_FOR_NO_INTERVAL in r.summary()
    assert_slice_naming_rule(r)


def one_label_slices():
    """Two slices, each graded with one label throughout by both sides.

    Neither slice varies, so neither carries an agreement figure. The two
    used different labels, so the overall figure exists, with an interval.
    """
    human = np.array([0] * 40 + [1] * 40)
    slices = np.array(["refusals"] * 40 + ["answers"] * 40, dtype=object)
    return human, human.copy(), slices


def thin_slice_with_no_interval():
    """A thin slice with an agreement figure and no interval on it, beside a
    slice that used one label throughout.

    Six items and one disagreement. About a third of resamples miss that
    item and have no variance, so the interval is refused. The other slice
    has no figure and sorts last, which leaves the thin one on top.
    """
    human = np.array([0, 0, 0, 0, 0, 1] + [1] * 60)
    judge = np.array([0, 0, 0, 0, 0, 0] + [1] * 60)
    slices = np.array(["thin"] * 6 + ["answers"] * 60, dtype=object)
    return human, judge, slices


@pytest.mark.parametrize(
    "build, top_row_has_agreement",
    [(one_label_slices, False), (thin_slice_with_no_interval, True)],
    ids=["no-slice-has-agreement", "top-slice-has-no-interval"],
)
def test_summary_refuses_a_slice_when_the_top_row_has_no_figure(
    build, top_row_has_agreement
):
    """The refusal for a table whose top row cannot be measured.

    One fixture for each half of the condition. A missing agreement figure
    is enough on its own, and so is a missing interval beside a figure that
    exists.
    """
    human, judge, slices = build()
    r = judge_validation(human, judge, slices=slices, n_boot=500, seed=1)

    top = r.by_slice.iloc[0]
    assert r.has_interval, "the fixture lost the interval on the overall figure"
    assert bool(np.isfinite(top["agreement"])) is top_row_has_agreement
    assert np.isnan(top["ci_high"])
    assert r.worst_slice is None
    assert REFUSES_FOR_NO_FIGURE in r.summary()
    assert_slice_naming_rule(r)


def test_summary_hedges_a_thin_slice_it_does_name():
    """A slice can clear the interval on 15 items and still be thin enough
    that the reader should hear the count before acting.

    The fixture names the slice with room to spare. The overall lower bound
    sits near 0.94 and the thin slice near 0.0, checked across ten seeds
    before this assertion was written, so the naming branch is reached and
    the hedge is what is actually under test.
    """
    rng = np.random.default_rng(35)
    human, judge, slices = sliced_data(
        [("clear", 600, 0.01), ("borderline", 15, 1.0)], rng
    )

    r = judge_validation(human, judge, slices=slices, n_boot=500, seed=36)

    assert r.worst_slice == "borderline"
    row = r.by_slice.iloc[0]
    assert int(row["n_items"]) == 15
    assert row["n_items"] < THIN_SLICE
    assert NAMES_A_SLICE in r.summary()
    assert "15 items" in r.summary()
    assert "provisional" in r.summary().lower()


def test_a_slice_wide_enough_to_act_on_gets_no_hedge():
    """The other side of the hedge, so the word is not simply always there."""
    rng = np.random.default_rng(38)
    human, judge, slices = sliced_data(
        [("clear", 600, 0.01), ("borderline", 200, 1.0)], rng
    )

    r = judge_validation(human, judge, slices=slices, n_boot=500, seed=39)

    assert r.worst_slice == "borderline"
    assert int(r.by_slice.iloc[0]["n_items"]) >= THIN_SLICE
    assert "provisional" not in r.summary().lower()


def test_summary_is_a_string_in_every_branch():
    rng = np.random.default_rng(37)
    cases = [
        judge_validation(*judge_and_human(50, 0.3, rng), n_boot=100, seed=1),
        judge_validation(*judge_and_human(50, 0.0, rng), n_boot=100, seed=1),
        judge_validation([1, 1, 1, 1], [1, 1, 1, 1], n_boot=100, seed=1),
        judge_validation([0, 1], [1, 0], n_boot=100, seed=1),
    ]
    human, judge, slices = sliced_data([("a", 30, 0.2), ("b", 30, 0.8)], rng)
    cases.append(
        judge_validation(human, judge, slices=slices, n_boot=200, seed=1)
    )
    cases.append(judge_validation(human, judge, slices=slices, n_boot=0))

    for r in cases:
        assert isinstance(r.summary(), str)
        assert r.summary().strip()


def test_agreement_is_undefined_when_nobody_varied():
    """Both sides gave every item the same label. Not perfect agreement, a
    scale nobody used, and the summary has to say which."""
    r = judge_validation([1] * 40, [1] * 40, n_boot=200, seed=1)

    assert np.isnan(r.agreement)
    assert r.accuracy == pytest.approx(1.0)
    text = r.summary().lower()
    assert "undefined" in text
    assert "perfect agreement" not in text.replace("not perfect agreement", "")


def test_the_nobody_varied_summary_is_in_plain_sentences():
    """The sentence used to read "That is not perfect agreement, it is a
    rubric with one label in it." That construction is banned in this
    codebase. Both halves of the meaning stay: the cause is a rubric with one
    label in it, and the result is not perfect agreement."""
    text = judge_validation([1] * 40, [1] * 40, n_boot=200, seed=1).summary()
    assert (
        "This comes from a rubric with one label in it. It is not perfect "
        "agreement." in text
    )
    assert "not perfect agreement, it is" not in text


def test_one_item_is_not_called_a_rubric_nobody_varied():
    """One item, the labels differ, accuracy 0.0%.

    The summary said every label that could be compared was identical, which
    the accuracy printed after it contradicts, and it said "1 items". Alpha
    is undefined here because one item cannot carry an estimate.
    """
    r = judge_validation(["a"], ["b"], n_boot=0)
    assert np.isnan(r.agreement)
    assert r.accuracy == 0.0
    assert r.summary() == (
        "Judge and human agreement is undefined (nominal, 1 item). One item "
        "cannot carry a reliability estimate, so there is no number to report "
        "and no interval around it. Label more items and run this again. "
        "Plain accuracy is 0.0%."
    )


def test_one_item_is_too_few_whatever_its_labels():
    """The item count is checked before the variance, the order the slice
    notes use. One item is too small to measure whatever else is true."""
    text = judge_validation(["a"], ["a"], n_boot=0).summary()
    assert "One item cannot carry a reliability estimate" in text
    assert "identical" not in text
    assert "1 items" not in text


def test_one_set_aside_item_is_singular():
    r = judge_validation(["a", "b", None], ["a", "a", "b"], n_boot=0)
    assert r.n_dropped == 1
    assert " 1 item was set aside because one side had no label." in r.summary()
    assert "1 items" not in r.summary()


# --------------------------------------------------------------------------
# judge_validation input handling
# --------------------------------------------------------------------------

def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        judge_validation([1, 2, 3], [1, 2])


def test_rejects_mismatched_slice_length():
    with pytest.raises(ValueError, match="slices"):
        judge_validation([1, 2, 3], [1, 2, 3], slices=["a", "b"])


def test_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        judge_validation([], [])


def test_rejects_unknown_level():
    with pytest.raises(ValueError, match="level"):
        judge_validation([1, 0], [1, 1], level="ordinalish")


def test_drops_items_where_either_label_is_missing():
    """An item the judge could not label is not a disagreement."""
    human = [1, 0, 1, 0, 1, 0]
    judge = [1, 0, None, 0, 1, None]

    r = judge_validation(human, judge, n_boot=100, seed=1)

    assert r.n_items == 4
    assert r.n_dropped == 2
    assert r.accuracy == pytest.approx(1.0)
    assert "2" in r.summary()


def test_accepts_pandas_series_and_string_labels():
    human = pd.Series(["good", "bad", "good", "bad"] * 10)
    judge = pd.Series(["good", "bad", "bad", "bad"] * 10)

    r = judge_validation(human, judge, n_boot=100, seed=1)

    assert 0 < r.accuracy < 1
    assert np.isfinite(r.agreement)


def test_ordinal_level_rejects_labels_that_are_not_numbers():
    with pytest.raises(ValueError, match="numeric"):
        judge_validation(["good", "bad"] * 10, ["good", "good"] * 10,
                         level="ordinal")


def test_does_not_mutate_its_inputs():
    human = np.array([1, 0, 1, 0] * 10)
    judge = np.array([1, 1, 1, 0] * 10)
    slices = np.array(["a", "b"] * 20, dtype=object)
    before = (human.copy(), judge.copy(), slices.copy())

    judge_validation(human, judge, slices=slices, n_boot=100, seed=1)

    assert np.array_equal(human, before[0])
    assert np.array_equal(judge, before[1])
    assert np.array_equal(slices, before[2])


# --------------------------------------------------------------------------
# position_bias, detecting which design it was handed
# --------------------------------------------------------------------------

def test_detects_the_randomised_design():
    rng = np.random.default_rng(40)
    frame = randomised_frame(200, 0.5, rng)

    r = position_bias(frame)

    assert r.design == "randomised"
    assert r.n_pairs == 200
    assert r.n_judgements == 200
    assert r.n_both_orders == 0
    assert "randomised" in r.summary().lower()
    assert "position-a win rate" in r.summary().lower()


def test_detects_the_both_orders_design():
    rng = np.random.default_rng(41)
    frame = both_orders_frame(150, consistency=0.8, rng=rng)

    r = position_bias(frame)

    assert r.design == "both_orders"
    assert r.n_pairs == 150
    assert r.n_judgements == 300
    assert r.n_both_orders == 150
    assert "both order" in r.summary().lower()
    assert "consistency" in r.summary().lower()


def test_the_design_is_read_from_the_data_not_asked_for():
    """One call signature, two analyses. The caller never says which."""
    rng = np.random.default_rng(42)
    one = position_bias(randomised_frame(80, 0.5, rng))
    two = position_bias(both_orders_frame(80, 0.8, rng))

    assert one.design != two.design


def test_a_repeat_in_the_same_order_is_not_a_both_orders_design():
    """The same pair judged twice with the same output in front says nothing
    about position. Only a genuine swap does."""
    rng = np.random.default_rng(43)
    frame = randomised_frame(60, 0.5, rng)
    frame = pd.concat([frame, frame.head(30)], ignore_index=True)

    r = position_bias(frame)

    assert r.design == "randomised"
    assert r.n_both_orders == 0


def test_a_stray_swapped_pair_does_not_flip_the_design():
    rng = np.random.default_rng(44)
    frame = randomised_frame(100, 0.5, rng)
    swap = frame.head(3).rename(
        columns={"option_a": "option_b", "option_b": "option_a"}
    )[COMPARISON_COLUMNS]
    frame = pd.concat([frame, swap], ignore_index=True)

    r = position_bias(frame)

    assert r.design == "randomised"
    assert r.n_both_orders == 3
    assert "3" in r.summary()


def test_the_design_switches_at_half_the_pairs():
    """The boundary, pinned from both sides."""
    rng = np.random.default_rng(45)
    both = both_orders_frame(50, consistency=0.9, rng=rng)
    singles = randomised_frame(50, 0.5, rng)
    singles["pair_id"] = [f"solo{i}" for i in range(len(singles))]

    at_half = position_bias(pd.concat([both, singles], ignore_index=True))
    assert at_half.design == "both_orders"

    more_singles = randomised_frame(51, 0.5, rng)
    more_singles["pair_id"] = [f"solo{i}" for i in range(len(more_singles))]
    below_half = position_bias(
        pd.concat([both, more_singles], ignore_index=True)
    )
    assert below_half.design == "randomised"


# --------------------------------------------------------------------------
# position_bias, the randomised branch
# --------------------------------------------------------------------------

def test_position_a_rate_and_counts():
    rng = np.random.default_rng(46)
    frame = randomised_frame(200, 0.65, rng)
    wins = int((frame["winner"] == frame["option_a"]).sum())

    r = position_bias(frame)

    assert r.n_a_wins == wins
    assert r.n_decisive == 200
    assert r.position_a_rate == pytest.approx(wins / 200)
    assert r.estimate == pytest.approx(r.position_a_rate)


def test_binomial_p_value_matches_a_hand_written_exact_test():
    """Against binomial coefficients summed in pure Python. scipy cannot
    have taught this reference anything."""
    for seed, rate, n in [(47, 0.5, 60), (48, 0.62, 200), (49, 0.8, 45)]:
        rng = np.random.default_rng(seed)
        frame = randomised_frame(n, rate, rng)
        wins = int((frame["winner"] == frame["option_a"]).sum())

        r = position_bias(frame)

        assert r.p_value == pytest.approx(exact_two_sided_p(wins, n), abs=1e-12)


def test_interval_matches_statsmodels_clopper_pearson():
    """The randomised interval is Clopper-Pearson, the exact binomial
    interval, since the exact binomial p-value is printed beside it. It was
    Wilson, matched against statsmodels' Wilson at the same tolerance."""
    rng = np.random.default_rng(50)
    frame = randomised_frame(180, 0.6, rng)
    wins = int((frame["winner"] == frame["option_a"]).sum())
    lo, hi = proportion_confint(wins, 180, alpha=0.05, method="beta")

    r = position_bias(frame)

    assert r.ci_low == pytest.approx(lo, abs=1e-9)
    assert r.ci_high == pytest.approx(hi, abs=1e-9)


def test_an_unbiased_judge_covers_a_half():
    """Built with exactly 200 wins each way rather than drawn at 0.5.

    A 95% interval on data drawn from the null misses the null 5% of the
    time, so a drawn fixture would fail on one seed in twenty and the test
    would be reporting the draw rather than the arithmetic.
    """
    rng = np.random.default_rng(51)
    frame = randomised_frame(400, 0.5, rng)
    winners = list(frame["option_a"][:200]) + list(frame["option_b"][200:])
    frame["winner"] = winners

    r = position_bias(frame)

    assert r.n_a_wins == 200
    assert r.position_a_rate == pytest.approx(0.5)
    assert r.ci_low < 0.5 < r.ci_high
    assert r.p_value == pytest.approx(1.0)
    assert "cannot" in r.summary().lower() or "no position" in r.summary().lower()


def test_a_first_position_judge_is_caught():
    rng = np.random.default_rng(52)
    frame = randomised_frame(400, 0.72, rng)

    r = position_bias(frame)

    assert r.p_value < 0.001
    assert r.ci_low > 0.5
    assert r.has_position_effect
    assert RANDOMISED_CLEARS_A_HALF in r.summary()
    assert RANDOMISED_INCLUDES_A_HALF not in r.summary()


def test_the_randomised_branch_states_what_it_assumes():
    """The test cannot see whether order was randomised. If it was not, the
    same number appears when one system is simply better, and the summary
    has to say so.
    """
    rng = np.random.default_rng(53)
    r = position_bias(randomised_frame(120, 0.6, rng))
    assert "randomis" in r.summary().lower()
    assert "assum" in r.summary().lower()


def test_ties_are_dropped_and_counted():
    rng = np.random.default_rng(54)
    frame = randomised_frame(100, 0.6, rng)
    frame.loc[:9, "winner"] = None

    r = position_bias(frame)

    assert r.n_ties == 10
    assert r.n_decisive == 90
    assert "10" in r.summary()


def test_consistency_is_undefined_in_the_randomised_branch():
    """There is nothing to be consistent with when each pair was judged once."""
    rng = np.random.default_rng(55)
    r = position_bias(randomised_frame(100, 0.5, rng))

    assert np.isnan(r.consistency_rate)
    assert "consistency" not in r.summary().lower()


# --------------------------------------------------------------------------
# position_bias, the both-orders branch
# --------------------------------------------------------------------------

def test_consistency_rate_counts_pairs_where_the_same_output_won():
    rng = np.random.default_rng(56)
    frame = both_orders_frame(200, consistency=0.75, rng=rng)

    r = position_bias(frame)

    assert_consistency_counts_outputs(r, frame)
    assert r.estimate == pytest.approx(r.consistency_rate)


def test_a_perfectly_consistent_judge_scores_one():
    rng = np.random.default_rng(57)
    frame = both_orders_frame(100, consistency=1.0, rng=rng)

    r = position_bias(frame)

    assert r.consistency_rate == pytest.approx(1.0)
    assert r.n_decisive == 0
    assert np.isnan(r.position_a_rate)
    assert np.isnan(r.p_value)


def test_a_judge_that_always_picks_the_first_output_scores_zero():
    """The pure position case. It never picks the same output twice, and
    every flip goes to whatever was shown first."""
    rng = np.random.default_rng(58)
    frame = both_orders_frame(100, consistency=0.0, rng=rng, first_share=1.0)

    r = position_bias(frame)

    assert r.consistency_rate == pytest.approx(0.0)
    assert r.position_a_rate == pytest.approx(1.0)
    assert r.n_decisive == 100
    assert r.p_value < 1e-20

    # The verdict, not just the numbers behind it. Every flip went to the
    # first-shown output, so this is the strongest position effect the
    # design can produce and the report has to name it as one.
    assert r.has_position_effect
    assert CALLS_IT_POSITION_BIAS in r.summary()
    assert CANNOT_SHOW_A_DIRECTION not in r.summary()


def test_flips_that_are_noise_split_evenly_across_positions():
    """An inconsistent judge is not automatically a position-biased one.

    The flips have a direction and it is the direction that says which.

    The band is four standard errors on the flip count rather than a fixed
    0.4 to 0.6. At 2000 pairs about a thousand pairs flip, which puts four
    standard errors at roughly plus or minus 0.063, so the assertion is
    about six failures in a hundred thousand seeds. The earlier version
    asserted a fixed band worth 2.8 standard errors on 200 flips, which is
    around one seed in two hundred.
    """
    rng = np.random.default_rng(59)
    frame = both_orders_frame(2000, consistency=0.5, rng=rng, first_share=0.5)

    r = position_bias(frame)

    assert r.consistency_rate < 0.7
    assert r.n_decisive > 800
    margin = 4 * math.sqrt(0.25 / r.n_decisive)
    assert margin < 0.08, f"the band grew to {margin:.3f}, tighten the fixture"
    assert abs(r.position_a_rate - 0.5) < margin
    assert r.p_value > 1e-5

    # The other half of the claim. An unsteady judge must not be reported as
    # a position-biased one, and this is the pair of assertions that stops
    # the two branches collapsing into whichever one the fixture happens to
    # reach.
    assert not r.has_position_effect
    assert CANNOT_SHOW_A_DIRECTION in r.summary()
    assert CALLS_IT_POSITION_BIAS not in r.summary()


def test_the_both_orders_summary_reports_both_numbers():
    rng = np.random.default_rng(60)
    frame = both_orders_frame(200, consistency=0.6, rng=rng, first_share=0.9)

    r = position_bias(frame)
    text = r.summary().lower()

    assert "consistency" in text
    assert f"{r.consistency_rate * 100:.1f}%" in r.summary()
    assert "flip" in text
    assert str(r.n_decisive) in r.summary()


def test_consistency_interval_matches_statsmodels_wilson():
    rng = np.random.default_rng(61)
    frame = both_orders_frame(150, consistency=0.7, rng=rng)
    same = sum(
        rows["winner"].iloc[0] == rows["winner"].iloc[1]
        for _, rows in frame.groupby("pair_id", sort=False)
    )
    lo, hi = proportion_confint(same, 150, alpha=0.05, method="wilson")

    r = position_bias(frame)

    assert r.ci_low == pytest.approx(lo, abs=1e-9)
    assert r.ci_high == pytest.approx(hi, abs=1e-9)


def test_a_tied_judgement_removes_its_pair_from_the_consistency_rate():
    rng = np.random.default_rng(62)
    frame = both_orders_frame(60, consistency=0.8, rng=rng)
    frame.loc[0, "winner"] = None

    r = position_bias(frame)

    assert r.n_ties == 1
    assert r.n_pairs_scored == 59


# --------------------------------------------------------------------------
# position_bias input handling
# --------------------------------------------------------------------------

def test_position_bias_rejects_missing_columns():
    frame = pd.DataFrame({"pair_id": ["p"], "winner": ["x"]})
    with pytest.raises(ValueError, match="option_a"):
        position_bias(frame)


def test_position_bias_rejects_a_winner_that_was_not_on_offer():
    rng = np.random.default_rng(63)
    frame = randomised_frame(20, 0.5, rng)
    frame.loc[5, "winner"] = "somebody-else"

    with pytest.raises(ValueError, match="winner"):
        position_bias(frame)


def test_position_bias_rejects_a_pair_against_itself():
    frame = pd.DataFrame(
        [("p0", "x", "x", "x")], columns=COMPARISON_COLUMNS
    )
    with pytest.raises(ValueError, match="same output"):
        position_bias(frame)


def test_position_bias_rejects_an_empty_frame():
    with pytest.raises(ValueError, match="empty"):
        position_bias(pd.DataFrame(columns=COMPARISON_COLUMNS))


def test_position_bias_ignores_the_seed():
    """Neither interval here is drawn by resampling, so nothing is random.
    The argument exists to match the rest of the package and must not change
    an answer.
    """
    rng = np.random.default_rng(64)
    frame = randomised_frame(120, 0.6, rng)

    one = position_bias(frame, seed=1)
    two = position_bias(frame, seed=999)

    # field by field rather than == on the dataclass, because the unused
    # half of the result is NaN and NaN is not equal to itself
    for field in one.__dataclass_fields__:
        a, b = getattr(one, field), getattr(two, field)
        if isinstance(a, float) and np.isnan(a):
            assert np.isnan(b), field
        else:
            assert a == b, field


def test_position_bias_does_not_mutate_the_frame():
    rng = np.random.default_rng(65)
    frame = both_orders_frame(30, 0.7, rng)
    before = frame.copy()

    position_bias(frame)

    pd.testing.assert_frame_equal(frame, before)


# --------------------------------------------------------------------------
# length_bias, the preference model
# --------------------------------------------------------------------------

def test_preference_coefficient_matches_statsmodels():
    rng = np.random.default_rng(70)
    human, judge, lengths = quality_pairs(400, rng, judge_length_weight=0.004)

    r = length_bias(judge, lengths)
    coef, lo, hi, p = statsmodels_logit(lengths[:, 0] - lengths[:, 1], judge)

    assert r.coefficient == pytest.approx(coef, rel=1e-6, abs=1e-10)
    assert r.ci_low == pytest.approx(lo, rel=1e-5, abs=1e-9)
    assert r.ci_high == pytest.approx(hi, rel=1e-5, abs=1e-9)
    assert r.p_value == pytest.approx(p, rel=1e-5, abs=1e-12)


def test_the_predictor_is_the_signed_length_difference():
    """Not the absolute difference. A judge that favours longer answers and
    one that favours shorter ones are opposite findings, and an unsigned
    predictor cannot tell them apart.
    """
    rng = np.random.default_rng(71)
    _, judge, lengths = quality_pairs(500, rng, judge_length_weight=0.01)
    longer = length_bias(judge, lengths)

    swapped = length_bias(1 - judge, lengths)

    assert longer.coefficient > 0
    assert swapped.coefficient == pytest.approx(-longer.coefficient, rel=1e-6)


def test_a_judge_that_chases_length_gets_a_positive_coefficient():
    rng = np.random.default_rng(72)
    _, judge, lengths = quality_pairs(600, rng, judge_length_weight=0.02)

    r = length_bias(judge, lengths)

    assert r.coefficient > 0
    assert r.ci_low > 0
    assert r.p_value < 0.001


def test_a_judge_indifferent_to_length_reports_no_effect():
    """The claim is that the effect is small, not that a 95% interval
    happened to cover zero on this draw.

    An interval covering the null covers it 95% of the time, so asserting
    that directly is a one-in-twenty failure across seeds. The assertion is
    on the size of the effect instead: length has to move the odds by less
    than a quarter across a full standard deviation of length difference,
    which is about four standard errors out at this sample size.
    """
    rng = np.random.default_rng(73)
    n = 3000
    lengths = np.column_stack([
        rng.normal(400, 150, n), rng.normal(400, 150, n)
    ])
    judge = rng.integers(0, 2, n)

    r = length_bias(judge, lengths)

    assert 0.8 < r.odds_ratio_per_sd < 1.25
    assert r.p_value > 1e-4
    assert "cannot" in r.summary().lower()


def test_the_coefficient_is_reported_against_a_readable_length_step():
    """A per-character log-odds is unreadable. The summary translates it to
    the odds change over one standard deviation of the observed differences,
    and the result carries that standard deviation so the reader can check.
    """
    rng = np.random.default_rng(74)
    _, judge, lengths = quality_pairs(400, rng, judge_length_weight=0.01)

    r = length_bias(judge, lengths)

    expected = float(np.std(lengths[:, 0] - lengths[:, 1], ddof=1))
    assert r.sd_difference == pytest.approx(expected)
    assert r.odds_ratio_per_sd == pytest.approx(
        math.exp(r.coefficient * expected)
    )
    assert f"{expected:.0f}" in r.summary()


def test_longer_rate_is_reported():
    rng = np.random.default_rng(75)
    _, judge, lengths = quality_pairs(300, rng, judge_length_weight=0.01)

    r = length_bias(judge, lengths)

    picked_longer = np.where(judge == 1, lengths[:, 0] > lengths[:, 1],
                             lengths[:, 1] > lengths[:, 0])
    assert r.longer_rate == pytest.approx(float(np.mean(picked_longer)))


# --------------------------------------------------------------------------
# length_bias, the disagreement model
# --------------------------------------------------------------------------

def test_disagreement_model_matches_statsmodels():
    rng = np.random.default_rng(76)
    human, judge, lengths = quality_pairs(500, rng, judge_length_weight=0.01)

    r = length_bias(judge, lengths, human_preferences=human)

    coef, lo, hi, p = statsmodels_logit(
        human_oriented_difference(human, lengths), (judge != human).astype(int)
    )

    assert r.disagreement_coefficient == pytest.approx(coef, rel=1e-6, abs=1e-10)
    assert r.disagreement_ci_low == pytest.approx(lo, rel=1e-5, abs=1e-9)
    assert r.disagreement_ci_high == pytest.approx(hi, rel=1e-5, abs=1e-9)
    assert r.disagreement_p_value == pytest.approx(p, rel=1e-5, abs=1e-12)


def test_disagreement_model_is_oriented_by_the_human_choice():
    """The predictor is the length the humans passed over minus the length
    they picked, so a positive coefficient means the judge breaks with them
    on the pairs where they went short.

    Orienting by the judge's own choice instead makes the predictor a
    consequence of the outcome. Every disagreement is a pair where the judge
    took the option the humans rejected, which on a corpus where length
    tracks quality is the shorter one, and that swamps the bias. Orienting
    by option A asks whether disagreement rises when A happens to be longer,
    which is a question about column order in the file. Neither can be told
    from this one by shape alone, so the invariance is the test: relabelling
    A and B must not move either coefficient.
    """
    rng = np.random.default_rng(77)
    human, judge, lengths = quality_pairs(500, rng, judge_length_weight=0.015)

    straight = length_bias(judge, lengths, human_preferences=human)
    flipped = length_bias(
        1 - judge, lengths[:, ::-1], human_preferences=1 - human
    )

    assert flipped.disagreement_coefficient == pytest.approx(
        straight.disagreement_coefficient, rel=1e-6
    )
    assert flipped.coefficient == pytest.approx(straight.coefficient, rel=1e-6)


def test_both_models_read_flat_when_length_carries_no_quality():
    """The corpus where the second model is exact.

    Length here is noise, unrelated to quality, so an unbiased judge has
    nothing to correlate with and both models sit on zero.
    """
    rng = np.random.default_rng(78)
    human, judge, lengths = quality_pairs(
        3000, rng, length_per_quality=0.0, length_noise=150.0,
        judge_length_weight=0.0,
    )

    r = length_bias(judge, lengths, human_preferences=human)

    assert 0.8 < r.odds_ratio_per_sd < 1.25
    assert 0.8 < r.disagreement_odds_ratio_per_sd < 1.25


def test_a_length_biased_judge_shows_up_in_both_models():
    rng = np.random.default_rng(79)
    human, judge, lengths = quality_pairs(
        1500, rng, length_per_quality=140.0, judge_length_weight=0.02
    )

    r = length_bias(judge, lengths, human_preferences=human)

    assert r.coefficient > 0
    assert r.ci_low > 0
    assert r.disagreement_coefficient > 0
    assert r.disagreement_ci_low > 0


def test_bias_on_a_corpus_where_length_carries_no_quality():
    """A biased judge, and nothing for the bias to hide behind."""
    rng = np.random.default_rng(80)
    human, judge, lengths = quality_pairs(
        3000, rng, length_per_quality=0.0, length_noise=150.0,
        judge_length_weight=0.01,
    )

    r = length_bias(judge, lengths, human_preferences=human)

    assert r.coefficient > 0 and r.ci_low > 0
    assert r.disagreement_coefficient > 0 and r.disagreement_ci_low > 0


def test_the_disagreement_model_shrinks_the_confound_without_removing_it():
    """The honest size of the second model's claim, measured.

    The judge here has no length preference at all. Quality drives length,
    so the judge's preference tracks length and the first model reports a
    strong effect that is not bias. Holding the human verdict fixed pulls
    the coefficient down by roughly a third and it stays clear of zero,
    because a binary human label is a coarse measure of quality and the
    length difference still carries quality information the label missed.

    So the second model is the sharper of the two and it is not a clean
    separation. The summary has to say that rather than sell it as one.
    """
    rng = np.random.default_rng(81)
    human, judge, lengths = quality_pairs(
        3000, rng, length_per_quality=140.0, judge_length_weight=0.0
    )

    r = length_bias(judge, lengths, human_preferences=human)

    assert r.coefficient > 0
    assert 0 < r.disagreement_coefficient < r.coefficient
    assert r.disagreement_ci_low > 0, (
        "the fixture no longer shows the residual confound this test is about"
    )
    assert "does not remove" in r.summary().lower()


def test_the_summary_says_which_model_is_which():
    rng = np.random.default_rng(82)
    human, judge, lengths = quality_pairs(600, rng, judge_length_weight=0.01)

    text = length_bias(judge, lengths, human_preferences=human).summary()

    assert "preference" in text.lower()
    assert "disagree" in text.lower()
    assert "longer answers may simply be better" in text.lower()


def test_without_human_labels_only_the_first_model_is_reported():
    rng = np.random.default_rng(81)
    _, judge, lengths = quality_pairs(300, rng, judge_length_weight=0.01)

    r = length_bias(judge, lengths)

    assert not r.has_human
    assert np.isnan(r.disagreement_coefficient)
    assert np.isnan(r.disagreement_ci_low)
    assert np.isnan(r.disagreement_p_value)
    assert np.isfinite(r.coefficient)
    text = r.summary().lower()
    assert "without human labels" in text or "no human labels" in text
    assert "cannot separate" in text


def test_the_disagreement_caveat_does_not_claim_to_be_the_sharper_number():
    """The caveat hedges the second fit without ranking it.

    "The sharper of the two" belongs in the audit's lead, where it tells a
    reader which of two fits the verdict was written from before they meet
    the numbers. Repeating it here says the same thing a second time in the
    same paragraph, and the second time carries no information the first
    did not.

    The string is written out rather than imported, so rewording it fails
    here and the new wording gets read.
    """
    rng = np.random.default_rng(82)
    human, judge, lengths = quality_pairs(300, rng, judge_length_weight=0.01)

    text = length_bias(judge, lengths, human_preferences=human).summary()

    assert "Read this as an indication rather than as proof." in text
    assert "sharper of the two" not in text


def test_n_disagreements_is_reported():
    rng = np.random.default_rng(82)
    human, judge, lengths = quality_pairs(300, rng, judge_length_weight=0.01)

    r = length_bias(judge, lengths, human_preferences=human)

    assert r.n_disagreements == int(np.sum(human != judge))
    assert str(r.n_disagreements) in r.summary()


# --------------------------------------------------------------------------
# length_bias, the fits that cannot be made
# --------------------------------------------------------------------------

def test_a_judge_that_always_picks_a_has_nothing_to_model():
    rng = np.random.default_rng(83)
    lengths = np.column_stack([rng.normal(400, 100, 50), rng.normal(400, 100, 50)])

    r = length_bias(np.ones(50, dtype=int), lengths)

    assert np.isnan(r.coefficient)
    assert r.note == NOTE_ONE_OUTCOME
    assert isinstance(r.summary(), str)


def test_identical_lengths_leave_no_predictor():
    rng = np.random.default_rng(84)
    lengths = np.column_stack([np.full(60, 300.0), np.full(60, 300.0)])
    judge = rng.integers(0, 2, 60)

    r = length_bias(judge, lengths)

    assert np.isnan(r.coefficient)
    assert r.note == NOTE_NO_LENGTH_VARIATION


def test_perfect_separation_is_refused_not_reported():
    """Every long answer won and every short one lost. The maximum
    likelihood coefficient is infinite, and any finite number printed here
    would be an artefact of where the optimiser gave up.
    """
    n = 60
    diff = np.concatenate([np.linspace(-200, -10, n // 2),
                           np.linspace(10, 200, n // 2)])
    lengths = np.column_stack([400 + diff / 2, 400 - diff / 2])
    judge = (diff > 0).astype(int)

    r = length_bias(judge, lengths)

    assert np.isnan(r.coefficient)
    assert np.isnan(r.ci_low)
    assert r.note == NOTE_SEPARATED
    assert "unbounded" in r.summary().lower()


def test_a_judge_that_never_disagrees_leaves_the_second_model_empty():
    rng = np.random.default_rng(85)
    _, judge, lengths = quality_pairs(200, rng, judge_length_weight=0.01)

    r = length_bias(judge, lengths, human_preferences=judge.copy())

    assert r.n_disagreements == 0
    assert np.isnan(r.disagreement_coefficient)
    assert r.disagreement_note == NOTE_ONE_OUTCOME


# --------------------------------------------------------------------------
# length_bias input handling
# --------------------------------------------------------------------------

def test_accepts_a_and_b_labels():
    rng = np.random.default_rng(86)
    human, judge, lengths = quality_pairs(200, rng, judge_length_weight=0.01)
    letters = np.where(judge == 1, "A", "B")
    human_letters = np.where(human == 1, "A", "B")

    numeric = length_bias(judge, lengths, human_preferences=human)
    lettered = length_bias(letters, lengths, human_preferences=human_letters)

    assert lettered.coefficient == pytest.approx(numeric.coefficient)
    assert lettered.disagreement_coefficient == pytest.approx(
        numeric.disagreement_coefficient
    )


def test_accepts_a_dataframe_of_lengths():
    rng = np.random.default_rng(87)
    _, judge, lengths = quality_pairs(200, rng, judge_length_weight=0.01)
    frame = pd.DataFrame(lengths, columns=["a", "b"])

    assert length_bias(judge, frame).coefficient == pytest.approx(
        length_bias(judge, lengths).coefficient
    )


def test_length_bias_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        length_bias([1, 0, 1], np.zeros((2, 2)))


def test_length_bias_rejects_a_bad_length_shape():
    with pytest.raises(ValueError, match="two columns"):
        length_bias([1, 0], np.zeros((2, 3)))


def test_length_bias_rejects_three_valued_preferences():
    with pytest.raises(ValueError, match="two"):
        length_bias([1, 0, 2], np.zeros((3, 2)))


def test_length_bias_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        length_bias([], np.zeros((0, 2)))


def test_length_bias_rejects_mismatched_human_labels():
    with pytest.raises(ValueError, match="human_preferences"):
        length_bias([1, 0], np.zeros((2, 2)), human_preferences=[1])


# --------------------------------------------------------------------------
# Mutation tests
#
# Each one applies a plausible wrong implementation and asserts that a named
# guard fails on it. The docstring says which test in this file is the one
# that would catch the mutation in the ordinary course of running the suite.
# The pattern is the one test_the_ordinal_metric_moves_between_resamples
# uses in test_agreement.py: show the discriminating condition really does
# discriminate, rather than trusting that it would.
# --------------------------------------------------------------------------

def test_mutation_sorting_slices_descending():
    """Mutation: sort by_slice by agreement descending.

    Caught by test_by_slice_is_sorted_ascending_by_agreement, through
    assert_slices_ascending.
    """
    rng = np.random.default_rng(90)
    human, judge, slices = sliced_data(
        [("easy", 120, 0.02), ("medium", 120, 0.25), ("hard", 120, 0.7)], rng
    )
    r = judge_validation(human, judge, slices=slices, n_boot=200, seed=1)

    mutant = r.by_slice.sort_values(
        "agreement", ascending=False, na_position="last"
    ).reset_index(drop=True)

    assert_slices_ascending(r.by_slice)
    with pytest.raises(AssertionError, match="ascending order"):
        assert_slices_ascending(mutant)


def test_mutation_sorting_undefined_slices_first():
    """Mutation: let a slice with no variance sort to the top, where
    ascending order with NaN treated as the smallest value would put it.

    Caught by test_a_slice_where_nobody_varied_is_undefined_not_perfect,
    through assert_slices_ascending.
    """
    rng = np.random.default_rng(91)
    human, judge, slices = sliced_data([("mixed", 100, 0.3)], rng)
    human = np.concatenate([human, np.ones(30, dtype=int)])
    judge = np.concatenate([judge, np.ones(30, dtype=int)])
    slices = np.concatenate([slices, np.array(["flat"] * 30, dtype=object)])
    r = judge_validation(human, judge, slices=slices, n_boot=200, seed=1)

    mutant = r.by_slice.sort_values(
        "agreement", ascending=True, na_position="first"
    ).reset_index(drop=True)

    assert_slices_ascending(r.by_slice)
    with pytest.raises(AssertionError, match="sorted above"):
        assert_slices_ascending(mutant)
    assert mutant.iloc[0]["slice"] == "flat"


def test_mutation_naming_a_slice_below_the_point_estimate(monkeypatch):
    """Mutation: name the worst slice whenever it falls below the overall
    agreement, rather than below the lower bound.

    Caught by test_summary_refuses_a_slice_cut_from_one_population and by
    test_the_slice_rule_needs_the_slice_interval_to_clear_as_well, both
    through assert_slice_naming_rule.
    """
    rng = np.random.default_rng(92)
    human, judge = judge_and_human(300, 0.3, rng)
    slices = np.array([f"s{i % 5}" for i in range(300)], dtype=object)

    real = judge_validation(human, judge, slices=slices, n_boot=600, seed=33)
    assert_slice_naming_rule(real)

    def mutant(self):
        if self.by_slice.empty or len(self.by_slice) < 2:
            return False
        worst = self.by_slice.iloc[0]["agreement"]
        return bool(np.isfinite(worst) and worst < self.agreement)

    monkeypatch.setattr(
        JudgeValidation, "slice_is_distinguishable", property(mutant)
    )
    mutated = judge_validation(human, judge, slices=slices, n_boot=600, seed=33)

    assert mutated.slice_is_distinguishable
    with pytest.raises(AssertionError, match="overlaps the overall interval"):
        assert_slice_naming_rule(mutated)


def test_mutation_naming_a_slice_below_the_upper_bound(monkeypatch):
    """Mutation: compare the worst slice to ci_high instead of ci_low.

    The loosest of the three bars. The fixture is picked so the worst slice
    sits squarely inside the overall interval, above the lower bound, which
    no defensible rule can name and this one does.

    Caught by test_summary_refuses_a_slice_cut_from_one_population and by
    test_the_slice_rule_needs_the_slice_interval_to_clear_as_well.
    """
    rng = np.random.default_rng(93)
    human, judge = judge_and_human(300, 0.3, rng)
    slices = np.array([f"s{i % 3}" for i in range(300)], dtype=object)

    real = judge_validation(human, judge, slices=slices, n_boot=600, seed=33)
    worst = real.by_slice.iloc[0]["agreement"]
    assert real.ci_low < worst < real.ci_high, (
        "the fixture no longer puts the worst slice inside the overall interval"
    )
    assert real.worst_slice is None

    def mutant(self):
        if self.by_slice.empty or len(self.by_slice) < 2 or not self.has_interval:
            return False
        lowest = self.by_slice.iloc[0]["agreement"]
        return bool(np.isfinite(lowest) and lowest < self.ci_high)

    monkeypatch.setattr(
        JudgeValidation, "slice_is_distinguishable", property(mutant)
    )
    mutated = judge_validation(human, judge, slices=slices, n_boot=600, seed=33)

    assert mutated.slice_is_distinguishable
    with pytest.raises(AssertionError, match="inside the overall interval"):
        assert_slice_naming_rule(mutated)


def test_mutation_ignoring_the_slice_interval(monkeypatch):
    """Mutation: name the worst slice on its point estimate falling below
    the overall lower bound, without asking what its own interval does.

    The rule that reads correctly and is not. A slice carries a fraction of
    the items and a correspondingly wider sampling error, so on a population
    with no structure in it the lowest of five slices drops under a lower
    bound computed on all of them. This is the mutation the shipped rule
    exists to avoid, and it is the one a careful reader is most likely to
    propose.

    Caught by test_summary_refuses_a_slice_cut_from_one_population and by
    test_the_slice_rule_needs_the_slice_interval_to_clear_as_well.
    """
    rng = np.random.default_rng(94)
    human, judge = judge_and_human(300, 0.3, rng)
    slices = np.array([f"s{i % 5}" for i in range(300)], dtype=object)

    real = judge_validation(human, judge, slices=slices, n_boot=600, seed=33)
    assert real.worst_slice is None
    assert_slice_naming_rule(real)

    def mutant(self):
        if self.by_slice.empty or len(self.by_slice) < 2 or not self.has_interval:
            return False
        worst = self.by_slice.iloc[0]["agreement"]
        return bool(np.isfinite(worst) and worst < self.ci_low)

    monkeypatch.setattr(
        JudgeValidation, "slice_is_distinguishable", property(mutant)
    )
    mutated = judge_validation(human, judge, slices=slices, n_boot=600, seed=33)

    assert mutated.slice_is_distinguishable, (
        "the fixture no longer reaches past the overall lower bound"
    )
    with pytest.raises(AssertionError, match="overlaps the overall interval"):
        assert_slice_naming_rule(mutated)


def test_mutation_reporting_accuracy_as_the_slice_agreement():
    """Mutation: fill the agreement column with the raw match rate.

    Caught by test_by_slice_agreement_is_alpha_on_the_slice, through
    assert_slice_agreement_is_chance_corrected. Also flips the ordering when
    label balance differs between slices, which
    test_by_slice_is_sorted_ascending_by_agreement then sees.
    """
    rng = np.random.default_rng(94)
    human, judge, slices = sliced_data(
        [("easy", 90, 0.05), ("hard", 90, 0.6)], rng, labels=(1, 2, 3, 4, 5)
    )
    r = judge_validation(human, judge, slices=slices, n_boot=200, seed=1)

    mutant = r.by_slice.assign(agreement=r.by_slice["accuracy"])

    assert_slice_agreement_is_chance_corrected(
        r.by_slice, human, judge, slices, "nominal"
    )
    with pytest.raises(AssertionError):
        assert_slice_agreement_is_chance_corrected(
            mutant, human, judge, slices, "nominal"
        )


def test_mutation_reporting_the_overall_count_on_every_slice():
    """Mutation: fill n_items with the total item count.

    Caught by test_by_slice_reports_the_item_count_per_slice, through
    assert_slice_counts_are_per_slice. This is the mutation that makes a
    twelve-item slice look like a finding.
    """
    rng = np.random.default_rng(95)
    human, judge, slices = sliced_data(
        [("easy", 200, 0.05), ("hard", 12, 0.6), ("medium", 45, 0.3)], rng
    )
    r = judge_validation(human, judge, slices=slices, n_boot=200, seed=1)

    mutant = r.by_slice.assign(n_items=r.n_items)

    assert_slice_counts_are_per_slice(r.by_slice, slices)
    with pytest.raises(AssertionError, match="reports"):
        assert_slice_counts_are_per_slice(mutant, slices)


def test_mutation_always_reading_the_data_as_randomised(monkeypatch):
    """Mutation: skip the detection and always run the binomial test.

    Caught by test_detects_the_both_orders_design.

    The frame is built by hand so the demonstration is exact. Two hundred
    pairs, every one of them flipped, half of the flips toward the output
    shown first and half toward the one shown second. Counting rows, the
    A-wins come to exactly half, so the randomised branch reports a rate of
    0.5 and a p-value of 1. A clean bill of health for a judge that never
    once named the same output twice.
    """
    import evalaudit.judge as judge_module

    rows = []
    for i in range(200):
        x, y = f"p{i}x", f"p{i}y"
        if i < 100:
            w1, w2 = x, y      # went to whatever was in front, both times
        else:
            w1, w2 = y, x      # went to whatever came second, both times
        rows.append((f"p{i}", x, y, w1))
        rows.append((f"p{i}", y, x, w2))
    frame = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)

    real = position_bias(frame)
    assert real.design == "both_orders"
    assert real.consistency_rate == pytest.approx(0.0)
    assert real.position_a_rate == pytest.approx(0.5)

    monkeypatch.setattr(
        judge_module, "_detect_design", lambda data: ("randomised", 0)
    )
    mutated = position_bias(frame)

    assert mutated.design == "randomised"
    assert mutated.n_decisive == 400
    assert mutated.n_a_wins == 200
    assert mutated.position_a_rate == pytest.approx(0.5)
    assert mutated.p_value == pytest.approx(1.0)
    assert np.isnan(mutated.consistency_rate)


def test_mutation_consistency_counted_by_position(monkeypatch):
    """Mutation: count a pair as consistent when the same position won twice
    rather than when the same output won twice.

    Caught by test_consistency_rate_counts_pairs_where_the_same_output_won,
    through assert_consistency_counts_outputs, and by
    test_a_judge_that_always_picks_the_first_output_scores_zero, where the
    mutation turns the worst possible judge into a perfect score of 1.0.
    """
    import evalaudit.judge as judge_module

    rng = np.random.default_rng(97)
    frame = both_orders_frame(120, consistency=0.0, rng=rng, first_share=1.0)

    real = position_bias(frame)
    assert_consistency_counts_outputs(real, frame)
    assert real.consistency_rate == pytest.approx(0.0)

    monkeypatch.setattr(
        judge_module,
        "_pair_is_consistent",
        lambda rows: (rows["winner"] == rows["option_a"]).nunique() == 1,
    )
    mutated = position_bias(frame)

    assert mutated.consistency_rate == pytest.approx(1.0)
    with pytest.raises(AssertionError):
        assert_consistency_counts_outputs(mutated, frame)


def test_mutation_unsigned_length_difference(monkeypatch):
    """Mutation: regress preference on the absolute length difference.

    Caught by test_a_judge_that_chases_length_gets_a_positive_coefficient
    and by the first assertion in
    test_the_predictor_is_the_signed_length_difference. Under the mutation a
    judge that plainly reaches for longer answers returns a slope near zero
    with an interval covering it, because an unsigned predictor cannot tell
    a long winner from a short one.

    The mirror half of that test does not catch this. Flipping the outcome
    negates the slope whatever the predictor is, so the symmetry survives
    the mutation. Worth stating, because the mirror assertion looks like the
    one doing the work and it is not.
    """
    import evalaudit.judge as judge_module

    rng = np.random.default_rng(98)
    _, judge, lengths = quality_pairs(500, rng, judge_length_weight=0.01)

    real = length_bias(judge, lengths)
    assert real.coefficient > 0 and real.ci_low > 0

    monkeypatch.setattr(
        judge_module, "_length_difference", lambda a, b: np.abs(a - b)
    )
    mutated = length_bias(judge, lengths)
    mutated_mirror = length_bias(1 - judge, lengths)

    assert not (mutated.coefficient > 0 and mutated.ci_low > 0), (
        "the mutation should destroy the direction the real fit finds"
    )
    assert mutated.ci_low < 0 < mutated.ci_high
    # and the half of the sibling test that the mutation survives
    assert mutated_mirror.coefficient == pytest.approx(
        -mutated.coefficient, rel=1e-6
    )


def test_mutation_disagreement_oriented_by_option_a(monkeypatch):
    """Mutation: regress disagreement on len_a - len_b instead of on the
    length the humans passed over minus the length they picked.

    Caught by test_disagreement_model_is_oriented_by_the_human_choice, since
    the mutated coefficient moves when A and B are relabelled, and by
    test_disagreement_model_matches_statsmodels.
    """
    import evalaudit.judge as judge_module

    rng = np.random.default_rng(99)
    human, judge, lengths = quality_pairs(500, rng, judge_length_weight=0.015)

    real = length_bias(judge, lengths, human_preferences=human)
    flipped = length_bias(1 - judge, lengths[:, ::-1], human_preferences=1 - human)
    assert flipped.disagreement_coefficient == pytest.approx(
        real.disagreement_coefficient, rel=1e-6
    )

    monkeypatch.setattr(
        judge_module,
        "_oriented_length_difference",
        lambda human_chose_a, len_a, len_b: len_a - len_b,
    )
    mutated = length_bias(judge, lengths, human_preferences=human)
    mutated_flipped = length_bias(
        1 - judge, lengths[:, ::-1], human_preferences=1 - human
    )

    assert mutated_flipped.disagreement_coefficient != pytest.approx(
        mutated.disagreement_coefficient, rel=1e-6
    )


def test_mutation_disagreement_oriented_by_the_judge_choice(monkeypatch):
    """Mutation: orient by the judge's own pick rather than the humans'.

    The one that looks right and is not. Every disagreement is by definition
    a pair where the judge took what the humans rejected, so on a corpus
    where length tracks quality the predictor is negative on exactly the
    rows where the outcome is one, and the coefficient comes out negative
    whether or not the judge is biased. It is invariant under relabelling A
    and B, so the orientation test above does not see it.

    Caught by test_the_disagreement_model_shrinks_the_confound_without_
    removing_it, where the sign flips, and by
    test_disagreement_model_matches_statsmodels.
    """
    import evalaudit.judge as judge_module

    rng = np.random.default_rng(100)
    human, judge, lengths = quality_pairs(
        3000, rng, length_per_quality=140.0, judge_length_weight=0.0
    )

    real = length_bias(judge, lengths, human_preferences=human)
    assert 0 < real.disagreement_coefficient < real.coefficient

    judge_pick = {"value": None}

    def mutant(human_chose_a, len_a, len_b):
        chose = np.asarray(judge_pick["value"], dtype=bool)
        return np.where(chose, len_a - len_b, len_b - len_a)

    judge_pick["value"] = judge == 1
    monkeypatch.setattr(judge_module, "_oriented_length_difference", mutant)
    mutated = length_bias(judge, lengths, human_preferences=human)

    assert mutated.disagreement_coefficient < 0, (
        "the judge-oriented predictor should invert the sign on this corpus"
    )
    with pytest.raises(AssertionError):
        assert 0 < mutated.disagreement_coefficient < mutated.coefficient


def test_mutation_reporting_the_preference_fit_twice():
    """Mutation: copy the preference coefficient into the disagreement slot.

    Caught by test_the_disagreement_model_shrinks_the_confound_without_
    removing_it, which asserts the second coefficient is strictly below the
    first on a corpus built so that holding the human verdict fixed has to
    pull it down.
    """
    rng = np.random.default_rng(101)
    human, judge, lengths = quality_pairs(
        3000, rng, length_per_quality=140.0, judge_length_weight=0.0
    )

    r = length_bias(judge, lengths, human_preferences=human)
    mutated = r.disagreement_coefficient

    assert 0 < mutated < r.coefficient
    with pytest.raises(AssertionError):
        assert 0 < r.coefficient < r.coefficient


# --------------------------------------------------------------------------
# Types and the package surface
# --------------------------------------------------------------------------

def test_results_are_frozen():
    rng = np.random.default_rng(102)
    human, judge, lengths = quality_pairs(60, rng, judge_length_weight=0.01)
    frame = randomised_frame(30, 0.5, rng)

    results = [
        judge_validation(human, judge, n_boot=50, seed=1),
        position_bias(frame),
        length_bias(judge, lengths),
    ]
    for r in results:
        with pytest.raises(Exception):
            r.n_items = 0


def test_result_types():
    rng = np.random.default_rng(103)
    human, judge, lengths = quality_pairs(60, rng, judge_length_weight=0.01)

    assert isinstance(
        judge_validation(human, judge, n_boot=50, seed=1), JudgeValidation
    )
    assert isinstance(
        position_bias(randomised_frame(30, 0.5, np.random.default_rng(1))),
        PositionBias,
    )
    assert isinstance(length_bias(judge, lengths), LengthBias)


def test_judge_is_importable_from_the_package_root():
    import evalaudit

    for name in [
        "judge_validation", "position_bias", "length_bias",
        "JudgeValidation", "PositionBias", "LengthBias",
    ]:
        assert hasattr(evalaudit, name)
        assert name in evalaudit.__all__


def test_every_result_has_a_summary():
    rng = np.random.default_rng(104)
    human, judge, lengths = quality_pairs(80, rng, judge_length_weight=0.01)

    for r in [
        judge_validation(human, judge, n_boot=50, seed=1),
        position_bias(randomised_frame(40, 0.5, rng)),
        position_bias(both_orders_frame(40, 0.8, rng)),
        length_bias(judge, lengths),
        length_bias(judge, lengths, human_preferences=human),
    ]:
        assert isinstance(r.summary(), str)
        assert len(r.summary()) > 40
        assert str(r) == r.summary()


# --------------------------------------------------------------------------
# The both-orders design prints the interval on the flip share, and decides
# on it
#
# The interval on the flip share was computed and thrown away, and the
# summary printed the share with a bare p-value. It is printed now in the
# form the randomised design uses, and the verdict reads it. It is
# Clopper-Pearson, which inverts the exact test, so the verdict and the
# p-value cannot disagree about 50%.
# --------------------------------------------------------------------------

def flips_frame(first, flipped, consistent):
    """Pairs run both ways. ``first`` of the ``flipped`` pairs went to the
    output shown first, and ``consistent`` more pairs never flipped."""
    rows = []
    for p in range(flipped):
        toward_first = p < first
        rows += [
            (p, "x", "y", "x" if toward_first else "y"),
            (p, "y", "x", "y" if toward_first else "x"),
        ]
    for p in range(flipped, flipped + consistent):
        rows += [(p, "x", "y", "x"), (p, "y", "x", "x")]
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def test_both_orders_prints_the_interval_on_the_flip_share():
    r = position_bias(flips_frame(7, 9, 21))
    lo, hi = proportion_confint(7, 9, method="beta")
    assert r.position_a_ci_low == pytest.approx(lo, abs=1e-12)
    assert r.position_a_ci_high == pytest.approx(hi, abs=1e-12)
    assert (
        " Of the 9 pairs it flipped on, 7 went to whichever output was shown "
        "first (77.8%, 95% CI: 40.0% to 97.2%, exact binomial p=0.1797)."
    ) in r.summary()
    assert "(77.8%, exact binomial" not in r.summary()


def test_both_orders_decides_on_the_interval_it_prints():
    """11 of 14 flips toward the first answer, exact binomial p=0.0574.

    Written for the Wilson interval, 52.4% to 92.4%, which cleared 50% and
    called position bias beside that p-value. The interval is
    Clopper-Pearson now, 49.2% to 95.3%, which inverts the exact test and so
    includes 50% whenever p is above 0.05. The verdict flips back to what
    0.3.0 said, and now the interval printed beside it agrees.
    """
    r = position_bias(flips_frame(11, 14, 16))
    assert r.p_value > 0.05
    assert r.position_a_ci_low < 0.5 < r.position_a_ci_high
    assert not r.has_position_effect
    assert CANNOT_SHOW_A_DIRECTION in r.summary()
    assert CALLS_IT_POSITION_BIAS not in r.summary()
    assert (
        "(78.6%, 95% CI: 49.2% to 95.3%, exact binomial p=0.0574)."
    ) in r.summary()


def test_the_both_orders_verdict_always_agrees_with_its_interval():
    for flipped in range(1, 17):
        for first in range(flipped + 1):
            r = position_bias(flips_frame(first, flipped, 4))
            clears = not (r.position_a_ci_low <= 0.5 <= r.position_a_ci_high)
            assert r.has_position_effect == clears, (first, flipped)
            assert (CANNOT_SHOW_A_DIRECTION in r.summary()) is (not clears), (
                first, flipped
            )


def test_the_randomised_design_carries_the_same_interval_fields():
    r = position_bias(randomised_frame(200, 0.5, np.random.default_rng(3)))
    assert r.design == "randomised"
    assert (r.position_a_ci_low, r.position_a_ci_high) == (r.ci_low, r.ci_high)


def test_no_flips_leaves_no_interval_on_the_share():
    r = position_bias(flips_frame(0, 0, 10))
    assert r.design == "both_orders"
    assert np.isnan(r.position_a_ci_low)
    assert np.isnan(r.position_a_ci_high)
    assert not r.has_position_effect



# --------------------------------------------------------------------------
# Clopper-Pearson where an exact p is printed
#
# The rate tested against a half now gets the Clopper-Pearson interval,
# which inverts the exact binomial test whose p-value is printed beside it.
# Wilson and that test disagree at a half in 188 of the 20,300 (n, k) cells
# up to n=200, with Wilson on the permissive side every time.
# --------------------------------------------------------------------------

def exact_randomised_frame(k, n):
    """n singly judged pairs, k of them won by the output shown first."""
    return pd.DataFrame(
        [(f"p{i}", "x", "y", "x" if i < k else "y") for i in range(n)],
        columns=COMPARISON_COLUMNS,
    )


def test_the_verdict_and_the_p_value_never_disagree_about_a_half():
    """Every k <= n <= 30, in both designs. The interval matches scipy's
    exact interval, and the verdict read off it matches the p-value printed
    beside it. That holds by construction, since the interval inverts the
    test, so this pins the construction."""
    from scipy import stats as sp_stats

    for n in range(1, 31):
        for k in range(n + 1):
            exact = sp_stats.binomtest(k, n, 0.5)
            ci = exact.proportion_ci(confidence_level=0.95, method="exact")
            for r in (
                position_bias(exact_randomised_frame(k, n)),
                position_bias(flips_frame(k, n, 2)),
            ):
                where = (r.design, k, n)
                assert r.position_a_ci_low == pytest.approx(ci.low, abs=1e-12), where
                assert r.position_a_ci_high == pytest.approx(ci.high, abs=1e-12), where
                assert r.p_value == pytest.approx(exact.pvalue, abs=1e-12), where
                assert r.has_position_effect is (r.p_value < 0.05), where


def test_the_randomised_eleven_of_fourteen_flips_to_no_direction():
    """11 of 14 position-A wins, exact binomial p=0.0574. The Wilson
    interval, 52.4% to 92.4%, cleared 50%, so 0.3.0 said the judge favours
    the first output beside that p-value. Clopper-Pearson runs from 49.2% to
    95.3% and includes 50%."""
    r = position_bias(exact_randomised_frame(11, 14))
    assert r.design == "randomised"
    assert r.p_value > 0.05
    assert r.ci_low < 0.5 < r.ci_high
    assert not r.has_position_effect
    assert (
        "Position A won 78.6% of 14 judgements (95% CI: 49.2% to 95.3%, "
        "exact binomial p=0.0574)."
    ) in r.summary()
    assert RANDOMISED_INCLUDES_A_HALF in r.summary()
    assert RANDOMISED_CLEARS_A_HALF not in r.summary()


def test_the_consistency_interval_stays_wilson():
    """No test sits beside the consistency rate, so it keeps Wilson."""
    r = position_bias(flips_frame(7, 9, 21))
    lo, hi = proportion_confint(21, 30, method="wilson")
    assert r.ci_low == pytest.approx(lo, abs=1e-12)
    assert r.ci_high == pytest.approx(hi, abs=1e-12)


# --------------------------------------------------------------------------
# judge_validation against a human baseline
#
# Written before the implementation. The baseline is a long frame of items
# graded by two or more humans, with item_id, cluster_id, rater_id and
# rating. The function reports judge-human alpha and human-human alpha on
# the same items, and a percentile interval on judge-human minus
# human-human that resamples cluster_id. The verdict reads that interval
# and nothing else.
#
# validate_with_baseline makes the call for every test here except the
# refusals, which call judge_validation directly. item_ids names the item
# at each position of human and judge, and the baseline's item_id column is
# matched against it to find the judge's label on each baseline item. ties
# defaults to None. "category" keeps a tie as a label of its own, and
# "drop" sets it aside as no label. With a baseline, ties sets the coding
# for every figure in the result, the headline included. Under "drop" a
# headline item with a tie on either side is set aside and counted in
# n_dropped_ties, apart from the items with no label that n_dropped counts.
#
# A tie is a rating that reads "tie" once case and surrounding spaces are
# ignored, as bradley_terry's _is_tie reads one. A missing rating stays no
# label. It is never a tie here, where bradley_terry would count it as one.
#
# judge_validation raises ValueError for a baseline with no ties, for ties
# or item_ids with no baseline, for a repeated value in item_ids, for a
# baseline item that item_ids does not name, for a repeated (item_id,
# rater_id) pair in the baseline, and for a baseline at any level other
# than "nominal".
#
# Baseline items are set aside before anything is computed, by three rules
# in this order. An item with no judge label goes first. An item with fewer
# than two human labels goes next, and under "drop" only decisive labels
# count toward the two. Last, under "drop", an item whose judge label is a
# tie goes. Each item is counted once, under the first rule it fails. Both
# alphas use the items that remain. A baseline row with a missing rating is
# no label, so it counts toward nothing and changes nothing.
#
# The result carries baseline_judge_human, baseline_human_human,
# baseline_difference, baseline_ci_low, baseline_ci_high, baseline_n_items,
# baseline_n_no_judge_label, baseline_n_too_few_humans,
# baseline_n_judge_ties and baseline_n_boot_usable, and n_dropped_ties
# beside the existing n_dropped.
#
# The bootstrap is the one analysis/q5_judge_against_human_baseline.py ran.
# It is written down here because the published MT-Bench intervals pin it
# to the third decimal. Clusters are numbered in order of first appearance
# among the baseline rows that remain. One index matrix comes from
# default_rng(seed).integers(0, n_clusters, size=(n_boot, n_clusters)),
# and both alphas are computed on every draw from it. A draw where either
# alpha is undefined is left out, and the bounds are numpy's default
# percentiles of the difference over the draws that remain. Human-human is
# undefined exactly when every human label in a draw is the same, and
# judge-human can only be undefined then too. When the usable draws fall
# below _MIN_USABLE_SHARE there is no interval on the difference and no
# verdict. baseline_reference below does all of this apart from the
# package, and the intervals in the pinned summaries are the ones it gives.
#
# With a baseline, the summary has no band sentence. Under "drop" it says
# the judge's own ties decide which items it is scored on whenever the
# judge-tie rule set an item aside. When the interval on the difference
# excludes zero and the items that remain span fewer than THIN_SLICE
# clusters, a sentence after the verdict gives the cluster count and calls
# the verdict provisional. THIN_SLICE is 30, the line ScoreCI and the slice
# table use. The pinned summaries below show every sentence it can carry.
# --------------------------------------------------------------------------

PREFERENCE = np.array(["first", "second"], dtype=object)
TIE = "tie"
BASELINE_BOOT = 2000
BASELINE_SEED = 0

# The package's _MIN_USABLE_SHARE, copied out by hand. Below this share of
# usable draws there is no interval on the difference.
USABLE_SHARE = 0.90

MT_BENCH_DATA = Path(__file__).resolve().parent / "data"
MT_BENCH_BOOT = 5000
MT_BENCH_SEED = 0

# As analysis/mt-bench.md prints them. Judge-human, human-human, the
# difference, and the two bounds on the difference.
PUBLISHED_ALL_COMPARISONS = ("0.479", "0.478", "0.001", "-0.055", "0.054")
PUBLISHED_DECISIVE_ONLY = ("0.737", "0.692", "0.045", "0.002", "0.088")

# The same five at full precision, from run() in
# analysis/q5_judge_against_human_baseline.py, coded by alphabetical
# position, with 5,000 question resamples at seed 0. The published bounds
# were produced under NumPy 2.5.3, the version analysis/README.md records.
# These values come from q5 run again under NumPy 2.5.3 with evalaudit
# 0.4.0 from PyPI.
Q5_ALL_COMPARISONS = (
    0.4785201983909394, 0.4779214095927723, 0.0005987887981671047,
    -0.05499709908783848, 0.05449167095807316,
)
Q5_DECISIVE_ONLY = (
    0.7368054671588595, 0.6915237503781908, 0.045281716780668724,
    0.0019998554309556714, 0.0878113776308402,
)


def graded_items(human_accuracy, judge_accuracy, items_per_cluster, seed,
                 humans_per_item=3, human_tie=0.0, judge_tie=0.0):
    """A human baseline and the judge's label on each of its items.

    One cluster per entry of the two accuracy arrays. Each item has a true
    preference, first or second. Every human and the judge report it at
    their cluster's accuracy and report the other label otherwise. Raters
    h0, h1 and so on grade every item.

    human_tie and judge_tie are the chance that a label is replaced by
    "tie". Those draws come after all the others, so a fixture with no ties
    is the same whether or not the arguments exist.

    Returns the baseline frame and the judge's labels as a Series indexed
    by item_id.
    """
    rng = np.random.default_rng(seed)
    human_accuracy = np.asarray(human_accuracy, dtype=float)
    judge_accuracy = np.asarray(judge_accuracy, dtype=float)
    cluster = np.repeat(np.arange(human_accuracy.size), items_per_cluster)
    truth = rng.integers(0, 2, size=cluster.size)
    human_right = (
        rng.random((cluster.size, humans_per_item))
        < human_accuracy[cluster][:, None]
    )
    human = np.where(human_right, truth[:, None], 1 - truth[:, None])
    judge_right = rng.random(cluster.size) < judge_accuracy[cluster]
    judge = np.where(judge_right, truth, 1 - truth)
    human_labels = PREFERENCE[human]
    judge_labels = PREFERENCE[judge]
    human_labels[rng.random(human_labels.shape) < human_tie] = TIE
    judge_labels[rng.random(judge_labels.shape) < judge_tie] = TIE

    items = np.array([f"i{u}" for u in range(cluster.size)], dtype=object)
    frame = pd.DataFrame({
        "item_id": np.repeat(items, humans_per_item),
        "cluster_id": np.repeat([f"c{c}" for c in cluster], humans_per_item),
        "rater_id": np.tile(
            [f"h{r}" for r in range(humans_per_item)], cluster.size
        ),
        "rating": human_labels.ravel(),
    })
    return frame, pd.Series(judge_labels, index=items, name="judge")


def is_tie(values):
    """True where a rating is a tie.

    Read the way bradley_terry's _is_tie reads one, with case and
    surrounding spaces ignored, except that a missing rating is never a tie.
    Keeps the index of a Series.
    """
    values = pd.Series(values, dtype=object)
    present = values.notna()
    words = values.where(present, "").astype(str).str.strip().str.lower()
    return present & (words == TIE)


def headline_ties(frame, judge_labels):
    """How many headline items ties="drop" sets aside for a tie.

    An item where either side has no label is counted by n_dropped
    instead, so it is left out here.
    """
    human, judge, _ = headline_inputs(frame, judge_labels)
    present = pd.Series(human).notna() & pd.Series(judge).notna()
    return int((present & (is_tie(human) | is_tie(judge))).sum())


def kref_human_human(frame):
    """Human-human alpha from the krippendorff package, nominal, on exactly
    the ratings in the frame."""
    item_codes, items = pd.factorize(frame["item_id"], sort=False)
    rater_codes, raters = pd.factorize(frame["rater_id"], sort=False)
    values, coded = np.unique(frame["rating"].to_numpy(dtype=str),
                              return_inverse=True)
    matrix = np.full((raters.size, items.size), np.nan)
    matrix[rater_codes, item_codes] = coded
    return float(kref.alpha(reliability_data=matrix,
                            level_of_measurement="nominal"))


def kref_figures(frame, judge_labels):
    """Judge-human and human-human alpha from the krippendorff package,
    nominal, on exactly the items and ratings given.

    Judge-human pairs every human rating with the judge's label on its
    item. The caller builds the item set by hand, so this is the reference
    for which items the package keeps.
    """
    item_codes, items = pd.factorize(frame["item_id"], sort=False)
    ratings = frame["rating"].to_numpy(dtype=str)
    judged = judge_labels.loc[items].to_numpy(dtype=str)[item_codes]
    values = np.unique(np.concatenate([ratings, judged]))
    pairs = np.vstack([np.searchsorted(values, ratings),
                       np.searchsorted(values, judged)]).astype(float)
    judge_human = float(kref.alpha(reliability_data=pairs,
                                   level_of_measurement="nominal"))
    return judge_human, kref_human_human(frame)


def nominal_alpha(numerator, marginals):
    """Alpha from summed per-item disagreement and summed value counts."""
    n = marginals.sum(axis=-1)
    expected = n ** 2 - (marginals ** 2).sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = 1.0 - (n - 1) * numerator / expected
    return np.where(expected > 0, alpha, np.nan)


def baseline_reference(frame, judge_labels, unit="cluster_id",
                       n_boot=BASELINE_BOOT, seed=BASELINE_SEED,
                       confidence=0.95):
    """Both alphas, their difference, and percentile intervals on all
    three, computed apart from the package on the items given.

    The caller sets items aside first. Every item in the frame is used.

    ``unit`` is what one draw picks up. "cluster_id" is the design under
    test. "item_id" draws whole items. "row" draws single human ratings,
    each carrying the judge's label on its item, and rebuilds the items
    from whatever rows came up. The last two are the wrong units, and the
    tests use them to show that a fixture can tell them apart.

    Nominal only. The point figures are checked against kref_figures
    before anything is returned. The counts of undefined draws are split
    by which alpha was undefined, so a fixture can say which one it
    leaves undefined.
    """
    item_codes, items = pd.factorize(frame["item_id"], sort=False)
    ratings = frame["rating"].to_numpy(dtype=str)
    judged = judge_labels.loc[items].to_numpy(dtype=str)[item_codes]
    values = np.unique(np.concatenate([ratings, judged]))
    human = np.eye(values.size)[np.searchsorted(values, ratings)]
    judge = np.eye(values.size)[np.searchsorted(values, judged)]
    disagree = 1.0 - (human * judge).sum(axis=1)
    member = np.zeros((items.size, len(frame)))
    member[item_codes, np.arange(len(frame))] = 1.0

    def from_items(copies):
        """Both alphas for each row of ``copies``, a count per item."""
        counts = member @ human
        size = counts.sum(axis=1)
        hh_num = copies @ ((size ** 2 - (counts ** 2).sum(axis=1)) / (size - 1))
        jh_num = copies @ (2.0 * (member @ disagree))
        return (
            nominal_alpha(jh_num, copies @ (counts + member @ judge)),
            nominal_alpha(hh_num, copies @ counts),
        )

    def from_rows(copies):
        """Both alphas for each row of ``copies``, a count per rating."""
        counts = np.stack(
            [(copies * human[:, v]) @ member.T for v in range(values.size)],
            axis=-1,
        )
        size = counts.sum(axis=-1)
        pairable = size >= 2
        with np.errstate(divide="ignore", invalid="ignore"):
            per_item = (size ** 2 - (counts ** 2).sum(axis=-1)) / (size - 1)
        hh_num = np.where(pairable, per_item, 0.0).sum(axis=1)
        hh_marg = (counts * pairable[..., None]).sum(axis=1)
        jh_num = copies @ (2.0 * disagree)
        return (
            nominal_alpha(jh_num, copies @ (human + judge)),
            nominal_alpha(hh_num, hh_marg),
        )

    jh, hh = (float(a[0]) for a in from_items(np.ones((1, items.size))))
    assert (jh, hh) == pytest.approx(kref_figures(frame, judge_labels), abs=1e-9)

    group = {
        "cluster_id": pd.factorize(frame["cluster_id"], sort=False)[0],
        "item_id": item_codes,
        "row": np.arange(len(frame)),
    }[unit]
    n_groups = int(group.max()) + 1
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_groups, size=(n_boot, n_groups))
    offset = idx + np.arange(n_boot)[:, None] * n_groups
    drawn = np.bincount(offset.ravel(), minlength=n_boot * n_groups)
    drawn = drawn.reshape(n_boot, n_groups).astype(float)

    first_row = pd.Series(np.arange(len(frame))).groupby(item_codes).first()
    jh_b, hh_b = (
        from_rows(drawn) if unit == "row"
        else from_items(drawn[:, group[first_row.to_numpy()]])
    )
    judge_defined = np.isfinite(jh_b)
    human_defined = np.isfinite(hh_b)
    usable = judge_defined & human_defined
    tail = 100 * (1 - confidence) / 2

    def bounds(x):
        low, high = np.percentile(x[usable], [tail, 100 - tail])
        return float(low), float(high)

    return {
        "judge_human": jh,
        "human_human": hh,
        "difference": jh - hh,
        "difference_ci": bounds(jh_b - hh_b),
        "judge_human_ci": bounds(jh_b),
        "human_human_ci": bounds(hh_b),
        "n_undefined": int((~usable).sum()),
        "n_usable": int(usable.sum()),
        "n_judge_human_undefined": int((~judge_defined).sum()),
        "n_human_human_undefined": int((~human_defined).sum()),
    }


def baseline_items(frame, judge_labels, ties):
    """The baseline items the package keeps, built by hand.

    The three rules run in their fixed order, and each item counts under
    the first one it fails. Items with no judge label go first. A row with
    a missing rating is no label and never counts. Under "drop" the tie
    ratings go as no label too. Items left with fewer than two human labels
    go next. Last, under "drop", items whose judge label is a tie go. Every
    tie comes back spelled "tie", whatever its case and spacing were.

    Returns the remaining frame in its original row order, the judge's
    labels on its items, and the three counts in rule order.
    """
    judge_labels = judge_labels.where(~is_tie(judge_labels), TIE)
    frame = frame.assign(
        rating=frame["rating"].where(~is_tie(frame["rating"]), TIE)
    )
    no_judge = judge_labels.index[judge_labels.isna()]
    rest = frame[~frame["item_id"].isin(no_judge)]
    candidates = pd.unique(rest["item_id"])
    labelled = rest[rest["rating"].notna()]
    labelled = labelled[labelled["rating"] != TIE] if ties == "drop" else labelled
    raters = labelled.groupby("item_id")["rater_id"].nunique()
    enough = raters.index[raters >= 2]
    labelled = labelled[labelled["item_id"].isin(enough)]
    judged = judge_labels.loc[pd.unique(labelled["item_id"])]
    judge_ties = judged.index[(judged == TIE) & (ties == "drop")]
    kept = labelled[~labelled["item_id"].isin(judge_ties)].reset_index(drop=True)
    kept_labels = judge_labels.loc[pd.unique(kept["item_id"])]
    counts = (no_judge.size, candidates.size - enough.size, judge_ties.size)
    return kept, kept_labels, counts


def reference_figures(reference):
    low, high = reference["difference_ci"]
    return (reference["judge_human"], reference["human_human"],
            reference["difference"], low, high)


def baseline_figures(result):
    return (result.baseline_judge_human, result.baseline_human_human,
            result.baseline_difference, result.baseline_ci_low,
            result.baseline_ci_high)


def three_decimals(figures):
    return tuple(f"{x:.3f}" for x in figures)


def interval_width(reference):
    low, high = reference["difference_ci"]
    return high - low


def headline_inputs(frame, judge_labels):
    """The first human listed on each item, the judge's labels, and the
    item ids, one position per item."""
    first_human = (
        frame.drop_duplicates("item_id").set_index("item_id")["rating"]
    )
    return (
        first_human.loc[judge_labels.index].to_numpy(),
        judge_labels.to_numpy(),
        judge_labels.index.to_numpy(),
    )


def validate_with_baseline(frame, judge_labels, ties, n_boot=BASELINE_BOOT,
                           seed=BASELINE_SEED, confidence=0.95):
    """judge_validation with a human baseline, the call every test here
    makes except the refusals.

    The headline pair is the judge against the first human listed on each
    item. The baseline is every human rating in the frame.
    """
    human, judge, item_ids = headline_inputs(frame, judge_labels)
    return judge_validation(
        human, judge, item_ids=item_ids, human_baseline=frame, ties=ties,
        n_boot=n_boot, seed=seed, confidence=confidence,
    )


def every_figure(result):
    """Every number the result carries about the headline and the
    baseline, for comparing two results whole."""
    return (
        result.agreement, result.ci_low, result.ci_high, result.accuracy,
        result.n_items, result.n_dropped, result.n_dropped_ties,
        *baseline_figures(result),
        result.baseline_n_items, result.baseline_n_no_judge_label,
        result.baseline_n_too_few_humans, result.baseline_n_judge_ties,
        result.baseline_n_boot_usable,
    )


# Fixtures. Each docstring states the property the tests rely on, and each
# test asserts that property on the fixture before it calls the package.

def judge_like_a_fourth_human():
    """Three humans and a judge, all drawn the same way.

    Every rater, the judge included, reports an item's true preference 80%
    of the time, over 24 clusters of six items. Judge-human and human-human
    agreement estimate the same number.

    The property the tests rely on is that the interval on the difference
    includes zero, and that the bounds move when the seed does.
    """
    return graded_items([0.8] * 24, [0.8] * 24, items_per_cluster=6, seed=1)


def judge_more_reliable_than_the_humans():
    """The judge is right 98% of the time and each human 72%, over 24
    clusters of ten items.

    The property the test relies on is that the interval on the difference
    sits wholly above zero. The separate intervals on the two alphas do not
    overlap either, and the test checks that too. So this fixture cannot
    tell a verdict read off the difference from one read off the two
    intervals. The fixture for test_baseline_verdict_reads_the_difference
    can.
    """
    return graded_items([0.72] * 24, [0.98] * 24, items_per_cluster=10, seed=0)


def judge_less_reliable_than_the_humans():
    """The humans are right 90% of the time and the judge 60%, over 24
    clusters of six items.

    The property the test relies on is that the interval on the difference
    sits wholly below zero.
    """
    return graded_items([0.9] * 24, [0.6] * 24, items_per_cluster=6, seed=0)


def judge_that_guesses_in_half_the_clusters():
    """The humans are right 85% of the time everywhere. The judge is right
    95% of the time in the even clusters and guesses in the odd ones.

    Twelve clusters of 25 items, three ratings per item. Every item in a
    cluster gets the same judge, so the clusters differ from each other far
    more than the items inside one do. A bootstrap that draws items or rows
    treats 300 items or 900 ratings as independent evidence about a
    difference that mostly varies between 12 clusters.

    The property the test relies on is that resampling items and resampling
    rows each give an interval on the difference less than half as wide as
    resampling clusters. Both of those intervals also sit wholly below zero,
    where the cluster interval includes it.
    """
    return graded_items(
        [0.85] * 12, [0.95, 0.5] * 6, items_per_cluster=25, seed=0
    )


def judge_a_little_ahead_in_every_cluster():
    """Human accuracy runs from 60% to 94% over ten clusters of 30 items,
    and the judge is 8 points more accurate than the humans in each.

    The spread between clusters moves both alphas together, so each one has
    a wide interval. Their difference moves much less, because what the two
    share cancels in it.

    The property the test relies on is that the separate intervals on the
    two alphas overlap, and that the interval on the difference sits wholly
    above zero.
    """
    accuracy = np.linspace(0.6, 0.94, 10)
    return graded_items(
        accuracy, np.minimum(accuracy + 0.08, 1.0), items_per_cluster=30, seed=0
    )


def ties_on_both_sides():
    """Humans right 80% of the time and the judge 85%, over 24 clusters of
    six items, with 30% of human labels and 20% of judge labels replaced by
    a tie.

    The property the test relies on is that ties are on both sides, 131
    human labels and 29 judge labels, and that the two codings give
    different alphas. Under ties="drop" the too-few-humans rule sets aside
    30 items and the judge-tie rule 23 more. Six items fail both rules, so
    the counts would read 24 and 29 if the judge-tie rule ran first.
    """
    return graded_items(
        [0.8] * 24, [0.85] * 24, items_per_cluster=6, seed=2,
        human_tie=0.3, judge_tie=0.2,
    )


def judge_silent_on_some_items():
    """Humans right 80% of the time and the judge 85%, over 16 clusters of
    six items. The judge gives no label on the first twelve items where all
    three humans agree.

    The property the test relies on is that human-human alpha over every
    baseline item differs from human-human alpha over the items the judge
    labelled. Taking unanimous items out lowers it from 0.308 to 0.209.
    """
    frame, judge_labels = graded_items(
        [0.8] * 16, [0.85] * 16, items_per_cluster=6, seed=5
    )
    distinct = frame.groupby("item_id", sort=False)["rating"].nunique()
    silent = distinct.index[distinct == 1][:12]
    judge_labels = judge_labels.astype(object)
    judge_labels.loc[silent] = None
    return frame, judge_labels


def one_decisive_human_and_a_tie():
    """Two humans per item, right 80% of the time, and a judge right 80% of
    the time, over 16 clusters of six items. On the first twelve items
    where the judge disagrees with h0, h1's label is replaced by a tie.

    The property the test relies on is that those twelve items each hold
    one decisive human label and one tie, the judge gives no tie anywhere,
    and keeping the twelve items with their one decisive label would lower
    judge-human alpha from 0.417 to 0.323. Human-human alpha is the same
    either way, since an item with one label has no pair to compare.
    """
    frame, judge_labels = graded_items(
        [0.8] * 16, [0.8] * 16, items_per_cluster=6, seed=7, humans_per_item=2
    )
    h0 = frame[frame["rater_id"] == "h0"].set_index("item_id")["rating"]
    against = h0.index[(h0 != judge_labels.loc[h0.index]).to_numpy()][:12]
    tied = frame["item_id"].isin(against) & (frame["rater_id"] == "h1")
    frame.loc[tied, "rating"] = TIE
    return frame, judge_labels


def one_cluster_where_every_human_says_first():
    """Three clusters of ten items, humans and judge right 80% of the time.
    In c0 every human label is "first", and the judge alternates between
    "first" and "second".

    A draw that picks c0 three times holds human labels of one value only,
    so human-human alpha is undefined there. The judge's labels still vary,
    so judge-human alpha is defined.

    The property the test relies on is that 68 of 2,000 draws leave
    human-human alpha undefined and none leave judge-human undefined. That
    is 3.4%, under the 10% past which the package refuses an interval.
    """
    return clusters_where_every_human_says_first(["c0"])


def two_clusters_where_every_human_says_first():
    """The same three clusters, with every human label "first" in c1 as
    well as c0, and the judge alternating in both.

    A draw that picks only c0 and c1 leaves human-human alpha undefined.
    That happens in about (2/3) ** 3 of draws, near 30%.

    The property the test relies on is that 591 of 2,000
    draws leave human-human alpha undefined, which is past the 10% line,
    and none leave judge-human undefined.
    """
    return clusters_where_every_human_says_first(["c0", "c1"])


def clusters_where_every_human_says_first(clusters):
    """Three clusters of ten items, humans and judge right 80% of the time,
    and in each named cluster every human label set to "first" and the
    judge alternating between "first" and "second"."""
    frame, judge_labels = graded_items(
        [0.8] * 3, [0.8] * 3, items_per_cluster=10, seed=11
    )
    judge_labels = judge_labels.copy()
    for cluster in clusters:
        inside = frame["cluster_id"] == cluster
        frame.loc[inside, "rating"] = "first"
        items = pd.unique(frame.loc[inside, "item_id"])
        judge_labels.loc[items] = np.where(
            np.arange(items.size) % 2 == 0, "first", "second"
        )
    return frame, judge_labels


def some_items_graded_by_one_human():
    """judge_like_a_fourth_human with h1 and h2 removed from the first item
    of every third cluster, so eight items keep one human label each.

    The property the test relies on is that those eight items hold one
    human label, and that keeping them would move judge-human alpha from
    0.427 to 0.419. Human-human alpha is the
    same either way, since an item with one label has no pair to compare.
    """
    frame, judge_labels = judge_like_a_fourth_human()
    first_items = frame.drop_duplicates("cluster_id")["item_id"].iloc[::3]
    thinned = frame["item_id"].isin(first_items) & (frame["rater_id"] != "h0")
    return frame[~thinned].reset_index(drop=True), judge_labels


def judge_silent_among_ties():
    """ties_on_both_sides, with the judge's label removed on the first four
    items that have fewer than two decisive human labels and a decisive
    judge label.

    The property the test relies on is that four items fail both the
    no-judge-label rule and the too-few-humans rule, and six items fail
    both the too-few-humans rule and the judge-tie rule. An item cannot
    fail the first and last rules together, since a missing label is not a
    tie. So the counts come out 4, 26 and 23 only when the rules run
    no-judge-label, then too-few-humans, then judge-tie. Every other order
    moves one of the two groups of overlapping items to another count.
    """
    frame, judge_labels = ties_on_both_sides()
    decisive = frame[frame["rating"] != TIE].groupby("item_id")["rater_id"]
    decisive_humans = decisive.nunique().reindex(judge_labels.index, fill_value=0)
    thin = judge_labels.index[
        (decisive_humans < 2).to_numpy() & (judge_labels != TIE).to_numpy()
    ]
    judge_labels = judge_labels.astype(object)
    judge_labels.loc[thin[:4]] = None
    return frame, judge_labels


def ties_spelled_two_ways():
    """ties_on_both_sides, with every tie respelled. Human ties alternate
    between "Tie" and " tie ", and judge ties between " tie " and "Tie".

    The property the test relies on is that no tie is spelled "tie", both
    spellings appear on both sides, and reading the spellings as labels of
    their own would move the alphas. kref on the respelled data gives
    judge-human 0.144 and human-human 0.071, against 0.149 and 0.122 on the
    data as generated.
    """
    frame, judge_labels = ties_on_both_sides()
    human_ties = np.flatnonzero((frame["rating"] == TIE).to_numpy())
    frame.loc[human_ties, "rating"] = np.where(
        np.arange(human_ties.size) % 2 == 0, "Tie", " tie "
    )
    judge_labels = judge_labels.astype(object)
    judge_ties = np.flatnonzero((judge_labels == TIE).to_numpy())
    judge_labels.iloc[judge_ties] = np.where(
        np.arange(judge_ties.size) % 2 == 0, " tie ", "Tie"
    )
    return frame, judge_labels


def ties_with_a_missing_rating():
    """ties_on_both_sides with one row added at the end. Rater h3, who
    grades nothing else, has a row with no rating on i5, the
    first item with exactly one decisive human label and a decisive judge
    label.

    A missing rating is no label, so every figure, count and summary should
    be the one ties_on_both_sides gives. The property the test relies on is
    that reading the missing rating any other way would show. Read as a
    decisive label under "drop", it gives i5 a second decisive
    label and keeps it, so the too-few-humans count falls from 30 to 29.
    Under "category", read as a label of its own it moves human-human alpha
    from 0.122 to 0.121, and read as a tie, the way bradley_terry
    reads a missing winner, it moves it to 0.124.
    """
    frame, judge_labels = ties_on_both_sides()
    decisive = frame[frame["rating"] != TIE].groupby("item_id")["rater_id"]
    decisive_humans = decisive.nunique().reindex(judge_labels.index, fill_value=0)
    item = judge_labels.index[
        (decisive_humans == 1).to_numpy() & (judge_labels != TIE).to_numpy()
    ][0]
    cluster = frame.loc[frame["item_id"] == item, "cluster_id"].iloc[0]
    extra = pd.DataFrame({
        "item_id": [item], "cluster_id": [cluster], "rater_id": ["h3"],
        "rating": pd.Series([None], dtype=object),
    })
    return pd.concat([frame, extra], ignore_index=True), judge_labels


def headline_items_with_a_tie_and_no_label():
    """ties_on_both_sides with two headline items changed. On
    i1, where h0 says tie and the judge does not, the
    judge's label is removed. On i4, where the judge says tie
    and h0 does not, h0's rating is removed.

    Both were among the 64 headline items set aside for a tie. The property
    the test relies on is that each now has no label on one side and a tie
    on the other, so under "drop" they belong in n_dropped, which becomes 2,
    and n_dropped_ties falls to 62.
    """
    frame, judge_labels = ties_on_both_sides()
    human, judge, item_ids = headline_inputs(frame, judge_labels)
    human_tie = is_tie(human).to_numpy()
    judge_tie = is_tie(judge).to_numpy()
    tie_then_none = item_ids[human_tie & ~judge_tie][0]
    none_then_tie = item_ids[judge_tie & ~human_tie][0]
    judge_labels = judge_labels.astype(object)
    judge_labels.loc[tie_then_none] = None
    first_row = frame.index[frame["item_id"] == none_then_tie][0]
    frame = frame.astype({"rating": object})
    frame.loc[first_row, "rating"] = None
    return frame, judge_labels


def one_item_under_each_rule():
    """judge_like_a_fourth_human, which has no ties, with three items
    changed. The judge gives no label on i0. h1 and h2 say tie on i6, so it
    keeps one decisive human label. The judge says tie on i12.

    The property the test relies on is that under "drop" each baseline rule
    sets aside exactly one item, i0, i6 and i12 in rule order, one headline
    item has no label, i0, and one headline item has a tie, i12. Every count
    in the summary is one, so every sentence that gives a count is in the
    singular.
    """
    frame, judge_labels = judge_like_a_fourth_human()
    judge_labels = judge_labels.astype(object)
    judge_labels.loc["i0"] = None
    judge_labels.loc["i12"] = TIE
    frame.loc[(frame["item_id"] == "i6") & (frame["rater_id"] != "h0"), "rating"] = TIE
    return frame, judge_labels


def six_clusters_where_everyone_says_first():
    """Eight clusters of three items, three humans per item. In c0 to c5
    every human and the judge say "first". c6 and c7 come from graded_items
    and each hold at least one other human label.

    A draw is undefined when it picks only from c0 to c5, which happens
    with probability 0.75 ** 8, just over 10%. So the usable share of 2,000
    draws falls either side of the 90% line depending on the seed.

    The property the test relies on is that seed 1 leaves exactly 1,800
    usable draws, on the line, and seed 20 leaves exactly 1,799, under it.
    """
    frame, judge_labels = graded_items(
        [0.8] * 8, [0.8] * 8, items_per_cluster=3, seed=3
    )
    unanimous = frame["cluster_id"].isin([f"c{c}" for c in range(6)])
    frame.loc[unanimous, "rating"] = "first"
    judge_labels = judge_labels.copy()
    judge_labels.loc[pd.unique(frame.loc[unanimous, "item_id"])] = "first"
    return frame, judge_labels


def three_items_two_of_them_unanimous():
    """Three items in three clusters, three humans each. On i0 and i1 every
    human and the judge say "first". On i2 the humans say "second", "first"
    and "second", and the judge says "first".

    A draw that picks only i0 and i1 holds one label, on the headline pair
    and in the baseline alike, and that happens in about (2/3) ** 3 of
    draws. The property the test relies on is that both the headline
    interval and the baseline interval are refused.
    """
    frame = pd.DataFrame({
        "item_id": np.repeat(["i0", "i1", "i2"], 3),
        "cluster_id": np.repeat(["c0", "c1", "c2"], 3),
        "rater_id": np.tile(["h0", "h1", "h2"], 3),
        "rating": ["first"] * 6 + ["second", "first", "second"],
    })
    judge_labels = pd.Series(
        ["first", "first", "first"], index=["i0", "i1", "i2"], dtype=object,
        name="judge",
    )
    return frame, judge_labels


def judge_ahead_on_29_clusters():
    """The judge right 98% of the time and each human 72%, over 29 clusters
    of ten items.

    The property the test relies on is that the interval on the difference
    sits wholly above zero and rests on 29 clusters, one under THIN_SLICE.
    """
    return graded_items([0.72] * 29, [0.98] * 29, items_per_cluster=10, seed=0)


def judge_ahead_on_30_clusters():
    """The same design over 30 clusters of ten items.

    The property the test relies on is that the interval on the difference
    sits wholly above zero and rests on 30 clusters, exactly THIN_SLICE.
    """
    return graded_items([0.72] * 30, [0.98] * 30, items_per_cluster=10, seed=0)


def judge_ahead_on_30_clusters_with_one_unjudged():
    """judge_ahead_on_30_clusters with the judge's label removed on every
    item of c0.

    The property the test relies on is that the frame spans 30 clusters,
    the no-judge-label rule sets aside all ten items of c0, so the items
    that remain span 29, and the interval on the difference still sits
    wholly above zero.
    """
    frame, judge_labels = judge_ahead_on_30_clusters()
    judge_labels = judge_labels.astype(object)
    judge_labels.loc[pd.unique(frame.loc[frame["cluster_id"] == "c0", "item_id"])] = None
    return frame, judge_labels


def mt_bench_splits():
    """The human and gpt4_pair splits of lmsys/mt_bench_human_judgments at
    revision f7d2896, as committed under tests/data.

    tests/data/make_mt_bench_csv.py wrote them from the parquet files by
    selecting six columns. winner keeps its raw strings. Every string is
    read as it is, so no value turns into a missing one.
    """
    return tuple(
        pd.read_csv(MT_BENCH_DATA / f"mt_bench_{split}.csv",
                    keep_default_na=False, na_values=[])
        for split in ("human", "gpt4_pair")
    )


def mt_bench_baseline(human, gpt4):
    """Every MT-Bench comparison two or more humans judged, ties kept.

    Built the way analysis/q5_judge_against_human_baseline.py builds its
    units. An item is one question, one ordered pair of models and one
    turn. A human's first vote on an item is kept, and an item needs two or
    more distinct humans. The label is the winner's alphabetical position
    in the pair, "first" or "second", or "tie". gpt4_pair's "tie" and "tie
    (inconsistent)" both become "tie". The cluster is question_id. Rows
    stay in the order of the human split.

    The property the MT-Bench tests rely on is checked by
    assert_mt_bench_is_the_published_data.
    """
    def labelled(votes):
        lo = votes[["model_a", "model_b"]].min(axis=1).to_numpy()
        winner = np.where(
            votes["winner"] == "model_a", votes["model_a"],
            np.where(votes["winner"] == "model_b", votes["model_b"], TIE),
        )
        return votes.assign(
            lo=lo,
            hi=votes[["model_a", "model_b"]].max(axis=1),
            label=np.where(
                winner == TIE, TIE, np.where(winner == lo, "first", "second")
            ),
        )

    judged = labelled(gpt4)[["question_id", "lo", "hi", "turn", "label"]]
    votes = labelled(human).merge(
        judged.rename(columns={"label": "judge_label"}),
        on=["question_id", "lo", "hi", "turn"], how="left", validate="m:1",
    )
    votes["item_id"] = (
        votes["question_id"].astype(str) + "|" + votes["model_a"] + "|"
        + votes["model_b"] + "|t" + votes["turn"].astype(str)
    )
    votes = votes.drop_duplicates(["item_id", "judge"], keep="first")
    raters = votes.groupby("item_id")["judge"].nunique()
    votes = votes[votes["item_id"].isin(raters.index[raters >= 2])]
    votes = votes.reset_index(drop=True)

    frame = pd.DataFrame({
        "item_id": votes["item_id"],
        "cluster_id": votes["question_id"],
        "rater_id": votes["judge"],
        "rating": votes["label"],
    })
    judge_labels = votes.groupby("item_id", sort=False)["judge_label"].first()
    return frame, judge_labels


def assert_mt_bench_is_the_published_data(human, gpt4, frame, judge_labels):
    """The committed files are the two splits, with raw winner strings, and
    the frame built from them is the one the write-up measured.

    Ties are on both sides, 386 human labels and 193 judge labels. The 80
    clusters hold between 2 and 18 items each, and the 761 items carry
    between two and five human labels, 1,720 in all. So clusters, items and
    rows are three different units. Every item has a judge label.
    """
    assert (len(human), len(gpt4)) == (3355, 2400)
    assert set(human["winner"]) == {"model_a", "model_b", "tie"}
    assert set(gpt4["winner"]) == {
        "model_a", "model_b", "tie", "tie (inconsistent)"
    }
    labels_per_item = frame.groupby("item_id").size()
    items_per_cluster = frame.groupby("cluster_id")["item_id"].nunique()
    assert (len(frame), len(labels_per_item), len(items_per_cluster)) == (
        1720, 761, 80
    )
    assert (
        int((frame["rating"] == TIE).sum()), int((judge_labels == TIE).sum())
    ) == (386, 193)
    assert sorted(labels_per_item.unique()) == [2, 3, 4, 5]
    assert (items_per_cluster.min(), items_per_cluster.max()) == (2, 18)
    assert set(judge_labels) == {"first", "second", TIE}


# The full summaries. Written out whole and compared whole. The wording is
# new, the figures are what baseline_reference gives on each fixture, and
# the headline is what judge_validation already prints for the first human
# on each item, on the items the tie coding keeps.

SUMMARY_MT_BENCH_ALL_COMPARISONS = (
    "Judge and human agree at alpha 0.497 (95% CI: 0.448 to 0.547, nominal,"
    " 761 items). Plain accuracy is 66.9%. Against the human baseline, on "
    "761 items, the judge agrees with the human labels at alpha 0.479 and "
    "the humans agree with each other at alpha 0.478. Judge-human minus "
    "human-human is +0.001 (95% CI: -0.055 to +0.054, resampling 80 "
    "clusters). Ties count as a label of their own on both sides. The "
    "interval includes zero, so the data cannot show that the judge agrees "
    "with a human any more or less than a second human does. That does not "
    "mean it agrees equally well."
)


SUMMARY_MT_BENCH_DECISIVE_ONLY = (
    "Judge and human agree at alpha 0.761 (95% CI: 0.701 to 0.816, nominal,"
    " 468 items). Plain accuracy is 88.0%. 293 items were set aside because"
    " one side called it a tie. Against the human baseline, on 453 items, "
    "the judge agrees with the human labels at alpha 0.737 and the humans "
    "agree with each other at alpha 0.692. Judge-human minus human-human is"
    " +0.045 (95% CI: +0.002 to +0.088, resampling 78 clusters). Ties are "
    "set aside as no label on both sides. 225 baseline items were set aside"
    " because fewer than two humans gave a decisive label. 83 baseline "
    "items were set aside because the judge's label was a tie. The judge's "
    "own ties decide which items it is scored on. The whole interval sits "
    "above zero, so the judge agrees with a human more than a second human "
    "does. A judge that sits nearer the middle of the human spread than a "
    "typical human does will score this way, so this does not show that the"
    " judge is better than a human."
)


SUMMARY_INCLUDES_ZERO = (
    "Judge and human agree at alpha 0.363 (95% CI: 0.211 to 0.505, nominal,"
    " 144 items). Plain accuracy is 68.1%. Against the human baseline, on "
    "144 items, the judge agrees with the human labels at alpha 0.408 and "
    "the humans agree with each other at alpha 0.427. Judge-human minus "
    "human-human is -0.019 (95% CI: -0.106 to +0.074, resampling 24 "
    "clusters). Ties count as a label of their own on both sides. The "
    "interval includes zero, so the data cannot show that the judge agrees "
    "with a human any more or less than a second human does. That does not "
    "mean it agrees equally well."
)


SUMMARY_ABOVE_ZERO = (
    "Judge and human agree at alpha 0.409 (95% CI: 0.292 to 0.526, nominal,"
    " 240 items). Plain accuracy is 70.4%. Against the human baseline, on "
    "240 items, the judge agrees with the human labels at alpha 0.414 and "
    "the humans agree with each other at alpha 0.115. Judge-human minus "
    "human-human is +0.299 (95% CI: +0.237 to +0.357, resampling 24 "
    "clusters). Ties count as a label of their own on both sides. The whole"
    " interval sits above zero, so the judge agrees with a human more than "
    "a second human does. A judge that sits nearer the middle of the human "
    "spread than a typical human does will score this way, so this does not"
    " show that the judge is better than a human. That interval rests on "
    "only 24 clusters, so treat the verdict as provisional."
)


SUMMARY_BELOW_ZERO = (
    "Judge and human agree at alpha 0.156 (95% CI: -0.012 to 0.308, "
    "nominal, 144 items). Plain accuracy is 57.6%. Against the human "
    "baseline, on 144 items, the judge agrees with the human labels at "
    "alpha 0.153 and the humans agree with each other at alpha 0.551. "
    "Judge-human minus human-human is -0.398 (95% CI: -0.569 to -0.246, "
    "resampling 24 clusters). Ties count as a label of their own on both "
    "sides. The whole interval sits below zero, so the judge agrees with a "
    "human less than a second human does. On these items the judge is not a"
    " stand-in for a second human. That interval rests on only 24 clusters,"
    " so treat the verdict as provisional."
)


SUMMARY_OVERLAPPING_INTERVALS = (
    "Judge and human agree at alpha 0.418 (95% CI: 0.317 to 0.520, nominal,"
    " 300 items). Plain accuracy is 71.0%. Against the human baseline, on "
    "300 items, the judge agrees with the human labels at alpha 0.436 and "
    "the humans agree with each other at alpha 0.362. Judge-human minus "
    "human-human is +0.074 (95% CI: +0.036 to +0.105, resampling 10 "
    "clusters). Ties count as a label of their own on both sides. The whole"
    " interval sits above zero, so the judge agrees with a human more than "
    "a second human does. A judge that sits nearer the middle of the human "
    "spread than a typical human does will score this way, so this does not"
    " show that the judge is better than a human. That interval rests on "
    "only 10 clusters, so treat the verdict as provisional."
)


SUMMARY_TIES_AS_CATEGORY = (
    "Judge and human agree at alpha 0.103 (95% CI: -0.022 to 0.220, "
    "nominal, 144 items). Plain accuracy is 41.0%. Against the human "
    "baseline, on 144 items, the judge agrees with the human labels at "
    "alpha 0.149 and the humans agree with each other at alpha 0.122. "
    "Judge-human minus human-human is +0.027 (95% CI: -0.077 to +0.125, "
    "resampling 24 clusters). Ties count as a label of their own on both "
    "sides. The interval includes zero, so the data cannot show that the "
    "judge agrees with a human any more or less than a second human does. "
    "That does not mean it agrees equally well."
)


SUMMARY_TIES_DROPPED = (
    "Judge and human agree at alpha 0.277 (95% CI: 0.054 to 0.464, nominal,"
    " 80 items). Plain accuracy is 63.7%. 64 items were set aside because "
    "one side called it a tie. Against the human baseline, on 91 items, the"
    " judge agrees with the human labels at alpha 0.374 and the humans "
    "agree with each other at alpha 0.360. Judge-human minus human-human is"
    " +0.014 (95% CI: -0.185 to +0.194, resampling 24 clusters). Ties are "
    "set aside as no label on both sides. 30 baseline items were set aside "
    "because fewer than two humans gave a decisive label. 23 baseline items"
    " were set aside because the judge's label was a tie. The judge's own "
    "ties decide which items it is scored on. The interval includes zero, "
    "so the data cannot show that the judge agrees with a human any more or"
    " less than a second human does. That does not mean it agrees equally "
    "well."
)


SUMMARY_UNDEFINED_RESAMPLES = (
    "Judge and human agree at alpha 0.409 (95% CI: 0.066 to 0.718, nominal,"
    " 30 items). Plain accuracy is 70.0%. Against the human baseline, on 30"
    " items, the judge agrees with the human labels at alpha 0.381 and the "
    "humans agree with each other at alpha 0.546. Judge-human minus human-"
    "human is -0.165 (95% CI: -0.571 to +0.359, resampling 3 clusters). 68 "
    "of 2000 resamples left an alpha undefined, and the interval comes from"
    " the other 1932. Those are the resamples where every human label was "
    "the same, so leaving them out shifts the interval in the judge's "
    "favour. Ties count as a label of their own on both sides. The interval"
    " includes zero, so the data cannot show that the judge agrees with a "
    "human any more or less than a second human does. That does not mean it"
    " agrees equally well."
)


SUMMARY_NO_INTERVAL = (
    "Judge and human agree at alpha 0.063 (95% CI: -0.317 to 0.388, "
    "nominal, 30 items). Plain accuracy is 56.7%. Against the human "
    "baseline, on 30 items, the judge agrees with the human labels at alpha"
    " 0.070 and the humans agree with each other at alpha 0.641. Judge-"
    "human minus human-human is -0.571. Ties count as a label of their own "
    "on both sides. There is no interval on the difference, and so no "
    "verdict. 591 of 2000 resamples left an alpha undefined, above the 10% "
    "this reports through. Those are the resamples where every human label "
    "was the same, so an interval from the rest would lean in the judge's "
    "favour."
)


SUMMARY_ONE_HUMAN = (
    "Judge and human agree at alpha 0.363 (95% CI: 0.211 to 0.505, nominal,"
    " 144 items). Plain accuracy is 68.1%. Against the human baseline, on "
    "136 items, the judge agrees with the human labels at alpha 0.427 and "
    "the humans agree with each other at alpha 0.432. Judge-human minus "
    "human-human is -0.006 (95% CI: -0.097 to +0.084, resampling 24 "
    "clusters). Ties count as a label of their own on both sides. 8 "
    "baseline items were set aside because fewer than two humans graded "
    "them. The interval includes zero, so the data cannot show that the "
    "judge agrees with a human any more or less than a second human does. "
    "That does not mean it agrees equally well."
)


SUMMARY_JUDGE_SILENT = (
    "Judge and human agree at alpha 0.286 (95% CI: 0.077 to 0.479, nominal,"
    " 84 items). Plain accuracy is 64.3%. 12 items were set aside because "
    "one side had no label. Against the human baseline, on 84 items, the "
    "judge agrees with the human labels at alpha 0.348 and the humans agree"
    " with each other at alpha 0.209. Judge-human minus human-human is "
    "+0.139 (95% CI: -0.030 to +0.293, resampling 16 clusters). Ties count "
    "as a label of their own on both sides. 12 baseline items were set "
    "aside because the judge gave no label. The interval includes zero, so "
    "the data cannot show that the judge agrees with a human any more or "
    "less than a second human does. That does not mean it agrees equally "
    "well."
)


SUMMARY_ON_THE_LINE = (
    "Judge and human agree at alpha 0.781 (95% CI: -0.022 to 1.000, "
    "nominal, 24 items). Plain accuracy is 95.8%. Against the human "
    "baseline, on 24 items, the judge agrees with the human labels at alpha"
    " 0.580 and the humans agree with each other at alpha 0.376. Judge-"
    "human minus human-human is +0.204 (95% CI: -0.328 to +0.341, "
    "resampling 8 clusters). 200 of 2000 resamples left an alpha undefined,"
    " and the interval comes from the other 1800. Those are the resamples "
    "where every human label was the same, so leaving them out shifts the "
    "interval in the judge's favour. Ties count as a label of their own on "
    "both sides. The interval includes zero, so the data cannot show that "
    "the judge agrees with a human any more or less than a second human "
    "does. That does not mean it agrees equally well."
)

SUMMARY_UNDER_THE_LINE = (
    "Judge and human agree at alpha 0.781 (95% CI: -0.022 to 1.000, "
    "nominal, 24 items). Plain accuracy is 95.8%. Against the human "
    "baseline, on 24 items, the judge agrees with the human labels at alpha"
    " 0.580 and the humans agree with each other at alpha 0.376. Judge-"
    "human minus human-human is +0.204. Ties count as a label of their own "
    "on both sides. There is no interval on the difference, and so no "
    "verdict. 201 of 2000 resamples left an alpha undefined, above the 10% "
    "this reports through. Those are the resamples where every human label "
    "was the same, so an interval from the rest would lean in the judge's "
    "favour."
)

SUMMARY_ONE_OF_EACH = (
    "Judge and human agree at alpha 0.368 (95% CI: 0.206 to 0.521, nominal,"
    " 142 items). Plain accuracy is 68.3%. 1 item was set aside because one"
    " side had no label. 1 item was set aside because one side called it a "
    "tie. Against the human baseline, on 141 items, the judge agrees with "
    "the human labels at alpha 0.410 and the humans agree with each other "
    "at alpha 0.434. Judge-human minus human-human is -0.024 (95% CI: "
    "-0.111 to +0.070, resampling 24 clusters). Ties are set aside as no "
    "label on both sides. 1 baseline item was set aside because the judge "
    "gave no label. 1 baseline item was set aside because fewer than two "
    "humans gave a decisive label. 1 baseline item was set aside because "
    "the judge's label was a tie. The judge's own ties decide which items "
    "it is scored on. The interval includes zero, so the data cannot show "
    "that the judge agrees with a human any more or less than a second "
    "human does. That does not mean it agrees equally well."
)

SUMMARY_BOTH_REFUSE = (
    "Judge and human agree at alpha 0.000 (nominal, 3 items). There is no "
    "interval on that alpha. 591 of 2000 resamples left it undefined, above"
    " the 10% this reports through. Those are the resamples where every "
    "label was the same, so percentiles of the rest would understate the "
    "upper bound. Plain accuracy is 66.7%. Against the human baseline, on 3"
    " items, the judge agrees with the human labels at alpha -0.062 and the"
    " humans agree with each other at alpha 0.429. Judge-human minus human-"
    "human is -0.491. Ties count as a label of their own on both sides. "
    "There is no interval on the difference, and so no verdict. 591 of 2000"
    " resamples left an alpha undefined, above the 10% this reports "
    "through. Those are the resamples where every human label was the same,"
    " so an interval from the rest would lean in the judge's favour."
)


SUMMARY_ONE_DECISIVE_HUMAN = (
    "Judge and human agree at alpha 0.398 (95% CI: 0.211 to 0.567, nominal,"
    " 96 items). Plain accuracy is 69.8%. Against the human baseline, on 84"
    " items, the judge agrees with the human labels at alpha 0.417 and the "
    "humans agree with each other at alpha 0.302. Judge-human minus human-"
    "human is +0.116 (95% CI: -0.100 to +0.314, resampling 16 clusters). "
    "Ties are set aside as no label on both sides. 12 baseline items were "
    "set aside because fewer than two humans gave a decisive label. The "
    "interval includes zero, so the data cannot show that the judge agrees "
    "with a human any more or less than a second human does. That does not "
    "mean it agrees equally well."
)

HEADLINE_WITHOUT_RESAMPLES = (
    "Judge and human agree at alpha 0.363 (nominal, 144 items). No interval"
    " was computed, so nothing here is placed against sampling error. Plain"
    " accuracy is 68.1%. Without an interval there is nothing to place the "
    "judge's agreement with the humans against the conventional lines at "
    "0.667 and 0.800, so the data cannot show that it clears either. That "
    "does not mean it falls short of them."
)

HEADLINE_REFUSED = (
    "Judge and human agree at alpha 0.000 (nominal, 3 items). There is no "
    "interval on that alpha. 591 of 2000 resamples left it undefined, above"
    " the 10% this reports through. Those are the resamples where every "
    "label was the same, so percentiles of the rest would understate the "
    "upper bound. Plain accuracy is 66.7%. Without an interval there is "
    "nothing to place the judge's agreement with the humans against the "
    "conventional lines at 0.667 and 0.800, so the data cannot show that it"
    " clears either. That does not mean it falls short of them."
)


SUMMARY_29_CLUSTERS = (
    "Judge and human agree at alpha 0.382 (95% CI: 0.270 to 0.492, nominal,"
    " 290 items). Plain accuracy is 69.3%. Against the human baseline, on "
    "290 items, the judge agrees with the human labels at alpha 0.410 and "
    "the humans agree with each other at alpha 0.156. Judge-human minus "
    "human-human is +0.255 (95% CI: +0.202 to +0.309, resampling 29 "
    "clusters). Ties count as a label of their own on both sides. The whole"
    " interval sits above zero, so the judge agrees with a human more than "
    "a second human does. A judge that sits nearer the middle of the human "
    "spread than a typical human does will score this way, so this does not"
    " show that the judge is better than a human. That interval rests on "
    "only 29 clusters, so treat the verdict as provisional."
)

SUMMARY_30_CLUSTERS = (
    "Judge and human agree at alpha 0.418 (95% CI: 0.317 to 0.519, nominal,"
    " 300 items). Plain accuracy is 71.0%. Against the human baseline, on "
    "300 items, the judge agrees with the human labels at alpha 0.417 and "
    "the humans agree with each other at alpha 0.142. Judge-human minus "
    "human-human is +0.274 (95% CI: +0.224 to +0.323, resampling 30 "
    "clusters). Ties count as a label of their own on both sides. The whole"
    " interval sits above zero, so the judge agrees with a human more than "
    "a second human does. A judge that sits nearer the middle of the human "
    "spread than a typical human does will score this way, so this does not"
    " show that the judge is better than a human."
)


SUMMARY_30_CLUSTERS_ONE_UNJUDGED = (
    "Judge and human agree at alpha 0.432 (95% CI: 0.327 to 0.529, nominal,"
    " 290 items). Plain accuracy is 71.7%. 10 items were set aside because "
    "one side had no label. Against the human baseline, on 290 items, the "
    "judge agrees with the human labels at alpha 0.424 and the humans agree"
    " with each other at alpha 0.140. Judge-human minus human-human is "
    "+0.284 (95% CI: +0.233 to +0.331, resampling 29 clusters). Ties count "
    "as a label of their own on both sides. 10 baseline items were set "
    "aside because the judge gave no label. The whole interval sits above "
    "zero, so the judge agrees with a human more than a second human does. "
    "A judge that sits nearer the middle of the human spread than a typical"
    " human does will score this way, so this does not show that the judge "
    "is better than a human. That interval rests on only 29 clusters, so "
    "treat the verdict as provisional."
)


# baseline_reference on judge_like_a_fourth_human at seed 1. At seed 0
# the bounds are -0.10578 and 0.07356.
SEED_1_BOUNDS = (-0.11088995448993597, 0.06908082521187421)


# baseline_reference on judge_like_a_fourth_human at confidence 0.90 and
# seed 0. At 0.95 the bounds are -0.10578 and 0.07356.
CONFIDENCE_90_BOUNDS = (-0.0923096724775582, 0.0595140973148152)


def test_baseline_reproduces_the_published_mt_bench_figures():
    """The ties-as-a-label row of analysis/mt-bench.md, coded by the
    winner's alphabetical position.

    The write-up reports judge-human alpha 0.479, human-human 0.478, and a
    difference of +0.001 with a 95% interval from -0.055 to +0.054, from
    5,000 question resamples at seed 0. The fixture is built the way q5
    built it, and the published figures are the reference, at three
    decimals and at full precision from q5's run().

    These are the inputs on which a wrong implementation would still match.

    - The ties argument ignored and "tie" read as an ordinary label. Under
      ties="category" that is the same computation. The decisive-only test
      catches it, since there a tie has to leave.
    - Clusters in sorted order. The human split is sorted by question_id,
      so sorted order and first-appearance order draw the same resamples.
      The synthetic fixtures name clusters c0 to c23, where the two orders
      differ.
    - A seed ignored in favour of default_rng(0). The seed-1 test catches
      it.
    - A confidence argument ignored. Only 0.95 runs here. The
      confidence-0.90 test catches it.
    - Items without a judge label. Every item here has one. The silent-judge
      test covers them.
    - Items graded by one human. The frame is filtered to two or more
      before the call, as q5 did. The one-human test covers them.
    - A tie recognised only when spelled exactly "tie". Every tie here is
      spelled that way. The spelling test catches it.
    - A rule for undefined resamples. All 5,000 resamples define both
      alphas here. The two undefined-resample tests cover it, on each side
      of the refusal line.

    These are the inputs that do tell the implementations apart.

    - Ties dropped on both sides. That gives 453 items at 0.737 against
      0.692, a difference of +0.045 (+0.002 to +0.088).
    - Clusters equal to items or to rows. Here 80 clusters hold 761 items
      and 1,720 ratings. Resampling items gives -0.045 to +0.046, and
      resampling rows gives -0.262 to -0.180.
    - The judge added as one more rater on each item gives 0.481.
    - Human-human from the first two humans on each item gives 0.497.

    Three margins are thin. The upper bound is 0.054492, which is 0.000008
    under the point where it prints as 0.055, so a percentile rule other
    than numpy's default can move it. An n_boot of 1,000 or 2,000 in place
    of 5,000 is caught by that bound alone. And the sign rests on the third
    decimal. With the subtraction reversed the result prints as -0.001, from
    -0.054 to 0.055. The full-precision pins close all three.
    """
    human, gpt4 = mt_bench_splits()
    frame, judge_labels = mt_bench_baseline(human, gpt4)
    assert_mt_bench_is_the_published_data(human, gpt4, frame, judge_labels)
    reference = baseline_reference(
        frame, judge_labels, n_boot=MT_BENCH_BOOT, seed=MT_BENCH_SEED
    )
    assert three_decimals(reference_figures(reference)) == PUBLISHED_ALL_COMPARISONS
    assert reference_figures(reference) == pytest.approx(
        Q5_ALL_COMPARISONS, abs=1e-9
    )

    result = validate_with_baseline(
        frame, judge_labels, ties="category",
        n_boot=MT_BENCH_BOOT, seed=MT_BENCH_SEED,
    )

    assert three_decimals(baseline_figures(result)) == PUBLISHED_ALL_COMPARISONS
    assert baseline_figures(result) == pytest.approx(
        Q5_ALL_COMPARISONS, abs=1e-9
    )
    assert result.summary() == SUMMARY_MT_BENCH_ALL_COMPARISONS


def test_baseline_reproduces_the_published_decisive_only_figures():
    """The decisive-only row of analysis/mt-bench.md, coded by the winner's
    alphabetical position, under ties="drop".

    The package gets all 761 items with their ties and has to set aside the
    same ones q5 did. The too-few-humans rule takes 225 items with fewer
    than two decisive human labels, and the judge-tie rule then takes 83
    the judge called a tie, which leaves 453 items in 78 clusters. The
    write-up reports judge-human 0.737, human-human 0.692, and a difference
    of +0.045 from +0.002 to +0.088. Its separate intervals, 0.667 to 0.800
    and 0.622 to 0.757, overlap, so only the interval on the difference
    separates the two figures.

    ties="drop" also sets the headline's coding. 293 of the
    761 headline items have a tie from the first human or the judge, and
    they leave the headline too.

    These are the inputs on which a wrong implementation would still match.

    - Rule order, on the figures. Whichever rule runs first, the same 453
      items remain. The counts catch a reversed order, which reads 193
      judge ties and then 115 items with too few humans. No item here lacks
      a judge label, so the place of that rule shows only in the rule-order
      test.
    - A judge tie counted as a missing judge label. The figures are the
      same. The counts catch it, since baseline_n_judge_ties would be zero.
    - Clusters in sorted order. The 78 question_ids appear in sorted order.
    - A seed ignored in favour of default_rng(0), or a confidence argument
      ignored. The seed-1 and confidence-0.90 tests catch them.
    - A rule for undefined resamples. All 5,000 resamples define both
      alphas here.
    - Items without a judge label, and items graded by one human. Neither
      occurs here.
    - Which of gpt4_pair's two tie strings counts as a tie, and ties spelled
      with another case or spacing. Both of gpt4_pair's strings become
      "tie" before the call, so the package never sees any of these.

    These are the inputs that do tell the implementations apart.

    - Items with one decisive human label kept for judge-human. 91 of the
      225 have exactly one decisive label and a decisive judge label, and
      keeping them moves judge-human from 0.737 to 0.719.
    - The judge's tie kept as a label. That puts the 83 items back and
      moves both figures.
    - Ties kept as a category, which is the other MT-Bench test.
    - Headline ties kept as a label, or counted with the items that have no
      label. The pinned summary and n_dropped_ties catch both.

    One margin is thin. The upper bound on judge-human alpha is 0.800456,
    which is 0.000044 under the point where it prints as 0.801.
    """
    human, gpt4 = mt_bench_splits()
    frame, judge_labels = mt_bench_baseline(human, gpt4)
    assert_mt_bench_is_the_published_data(human, gpt4, frame, judge_labels)
    kept, kept_labels, counts = baseline_items(frame, judge_labels, "drop")
    assert (kept_labels.size, *counts) == (453, 0, 225, 83)
    assert kept["cluster_id"].nunique() == 78
    assert headline_ties(frame, judge_labels) == 293
    reference = baseline_reference(
        kept, kept_labels, n_boot=MT_BENCH_BOOT, seed=MT_BENCH_SEED
    )
    assert three_decimals(reference_figures(reference)) == PUBLISHED_DECISIVE_ONLY
    assert reference_figures(reference) == pytest.approx(
        Q5_DECISIVE_ONLY, abs=1e-9
    )
    judge_low, judge_high = reference["judge_human_ci"]
    human_low, human_high = reference["human_human_ci"]
    assert three_decimals((judge_low, judge_high)) == ("0.667", "0.800")
    assert three_decimals((human_low, human_high)) == ("0.622", "0.757")
    assert judge_low < human_high

    result = validate_with_baseline(
        frame, judge_labels, ties="drop",
        n_boot=MT_BENCH_BOOT, seed=MT_BENCH_SEED,
    )

    assert (
        result.baseline_n_items,
        result.baseline_n_judge_ties,
        result.baseline_n_too_few_humans,
    ) == (453, 83, 225)
    assert (result.n_dropped_ties, result.n_dropped) == (293, 0)
    assert three_decimals(baseline_figures(result)) == PUBLISHED_DECISIVE_ONLY
    assert baseline_figures(result) == pytest.approx(Q5_DECISIVE_ONLY, abs=1e-9)
    assert result.summary() == SUMMARY_MT_BENCH_DECISIVE_ONLY


def test_baseline_summary_when_the_interval_includes_zero():
    frame, judge_labels = judge_like_a_fourth_human()
    reference = baseline_reference(frame, judge_labels)
    low, high = reference["difference_ci"]
    assert low < 0 < high
    assert three_decimals(reference_figures(reference)) == (
        "0.408", "0.427", "-0.019", "-0.106", "0.074"
    )

    result = validate_with_baseline(frame, judge_labels, ties="category")

    assert result.summary() == SUMMARY_INCLUDES_ZERO


def test_baseline_summary_when_the_interval_sits_above_zero():
    frame, judge_labels = judge_more_reliable_than_the_humans()
    reference = baseline_reference(frame, judge_labels)
    assert reference["difference_ci"][0] > 0
    assert reference["judge_human_ci"][0] > reference["human_human_ci"][1]
    assert three_decimals(reference_figures(reference)) == (
        "0.414", "0.115", "0.299", "0.237", "0.357"
    )

    result = validate_with_baseline(frame, judge_labels, ties="category")

    assert result.summary() == SUMMARY_ABOVE_ZERO


def test_baseline_summary_when_the_interval_sits_below_zero():
    frame, judge_labels = judge_less_reliable_than_the_humans()
    reference = baseline_reference(frame, judge_labels)
    assert reference["difference_ci"][1] < 0
    assert three_decimals(reference_figures(reference)) == (
        "0.153", "0.551", "-0.398", "-0.569", "-0.246"
    )

    result = validate_with_baseline(frame, judge_labels, ties="category")

    assert result.summary() == SUMMARY_BELOW_ZERO


def test_baseline_interval_resamples_clusters():
    """The interval on the difference is as wide as the clusters make it.

    On this fixture the reference interval on the difference is 0.449 wide
    when clusters are drawn, 0.198 when items are, and 0.148 when rows are.
    An implementation that draws items or rows cannot print 0.449, so this
    fails whichever of the two it does.
    """
    frame, judge_labels = judge_that_guesses_in_half_the_clusters()
    clusters = baseline_reference(frame, judge_labels, unit="cluster_id")
    items = baseline_reference(frame, judge_labels, unit="item_id")
    rows = baseline_reference(frame, judge_labels, unit="row")
    assert interval_width(items) < interval_width(clusters) / 2
    assert interval_width(rows) < interval_width(clusters) / 2
    assert clusters["difference_ci"][0] < 0 < clusters["difference_ci"][1]
    assert items["difference_ci"][1] < 0
    assert rows["difference_ci"][1] < 0
    assert f"{interval_width(clusters):.3f}" == "0.449"

    result = validate_with_baseline(frame, judge_labels, ties="category")

    assert f"{result.baseline_ci_high - result.baseline_ci_low:.3f}" == "0.449"


def test_baseline_verdict_reads_the_difference():
    """Both alphas come from the same draws, so what they share cancels in
    the difference. Their separate intervals can overlap while the
    difference is well measured. The write-up makes the same point about
    leaderboard ratings. A verdict read off the overlap would call these two
    the same, and the interval on the difference says they differ.
    """
    frame, judge_labels = judge_a_little_ahead_in_every_cluster()
    reference = baseline_reference(frame, judge_labels)
    judge_low, judge_high = reference["judge_human_ci"]
    human_low, human_high = reference["human_human_ci"]
    assert judge_low < human_high
    assert human_low < judge_high
    assert reference["difference_ci"][0] > 0
    assert three_decimals(reference_figures(reference)) == (
        "0.436", "0.362", "0.074", "0.036", "0.105"
    )

    result = validate_with_baseline(frame, judge_labels, ties="category")

    assert result.summary() == SUMMARY_OVERLAPPING_INTERVALS


def test_baseline_calls_a_verdict_on_few_clusters_provisional():
    """An interval that excludes zero on fewer than THIN_SLICE clusters gets
    a sentence after the verdict saying how many clusters it rests on and
    that the verdict is provisional. On 29 clusters it appears, and on 30
    it does not.
    """
    few, few_labels = judge_ahead_on_29_clusters()
    enough, enough_labels = judge_ahead_on_30_clusters()
    on_few = baseline_reference(few, few_labels)
    on_enough = baseline_reference(enough, enough_labels)
    assert (few["cluster_id"].nunique(), enough["cluster_id"].nunique()) == (
        THIN_SLICE - 1, THIN_SLICE
    )
    assert on_few["difference_ci"][0] > 0
    assert on_enough["difference_ci"][0] > 0

    few_result = validate_with_baseline(few, few_labels, ties="category")
    enough_result = validate_with_baseline(enough, enough_labels, ties="category")

    assert few_result.summary() == SUMMARY_29_CLUSTERS
    assert enough_result.summary() == SUMMARY_30_CLUSTERS


def test_baseline_counts_clusters_after_items_are_set_aside():
    """The thin-cluster count is taken on the items that remain. This frame
    spans 30 clusters and the rules leave 29, so the summary carries the
    29-cluster sentence. Counting before the rules would give 30 and leave
    it out.
    """
    frame, judge_labels = judge_ahead_on_30_clusters_with_one_unjudged()
    kept, kept_labels, counts = baseline_items(frame, judge_labels, "category")
    assert counts == (10, 0, 0)
    assert (frame["cluster_id"].nunique(), kept["cluster_id"].nunique()) == (
        THIN_SLICE, THIN_SLICE - 1
    )
    reference = baseline_reference(kept, kept_labels)
    assert reference["difference_ci"][0] > 0

    result = validate_with_baseline(frame, judge_labels, ties="category")

    assert result.summary() == SUMMARY_30_CLUSTERS_ONE_UNJUDGED


def test_baseline_summary_names_the_tie_coding():
    """Ties are on both sides, so the two codings see different data. Each
    summary names its coding, and the drop summary gives the count each
    rule set aside and says the judge's ties pick its items.
    """
    frame, judge_labels = ties_on_both_sides()
    assert (
        int((frame["rating"] == TIE).sum()), int((judge_labels == TIE).sum())
    ) == (131, 29)
    kept, kept_labels, counts = baseline_items(frame, judge_labels, "drop")
    assert counts == (0, 30, 23)
    assert headline_ties(frame, judge_labels) == 64
    as_category = kref_figures(frame, judge_labels)
    as_dropped = kref_figures(kept, kept_labels)
    assert abs(as_category[0] - as_dropped[0]) > 0.1
    assert abs(as_category[1] - as_dropped[1]) > 0.1

    kept_result = validate_with_baseline(frame, judge_labels, ties="category")
    dropped_result = validate_with_baseline(frame, judge_labels, ties="drop")

    assert (kept_result.n_dropped_ties, dropped_result.n_dropped_ties) == (
        0, 64
    )
    assert kept_result.summary() == SUMMARY_TIES_AS_CATEGORY
    assert dropped_result.summary() == SUMMARY_TIES_DROPPED


def test_baseline_sets_aside_items_the_judge_did_not_label():
    """Twelve baseline items have no judge label. They leave both figures
    and are counted, and both alphas come from the other 84 items.
    """
    frame, judge_labels = judge_silent_on_some_items()
    silent = judge_labels.index[judge_labels.isna()]
    labelled = frame[~frame["item_id"].isin(silent)].reset_index(drop=True)
    expected = kref_figures(labelled, judge_labels.drop(silent))
    assert silent.size == 12
    assert three_decimals(
        (kref_human_human(frame), expected[1])
    ) == ("0.308", "0.209")

    result = validate_with_baseline(frame, judge_labels, ties="category")

    assert (
        result.baseline_n_no_judge_label,
        result.baseline_n_too_few_humans,
        result.baseline_n_items,
    ) == (12, 0, 84)
    assert (result.baseline_judge_human, result.baseline_human_human) == (
        pytest.approx(expected, abs=1e-9)
    )
    assert result.summary() == SUMMARY_JUDGE_SILENT


def test_baseline_drop_sets_aside_items_left_with_one_decisive_human():
    """Under ties="drop" an item with one decisive human label and one tie
    has no human pair, and it leaves judge-human alpha as well. Keeping it
    there would score the judge on items the humans never agreed on.

    This is the one drop fixture with no judge ties, so its summary is the
    one that pins the judge-ties sentence as absent.
    """
    frame, judge_labels = one_decisive_human_and_a_tie()
    one_label = frame.loc[frame["rating"] == TIE, "item_id"]
    on_them = frame[frame["item_id"].isin(one_label)]
    assert (one_label.nunique(), len(on_them)) == (12, 24)
    assert set(judge_labels) == {"first", "second"}
    expected = kref_figures(
        frame[~frame["item_id"].isin(one_label)].reset_index(drop=True),
        judge_labels.drop(one_label),
    )
    kept_anyway = kref_figures(
        frame[frame["rating"] != TIE].reset_index(drop=True), judge_labels
    )
    assert three_decimals((expected[0], kept_anyway[0])) == ("0.417", "0.323")
    assert kept_anyway[1] == pytest.approx(expected[1], abs=1e-12)

    result = validate_with_baseline(frame, judge_labels, ties="drop")

    assert (
        result.baseline_n_items,
        result.baseline_n_too_few_humans,
        result.baseline_n_judge_ties,
    ) == (84, 12, 0)
    assert (result.baseline_judge_human, result.baseline_human_human) == (
        pytest.approx(expected, abs=1e-9)
    )
    assert result.summary() == SUMMARY_ONE_DECISIVE_HUMAN


def test_baseline_drops_resamples_that_leave_an_alpha_undefined():
    """A draw that leaves either alpha undefined leaves the interval, and
    the summary says how many did and which way that pushes the interval.

    Here only human-human is undefined on those draws. An implementation
    that dropped a draw only when both alphas were undefined would keep
    them, and a NaN in the difference would reach the percentiles.
    """
    frame, judge_labels = one_cluster_where_every_human_says_first()
    reference = baseline_reference(frame, judge_labels)
    assert (
        reference["n_undefined"],
        reference["n_human_human_undefined"],
        reference["n_judge_human_undefined"],
    ) == (68, 68, 0)
    assert reference["n_usable"] / BASELINE_BOOT >= USABLE_SHARE

    result = validate_with_baseline(frame, judge_labels, ties="category")

    assert result.baseline_n_boot_usable == 1932
    assert result.summary() == SUMMARY_UNDEFINED_RESAMPLES


def test_baseline_refuses_an_interval_past_the_usable_share():
    """Past the usable share there is no interval on the difference and no
    verdict, and the summary says why. The point figures still stand.
    """
    frame, judge_labels = two_clusters_where_every_human_says_first()
    reference = baseline_reference(frame, judge_labels)
    assert (
        reference["n_undefined"],
        reference["n_human_human_undefined"],
        reference["n_judge_human_undefined"],
    ) == (591, 591, 0)
    assert reference["n_usable"] / BASELINE_BOOT < USABLE_SHARE

    result = validate_with_baseline(frame, judge_labels, ties="category")

    assert result.baseline_n_boot_usable == 1409
    assert np.isnan(result.baseline_ci_low)
    assert np.isnan(result.baseline_ci_high)
    assert result.summary() == SUMMARY_NO_INTERVAL


def test_baseline_bootstrap_follows_the_seed():
    """One call at seed 1. The bounds come from baseline_reference at seed
    1, and they differ from the seed-0 bounds, so an implementation that
    ignores the seed fails here.
    """
    frame, judge_labels = judge_like_a_fourth_human()
    at_seed_0 = baseline_reference(frame, judge_labels, seed=0)
    at_seed_1 = baseline_reference(frame, judge_labels, seed=1)
    assert at_seed_1["difference_ci"] == pytest.approx(SEED_1_BOUNDS, abs=1e-9)
    assert abs(at_seed_1["difference_ci"][0] - at_seed_0["difference_ci"][0]) > 1e-3
    assert abs(at_seed_1["difference_ci"][1] - at_seed_0["difference_ci"][1]) > 1e-3

    result = validate_with_baseline(frame, judge_labels, ties="category", seed=1)

    assert (result.baseline_ci_low, result.baseline_ci_high) == pytest.approx(
        SEED_1_BOUNDS, abs=1e-9
    )


def test_baseline_interval_follows_the_confidence_level():
    """One call at confidence=0.90. The bounds come from baseline_reference
    at 0.90, and they sit inside the 0.95 bounds, so an implementation that
    ignores the confidence level fails here.
    """
    frame, judge_labels = judge_like_a_fourth_human()
    at_95 = baseline_reference(frame, judge_labels)["difference_ci"]
    at_90 = baseline_reference(frame, judge_labels, confidence=0.90)
    assert at_90["difference_ci"] == pytest.approx(CONFIDENCE_90_BOUNDS, abs=1e-9)
    assert at_95[0] < at_90["difference_ci"][0]
    assert at_90["difference_ci"][1] < at_95[1]

    result = validate_with_baseline(
        frame, judge_labels, ties="category", confidence=0.90
    )

    assert (result.baseline_ci_low, result.baseline_ci_high) == pytest.approx(
        CONFIDENCE_90_BOUNDS, abs=1e-9
    )


def test_baseline_sets_aside_items_graded_by_one_human():
    """Under ties="category" an item with one human label has no human pair,
    and it leaves judge-human alpha as well. It counts as too few humans.
    """
    frame, judge_labels = some_items_graded_by_one_human()
    labels_per_item = frame.groupby("item_id").size()
    assert int((labels_per_item == 1).sum()) == 8
    kept, kept_labels, counts = baseline_items(frame, judge_labels, "category")
    assert counts == (0, 8, 0)
    expected = kref_figures(kept, kept_labels)
    kept_anyway = kref_figures(frame, judge_labels)
    assert three_decimals((expected[0], kept_anyway[0])) == (
        "0.427", "0.419"
    )
    assert kept_anyway[1] == pytest.approx(expected[1], abs=1e-12)

    result = validate_with_baseline(frame, judge_labels, ties="category")

    assert (
        result.baseline_n_no_judge_label,
        result.baseline_n_too_few_humans,
        result.baseline_n_items,
    ) == (0, 8, 136)
    assert (result.baseline_judge_human, result.baseline_human_human) == (
        pytest.approx(expected, abs=1e-9)
    )
    assert result.summary() == SUMMARY_ONE_HUMAN


def test_baseline_counts_each_item_under_the_first_rule_it_fails():
    """Under ties="drop", four items have no judge label and too few
    decisive humans, and they count as having no judge label. Six items
    have too few decisive humans and a judge tie, and they count as too few
    humans. Any other order of the rules moves one of those groups.
    """
    frame, judge_labels = judge_silent_among_ties()
    decisive = frame[frame["rating"] != TIE].groupby("item_id")["rater_id"]
    too_few = (
        decisive.nunique().reindex(judge_labels.index, fill_value=0) < 2
    ).to_numpy()
    silent = judge_labels.isna().to_numpy()
    tied = (judge_labels == TIE).to_numpy()
    assert (int((silent & too_few).sum()), int((too_few & tied).sum())) == (4, 6)
    assert baseline_items(frame, judge_labels, "drop")[2] == (4, 26, 23)

    result = validate_with_baseline(frame, judge_labels, ties="drop")

    assert (
        result.baseline_n_no_judge_label,
        result.baseline_n_too_few_humans,
        result.baseline_n_judge_ties,
        result.baseline_n_items,
    ) == (4, 26, 23, 91)


def test_baseline_reads_ties_whatever_their_case_and_spacing():
    """Ties spelled "Tie" and " tie " are ties. Under ties="drop" they are
    set aside, headline and baseline alike. Under ties="category" every
    figure equals the figure on the same data with each tie spelled "tie".
    """
    respelled, respelled_labels = ties_spelled_two_ways()
    plain, plain_labels = ties_on_both_sides()
    human_spellings = set(respelled.loc[is_tie(respelled["rating"]), "rating"])
    judge_spellings = set(respelled_labels[is_tie(respelled_labels)])
    assert human_spellings == judge_spellings == {"Tie", " tie "}
    assert TIE not in set(respelled["rating"]) | set(respelled_labels)
    assert three_decimals(kref_figures(respelled, respelled_labels)) == (
        "0.144", "0.071"
    )
    assert three_decimals(kref_figures(plain, plain_labels)) == ("0.149", "0.122")
    assert headline_ties(respelled, respelled_labels) == 64

    dropped = validate_with_baseline(respelled, respelled_labels, ties="drop")
    kept = validate_with_baseline(respelled, respelled_labels, ties="category")
    plain_dropped = validate_with_baseline(plain, plain_labels, ties="drop")
    plain_kept = validate_with_baseline(plain, plain_labels, ties="category")

    assert (
        dropped.n_dropped_ties,
        dropped.baseline_n_too_few_humans,
        dropped.baseline_n_judge_ties,
        dropped.baseline_n_items,
    ) == (64, 30, 23, 91)
    assert every_figure(dropped) == pytest.approx(
        every_figure(plain_dropped), abs=1e-12
    )
    assert every_figure(kept) == pytest.approx(
        every_figure(plain_kept), abs=1e-12
    )


def test_baseline_keeps_the_interval_at_exactly_the_usable_share():
    """1,800 usable draws of 2,000 is exactly 90%, which keeps the interval
    and the verdict. 1,799 is under it and refuses both. Same data, two
    seeds, so nothing but the count of usable draws differs.
    """
    frame, judge_labels = six_clusters_where_everyone_says_first()
    clusters = frame.groupby("cluster_id", sort=False)
    labels_in = clusters["rating"].agg(lambda r: frozenset(r))
    judged_in = clusters["item_id"].agg(
        lambda items: frozenset(judge_labels.loc[pd.unique(items)])
    )
    assert list(labels_in.index) == [f"c{c}" for c in range(8)]
    assert set(labels_in.iloc[:6]) == set(judged_in.iloc[:6]) == {
        frozenset({"first"})
    }
    assert list(labels_in.iloc[6:]) == [frozenset({"first", "second"})] * 2
    on_the_line = baseline_reference(frame, judge_labels, seed=1)
    under_the_line = baseline_reference(frame, judge_labels, seed=20)
    assert (on_the_line["n_usable"], under_the_line["n_usable"]) == (1800, 1799)
    assert (
        on_the_line["n_usable"]
        >= USABLE_SHARE * BASELINE_BOOT
        > under_the_line["n_usable"]
    )

    kept = validate_with_baseline(frame, judge_labels, ties="category", seed=1)
    refused = validate_with_baseline(frame, judge_labels, ties="category", seed=20)

    assert (kept.baseline_n_boot_usable, refused.baseline_n_boot_usable) == (
        1800, 1799
    )
    assert kept.summary() == SUMMARY_ON_THE_LINE
    assert refused.summary() == SUMMARY_UNDER_THE_LINE


def test_baseline_ignores_a_row_with_a_missing_rating():
    """A missing rating is no label. Under each coding every figure, count
    and summary is the one the tie fixture gives without the extra row.
    """
    frame, judge_labels = ties_with_a_missing_rating()
    plain, plain_labels = ties_on_both_sides()
    added = frame.iloc[len(plain):]
    assert (len(added), added["rater_id"].tolist()) == (1, ["h3"])
    assert added["rating"].isna().all()
    assert (frame.iloc[:len(plain)].to_numpy() == plain.to_numpy()).all()
    as_label = frame.assign(rating=frame["rating"].fillna("missing"))
    as_tie = frame.assign(rating=frame["rating"].fillna(TIE))
    assert baseline_items(frame, judge_labels, "drop")[2] == (0, 30, 23)
    assert baseline_items(as_label, judge_labels, "drop")[2] == (0, 29, 23)
    assert three_decimals(
        (kref_human_human(plain), kref_human_human(as_label),
         kref_human_human(as_tie))
    ) == ("0.122", "0.121", "0.124")

    kept = validate_with_baseline(frame, judge_labels, ties="category")
    dropped = validate_with_baseline(frame, judge_labels, ties="drop")
    plain_kept = validate_with_baseline(plain, plain_labels, ties="category")
    plain_dropped = validate_with_baseline(plain, plain_labels, ties="drop")

    assert kept.summary() == SUMMARY_TIES_AS_CATEGORY
    assert dropped.summary() == SUMMARY_TIES_DROPPED
    assert (
        kept.baseline_n_too_few_humans, kept.baseline_n_items,
        dropped.baseline_n_too_few_humans, dropped.baseline_n_judge_ties,
        dropped.baseline_n_items,
    ) == (0, 144, 30, 23, 91)
    assert every_figure(kept) == pytest.approx(
        every_figure(plain_kept), abs=1e-12
    )
    assert every_figure(dropped) == pytest.approx(
        every_figure(plain_dropped), abs=1e-12
    )


def test_baseline_counts_no_label_before_a_headline_tie():
    """A headline item with no label on one side and a tie on the other
    counts in n_dropped, once, and not in n_dropped_ties.
    """
    frame, judge_labels = headline_items_with_a_tie_and_no_label()
    human, judge, item_ids = headline_inputs(frame, judge_labels)
    tie_then_none = int(np.flatnonzero(item_ids == "i1")[0])
    none_then_tie = int(np.flatnonzero(item_ids == "i4")[0])
    assert human[tie_then_none] == TIE
    assert pd.isna(judge[tie_then_none])
    assert pd.isna(human[none_then_tie])
    assert judge[none_then_tie] == TIE
    missing = (pd.Series(human).isna() | pd.Series(judge).isna()).to_numpy()
    assert list(item_ids[missing]) == ["i1", "i4"]
    assert headline_ties(frame, judge_labels) == 62

    result = validate_with_baseline(frame, judge_labels, ties="drop")

    assert (result.n_dropped, result.n_dropped_ties) == (2, 62)


def test_baseline_summary_counts_one_item_in_the_singular():
    frame, judge_labels = one_item_under_each_rule()
    assert baseline_items(frame, judge_labels, "drop")[2] == (1, 1, 1)
    assert int(judge_labels.isna().sum()) == 1
    assert headline_ties(frame, judge_labels) == 1

    result = validate_with_baseline(frame, judge_labels, ties="drop")

    assert (result.n_dropped, result.n_dropped_ties) == (1, 1)
    assert result.summary() == SUMMARY_ONE_OF_EACH


def test_baseline_summary_when_both_intervals_refuse():
    frame, judge_labels = three_items_two_of_them_unanimous()
    human, judge, _ = headline_inputs(frame, judge_labels)
    headline_alone = judge_validation(
        human, judge, n_boot=BASELINE_BOOT, seed=BASELINE_SEED
    )
    reference = baseline_reference(frame, judge_labels)
    assert headline_alone.n_boot_usable / BASELINE_BOOT < USABLE_SHARE
    assert reference["n_usable"] / BASELINE_BOOT < USABLE_SHARE

    result = validate_with_baseline(frame, judge_labels, ties="category")

    assert result.summary() == SUMMARY_BOTH_REFUSE


# The headline alone, with no baseline. Its head has two ways to have no
# interval. With n_boot=0 nothing was drawn, and the sentence is the one
# judge_validation prints today. When the draws were made and too few of
# them were usable, the head says how many were undefined and which way
# that biases an interval from the rest, in the form of the baseline's
# refusal sentence.

def test_headline_summary_without_resamples():
    """n_boot=0 draws nothing. This passes on the code as it stands."""
    frame, judge_labels = judge_like_a_fourth_human()
    human, judge, _ = headline_inputs(frame, judge_labels)
    assert (pd.Series(human).isna() | pd.Series(judge).isna()).sum() == 0

    result = judge_validation(human, judge, n_boot=0)

    assert (result.n_boot, result.n_boot_usable) == (0, 0)
    assert result.summary() == HEADLINE_WITHOUT_RESAMPLES


def test_headline_summary_when_its_interval_is_refused():
    """The three-item fixture on its own. A draw that picks only i0 and i1
    holds one label on both sides, so alpha is undefined there, and 591 of
    2,000 draws do that. The count is taken here from the same index matrix
    the package draws, apart from the package.
    """
    frame, judge_labels = three_items_two_of_them_unanimous()
    human, judge, _ = headline_inputs(frame, judge_labels)
    draws = np.random.default_rng(BASELINE_SEED).integers(
        0, 3, size=(BASELINE_BOOT, 3)
    )
    only_unanimous_items = (draws < 2).all(axis=1)
    assert int(only_unanimous_items.sum()) == 591
    assert (BASELINE_BOOT - 591) / BASELINE_BOOT < USABLE_SHARE

    result = judge_validation(human, judge, n_boot=BASELINE_BOOT, seed=BASELINE_SEED)

    assert (result.n_boot, result.n_boot_usable) == (BASELINE_BOOT, 1409)
    assert result.summary() == HEADLINE_REFUSED


# Refusals. Each message is compared whole.

def test_baseline_refuses_a_baseline_without_ties():
    frame, judge_labels = judge_like_a_fourth_human()
    human, judge, item_ids = headline_inputs(frame, judge_labels)

    with pytest.raises(ValueError) as excinfo:
        judge_validation(human, judge, item_ids=item_ids, human_baseline=frame)

    assert str(excinfo.value) == (
        "ties must be 'category' or 'drop' when human_baseline is given, got "
        "None. The two codings give different alphas, so the choice has to be "
        "made and stated."
    )


def test_baseline_refuses_ties_without_a_baseline():
    frame, judge_labels = judge_like_a_fourth_human()
    human, judge, _ = headline_inputs(frame, judge_labels)

    with pytest.raises(ValueError) as excinfo:
        judge_validation(human, judge, ties="category")

    assert str(excinfo.value) == (
        "ties applies only with a human_baseline, got ties='category' and no "
        "human_baseline."
    )


def test_baseline_refuses_item_ids_without_a_baseline():
    frame, judge_labels = judge_like_a_fourth_human()
    human, judge, item_ids = headline_inputs(frame, judge_labels)

    with pytest.raises(ValueError) as excinfo:
        judge_validation(human, judge, item_ids=item_ids)

    assert str(excinfo.value) == (
        "item_ids applies only with a human_baseline, got item_ids and no "
        "human_baseline."
    )


def test_baseline_refuses_repeated_item_ids():
    """Item i5's rows leave the baseline, so every baseline item is still
    named in item_ids and the repeated id is the only defect."""
    frame, judge_labels = judge_like_a_fourth_human()
    human, judge, item_ids = headline_inputs(frame, judge_labels)
    item_ids = item_ids.copy()
    item_ids[5] = item_ids[3]
    baseline = frame[frame["item_id"] != "i5"].reset_index(drop=True)
    assert (item_ids[3], item_ids[5]) == ("i3", "i3")
    assert set(baseline["item_id"]) <= set(item_ids)

    with pytest.raises(ValueError) as excinfo:
        judge_validation(
            human, judge, item_ids=item_ids, human_baseline=baseline,
            ties="category",
        )

    assert str(excinfo.value) == (
        "item_ids must name each item once, got 'i3' more than once. The "
        "judge's label on a baseline item is found by its id."
    )


def test_baseline_refuses_a_baseline_item_that_item_ids_does_not_name():
    """Items i5 and i9 are in the baseline and not in item_ids. The message
    names i5, the first of them in the baseline's order."""
    frame, judge_labels = judge_like_a_fourth_human()
    human, judge, item_ids = headline_inputs(
        frame, judge_labels.drop(["i5", "i9"])
    )
    assert set(frame["item_id"]) - set(item_ids) == {"i5", "i9"}
    assert len(set(item_ids)) == len(item_ids)

    with pytest.raises(ValueError) as excinfo:
        judge_validation(
            human, judge, item_ids=item_ids, human_baseline=frame,
            ties="category",
        )

    assert str(excinfo.value) == (
        "human_baseline names item 'i5', and item_ids does not. The usual "
        "cause is ids of different types, such as 5 in one and '5' in the "
        "other. An item the judge never graded still goes in item_ids, with "
        "no judge label."
    )


def test_baseline_refuses_a_rater_grading_one_item_twice():
    frame, judge_labels = judge_like_a_fourth_human()
    human, judge, item_ids = headline_inputs(frame, judge_labels)
    repeat = frame[(frame["item_id"] == "i0") & (frame["rater_id"] == "h1")]
    doubled = pd.concat([frame, repeat], ignore_index=True)
    assert int(doubled.duplicated(["item_id", "rater_id"]).sum()) == 1

    with pytest.raises(ValueError) as excinfo:
        judge_validation(
            human, judge, item_ids=item_ids, human_baseline=doubled,
            ties="category",
        )

    assert str(excinfo.value) == (
        "human_baseline has duplicate rater/item pairs, starting with item "
        "'i0' rated twice by 'h1'. Two rows for one rater on one item count "
        "that rater twice. Keep one row per rater and item."
    )


def test_baseline_refuses_levels_other_than_nominal():
    """Numeric labels, so the headline is valid at every level and only the
    baseline can be refused. The headline alone runs at both levels first,
    on every item."""
    frame, judge_labels = judge_like_a_fourth_human()
    as_number = {"first": 1, "second": 2}
    frame = frame.assign(rating=frame["rating"].map(as_number))
    judge_labels = judge_labels.map(as_number)
    human, judge, item_ids = headline_inputs(frame, judge_labels)

    ordinal_headline = judge_validation(human, judge, level="ordinal", n_boot=0)
    interval_headline = judge_validation(human, judge, level="interval", n_boot=0)
    assert (ordinal_headline.n_items, ordinal_headline.n_dropped) == (144, 0)
    assert (interval_headline.n_items, interval_headline.n_dropped) == (144, 0)

    with pytest.raises(ValueError) as ordinal:
        judge_validation(
            human, judge, level="ordinal", item_ids=item_ids,
            human_baseline=frame, ties="category",
        )
    with pytest.raises(ValueError) as interval:
        judge_validation(
            human, judge, level="interval", item_ids=item_ids,
            human_baseline=frame, ties="category",
        )

    assert str(ordinal.value) == (
        "human_baseline supports level='nominal' only, got level='ordinal'. "
        "Ordinal and interval baselines are not implemented yet."
    )
    assert str(interval.value) == (
        "human_baseline supports level='nominal' only, got level='interval'. "
        "Ordinal and interval baselines are not implemented yet."
    )


# --------------------------------------------------------------------------
# judge_validation against a human baseline, the branches the tests above
# do not reach
#
# Mutation testing of the implementation found five branches that no test
# above reaches. The first five tests here pin them as the code has them.
# They are a refusal for a baseline with no item_ids, a refusal for item_ids
# of the wrong length, "graded it" when exactly one item has too few humans,
# the refusal when the items that remain sit in one cluster, and the
# summary at n_boot=0.
#
# The rest are written before the code and fail until it changes. When
# human-human alpha is undefined, the summary names the cause, in the words
# JudgeValidation._undefined uses for its own two causes. A human_baseline
# that is not a DataFrame, lacks a column or is empty is refused in the
# words bradley_terry uses for the same three. An item whose rows carry more
# than one cluster_id is refused, because the interval moves an item with
# its cluster and such an item has no one cluster to move with.
# --------------------------------------------------------------------------

def one_item_graded_by_one_human():
    """judge_like_a_fourth_human with h1 and h2 removed from i0, so i0 holds
    one human label and every other item holds three.

    The property the test relies on is that under ties="category" the
    too-few-humans rule sets aside exactly one item and no other rule sets
    anything aside. Its sentence is the one place a summary says "graded
    it".
    """
    frame, judge_labels = judge_like_a_fourth_human()
    thinned = (frame["item_id"] == "i0") & (frame["rater_id"] != "h0")
    return frame[~thinned].reset_index(drop=True), judge_labels


def judge_silent_on_a_whole_cluster():
    """Two clusters of 20 items, humans and judge right 80% of the time, and
    no judge label on any item of c0.

    The property the test relies on is that the frame spans two clusters,
    the no-judge-label rule sets aside all 20 items of c0, and the 20 items
    that remain sit in c1 alone. Both alphas are defined on those items,
    judge-human 0.502 and human-human 0.631, so the difference has a value
    and only the interval is missing.
    """
    frame, judge_labels = graded_items(
        [0.8, 0.8], [0.8, 0.8], items_per_cluster=20, seed=3
    )
    judge_labels = judge_labels.astype(object)
    judge_labels.loc[
        pd.unique(frame.loc[frame["cluster_id"] == "c0", "item_id"])
    ] = None
    return frame, judge_labels


def both_alphas_defined():
    """judge_like_a_fourth_human, for a call with n_boot=0.

    The property the test relies on is that both alphas are defined on its
    144 items, judge-human 0.408 and human-human 0.427, so the difference
    has a value and the interval is the only thing missing.
    """
    return judge_like_a_fourth_human()


def one_item_left_with_humans_to_compare():
    """judge_like_a_fourth_human with h1 and h2 kept only on i0, so i0 is
    the one item with more than one human label.

    The property the test relies on is that under ties="category" the
    too-few-humans rule sets aside the other 143 items and one item
    remains. Its three humans do not all agree, so one item is the only
    reason human-human alpha is undefined. The headline pairs h0 with the
    judge on all 144 items and has an alpha of its own.
    """
    frame, judge_labels = judge_like_a_fourth_human()
    thinned = (frame["item_id"] != "i0") & (frame["rater_id"] != "h0")
    return frame[~thinned].reset_index(drop=True), judge_labels


def every_human_says_first():
    """Three clusters of ten items, every human label "first", and the
    judge alternating between "first" and "second".

    The property the test relies on is that every item keeps three human
    labels, so no rule sets anything aside, and every human label is the
    same, so identical labels are the only reason human-human alpha is
    undefined. The judge's labels vary, so the headline has an alpha.
    """
    return clusters_where_every_human_says_first(["c0", "c1", "c2"])


def items_in_two_clusters():
    """judge_like_a_fourth_human with h2's row on i8 moved to c5 and h1's
    row on i12 moved to c7.

    i8 sits in c1 and i12 in c2, so each of the two now has rows in two
    clusters. The property the test relies on is that they are the only
    such items, and that i8 comes first in the baseline's order while i12
    comes first in sorted order. Nothing else about the call is wrong.
    """
    frame, judge_labels = judge_like_a_fourth_human()
    frame = frame.copy()
    frame.loc[(frame["item_id"] == "i8") & (frame["rater_id"] == "h2"),
              "cluster_id"] = "c5"
    frame.loc[(frame["item_id"] == "i12") & (frame["rater_id"] == "h1"),
              "cluster_id"] = "c7"
    return frame, judge_labels


def every_item_left_with_one_decisive_human():
    """judge_like_a_fourth_human with h1 and h2 saying tie on every item.

    The property the test relies on is that under ties="drop" each item
    keeps one decisive human label, h0's, so the too-few-humans rule sets
    aside all 144 items and no baseline item remains. Neither h0 nor the
    judge ever says tie, so the headline keeps all 144 of its items.
    """
    frame, judge_labels = judge_like_a_fourth_human()
    frame = frame.copy()
    frame.loc[frame["rater_id"] != "h0", "rating"] = TIE
    return frame, judge_labels


# As the code has them now. The figures are what baseline_reference and
# kref_figures give on each fixture.

SUMMARY_ONE_ITEM_TOO_FEW_HUMANS = (
    "Judge and human agree at alpha 0.363 (95% CI: 0.211 to 0.505, nominal,"
    " 144 items). Plain accuracy is 68.1%. Against the human baseline, on "
    "143 items, the judge agrees with the human labels at alpha 0.408 and "
    "the humans agree with each other at alpha 0.432. Judge-human minus "
    "human-human is -0.024 (95% CI: -0.110 to +0.069, resampling 24 "
    "clusters). Ties count as a label of their own on both sides. 1 "
    "baseline item was set aside because fewer than two humans graded it. "
    "The interval includes zero, so the data cannot show that the judge "
    "agrees with a human any more or less than a second human does. That "
    "does not mean it agrees equally well."
)

SUMMARY_ONE_CLUSTER = (
    "Judge and human agree at alpha 0.594 (95% CI: 0.188 to 0.900, nominal,"
    " 20 items). Plain accuracy is 80.0%. 20 items were set aside because "
    "one side had no label. Against the human baseline, on 20 items, the "
    "judge agrees with the human labels at alpha 0.502 and the humans agree"
    " with each other at alpha 0.631. Judge-human minus human-human is "
    "-0.129. Ties count as a label of their own on both sides. 20 baseline "
    "items were set aside because the judge gave no label. There is no "
    "interval on the difference, and so no verdict. The items that remain "
    "sit in one cluster, and resampling one cluster draws the same items "
    "every time."
)

SUMMARY_BASELINE_WITHOUT_RESAMPLES = (
    "Judge and human agree at alpha 0.363 (nominal, 144 items). No interval"
    " was computed, so nothing here is placed against sampling error. Plain"
    " accuracy is 68.1%. Against the human baseline, on 144 items, the "
    "judge agrees with the human labels at alpha 0.408 and the humans agree"
    " with each other at alpha 0.427. Judge-human minus human-human is "
    "-0.019. Ties count as a label of their own on both sides. No interval "
    "was computed on the difference, so there is no verdict."
)

# Written before the code. The headline and the sentences after the first
# baseline sentence are what the code says now. The first baseline
# sentence names the cause in the words JudgeValidation._undefined uses.

SUMMARY_UNDEFINED_ON_ONE_ITEM = (
    "Judge and human agree at alpha 0.363 (95% CI: 0.211 to 0.505, nominal,"
    " 144 items). Plain accuracy is 68.1%. Against the human baseline, "
    "human-human agreement is undefined (nominal, 1 item). One item cannot "
    "carry a reliability estimate, so there is no difference to report and "
    "no interval around it. Have two or more humans grade more items and "
    "run this again. Ties count as a label of their own on both sides. 143 "
    "baseline items were set aside because fewer than two humans graded "
    "them."
)

SUMMARY_UNDEFINED_ON_IDENTICAL_LABELS = (
    "Judge and human agree at alpha -0.311 (95% CI: -0.475 to -0.180, "
    "nominal, 30 items). Plain accuracy is 50.0%. Against the human "
    "baseline, human-human agreement is undefined (nominal, 30 items). "
    "Every human label that could be compared was identical, so there is no"
    " disagreement to divide by and no difference to report. This comes "
    "from humans who used one label throughout. It is not perfect "
    "agreement. Ties count as a label of their own on both sides."
)

SUMMARY_NO_ITEM_REMAINS = (
    "Judge and human agree at alpha 0.363 (95% CI: 0.211 to 0.505, nominal,"
    " 144 items). Plain accuracy is 68.1%. Against the human baseline, "
    "every item was set aside, so there is no difference to report and no "
    "verdict. Ties are set aside as no label on both sides. 144 baseline "
    "items were set aside because fewer than two humans gave a decisive "
    "label."
)


# Pinned as the code has them now.

def test_baseline_refuses_a_baseline_without_item_ids():
    """The same call with item_ids runs, so the missing item_ids is the only
    defect."""
    frame, judge_labels = judge_like_a_fourth_human()
    human, judge, item_ids = headline_inputs(frame, judge_labels)
    judge_validation(
        human, judge, item_ids=item_ids, human_baseline=frame,
        ties="category", n_boot=0,
    )

    with pytest.raises(ValueError) as excinfo:
        judge_validation(human, judge, human_baseline=frame, ties="category")

    assert str(excinfo.value) == (
        "item_ids is required with a human_baseline. It names the item at "
        "each position of human and judge, which is how the judge's label on "
        "each baseline item is found."
    )


def test_baseline_refuses_item_ids_of_the_wrong_length():
    """One id too many, added at the end. Every baseline item is still named
    once, so the length is the only defect."""
    frame, judge_labels = judge_like_a_fourth_human()
    human, judge, item_ids = headline_inputs(frame, judge_labels)
    item_ids = np.append(item_ids, "i999")
    assert (len(item_ids), len(human)) == (145, 144)
    assert set(frame["item_id"]) <= set(item_ids)
    assert len(set(item_ids)) == len(item_ids)

    with pytest.raises(ValueError) as excinfo:
        judge_validation(
            human, judge, item_ids=item_ids, human_baseline=frame,
            ties="category",
        )

    assert str(excinfo.value) == (
        "item_ids must be the same length as human and judge, got 145 and 144"
    )


def test_baseline_summary_says_graded_it_for_one_item():
    frame, judge_labels = one_item_graded_by_one_human()
    labels_per_item = frame.groupby("item_id", sort=False).size()
    assert list(labels_per_item.index[labels_per_item != 3]) == ["i0"]
    assert labels_per_item["i0"] == 1
    kept, kept_labels, counts = baseline_items(frame, judge_labels, "category")
    assert counts == (0, 1, 0)
    reference = baseline_reference(kept, kept_labels)
    assert three_decimals(reference_figures(reference)) == (
        "0.408", "0.432", "-0.024", "-0.110", "0.069"
    )

    result = validate_with_baseline(frame, judge_labels, ties="category")

    assert result.summary() == SUMMARY_ONE_ITEM_TOO_FEW_HUMANS


def test_baseline_refuses_an_interval_on_one_cluster():
    """Resampling one cluster draws the same items every time, so an
    interval would have no width and a verdict would rest on nothing."""
    frame, judge_labels = judge_silent_on_a_whole_cluster()
    kept, kept_labels, counts = baseline_items(frame, judge_labels, "category")
    assert counts == (20, 0, 0)
    assert frame["cluster_id"].nunique() == 2
    assert list(pd.unique(kept["cluster_id"])) == ["c1"]
    assert three_decimals(kref_figures(kept, kept_labels)) == ("0.502", "0.631")

    result = validate_with_baseline(frame, judge_labels, ties="category")

    assert np.isnan(result.baseline_ci_low)
    assert np.isnan(result.baseline_ci_high)
    assert result.summary() == SUMMARY_ONE_CLUSTER


def test_baseline_summary_without_resamples():
    frame, judge_labels = both_alphas_defined()
    assert three_decimals(kref_figures(frame, judge_labels)) == (
        "0.408", "0.427"
    )

    result = validate_with_baseline(
        frame, judge_labels, ties="category", n_boot=0
    )

    assert np.isnan(result.baseline_ci_low)
    assert np.isnan(result.baseline_ci_high)
    assert result.summary() == SUMMARY_BASELINE_WITHOUT_RESAMPLES


# Written before the code. These fail until it changes.

def test_baseline_summary_names_one_item_as_the_cause():
    frame, judge_labels = one_item_left_with_humans_to_compare()
    kept, kept_labels, counts = baseline_items(frame, judge_labels, "category")
    assert counts == (0, 143, 0)
    assert list(kept_labels.index) == ["i0"]
    assert kept["rating"].nunique() == 2

    result = validate_with_baseline(frame, judge_labels, ties="category")

    assert result.summary() == SUMMARY_UNDEFINED_ON_ONE_ITEM


def test_baseline_summary_names_identical_labels_as_the_cause():
    frame, judge_labels = every_human_says_first()
    kept, kept_labels, counts = baseline_items(frame, judge_labels, "category")
    assert counts == (0, 0, 0)
    assert (frame.groupby("item_id").size() == 3).all()
    assert set(frame["rating"]) == {"first"}
    assert set(judge_labels) == {"first", "second"}

    result = validate_with_baseline(frame, judge_labels, ties="category")

    assert result.summary() == SUMMARY_UNDEFINED_ON_IDENTICAL_LABELS


def test_baseline_summary_when_no_item_remains():
    frame, judge_labels = every_item_left_with_one_decisive_human()
    h0 = frame["rater_id"] == "h0"
    assert set(frame.loc[~h0, "rating"]) == {TIE}
    assert TIE not in set(frame.loc[h0, "rating"])
    assert frame.loc[h0, "item_id"].nunique() == 144
    assert TIE not in set(judge_labels)
    kept, kept_labels, counts = baseline_items(frame, judge_labels, "drop")
    assert counts == (0, 144, 0)
    assert (len(kept), kept_labels.size) == (0, 0)
    assert headline_ties(frame, judge_labels) == 0

    result = validate_with_baseline(frame, judge_labels, ties="drop")

    assert result.summary() == SUMMARY_NO_ITEM_REMAINS


def test_baseline_refuses_a_baseline_that_is_not_a_frame():
    """The rows as a dict of columns, the usual slip. Every column is there,
    so the type is the only defect."""
    frame, judge_labels = judge_like_a_fourth_human()
    human, judge, item_ids = headline_inputs(frame, judge_labels)
    as_dict = frame.to_dict("list")
    assert set(as_dict) == {"item_id", "cluster_id", "rater_id", "rating"}

    with pytest.raises(ValueError) as excinfo:
        judge_validation(
            human, judge, item_ids=item_ids, human_baseline=as_dict,
            ties="category",
        )

    assert str(excinfo.value) == "human_baseline must be a pandas DataFrame"


def test_baseline_refuses_a_baseline_missing_columns():
    """cluster_id and rater_id are dropped. The message lists both, in the
    order the columns are expected."""
    frame, judge_labels = judge_like_a_fourth_human()
    human, judge, item_ids = headline_inputs(frame, judge_labels)
    thin = frame.drop(columns=["rater_id", "cluster_id"])
    assert list(thin.columns) == ["item_id", "rating"]

    with pytest.raises(ValueError) as excinfo:
        judge_validation(
            human, judge, item_ids=item_ids, human_baseline=thin,
            ties="category",
        )

    assert str(excinfo.value) == (
        "human_baseline is missing required column(s): cluster_id, rater_id. "
        "Expected item_id, cluster_id, rater_id, rating, where cluster_id "
        "groups items that are not independent of each other."
    )


def test_baseline_refuses_an_empty_baseline():
    """Every column is there and no row is."""
    frame, judge_labels = judge_like_a_fourth_human()
    human, judge, item_ids = headline_inputs(frame, judge_labels)
    empty = frame.iloc[:0]
    assert list(empty.columns) == ["item_id", "cluster_id", "rater_id", "rating"]
    assert len(empty) == 0

    with pytest.raises(ValueError) as excinfo:
        judge_validation(
            human, judge, item_ids=item_ids, human_baseline=empty,
            ties="category",
        )

    assert str(excinfo.value) == "human_baseline must not be empty"


def test_baseline_refuses_an_item_in_two_clusters():
    """The message names i8, the first split item in the baseline's order,
    which sorted order would not pick."""
    frame, judge_labels = items_in_two_clusters()
    human, judge, item_ids = headline_inputs(frame, judge_labels)
    clusters_per_item = frame.groupby("item_id", sort=False)["cluster_id"].nunique()
    split = list(clusters_per_item.index[clusters_per_item > 1])
    assert split == ["i8", "i12"]
    assert sorted(split) == ["i12", "i8"]
    assert set(frame["item_id"]) <= set(item_ids)
    assert len(set(item_ids)) == len(item_ids)
    assert not frame.duplicated(["item_id", "rater_id"]).any()

    with pytest.raises(ValueError) as excinfo:
        judge_validation(
            human, judge, item_ids=item_ids, human_baseline=frame,
            ties="category",
        )

    assert str(excinfo.value) == (
        "human_baseline puts item 'i8' in more than one cluster. The interval "
        "resamples whole clusters, so every row for an item needs the same "
        "cluster_id."
    )
