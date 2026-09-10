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
"""

import math

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
REFUSES_A_SLICE = "cannot single out a slice"

# The three sentences that carry REFUSES_A_SLICE, told apart. The phrase above
# is in all of them, so on its own it says a slice was refused and not why.
# The three refusals are different findings and a test that cannot separate
# them cannot tell a working rule from one that reports the wrong reason.
# Each constant runs through the shared tail and occurs once in the package,
# in one branch of JudgeValidation._slice_sentence, so pointing an assertion
# at one keeps the old check and adds the branch.
REFUSES_FOR_NO_INTERVAL = (
    "Without an interval on the overall figure there is nothing to place the "
    "slices against, so the data cannot single out a slice."
)
REFUSES_FOR_NO_FIGURE = (
    "No slice carries both an agreement figure and an interval, so the data "
    "cannot single out a slice."
)
REFUSES_FOR_OVERLAP = (
    "on the overall figure, so the two overlap and the data cannot single "
    "out a slice."
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
CALLS_IT_AN_UNSTEADY_JUDGE = (
    "The flips split evenly across the two positions, so this is an unsteady "
    "judge rather than a position-biased one."
)
# The randomised branch's own pair. Same reasoning.
RANDOMISED_CLEARS_A_HALF = (
    "The interval clears 50%, so the judge favours whichever output it sees "
    "first."
)
RANDOMISED_COVERS_A_HALF = (
    "The interval covers 50%, so the data cannot show that position moved "
    "the judge."
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


def test_interval_matches_statsmodels_wilson():
    rng = np.random.default_rng(50)
    frame = randomised_frame(180, 0.6, rng)
    wins = int((frame["winner"] == frame["option_a"]).sum())
    lo, hi = proportion_confint(wins, 180, alpha=0.05, method="wilson")

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
    assert RANDOMISED_COVERS_A_HALF not in r.summary()


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
    assert CALLS_IT_AN_UNSTEADY_JUDGE not in r.summary()


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
    assert CALLS_IT_AN_UNSTEADY_JUDGE in r.summary()
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
    """Both intervals here are Wilson, so nothing is drawn. The argument
    exists to match the rest of the package and must not change an answer.
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
