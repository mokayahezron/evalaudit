"""A result that does not clear its threshold, and how every module says so.

A test that fails to clear its threshold has failed to establish something.
It has not established the opposite. Every summary that describes such a
result says what was not shown, in the same two sentences whichever module
wrote it:

    "... so the data cannot show that X. That does not mean Y."

The first sentence says what the data failed to show. The second stops a
reader from turning that into a finding that nothing is there. Wording that
did the turning is listed in RETIRED and must not come back. "The flips split
evenly" was printed at 7 flips of 9, and "the judge does not depart from the
humans" beside an interval running both ways.

Each builder is guarded, so a fixture that drifts into significance fails
here rather than making the rule vacuous.
"""

import numpy as np
import pandas as pd
import pytest

from evalaudit import (
    compare_paired,
    judge_validation,
    length_bias,
    position_bias,
    rater_agreement,
)


NOT_SHOWN = "so the data cannot show that"
NOT_THE_OPPOSITE = "That does not mean"

# Phrases that read a result short of its threshold as a finding of nothing,
# or that said "not shown" in a different form in each module.
RETIRED = (
    "split evenly",
    "covers no effect",
    "does not depart",
    "cannot distinguish",
    "cannot single out",
)

COLUMNS = ["pair_id", "option_a", "option_b", "winner"]


# --------------------------------------------------------------------------
# One result short of its threshold from each module
# --------------------------------------------------------------------------

def a_margin_that_includes_zero():
    a = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    b = [0, 1, 0, 1, 1, 0, 1, 0, 1, 0]
    r = compare_paired(a, b, method="mcnemar")
    assert r.crosses_zero
    return r


def a_randomised_judge_at_a_half():
    frame = pd.DataFrame(
        [(f"p{i}", "x", "y", "x" if i % 2 == 0 else "y") for i in range(100)],
        columns=COLUMNS,
    )
    r = position_bias(frame)
    assert r.design == "randomised"
    assert not r.has_position_effect
    return r


def seven_flips_of_nine():
    """Thirty pairs run both ways. Nine flip, seven toward the first shown."""
    rows = []
    for p in range(9):
        first = p < 7
        rows += [
            (p, "x", "y", "x" if first else "y"),
            (p, "y", "x", "y" if first else "x"),
        ]
    for p in range(9, 30):
        rows += [(p, "x", "y", "x"), (p, "y", "x", "x")]
    r = position_bias(pd.DataFrame(rows, columns=COLUMNS))
    assert r.design == "both_orders"
    assert (r.n_a_wins, r.n_decisive) == (7, 9)
    assert not r.has_position_effect
    return r


def a_judge_indifferent_to_length():
    """Both fits run, and both intervals include an odds ratio of 1."""
    rng = np.random.default_rng(0)
    n = 300
    lengths = np.column_stack([rng.normal(800, 200, n), rng.normal(800, 200, n)])
    judge = rng.integers(0, 2, n)
    human = rng.integers(0, 2, n)
    r = length_bias(judge, lengths, human_preferences=human)
    assert r.ci_low < 0 < r.ci_high
    assert r.disagreement_ci_low < 0 < r.disagreement_ci_high
    return r


def raters_none_of_whom_stands_out():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(60):
        truth = int(rng.integers(0, 2))
        for rater in ("r1", "r2", "r3"):
            value = truth if rng.random() > 0.1 else 1 - truth
            rows.append({"item_id": f"i{i}", "rater_id": rater, "rating": value})
    r = rater_agreement(pd.DataFrame(rows), n_boot=500, seed=0)
    assert r.has_interval
    assert not r.dropout_is_distinguishable
    assert r.rater_dropout["alpha_without"].notna().all()
    return r


def slices_that_overlap():
    rng = np.random.default_rng(0)
    human = rng.integers(0, 2, 200)
    judge = human.copy()
    flip = rng.random(200) < 0.1
    judge[flip] = 1 - judge[flip]
    r = judge_validation(
        human, judge, slices=np.array(["a", "b"] * 100), n_boot=500, seed=0
    )
    assert r.has_interval
    assert not r.slice_is_distinguishable
    assert np.isfinite(r.by_slice.iloc[0]["ci_high"])
    return r


BUILDERS = [
    a_margin_that_includes_zero,
    a_randomised_judge_at_a_half,
    seven_flips_of_nine,
    a_judge_indifferent_to_length,
    raters_none_of_whom_stands_out,
    slices_that_overlap,
]


# --------------------------------------------------------------------------
# The rule, across modules
# --------------------------------------------------------------------------

@pytest.mark.parametrize("build", BUILDERS, ids=[b.__name__ for b in BUILDERS])
def test_every_result_short_of_its_threshold_says_what_was_not_shown(build):
    text = build().summary()
    assert NOT_SHOWN in text
    assert NOT_THE_OPPOSITE in text
    for phrase in RETIRED:
        assert phrase not in text, f"{phrase!r} is back in {build.__name__}"


def test_the_length_summary_follows_the_rule_in_both_fits():
    """Two verdicts in one summary, so each one gets both sentences."""
    text = a_judge_indifferent_to_length().summary()
    assert text.count(NOT_SHOWN) == 2
    assert text.count(NOT_THE_OPPOSITE) == 2


# --------------------------------------------------------------------------
# The exact sentences. Written out so that rewording fails and gets read.
# --------------------------------------------------------------------------

def test_compare_says_what_was_not_shown():
    assert (
        " The interval includes zero, so the data cannot show that either "
        "system is better. That does not mean the two are level."
    ) in a_margin_that_includes_zero().summary()


def test_seven_flips_of_nine_are_not_called_an_even_split():
    """7 of 9 flips went to the first-shown answer, p=0.18.

    Nine flips cannot rule a lean out, so the data cannot show a direction.
    The old sentence said the flips split evenly and concluded the judge was
    unsteady rather than position-biased. The data shows neither.
    """
    text = seven_flips_of_nine().summary()
    assert "(77.8%, 95% CI: 40.0% to 97.2%, exact binomial p=0.1797)." in text
    assert (
        " That share is not clear of 50% at this many flips, so the data "
        "cannot show that the flips have a direction. That does not mean "
        "the judge is free of position bias. Inconsistency is its own "
        "problem and does not become position bias without a direction."
    ) in text


def test_length_says_what_was_not_shown_in_both_fits():
    text = a_judge_indifferent_to_length().summary()
    assert (
        " The interval includes an odds ratio of 1, so the data cannot show "
        "that length moved the judge. That does not mean length plays no part "
        "in its choices."
    ) in text
    assert (
        " That interval includes an odds ratio of 1, so the data cannot show "
        "that the judge breaks with the humans toward longer or shorter "
        "answers. That does not mean it follows them on length."
    ) in text


# --------------------------------------------------------------------------
# Krippendorff's bands read the interval
#
# AgreementResult and JudgeValidation used to name a band from the point
# estimate. An audit finding that said the data cannot show the judge clears
# 0.667 carried a module summary, word for word, saying the judge was at or
# above 0.800. A band is now named only when the whole interval sits in it.
# --------------------------------------------------------------------------

FLOOR, BAR = 0.667, 0.800
BAND_ABOVE = (
    " The whole interval sits above 0.800, the conventional bar for treating "
    "coded data as reliable."
)
BAND_BETWEEN = (
    " The whole interval sits between 0.667 and 0.800, which supports "
    "tentative conclusions and no firm ones."
)
BAND_BELOW = (
    " The whole interval sits below 0.667, the conventional floor for drawing "
    "any conclusion from coded data."
)
RETIRED_BANDS = ("That is at or above", "That sits between", "That is below")
RATERS = "rater agreement"
JUDGE = "the judge's agreement with the humans"


def band_over_floor(subject):
    return (
        f" The interval runs both sides of 0.667, the conventional floor for "
        f"drawing any conclusion from coded data, so the data cannot show "
        f"that {subject} clears it. That does not mean it falls short of it."
    )


def band_over_bar(subject):
    return (
        f" The interval clears 0.667, which supports tentative conclusions. It "
        f"runs both sides of 0.800, so the data cannot show that {subject} "
        f"reaches the bar for reliable coded data. That does not mean it falls "
        f"short of it."
    )


def band_of(low, high):
    """Where an interval sits, for the guards."""
    if low > BAR:
        return "above"
    if high < FLOOR:
        return "below"
    if low > FLOOR:
        return "between" if high < BAR else "over bar"
    return "over floor"


def three_raters(seed, n, noise):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        truth = int(rng.integers(0, 2))
        for rater in ("r1", "r2", "r3"):
            value = truth if rng.random() > noise else 1 - truth
            rows.append({"item_id": f"i{i}", "rater_id": rater, "rating": value})
    return pd.DataFrame(rows)


def random_raters(seed, n):
    rng = np.random.default_rng(seed)
    return pd.DataFrame([
        {"item_id": f"i{i}", "rater_id": rater,
         "rating": int(rng.integers(0, 2))}
        for i in range(n) for rater in ("r1", "r2", "r3")
    ])


def judge_against_humans(seed, n, flip):
    rng = np.random.default_rng(seed)
    human = rng.integers(0, 2, n)
    judge = human.copy()
    flipped = rng.random(n) < flip
    judge[flipped] = 1 - judge[flipped]
    return human, judge


AGREEMENT_BANDS = [
    ("above", lambda: three_raters(0, 200, 0.02), BAND_ABOVE),
    ("between", lambda: three_raters(0, 400, 0.08), BAND_BETWEEN),
    ("over bar", lambda: three_raters(11, 80, 0.04), band_over_bar(RATERS)),
    ("over floor", lambda: three_raters(1, 20, 0.08), band_over_floor(RATERS)),
    ("over floor", lambda: three_raters(0, 20, 0.12), band_over_floor(RATERS)),
    ("below", lambda: random_raters(12, 80),
     BAND_BELOW + " Fix the rubric before reading anything into the scores "
     "it produced."),
]


@pytest.mark.parametrize(
    "band, build, sentence", AGREEMENT_BANDS,
    ids=[f"{b}-{i}" for i, (b, _, _) in enumerate(AGREEMENT_BANDS)],
)
def test_agreement_names_a_band_only_when_the_interval_sits_in_it(
    band, build, sentence
):
    r = rater_agreement(build(), n_boot=1000, seed=1)
    assert band_of(r.ci_low, r.ci_high) == band
    text = r.summary()
    assert sentence in text
    for old in RETIRED_BANDS:
        assert old not in text


def test_agreement_estimate_above_the_floor_with_an_interval_below_it():
    """The case from the audit. Alpha 0.730 used to read "sits between 0.667
    and 0.800, which supports tentative conclusions" beside an interval of
    0.466 to 0.933."""
    r = rater_agreement(three_raters(1, 20, 0.08), n_boot=1000, seed=1)
    assert r.alpha > FLOOR > r.ci_low
    assert "supports tentative conclusions" not in r.summary()


def test_agreement_without_an_interval_names_no_band():
    r = rater_agreement(three_raters(11, 80, 0.04), bootstrap_ci=False)
    assert not r.has_interval
    assert (
        " Without an interval there is nothing to place rater agreement "
        "against the conventional lines at 0.667 and 0.800, so the data "
        "cannot show that it clears either. That does not mean it falls short "
        "of them."
    ) in r.summary()


JUDGE_BANDS = [
    ("above", (0, 800, 0.07), BAND_ABOVE),
    ("over bar", (22, 200, 0.05), band_over_bar(JUDGE)),
    ("over floor", (0, 40, 0.1), band_over_floor(JUDGE)),
    ("below", (0, 200, 0.35),
     BAND_BELOW + " The judge is not a stand-in for the humans at this level."),
]


@pytest.mark.parametrize(
    "band, args, sentence", JUDGE_BANDS, ids=[b for b, _, _ in JUDGE_BANDS]
)
def test_the_judge_names_a_band_only_when_the_interval_sits_in_it(
    band, args, sentence
):
    r = judge_validation(*judge_against_humans(*args), n_boot=1000, seed=1)
    assert band_of(r.ci_low, r.ci_high) == band
    text = r.summary()
    assert sentence in text
    for old in RETIRED_BANDS:
        assert old not in text


def test_the_judge_case_from_the_audit():
    """Alpha 0.801 used to read "at or above 0.800, the conventional bar for
    treating coded data as reliable" beside an interval of 0.601 to 0.951,
    inside a finding saying the judge could not be shown to clear 0.667."""
    r = judge_validation(*judge_against_humans(0, 40, 0.1), n_boot=1000, seed=1)
    assert r.agreement > BAR > FLOOR > r.ci_low
    assert "conventional bar for treating coded data as reliable" not in (
        r.summary()
    )
