"""Tests for evalaudit.audit.

Written before the implementation. Make these pass.

The statistics under this module are already tested. What is tested here is
the wrapper: which checks fire on which data, how the findings are ranked,
what the prose says, and what the report does when the data for a check was
never supplied.

Three properties matter more than the rest.

Ranking. A report is read from the top. Severity comes first, so a critical
finding from power sits above an info finding from scores no matter which
module ran first. Inside one severity the order is CHECK_ORDER, which is
fixed and documented, and the sort is stable so equal keys keep the order
they were produced in.

Prose. Every finding's detail carries the underlying result's own
``summary()`` verbatim. That is the anti-drift rule. If audit wrote its own
sentences about the numbers they would age out of step with the modules that
compute them.

Absence. A check that could not run is reported as a check that could not
run, with the reason, and never as a finding. An audit of scores alone must
not produce a sentence about raters.

Every fixture here was confirmed against the module that scores it, so a
fixture named "poor" really does land below the threshold.
"""

import numpy as np
import pandas as pd
import pytest

from evalaudit import (
    AgreementResult,
    AuditConfig,
    AuditReport,
    ComparisonResult,
    Finding,
    JudgeValidation,
    LengthBias,
    PositionBias,
    PowerResult,
    ScoreCI,
    SkippedCheck,
    audit,
    length_bias,
)
from evalaudit.audit import (
    CHECK_ORDER,
    CHECK_TITLES,
    SEVERITY_ORDER,
    _detail,
    _ranked,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def _scores(n, k, seed):
    """n items, k of them passed, shuffled."""
    x = np.zeros(n, dtype=int)
    x[:k] = 1
    np.random.default_rng(seed).shuffle(x)
    return x


@pytest.fixture
def undecided_scores():
    """A 5 point margin on 120 items. The interval crosses zero."""
    return {"new": _scores(120, 66, 7), "old": _scores(120, 60, 8)}


@pytest.fixture
def decided_scores():
    """A 20 point margin on 300 items.

    The interval clears zero and the eval was large enough to find a margin
    this size with room to spare. The smallest difference it could have
    detected is 12.0 points against the 20.0 it reports, so the clean case
    built on this is comfortably clean rather than clean by a hair.
    """
    return {"new": _scores(300, 200, 3), "old": _scores(300, 140, 4)}


@pytest.fixture
def ratings_no_overlap():
    """Sixty items, one rater each. Agreement was never measurable."""
    rng = np.random.default_rng(5)
    return pd.DataFrame([
        {"item_id": f"i{i}", "rater_id": f"r{i % 3}",
         "rating": int(rng.integers(0, 2))}
        for i in range(60)
    ])


@pytest.fixture
def ratings_good():
    """Three raters, 80 items, 4% noise. Alpha lands near 0.82."""
    rng = np.random.default_rng(11)
    rows = []
    for i in range(80):
        truth = int(rng.integers(0, 2))
        for r in ("r1", "r2", "r3"):
            v = truth if rng.random() > 0.04 else 1 - truth
            rows.append({"item_id": f"i{i}", "rater_id": r, "rating": v})
    return pd.DataFrame(rows)


@pytest.fixture
def ratings_poor():
    """Three raters grading at random. Alpha lands near zero."""
    rng = np.random.default_rng(12)
    rows = []
    for i in range(80):
        for r in ("r1", "r2", "r3"):
            rows.append({"item_id": f"i{i}", "rater_id": r,
                         "rating": int(rng.integers(0, 2))})
    return pd.DataFrame(rows)


@pytest.fixture
def judge_slice_failure():
    """Alpha 0.68 overall, and the close calls come apart at zero."""
    rng = np.random.default_rng(11)
    n_easy, n_hard = 240, 80
    human_easy = rng.integers(0, 2, n_easy)
    judge_easy = human_easy.copy()
    flip = rng.random(n_easy) < 0.04
    judge_easy[flip] = 1 - judge_easy[flip]
    human_hard = rng.integers(0, 2, n_hard)
    judge_hard = rng.integers(0, 2, n_hard)
    return {
        "human": np.concatenate([human_easy, human_hard]),
        "judge": np.concatenate([judge_easy, judge_hard]),
        "slices": np.array(["clear"] * n_easy + ["close call"] * n_hard),
    }


@pytest.fixture
def judge_mediocre():
    """Alpha near 0.58 overall, no slices supplied."""
    rng = np.random.default_rng(21)
    human = rng.integers(0, 2, 200)
    judge = human.copy()
    flip = rng.random(200) < 0.25
    judge[flip] = 1 - judge[flip]
    return {"human": human, "judge": judge}


@pytest.fixture
def judge_good():
    """Alpha near 0.87 overall."""
    rng = np.random.default_rng(22)
    human = rng.integers(0, 2, 200)
    judge = human.copy()
    flip = rng.random(200) < 0.05
    judge[flip] = 1 - judge[flip]
    return {"human": human, "judge": judge}


@pytest.fixture
def comparisons_biased():
    """Position A wins 70 of 100. The interval clears a half."""
    return pd.DataFrame([
        {"pair_id": f"p{i}", "option_a": "alpha", "option_b": "beta",
         "winner": "alpha" if i < 70 else "beta"}
        for i in range(100)
    ])


@pytest.fixture
def comparisons_clean():
    """Position A wins half the time."""
    return pd.DataFrame([
        {"pair_id": f"p{i}", "option_a": "alpha", "option_b": "beta",
         "winner": "alpha" if i % 2 == 0 else "beta"}
        for i in range(100)
    ])


@pytest.fixture
def length_biased():
    """The judge goes for the longer answer and the humans do not."""
    rng = np.random.default_rng(5)
    n = 300
    len_a = rng.normal(800, 200, n)
    len_b = rng.normal(800, 200, n)
    d = (len_a - len_b) / 200.0
    pref = (rng.random(n) < 1 / (1 + np.exp(-1.5 * d))).astype(int)
    human = (rng.random(n) < 0.5).astype(int)
    return {
        "preferences": pref,
        "lengths": np.column_stack([len_a, len_b]),
        "human_preferences": human,
    }


@pytest.fixture
def length_biased_short():
    """The judge goes for the shorter answer and the humans do not.

    The mirror image of ``length_biased``. Built with the sign of the logit
    flipped, so both fits land clearly below zero rather than clearly above,
    and every sentence the audit writes about this judge has to point the
    other way. The suite had no fixture like this, which is how the
    long-answer wording survived on data that contradicts it.
    """
    rng = np.random.default_rng(5)
    n = 400
    len_a = rng.normal(800, 200, n)
    len_b = rng.normal(800, 200, n)
    d = (len_a - len_b) / 200.0
    pref = (rng.random(n) < 1 / (1 + np.exp(1.5 * d))).astype(int)
    human = (rng.random(n) < 0.5).astype(int)
    return {
        "preferences": pref,
        "lengths": np.column_stack([len_a, len_b]),
        "human_preferences": human,
    }


@pytest.fixture
def length_fits_disagree_in_sign():
    """The two fits point opposite ways, so only one of them can decide.

    ``length_biased`` makes both fits strongly positive and
    ``length_biased_short`` makes both strongly negative, so neither pins
    which fit the verdict reads. Deleting the human branch of
    ``_deciding_fit`` passes the whole suite against those two. This is the
    fixture that separates them.

    The construction leans on the two fits measuring different things. Let
    p(m) be the chance the judge takes the longer answer at length gap m.
    The preference fit asks whether p(m) sits above a half, which is a
    question about level. On a corpus where the humans took the longer
    answer every time, the disagreement fit reduces to regressing "the
    judge took the shorter one" on minus the gap, which asks whether p(m)
    rises or falls with m, a question about trend. Level and trend are free
    of each other, so they can point opposite ways.

    The judge here prefers the longer answer at every gap, 0.90 at no gap
    falling to 0.57 at three standard deviations, never below a half. It
    likes long answers throughout, its enthusiasm flattens as the gap
    widens, and the pairs it breaks with the humans on are the wide-gap
    ones, where it breaks toward the shorter answer.

    Read the limits before reusing this. The fixture needs the humans to
    pick the longer answer on 100% of pairs, and it is pinned there rather
    than merely happening to land there. At 95% the disagreement interval
    crosses zero on three of eight seeds and the fixture stops separating
    anything. So this is a corner of the space and not a typical corpus.
    What makes the corner worth testing is that it is the confound
    length_bias's own docstring is about, length and human-judged quality
    coinciding exactly.
    """
    rng = np.random.default_rng(11)
    n = 800
    len_a = rng.normal(800, 220, n)
    len_b = rng.normal(800, 220, n)
    gap = np.abs(len_a - len_b) / 220.0
    a_is_long = len_a > len_b

    p_long = 0.56 + (0.95 - 0.56) / (1 + np.exp(2.0 * (gap - 1.0)))
    judge_long = rng.random(n) < p_long

    return {
        "preferences": np.where(a_is_long, judge_long, ~judge_long).astype(int),
        "lengths": np.column_stack([len_a, len_b]),
        "human_preferences": a_is_long.astype(int),
    }


@pytest.fixture
def length_fits_disagree_toward_long():
    """The mirror of ``length_fits_disagree_in_sign``, signs swapped.

    The preference fit lands clearly negative and the disagreement fit
    clearly positive, so the verdict reads long while the judge's raw
    preference reads short. Without this the long-verdict half of the
    disagreeing lead would ship untested, which is the defect this fixture
    family exists to close.

    Same corner as its mirror, and the same limit. The humans take the
    shorter answer on 100% of pairs, and the fixture is pinned there.
    """
    rng = np.random.default_rng(11)
    n = 800
    len_a = rng.normal(800, 220, n)
    len_b = rng.normal(800, 220, n)
    gap = np.abs(len_a - len_b) / 220.0
    a_is_long = len_a > len_b

    p_short = 0.56 + (0.95 - 0.56) / (1 + np.exp(2.0 * (gap - 1.0)))
    judge_short = rng.random(n) < p_short

    return {
        "preferences": np.where(a_is_long, ~judge_short, judge_short).astype(int),
        "lengths": np.column_stack([len_a, len_b]),
        "human_preferences": (~a_is_long).astype(int),
    }


@pytest.fixture
def length_clean():
    """The judge picks at random with respect to length."""
    rng = np.random.default_rng(31)
    n = 300
    len_a = rng.normal(800, 200, n)
    len_b = rng.normal(800, 200, n)
    return {
        "preferences": (rng.random(n) < 0.5).astype(int),
        "lengths": np.column_stack([len_a, len_b]),
    }


def _by_check(report, check):
    """The findings a given check produced, in report order."""
    return [f for f in report.findings if f.check == check]


def _one(report, check):
    found = _by_check(report, check)
    assert len(found) == 1, f"expected one {check} finding, got {len(found)}"
    return found[0]


def _skipped(report):
    return {s.check for s in report.not_run}


def _reason(report, check):
    return next(s.reason for s in report.not_run if s.check == check)


# --------------------------------------------------------------------------
# Shape of the report
# --------------------------------------------------------------------------

def test_returns_an_audit_report(undecided_scores):
    r = audit(scores=undecided_scores, config={"seed": 1})
    assert isinstance(r, AuditReport)
    assert all(isinstance(f, Finding) for f in r.findings)
    assert all(isinstance(s, SkippedCheck) for s in r.not_run)


def test_report_and_finding_are_frozen(undecided_scores):
    r = audit(scores=undecided_scores, config={"seed": 1})
    with pytest.raises(Exception):
        r.findings = ()
    with pytest.raises(Exception):
        r.findings[0].severity = "info"


def test_every_severity_is_one_of_three(undecided_scores, ratings_poor,
                                        judge_slice_failure,
                                        comparisons_biased, length_biased):
    r = audit(
        scores=undecided_scores,
        ratings=ratings_poor,
        comparisons=comparisons_biased,
        judge={**judge_slice_failure, **length_biased},
        config={"seed": 1},
    )
    assert len(r.findings) > 0
    assert {f.severity for f in r.findings} <= set(SEVERITY_ORDER)


def test_severity_counts_always_carry_all_three_keys(undecided_scores):
    r = audit(scores=undecided_scores, config={"seed": 1})
    assert set(r.severity_counts) == set(SEVERITY_ORDER)
    assert sum(r.severity_counts.values()) == len(r.findings)


def test_severity_counts_match_the_findings(undecided_scores, ratings_poor,
                                            comparisons_biased):
    r = audit(scores=undecided_scores, ratings=ratings_poor,
              comparisons=comparisons_biased, config={"seed": 1})
    for severity in SEVERITY_ORDER:
        expected = sum(1 for f in r.findings if f.severity == severity)
        assert r.severity_counts[severity] == expected


def test_every_check_id_is_in_the_documented_order():
    assert set(CHECK_TITLES) == set(CHECK_ORDER)
    assert len(CHECK_ORDER) == len(set(CHECK_ORDER))


def test_no_finding_carries_a_check_outside_the_documented_order(
    undecided_scores, ratings_poor, judge_slice_failure, comparisons_biased,
    length_biased
):
    r = audit(
        scores=undecided_scores,
        ratings=ratings_poor,
        comparisons=comparisons_biased,
        judge={**judge_slice_failure, **length_biased},
        config={"seed": 1},
    )
    assert {f.check for f in r.findings} <= set(CHECK_ORDER)
    assert {s.check for s in r.not_run} <= set(CHECK_ORDER)


def test_a_check_is_either_a_finding_or_a_skip_and_never_both(
    undecided_scores, ratings_good
):
    r = audit(scores=undecided_scores, ratings=ratings_good,
              config={"seed": 1})
    assert {f.check for f in r.findings}.isdisjoint(_skipped(r))


def test_every_check_is_accounted_for(undecided_scores, ratings_good,
                                      judge_good, comparisons_clean,
                                      length_clean):
    """No check disappears quietly. It runs or it is listed as skipped."""
    r = audit(
        scores=undecided_scores,
        ratings=ratings_good,
        comparisons=comparisons_clean,
        judge={**judge_good, **length_clean},
        config={"seed": 1},
    )
    assert {f.check for f in r.findings} | _skipped(r) == set(CHECK_ORDER)


# --------------------------------------------------------------------------
# Check 1. No overlapping items, so agreement was never measurable
# --------------------------------------------------------------------------

def test_no_overlapping_items_is_critical(ratings_no_overlap):
    r = audit(ratings=ratings_no_overlap, config={"seed": 1})
    f = _one(r, "agreement")
    assert f.severity == "critical"
    assert isinstance(f.result, AgreementResult)
    assert f.result.n_overlapping_items == 0


def test_no_overlapping_items_reuses_the_agreement_summary(ratings_no_overlap):
    r = audit(ratings=ratings_no_overlap, config={"seed": 1})
    f = _one(r, "agreement")
    assert f.result.summary() in f.detail


def test_low_alpha_is_a_warning_not_a_critical(ratings_poor):
    r = audit(ratings=ratings_poor, config={"seed": 1})
    f = _one(r, "agreement")
    assert f.severity == "warning"
    assert f.result.alpha < 0.67


def test_healthy_alpha_is_info(ratings_good):
    r = audit(ratings=ratings_good, config={"seed": 1})
    f = _one(r, "agreement")
    assert f.severity == "info"
    assert f.result.alpha >= 0.67


def test_agreement_threshold_is_configurable(ratings_good):
    r = audit(ratings=ratings_good,
              config={"seed": 1, "agreement_threshold": 0.9})
    assert _one(r, "agreement").severity == "warning"


# --------------------------------------------------------------------------
# Check 2. The headline margin's interval crosses zero
# --------------------------------------------------------------------------

def test_margin_crossing_zero_is_critical_when_a_direction_is_claimed(
    undecided_scores
):
    r = audit(scores=undecided_scores, config={"seed": 1})
    f = _one(r, "compare")
    assert f.severity == "critical"
    assert isinstance(f.result, ComparisonResult)
    assert f.result.crosses_zero


def test_margin_crossing_zero_is_a_warning_when_no_direction_is_claimed(
    undecided_scores
):
    r = audit(scores=undecided_scores,
              config={"seed": 1, "claims_direction": False})
    assert _one(r, "compare").severity == "warning"


def test_margin_clear_of_zero_is_info(decided_scores):
    r = audit(scores=decided_scores, config={"seed": 1})
    f = _one(r, "compare")
    assert f.severity == "info"
    assert not f.result.crosses_zero


def test_compare_reuses_the_comparison_summary(undecided_scores):
    r = audit(scores=undecided_scores, config={"seed": 1})
    f = _one(r, "compare")
    assert f.result.summary() in f.detail


def test_two_equal_length_systems_are_read_as_paired(undecided_scores):
    r = audit(scores=undecided_scores, config={"seed": 1})
    assert _one(r, "compare").result.paired is True


def test_pairing_is_stated_when_it_was_inferred(undecided_scores):
    """An inferred pairing is an assumption, so the detail has to say so."""
    r = audit(scores=undecided_scores, config={"seed": 1})
    assert "paired=False" in _one(r, "compare").detail


def test_pairing_is_not_restated_when_it_was_given(undecided_scores):
    r = audit(scores=undecided_scores, config={"seed": 1, "paired": True})
    assert "paired=False" not in _one(r, "compare").detail


def test_unequal_lengths_are_read_as_independent():
    scores = {"new": _scores(120, 66, 7), "old": _scores(90, 40, 8)}
    r = audit(scores=scores, config={"seed": 1})
    assert _one(r, "compare").result.paired is False


def test_config_can_force_an_independent_comparison(undecided_scores):
    r = audit(scores=undecided_scores, config={"seed": 1, "paired": False})
    assert _one(r, "compare").result.paired is False


def test_difference_runs_first_system_minus_second(decided_scores):
    r = audit(scores=decided_scores, config={"seed": 1})
    assert _one(r, "compare").result.difference == pytest.approx(0.2)


# --------------------------------------------------------------------------
# Check 3. The sample could not have detected the effect at issue
# --------------------------------------------------------------------------

def test_underpowered_eval_is_critical(undecided_scores):
    r = audit(scores=undecided_scores, config={"seed": 1})
    f = _one(r, "power")
    assert f.severity == "critical"
    assert isinstance(f.result, PowerResult)
    assert f.result.difference > abs(_one(r, "compare").result.difference)


def test_power_reuses_the_power_summary(undecided_scores):
    r = audit(scores=undecided_scores, config={"seed": 1})
    f = _one(r, "power")
    assert f.result.summary() in f.detail


def test_effect_at_issue_defaults_to_the_observed_margin(undecided_scores):
    """With no stated effect, the claim being made is the effect at issue."""
    r = audit(scores=undecided_scores, config={"seed": 1})
    assert "5.0 points" in _one(r, "power").detail


def test_stated_effect_of_interest_overrides_the_observed_margin(
    decided_scores
):
    """A 20 point margin was detectable here. A 5 point one never was."""
    plain = audit(scores=decided_scores, config={"seed": 1})
    assert _one(plain, "power").severity == "info"

    stated = audit(scores=decided_scores,
                   config={"seed": 1, "effect_of_interest": 0.05})
    assert _one(stated, "power").severity == "critical"
    assert "5.0 points" in _one(stated, "power").detail


def test_power_uses_the_observed_discordance_rate(undecided_scores):
    """The rate is in the data, so nothing here is allowed to assume it."""
    a, b = undecided_scores["new"], undecided_scores["old"]
    f = _one(audit(scores=undecided_scores, config={"seed": 1}), "power")
    assert f.result.discordance_assumed is False
    assert f.result.discordance_rate == pytest.approx(np.mean(a != b))


def test_power_runs_on_one_system_when_an_effect_is_stated():
    r = audit(scores={"only": _scores(80, 40, 2)},
              config={"seed": 1, "effect_of_interest": 0.05})
    f = _one(r, "power")
    assert f.severity == "critical"
    assert f.result.n == 80


def test_power_cannot_run_without_an_effect_at_issue():
    r = audit(scores={"only": _scores(80, 40, 2)}, config={"seed": 1})
    assert "power" in _skipped(r)


def test_a_zero_margin_leaves_no_effect_to_size_against():
    scores = {"new": _scores(100, 50, 2), "old": _scores(100, 50, 3)}
    r = audit(scores=scores, config={"seed": 1})
    assert "power" in _skipped(r)
    assert "effect_of_interest" in _reason(r, "power")


# --------------------------------------------------------------------------
# Check 4. Judge against humans
# --------------------------------------------------------------------------

def test_a_failing_slice_is_critical(judge_slice_failure):
    r = audit(judge=judge_slice_failure, config={"seed": 3})
    f = _one(r, "judge")
    assert f.severity == "critical"
    assert isinstance(f.result, JudgeValidation)
    assert f.result.slice_is_distinguishable


def test_a_failing_slice_outranks_a_healthy_headline(judge_slice_failure):
    """The headline here clears the threshold. The slice is the finding."""
    f = _one(audit(judge=judge_slice_failure, config={"seed": 3}), "judge")
    assert f.result.agreement >= 0.67
    assert f.severity == "critical"


def test_low_overall_agreement_is_a_warning(judge_mediocre):
    f = _one(audit(judge=judge_mediocre, config={"seed": 3}), "judge")
    assert f.severity == "warning"
    assert f.result.agreement < 0.67
    assert f.result.slice_is_distinguishable is False


def test_healthy_judge_is_info(judge_good):
    r = audit(judge=judge_good, config={"seed": 3})
    assert _one(r, "judge").severity == "info"


def test_judge_threshold_is_configurable(judge_good):
    r = audit(judge=judge_good, config={"seed": 3, "judge_threshold": 0.95})
    assert _one(r, "judge").severity == "warning"


def test_judge_reuses_the_validation_summary(judge_slice_failure):
    f = _one(audit(judge=judge_slice_failure, config={"seed": 3}), "judge")
    assert f.result.summary() in f.detail


# --------------------------------------------------------------------------
# Check 5. Position and length bias
# --------------------------------------------------------------------------

def test_position_bias_is_a_warning(comparisons_biased):
    f = _one(audit(comparisons=comparisons_biased, config={"seed": 1}),
             "position")
    assert f.severity == "warning"
    assert isinstance(f.result, PositionBias)
    assert f.result.has_position_effect


def test_no_position_effect_is_info(comparisons_clean):
    f = _one(audit(comparisons=comparisons_clean, config={"seed": 1}),
             "position")
    assert f.severity == "info"
    assert not f.result.has_position_effect


def test_position_reuses_its_summary(comparisons_biased):
    f = _one(audit(comparisons=comparisons_biased, config={"seed": 1}),
             "position")
    assert f.result.summary() in f.detail


def test_length_bias_is_a_warning(judge_good, length_biased):
    r = audit(judge={**judge_good, **length_biased}, config={"seed": 1})
    f = _one(r, "length")
    assert f.severity == "warning"
    assert isinstance(f.result, LengthBias)


def test_the_fixture_really_is_a_short_preferring_judge(length_biased_short):
    """Guard the guard.

    Every assertion below is worthless if the fit does not actually land
    below zero, so pin that separately from the prose it drives.
    """
    r = length_bias(
        length_biased_short["preferences"],
        length_biased_short["lengths"],
        human_preferences=length_biased_short["human_preferences"],
    )
    assert r.ci_high < 0, f"preference interval is {(r.ci_low, r.ci_high)}"
    assert r.disagreement_ci_high < 0, (
        f"disagreement interval is "
        f"{(r.disagreement_ci_low, r.disagreement_ci_high)}"
    )


def test_a_short_preferring_judge_is_not_called_a_long_preferring_one(
    judge_good, length_biased_short
):
    """The title, the lead and the caveat all follow the sign of the fit.

    A judge that reaches for the terser answer is still a finding, and it is
    still a warning. What it is not is a judge pulled toward length, and the
    report must not say so while printing a coefficient that says otherwise.

    The three strings are written out here rather than imported, so that
    rewording the report fails this test and the new wording gets read.
    """
    r = audit(judge={**judge_good, **length_biased_short}, config={"seed": 1})
    f = _one(r, "length")

    assert f.severity == "warning"
    assert f.title == "The judge is pulled toward shorter answers"
    assert (
        "A judge that rewards brevity ranks the terser system higher "
        "whatever it says, and it can do that while agreeing with humans "
        "on most pairs."
    ) in f.detail
    assert (
        "A negative coefficient is harder to explain away than a positive "
        "one. Length can track quality, so a preference for long answers "
        "may be reading real content. Brevity rarely tracks quality in the "
        "same way, so a preference for short answers usually points at the "
        "judge."
    ) in f.detail

    # The two remaining directional sentences. The advice tells the reader
    # what to go and look at, and the disagreement verdict says which way
    # the judge left the humans. Both pointed at length regardless of sign.
    assert (
        "Check whether the winning system is simply the shorter one."
    ) in f.detail
    assert (
        "The judge departs from the humans in the direction of brevity."
    ) in f.detail

    assert "pulled by how long" not in f.title
    assert "rewards length ranks the wordier system" not in f.detail
    assert "A positive coefficient here is not bias on its own" not in f.detail
    assert "simply the longer one" not in f.detail
    assert "departs from the humans in the direction of length" not in f.detail


def test_a_long_preferring_judge_keeps_the_long_answer_wording(
    judge_good, length_biased
):
    """The positive branch is unchanged, and the two do not collapse.

    A fix that pointed every finding at brevity would pass the test above
    and be no better than the bug.
    """
    f = _one(
        audit(judge={**judge_good, **length_biased}, config={"seed": 1}),
        "length",
    )
    assert f.severity == "warning"
    assert f.title == "The judge is pulled by how long the answer is"
    assert (
        "A judge that rewards length ranks the wordier system higher "
        "whatever it says, and it can do that while agreeing with humans "
        "on most pairs."
    ) in f.detail
    assert (
        "A positive coefficient here is not bias on its own, because "
        "longer answers may simply be better."
    ) in f.detail
    assert (
        "Check whether the winning system is simply the longer one."
    ) in f.detail
    assert (
        "The judge departs from the humans in the direction of length."
    ) in f.detail
    assert "toward shorter answers" not in f.title
    assert "rewards brevity" not in f.detail
    assert "simply the shorter one" not in f.detail
    assert "direction of brevity" not in f.detail


def test_the_two_fits_really_do_disagree_in_sign(length_fits_disagree_in_sign):
    """Guard the guard.

    The test below says nothing unless the fixture actually splits the two
    fits, and the split is delicate enough to be worth pinning separately.
    """
    r = length_bias(
        length_fits_disagree_in_sign["preferences"],
        length_fits_disagree_in_sign["lengths"],
        human_preferences=length_fits_disagree_in_sign["human_preferences"],
    )
    assert r.coefficient > 0, f"preference coefficient is {r.coefficient}"
    assert r.ci_low > 0, f"preference interval is {(r.ci_low, r.ci_high)}"
    assert r.disagreement_ci_high < 0, (
        f"disagreement interval is "
        f"{(r.disagreement_ci_low, r.disagreement_ci_high)}"
    )


def test_the_verdict_follows_the_disagreement_fit_when_the_two_disagree(
    judge_good, length_fits_disagree_in_sign
):
    """The sharper fit decides the direction, not the louder one.

    On this fixture the preference fit says long and the disagreement fit
    says short. The disagreement fit is the one that holds the human
    verdict fixed, so it is the one the finding is written from. Reading
    the preference fit instead would point every sentence the other way.

    This is the assertion that deleting the human branch of _deciding_fit
    fails. The two fixtures either side of it pass with that branch gone.
    """
    f = _one(
        audit(
            judge={**judge_good, **length_fits_disagree_in_sign},
            config={"seed": 1},
        ),
        "length",
    )

    assert f.severity == "warning"
    assert f.title == "The judge is pulled toward shorter answers"
    assert "Check whether the winning system is simply the shorter one." in f.detail
    assert "Check whether the winning system is simply the longer one." not in f.detail

    # The lead on this fixture is the disagreeing one, not the plain
    # brevity lead. That is asserted in
    # test_the_lead_says_two_models_disagree_before_the_numbers_arrive.
    assert "Two models run here and they point opposite ways." in f.detail


# The lead that runs when the two fits point opposite ways. Written out
# here rather than imported, so rewording it fails these tests.
DISAGREEING_LEAD_SHORT = (
    "Two models run here and they point opposite ways. The judge picks the "
    "longer answer more often, and on the pairs where it breaks with the "
    "humans it breaks toward the shorter one. The verdict below runs on the "
    "second, which is the sharper of the two."
)
DISAGREEING_LEAD_LONG = (
    "Two models run here and they point opposite ways. The judge picks the "
    "shorter answer more often, and on the pairs where it breaks with the "
    "humans it breaks toward the longer one. The verdict below runs on the "
    "second, which is the sharper of the two."
)


def test_the_lead_says_two_models_disagree_before_the_numbers_arrive(
    judge_good, length_fits_disagree_in_sign
):
    """The disagreeing lead runs, in the form that matches the split.

    Why the bridge belongs in the lead. On this fixture the title reads
    short while the odds ratio and the long-answer rate read long, and
    every one of those is correct. A reader who is told nothing until the
    end meets the contradiction first and the explanation second. So the
    warning is the lead rather than another caveat behind the numbers.

    Four assertions carry that. The title reads short. The short-verdict
    lead is present. The long-verdict lead is absent, so the two forms
    cannot collapse into whichever one the fixture happens to reach. And
    the plain brevity lead is gone, so the disagreeing lead replaced it
    rather than stacking in front of it.

    Where the lead lands in the paragraph is not checked here. That is a
    property of _detail rather than of this finding, and it is pinned by
    test_detail_puts_the_lead_first_and_the_action_last.
    """
    f = _one(
        audit(
            judge={**judge_good, **length_fits_disagree_in_sign},
            config={"seed": 1},
        ),
        "length",
    )
    assert f.title == "The judge is pulled toward shorter answers"
    assert DISAGREEING_LEAD_SHORT in f.detail
    assert DISAGREEING_LEAD_LONG not in f.detail

    # And it replaces the plain lead rather than stacking on top of it.
    assert "A judge that rewards brevity ranks the terser system" not in f.detail


def test_the_mirror_fits_really_do_disagree_in_sign(
    length_fits_disagree_toward_long
):
    """Guard the guard, the other way round.

    Its sibling fixture has one of these and this one did not. Without it a
    mirror fixture that quietly stopped splitting the two fits would fail
    the test below on a lead assertion, which says nothing about why.
    """
    r = length_bias(
        length_fits_disagree_toward_long["preferences"],
        length_fits_disagree_toward_long["lengths"],
        human_preferences=length_fits_disagree_toward_long["human_preferences"],
    )
    assert r.coefficient < 0, f"preference coefficient is {r.coefficient}"
    assert r.ci_high < 0, f"preference interval is {(r.ci_low, r.ci_high)}"
    assert r.disagreement_ci_low > 0, (
        f"disagreement interval is "
        f"{(r.disagreement_ci_low, r.disagreement_ci_high)}"
    )


def test_the_disagreeing_lead_mirrors_when_the_verdict_reads_long(
    judge_good, length_fits_disagree_toward_long
):
    """Same case with the signs swapped, so neither half ships untested."""
    f = _one(
        audit(
            judge={**judge_good, **length_fits_disagree_toward_long},
            config={"seed": 1},
        ),
        "length",
    )
    assert f.severity == "warning"
    assert f.title == "The judge is pulled by how long the answer is"
    assert DISAGREEING_LEAD_LONG in f.detail
    assert DISAGREEING_LEAD_SHORT not in f.detail
    assert "A judge that rewards length ranks the wordier system" not in f.detail


@pytest.mark.parametrize(
    "fixture_name", ["length_biased", "length_biased_short"]
)
def test_agreeing_fits_keep_the_plain_lead(judge_good, fixture_name, request):
    """When the two fits agree there is nothing to warn the reader about.

    The disagreeing lead costs the reader a sentence and buys nothing here,
    and a lead that always says "two models point opposite ways" would be
    false on most data.
    """
    length_fixture = request.getfixturevalue(fixture_name)
    f = _one(
        audit(judge={**judge_good, **length_fixture}, config={"seed": 1}),
        "length",
    )
    assert DISAGREEING_LEAD_SHORT not in f.detail
    assert DISAGREEING_LEAD_LONG not in f.detail
    assert "Two models run here and they point opposite ways" not in f.detail


def test_length_bias_reads_the_disagreement_fit_when_humans_are_there(
    judge_good, length_biased
):
    """The sharper of the two numbers is the one the verdict runs on."""
    r = audit(judge={**judge_good, **length_biased}, config={"seed": 1})
    f = _one(r, "length")
    assert f.result.has_human
    assert f.result.disagreement_ci_low > 0


def test_no_length_effect_is_info(length_clean):
    f = _one(audit(judge=length_clean, config={"seed": 1}), "length")
    assert f.severity == "info"
    assert f.result.ci_low < 0 < f.result.ci_high


def test_a_judge_that_never_disagrees_is_not_flagged(length_clean):
    """No disagreements means no second fit, and a NaN is not a finding."""
    judge = dict(length_clean)
    judge["human_preferences"] = judge["preferences"]
    f = _one(audit(judge=judge, config={"seed": 1}), "length")
    coefficient = f.result.disagreement_coefficient
    assert coefficient != coefficient  # NaN
    assert f.severity == "info"


def test_length_reuses_its_summary(judge_good, length_biased):
    r = audit(judge={**judge_good, **length_biased}, config={"seed": 1})
    assert _one(r, "length").result.summary() in _one(r, "length").detail


# --------------------------------------------------------------------------
# Ranking. The part that decides whether the report is worth reading.
# --------------------------------------------------------------------------

def test_a_critical_from_power_outranks_an_info_from_scores(decided_scores):
    r = audit(scores=decided_scores,
              config={"seed": 1, "effect_of_interest": 0.05})
    assert r.findings[0].check == "power"
    assert r.findings[0].severity == "critical"
    positions = [i for i, f in enumerate(r.findings) if f.check == "scores"]
    assert min(positions) > 0
    assert all(r.findings[i].severity == "info" for i in positions)


def test_severity_beats_check_order(decided_scores):
    """compare precedes power in CHECK_ORDER. Severity still comes first."""
    r = audit(scores=decided_scores,
              config={"seed": 1, "effect_of_interest": 0.05})
    checks = [f.check for f in r.findings]
    assert checks.index("power") < checks.index("compare")
    assert _one(r, "compare").severity == "info"


def test_findings_are_sorted_by_severity(undecided_scores, ratings_poor,
                                         comparisons_biased, judge_good,
                                         length_biased):
    r = audit(
        scores=undecided_scores,
        ratings=ratings_poor,
        comparisons=comparisons_biased,
        judge={**judge_good, **length_biased},
        config={"seed": 1},
    )
    ranks = [SEVERITY_ORDER.index(f.severity) for f in r.findings]
    assert ranks == sorted(ranks)
    assert set(ranks) == {0, 1, 2}


def test_within_a_severity_the_order_is_check_order(ratings_poor,
                                                    comparisons_biased,
                                                    judge_good, length_biased):
    r = audit(
        ratings=ratings_poor,
        comparisons=comparisons_biased,
        judge={**judge_good, **length_biased},
        config={"seed": 1},
    )
    warnings = [f.check for f in r.findings if f.severity == "warning"]
    assert warnings == ["agreement", "position", "length"]


def test_ranking_is_documented_and_independent_of_production_order():
    findings = [
        Finding(check="scores", severity="info", title="s", detail="d"),
        Finding(check="length", severity="warning", title="l", detail="d"),
        Finding(check="power", severity="critical", title="p", detail="d"),
        Finding(check="agreement", severity="warning", title="a", detail="d"),
        Finding(check="compare", severity="info", title="c", detail="d"),
    ]
    order = [f.check for f in _ranked(findings)]
    assert order == ["power", "agreement", "length", "compare", "scores"]
    assert [f.check for f in _ranked(findings[::-1])] == order


def test_ranking_is_stable_within_a_check():
    """Two systems' score intervals keep the order they were supplied in."""
    findings = [
        Finding(check="scores", severity="info", title="new", detail="d"),
        Finding(check="scores", severity="info", title="old", detail="d"),
    ]
    assert [f.title for f in _ranked(findings)] == ["new", "old"]
    assert [f.title for f in _ranked(findings[::-1])] == ["old", "new"]


def test_score_findings_follow_the_order_the_systems_were_supplied_in(
    undecided_scores
):
    r = audit(scores=undecided_scores, config={"seed": 1})
    titles = [f.title for f in _by_check(r, "scores")]
    assert len(titles) == 2
    assert "new" in titles[0]
    assert "old" in titles[1]


def test_worst_severity_is_the_top_of_the_report(undecided_scores,
                                                 ratings_good):
    r = audit(scores=undecided_scores, ratings=ratings_good,
              config={"seed": 1})
    assert r.worst_severity == "critical"
    assert r.findings[0].severity == "critical"


def test_worst_severity_of_a_clean_report_is_info(ratings_good):
    r = audit(ratings=ratings_good, config={"seed": 1})
    assert r.worst_severity == "info"


# --------------------------------------------------------------------------
# Prose
# --------------------------------------------------------------------------

def test_every_finding_carries_the_result_summary_verbatim(
    undecided_scores, ratings_poor, judge_slice_failure, comparisons_biased,
    length_biased
):
    """The anti-drift rule. Audit never restates a number in its own words."""
    r = audit(
        scores=undecided_scores,
        ratings=ratings_poor,
        comparisons=comparisons_biased,
        judge={**judge_slice_failure, **length_biased},
        config={"seed": 1},
    )
    assert len(r.findings) >= 7
    for f in r.findings:
        assert f.result is not None
        assert f.result.summary() in f.detail


class _StubResult:
    """A result that carries nothing but a recognisable summary.

    No fixture, no arithmetic, no digits. The ordering contract is a
    property of _detail alone, so the test should not be able to fail
    because a coefficient moved.
    """

    def summary(self) -> str:
        return "MIDDLE"


def test_detail_puts_the_lead_first_and_the_action_last():
    """Lead, then the module's own summary, then what to do about it.

    The order is the contract. Every finding in the report is built by
    handing this function a sentence that frames the numbers, and framing
    only works before the reader meets them. The length finding leans on
    this hardest: when the two fits sign differently the lead is the only
    thing standing between a title and an odds ratio that point opposite
    ways, and it is worth nothing if it arrives after them.
    """
    text = _detail("LEAD", _StubResult(), "ACTION")

    assert "LEAD" in text
    assert "MIDDLE" in text
    assert "ACTION" in text
    assert text.index("LEAD") < text.index("MIDDLE") < text.index("ACTION")


def test_detail_says_more_than_the_summary_alone(undecided_scores):
    """What was measured, what it means for the claim, what to do."""
    r = audit(scores=undecided_scores, config={"seed": 1})
    for f in r.findings:
        assert len(f.detail) > len(f.result.summary())


def test_titles_are_short_and_present(undecided_scores, ratings_poor):
    r = audit(scores=undecided_scores, ratings=ratings_poor,
              config={"seed": 1})
    for f in r.findings:
        assert f.title.strip()
        assert len(f.title) < 90
        assert "\n" not in f.title


def test_no_em_dashes_in_the_prose(undecided_scores, ratings_poor,
                                   comparisons_biased):
    r = audit(scores=undecided_scores, ratings=ratings_poor,
              comparisons=comparisons_biased, config={"seed": 1})
    for f in r.findings:
        assert "—" not in f.title
        assert "—" not in f.detail
    for skipped in r.not_run:
        assert "—" not in skipped.reason
    assert "—" not in r.summary()


def test_report_summary_leads_with_the_critical_count(undecided_scores):
    r = audit(scores=undecided_scores, config={"seed": 1})
    assert r.severity_counts["critical"] >= 1
    assert "critical" in r.summary()


def test_report_summary_of_a_clean_report_does_not_claim_a_critical(
    ratings_good
):
    r = audit(ratings=ratings_good, config={"seed": 1})
    assert r.severity_counts["critical"] == 0
    assert "no critical" in r.summary().lower()


def test_report_summary_counts_the_checks_that_could_not_run(ratings_good):
    r = audit(ratings=ratings_good, config={"seed": 1})
    assert len(r.not_run) == 6
    assert "6" in r.summary()


def test_the_claim_is_carried_into_the_report(undecided_scores):
    r = audit(scores=undecided_scores,
              config={"seed": 1, "claim": "the new model is better"})
    assert "the new model is better" in r.summary()
    assert "the new model is better" in r.to_markdown()


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def test_markdown_carries_every_finding(undecided_scores, ratings_poor,
                                        comparisons_biased):
    r = audit(scores=undecided_scores, ratings=ratings_poor,
              comparisons=comparisons_biased, config={"seed": 1})
    md = r.to_markdown()
    for f in r.findings:
        assert f.title in md
        assert f.detail in md
        assert f.severity in md.lower()


def test_markdown_orders_findings_the_way_the_report_does(undecided_scores,
                                                          ratings_poor):
    r = audit(scores=undecided_scores, ratings=ratings_poor,
              config={"seed": 1})
    md = r.to_markdown()
    positions = [md.index(f.title) for f in r.findings]
    assert positions == sorted(positions)


def test_html_carries_every_finding(undecided_scores, ratings_poor):
    r = audit(scores=undecided_scores, ratings=ratings_poor,
              config={"seed": 1})
    html = r.to_html()
    assert html.strip().startswith("<")
    for f in r.findings:
        assert f.title in html


def test_html_escapes_the_data_it_was_given():
    """System names come from the client. They are not markup."""
    r = audit(scores={"<script>x</script>": _scores(40, 20, 1)},
              config={"seed": 1})
    html = r.to_html()
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_both_renderings_list_the_checks_that_could_not_run(ratings_good):
    r = audit(ratings=ratings_good, config={"seed": 1})
    md, html = r.to_markdown(), r.to_html()
    assert len(r.not_run) > 0
    for skipped in r.not_run:
        assert skipped.title in md
        assert skipped.reason in md
        assert skipped.reason in html


def test_a_full_report_has_nothing_to_say_about_missing_data(
    undecided_scores, ratings_good, judge_good, comparisons_clean,
    length_clean
):
    r = audit(
        scores=undecided_scores,
        ratings=ratings_good,
        comparisons=comparisons_clean,
        judge={**judge_good, **length_clean},
        config={"seed": 1},
    )
    assert r.not_run == ()
    assert "could not run" not in r.to_markdown().lower()


# --------------------------------------------------------------------------
# Absent data. No check may be invented and none may be passed over quietly.
# --------------------------------------------------------------------------

def test_scores_alone_runs_only_what_scores_supports(undecided_scores):
    r = audit(scores=undecided_scores, config={"seed": 1})
    assert {f.check for f in r.findings} == {"scores", "compare", "power"}
    assert _skipped(r) == {"agreement", "judge", "position", "length"}


def test_one_system_alone_runs_only_the_score_intervals():
    r = audit(scores={"only": _scores(60, 30, 1)}, config={"seed": 1})
    assert {f.check for f in r.findings} == {"scores"}
    assert _skipped(r) == {"compare", "power", "agreement", "judge",
                           "position", "length"}


def test_scores_alone_says_nothing_about_raters_or_judges(undecided_scores):
    r = audit(scores=undecided_scores, config={"seed": 1})
    prose = " ".join(f.title + " " + f.detail for f in r.findings).lower()
    for word in ("rater", "grader", "judge", "alpha", "position"):
        assert word not in prose


def test_every_skipped_check_says_why_and_names_the_input(undecided_scores):
    r = audit(scores=undecided_scores, config={"seed": 1})
    assert "ratings" in _reason(r, "agreement")
    assert "judge" in _reason(r, "judge")
    assert "comparisons" in _reason(r, "position")
    assert "lengths" in _reason(r, "length")
    for skipped in r.not_run:
        assert skipped.reason.strip()
        assert skipped.reason.endswith(".")


def test_skipped_checks_carry_the_documented_titles(undecided_scores):
    r = audit(scores=undecided_scores, config={"seed": 1})
    for skipped in r.not_run:
        assert skipped.title == CHECK_TITLES[skipped.check]


def test_skipped_checks_follow_check_order(undecided_scores):
    r = audit(scores=undecided_scores, config={"seed": 1})
    ranks = [CHECK_ORDER.index(s.check) for s in r.not_run]
    assert ranks == sorted(ranks)


def test_skipped_checks_sort_rather_than_arrive_in_order(ratings_good):
    """The scores check runs first and is reported last, so the sort has
    work to do here. Production order and CHECK_ORDER disagree."""
    r = audit(ratings=ratings_good, config={"seed": 1})
    assert [s.check for s in r.not_run] == [
        "compare", "power", "judge", "position", "length", "scores",
    ]


def test_ratings_alone_runs_only_agreement(ratings_good):
    r = audit(ratings=ratings_good, config={"seed": 1})
    assert {f.check for f in r.findings} == {"agreement"}
    assert "scores" in _skipped(r)


def test_judge_labels_without_lengths_skip_the_length_check(judge_good):
    r = audit(judge=judge_good, config={"seed": 1})
    assert {f.check for f in r.findings} == {"judge"}
    assert "length" in _skipped(r)


def test_lengths_without_labels_skip_the_validation_check(length_clean):
    r = audit(judge=length_clean, config={"seed": 1})
    assert {f.check for f in r.findings} == {"length"}
    assert "human" in _reason(r, "judge")


def test_comparisons_alone_runs_only_position_bias(comparisons_biased):
    r = audit(comparisons=comparisons_biased, config={"seed": 1})
    assert {f.check for f in r.findings} == {"position"}


# --------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------

def test_audit_with_no_data_at_all_is_an_error():
    with pytest.raises(ValueError, match="at least one"):
        audit()


def test_an_empty_score_mapping_is_an_error():
    with pytest.raises(ValueError, match="at least one"):
        audit(scores={})


def test_a_bare_sequence_of_scores_is_one_system():
    r = audit(scores=[1, 0, 1, 1, 0, 1, 1, 1, 0, 1], config={"seed": 1})
    f = _one(r, "scores")
    assert isinstance(f.result, ScoreCI)
    assert f.result.n == 10


def test_three_systems_report_intervals_and_skip_the_comparison():
    scores = {
        "a": _scores(60, 30, 1),
        "b": _scores(60, 33, 2),
        "c": _scores(60, 36, 3),
    }
    r = audit(scores=scores, config={"seed": 1})
    assert len(_by_check(r, "scores")) == 3
    assert "compare" in _skipped(r)
    assert "two" in _reason(r, "compare")


def test_config_accepts_a_dataclass(undecided_scores):
    from_dict = audit(scores=undecided_scores, config={"seed": 1})
    from_object = audit(scores=undecided_scores, config=AuditConfig(seed=1))
    assert from_dict.to_markdown() == from_object.to_markdown()


def test_config_defaults_when_omitted(undecided_scores):
    r = audit(scores=undecided_scores)
    assert isinstance(r.config, AuditConfig)
    assert r.config.confidence == 0.95
    assert r.config.claims_direction is True


def test_unknown_config_key_names_the_valid_ones(undecided_scores):
    with pytest.raises(ValueError, match="confidence"):
        audit(scores=undecided_scores, config={"confidance": 0.9})


def test_unknown_judge_key_is_an_error(judge_good):
    with pytest.raises(ValueError, match="human"):
        audit(judge={**judge_good, "humann": [1, 0]})


def test_bad_confidence_is_an_error(undecided_scores):
    with pytest.raises(ValueError, match="confidence"):
        audit(scores=undecided_scores, config={"confidence": 1.5})


def test_bad_effect_of_interest_is_an_error(undecided_scores):
    with pytest.raises(ValueError, match="effect_of_interest"):
        audit(scores=undecided_scores, config={"effect_of_interest": 0.0})


def test_confidence_reaches_every_interval(undecided_scores, ratings_good):
    r = audit(scores=undecided_scores, ratings=ratings_good,
              config={"seed": 1, "confidence": 0.9})
    assert _one(r, "compare").result.confidence == 0.9
    assert _one(r, "agreement").result.confidence == 0.9
    assert _by_check(r, "scores")[0].result.confidence == 0.9


def test_the_same_seed_gives_the_same_report(undecided_scores, ratings_good,
                                             judge_slice_failure):
    kwargs = dict(scores=undecided_scores, ratings=ratings_good,
                  judge=judge_slice_failure, config={"seed": 4})
    assert audit(**kwargs).to_markdown() == audit(**kwargs).to_markdown()


# --------------------------------------------------------------------------
# The clean case. A report that only knows how to find fault gets discounted,
# so an eval with nothing wrong has to come back saying so, in a document
# still worth sending.
# --------------------------------------------------------------------------

@pytest.fixture
def clean_inputs(decided_scores, ratings_good, judge_good, comparisons_clean,
                 length_clean):
    """Every one of the seven checks runs, and every one of them passes."""
    return dict(
        scores=decided_scores,
        ratings=ratings_good,
        comparisons=comparisons_clean,
        judge={**judge_good, **length_clean},
        config={"seed": 1},
    )


def test_a_clean_eval_produces_a_report_of_nothing_but_context(clean_inputs):
    r = audit(**clean_inputs)
    assert r.not_run == ()
    assert {f.check for f in r.findings} == set(CHECK_ORDER)
    assert [f.severity for f in r.findings] == ["info"] * len(r.findings)
    assert r.severity_counts == {"critical": 0, "warning": 0, "info": 8}
    assert r.worst_severity == "info"


def test_a_clean_report_does_not_manufacture_a_warning(clean_inputs):
    """Each check that fires on the dirty fixtures stays quiet here."""
    r = audit(**clean_inputs)
    for check in CHECK_ORDER:
        for f in _by_check(r, check):
            assert f.severity == "info", f"{check} invented a {f.severity}"


def test_a_clean_report_says_the_eval_holds_up(clean_inputs):
    text = audit(**clean_inputs).summary().lower()
    assert "no critical" in text
    assert "no warnings" in text
    assert "holds up" in text
    assert "every check ran" in text


def test_a_clean_report_is_still_a_document_worth_sending(clean_inputs):
    r = audit(**clean_inputs)
    md = r.to_markdown()
    assert "could not run" not in md.lower()
    assert md.startswith("# ")
    assert r.summary() in md
    for f in r.findings:
        assert f.title in md
        assert f.detail in md
        assert f.result.summary() in md
    assert len(md) > 1500


def test_a_clean_report_still_names_what_it_measured(clean_inputs):
    """The numbers are the value of a clean report. They stay in it."""
    r = audit(**clean_inputs)
    assert _one(r, "compare").result.crosses_zero is False
    assert _one(r, "power").result.difference <= abs(
        _one(r, "compare").result.difference
    )
    assert _one(r, "agreement").result.alpha >= 0.67
    assert _one(r, "judge").result.agreement >= 0.67
    assert not _one(r, "position").result.has_position_effect
    assert len(_by_check(r, "scores")) == 2
