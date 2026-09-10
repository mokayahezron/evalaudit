"""Every public function that takes a confidence level refuses one outside (0, 1).

Four functions checked it and three did not. At confidence=1.5 those three
returned NaN bounds, bounds clipped to -100% and 100%, or an error from
inside numpy, depending on the method, and the signed-rank interval in
compare_paired returned an ordinary-looking interval under a "150% CI"
label. Now all seven say the same sentence, which is the one bradley_terry
already used, copied out here by hand so a change to it anywhere shows up.

The list of functions is checked against the package rather than trusted.
A new public function that takes ``confidence`` fails the first test until
it is added here, which is how the three that had no check went unnoticed.
"""

import inspect
import math
import re

import pandas as pd
import pytest

import evalaudit


def ratings():
    return pd.DataFrame({
        "item_id": ["a", "a", "b", "b", "c", "c"],
        "rater_id": ["r1", "r2", "r1", "r2", "r1", "r2"],
        "rating": [1, 1, 0, 1, 0, 0],
    })


def comparisons():
    return pd.DataFrame({
        "item_id": ["q1", "q2", "q3"],
        "model_a": ["x", "y", "x"],
        "model_b": ["y", "x", "y"],
        "winner": ["x", "x", "y"],
    })


# One valid call per function, with the level swapped in. audit takes the
# level through its config rather than its signature, so it is listed by hand.
CALLS = {
    "score_ci": lambda c: evalaudit.score_ci([1, 0, 1, 1], confidence=c),
    "compare_paired": lambda c: evalaudit.compare_paired(
        [1, 0, 1, 1], [0, 0, 1, 0], confidence=c
    ),
    "compare_independent": lambda c: evalaudit.compare_independent(
        [1, 0, 1, 1], [0, 0, 1], confidence=c
    ),
    "rater_agreement": lambda c: evalaudit.rater_agreement(
        ratings(), confidence=c, bootstrap_ci=False
    ),
    "judge_validation": lambda c: evalaudit.judge_validation(
        [0, 1, 1, 0], [0, 1, 0, 0], confidence=c, n_boot=0
    ),
    "bradley_terry": lambda c: evalaudit.bradley_terry(
        comparisons(), confidence=c, n_boot=0
    ),
    "audit": lambda c: evalaudit.audit(
        scores=[1, 0, 1, 1], config={"confidence": c}
    ),
}

# Zero and one are the two a check written as 0 <= c <= 1 lets through.
BAD_LEVELS = [0.0, 1.0, 1.5, -0.2, math.nan]


def refusal(level):
    return re.escape(f"confidence must be in (0, 1), got {level}")


def test_every_public_function_taking_confidence_is_listed():
    taking = {
        name for name in evalaudit.__all__
        if inspect.isfunction(getattr(evalaudit, name))
        and "confidence" in inspect.signature(getattr(evalaudit, name)).parameters
    }
    assert taking == set(CALLS) - {"audit"}


@pytest.mark.parametrize("level", BAD_LEVELS, ids=str)
@pytest.mark.parametrize("name", sorted(CALLS))
def test_a_level_outside_the_unit_interval_is_refused(name, level):
    with pytest.raises(ValueError, match=refusal(level)):
        CALLS[name](level)


@pytest.mark.parametrize("name", sorted(CALLS))
def test_levels_just_inside_the_unit_interval_still_run(name):
    for level in (0.5, 0.999):
        assert CALLS[name](level) is not None


# The refusal has to come before the method is chosen. These data are 0/1,
# so a check placed after dispatch would meet the t or wilcoxon refusal first
# and the caller would be told about the wrong mistake.
METHOD_CALLS = [
    ("score_ci", m, lambda c, m=m: evalaudit.score_ci(
        [1, 0, 1, 1], method=m, confidence=c))
    for m in ("auto", "wilson", "bootstrap", "t")
] + [
    ("compare_paired", m, lambda c, m=m: evalaudit.compare_paired(
        [1, 0, 1, 1], [0, 0, 1, 0], method=m, confidence=c))
    for m in ("auto", "mcnemar", "bootstrap", "wilcoxon")
] + [
    ("compare_independent", m, lambda c, m=m: evalaudit.compare_independent(
        [1, 0, 1, 1], [0, 0, 1], method=m, confidence=c))
    for m in ("auto", "score", "bootstrap", "t")
]


@pytest.mark.parametrize(
    "name, method, call", METHOD_CALLS,
    ids=[f"{name}-{method}" for name, method, _ in METHOD_CALLS],
)
def test_the_refusal_comes_before_the_method_is_chosen(name, method, call):
    with pytest.raises(ValueError, match=refusal(1.5)):
        call(1.5)
