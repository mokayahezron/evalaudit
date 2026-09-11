"""One call over whatever data the client actually has.

This is the function the consulting work runs and the reason the other five
modules exist. You are handed a finished evaluation, some mixture of scores,
grader ratings, judge labels and pairwise judgements, and a conclusion
somebody has already written into a slide. The job is to say which parts of
that conclusion the data supports.

Seven checks run, each one on the data that supports it and never on data
that does not. A check with nothing behind it is reported as a check that
could not run, with the reason, because silence about a check reads as a
pass.

Two things here matter more than the arithmetic, which lives in the modules
and is tested there.

The ranking. A report that lists seven statistics in the order the modules
happen to run is worth nothing. Findings sort by severity first, so the
report opens with the thing that changes the decision, and by CHECK_ORDER
inside a severity.

The prose. Every finding carries the underlying result's own ``summary()``
word for word, framed by a line on what it means for the claim and a line on
what to do about it. The audit never restates a number in its own words,
because a second set of sentences about the same numbers drifts from the
first one the moment either changes.
"""

from __future__ import annotations

from dataclasses import fields as _fields
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from ._types import (
    _SEVERITIES,
    AuditConfig,
    AuditReport,
    Finding,
    SkippedCheck,
)
from .agreement import _LEVELS, rater_agreement
from .compare import compare_independent, compare_paired
from .judge import judge_validation, length_bias, position_bias
from .power import detectable_effect
from .scores import _is_binary, score_ci

__all__ = ["audit"]

SEVERITY_ORDER = _SEVERITIES

# How findings sort inside one severity, and it is not module order.
#
# The margin comes first because the margin is the number on the slide. If
# it does not survive a proper fit, nothing further down changes what the
# client has to be told. Power sits behind it because it says what size of
# difference the design could find. Then the two checks on whether the
# labels underneath mean anything, then the two judge biases, then the
# score intervals, which are context by construction.
#
# agreement and judge sitting below compare is deliberate and is not a
# statement about their importance. They are what makes this package worth
# building. But severity does the real work in this ordering: a critical
# agreement finding already outranks a compare warning, and this tuple only
# breaks ties inside one severity. Reordering it on the instinct that the
# best module should be first makes the report worse, because it moves the
# slide deck number down the page for reasons the reader does not share.
CHECK_ORDER = (
    "compare",
    "power",
    "agreement",
    "judge",
    "position",
    "length",
    "scores",
)

# What each check is called when it could not run. Findings carry their own
# titles, which say what was found rather than what was looked at.
CHECK_TITLES = {
    "compare": "Headline margin",
    "power": "What this eval could have detected",
    "agreement": "Rater agreement",
    "judge": "Judge against humans",
    "position": "Position bias",
    "length": "Length bias",
    "scores": "Score intervals",
}

_JUDGE_KEYS = (
    "human",
    "judge",
    "slices",
    "preferences",
    "lengths",
    "human_preferences",
)


def audit(
    scores=None,
    comparisons=None,
    ratings=None,
    judge=None,
    config=None,
) -> AuditReport:
    """Run every check the supplied data supports and rank what comes back.

    Parameters
    ----------
    scores
        Per-item scores, either a mapping of system name to scores or a bare
        sequence for a single system. Binary 0/1 or continuous. Two systems
        get a comparison and a power check on top of their intervals. Three
        or more get intervals only, since which pair to compare is not the
        audit's guess to make.
    comparisons
        A frame with ``pair_id``, ``option_a``, ``option_b`` and ``winner``,
        as :func:`evalaudit.position_bias` takes it. One row per pairwise
        judgement.
    ratings
        A long frame with ``item_id``, ``rater_id`` and ``rating``, as
        :func:`evalaudit.rater_agreement` takes it. One row per rating given.
    judge
        A mapping. ``human`` and ``judge`` are aligned label sequences and
        run the validation check, and ``slices`` alongside them is the
        argument worth supplying, since a judge that fails on the close
        calls posts a fine headline number. ``preferences`` and ``lengths``
        run the length check, with ``human_preferences`` sharpening it.
    config
        An :class:`evalaudit.AuditConfig` or a mapping of its fields. The
        ones worth setting are ``claim``, ``effect_of_interest`` and
        ``claims_direction``.

    Returns
    -------
    AuditReport

    Notes
    -----
    Nothing here is inferred from data that was not supplied. Every check
    either produces a finding or lands in ``not_run`` with the reason, so an
    audit of scores alone says out loud that it never looked at the graders.

    The paired versus independent question is read off the data when
    ``config.paired`` is None. Two systems with the same number of items are
    taken as paired, which is how most evals are built and which the finding
    says out loud so it can be corrected.

    When no effect_of_interest is stated, the power check reports the
    design's reach, the smallest difference the eval could find at the
    configured power, and makes no claim about the margin it measured. Power
    computed on an observed margin is a function of the p-value, so it would
    only repeat the comparison. The finding is a warning, because without a
    stated effect the audit does not know what size of difference matters.

    With a stated effect, the check answers a different question depending
    on what the comparison found. When the margin's interval includes zero,
    or there is no comparison to read, it asks whether the eval could have
    found the effect at issue at all. That is the question the power module
    was built for, and an eval that could not is a critical finding. When the
    interval clears zero the effect has been observed, and a figure computed
    before the first item was graded cannot overturn it. The finding is then
    context on what a tighter estimate would cost. It is always info, so it
    never decides the report's verdict. A null margin whose interval stops
    short of the stated effect in both directions is info for the same
    reason, since the data has already answered the question power asks.

    Rater agreement and judge agreement are held to their thresholds by the
    interval on alpha, the way the margin is held to zero. An interval wholly
    above the line passes, and one wholly below it is a warning. One running
    both sides of it, or no interval at all, is a warning that the data
    cannot show the grades clear the line.

    Examples
    --------
    >>> report = audit(scores={"new": new, "old": old})  # doctest: +SKIP
    >>> print(report.summary())  # doctest: +SKIP
    >>> print(report.to_markdown())  # doctest: +SKIP
    """
    cfg = _config(config)
    systems = _systems(scores)
    judge_data = _judge_data(judge)

    if not systems and comparisons is None and ratings is None and not judge_data:
        raise ValueError(
            "audit needs at least one of scores, comparisons, ratings or judge"
        )

    findings: list = []
    skipped: list = []

    def record(check, outcome):
        """Every check returns a Finding, a list of them, or a reason."""
        if isinstance(outcome, str):
            skipped.append(SkippedCheck(check, CHECK_TITLES[check], outcome))
        elif isinstance(outcome, list):
            findings.extend(outcome)
        else:
            findings.append(outcome)

    record("scores", _score_findings(systems, cfg))
    comparison = _comparison(systems, cfg)
    record("compare", _compare_finding(comparison, systems, cfg))
    record("power", _power_finding(comparison, systems, cfg))
    record("agreement", _agreement_finding(ratings, cfg))
    record("judge", _judge_finding(judge_data, cfg))
    record("position", _position_finding(comparisons, cfg))
    record("length", _length_finding(judge_data, cfg))

    skipped.sort(key=lambda s: CHECK_ORDER.index(s.check))
    return AuditReport(
        findings=_ranked(findings), not_run=tuple(skipped), config=cfg
    )


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------

def _ranked(findings) -> tuple:
    """Sort by severity, then by CHECK_ORDER, keeping ties as they came.

    The sort is stable, so two findings from one check stay in the order the
    check produced them. That is what keeps the score intervals in the order
    the systems were supplied in.
    """
    return tuple(
        sorted(
            findings,
            key=lambda f: (
                SEVERITY_ORDER.index(f.severity),
                CHECK_ORDER.index(f.check),
            ),
        )
    )


# --------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------

def _config(config) -> AuditConfig:
    if config is None:
        return _validated(AuditConfig())
    if isinstance(config, AuditConfig):
        return _validated(config)
    if not isinstance(config, Mapping):
        raise ValueError(
            "config must be an AuditConfig or a mapping of its fields, got "
            f"{type(config).__name__}"
        )
    valid = [f.name for f in _fields(AuditConfig)]
    unknown = sorted(set(config) - set(valid))
    if unknown:
        raise ValueError(
            f"unknown config keys {unknown}. Valid keys are {valid}"
        )
    return _validated(AuditConfig(**config))


def _validated(cfg: AuditConfig) -> AuditConfig:
    if not 0 < cfg.confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {cfg.confidence}")
    if not 0 < cfg.power < 1:
        raise ValueError(f"power must be in (0, 1), got {cfg.power}")
    if not 0 < cfg.alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {cfg.alpha}")
    if cfg.level not in _LEVELS:
        raise ValueError(
            f"level must be one of {sorted(_LEVELS)}, got {cfg.level!r}"
        )
    if cfg.effect_of_interest is not None and not cfg.effect_of_interest > 0:
        raise ValueError(
            "effect_of_interest must be greater than zero, got "
            f"{cfg.effect_of_interest}. It is the size of the difference the "
            "decision turns on, so zero names no difference at all"
        )
    if cfg.n_boot < 0:
        raise ValueError(f"n_boot must not be negative, got {cfg.n_boot}")
    return cfg


def _systems(scores) -> dict:
    """Score sequences by system name, in the order they were supplied."""
    if scores is None:
        return {}
    if isinstance(scores, pd.DataFrame):
        raise ValueError(
            "scores must be a sequence or a mapping of system name to "
            "scores. For a frame, pass a mapping of the columns you want "
            "compared, such as {name: frame[name] for name in (...)}"
        )
    if isinstance(scores, Mapping):
        return {str(name): np.asarray(list(v)) for name, v in scores.items()}
    return {"system": np.asarray(list(scores))}


def _judge_data(judge) -> dict:
    if judge is None:
        return {}
    if not isinstance(judge, Mapping):
        raise ValueError(
            f"judge must be a mapping with keys from {list(_JUDGE_KEYS)}, "
            f"got {type(judge).__name__}"
        )
    unknown = sorted(set(judge) - set(_JUDGE_KEYS))
    if unknown:
        raise ValueError(
            f"unknown judge keys {unknown}. Valid keys are {list(_JUDGE_KEYS)}"
        )
    return dict(judge)


# --------------------------------------------------------------------------
# Score intervals. Context, and the foundation the margin sits on.
# --------------------------------------------------------------------------

def _score_findings(systems: dict, cfg: AuditConfig):
    if not systems:
        return (
            "No scores supplied. Pass scores as a sequence, or as a mapping "
            "of system name to scores."
        )
    return [
        Finding(
            check="scores",
            severity="info",
            title=f"Score interval for {name}",
            detail=_detail(
                f"The headline number for {name}, with the interval around "
                f"it.",
                result,
                "Quote the interval wherever the number appears. A point "
                "estimate on its own invites a reader to compare it with "
                "something it cannot be compared with.",
            ),
            result=result,
        )
        for name, result in (
            (
                name,
                score_ci(values, confidence=cfg.confidence, seed=cfg.seed),
            )
            for name, values in systems.items()
        )
    ]


# --------------------------------------------------------------------------
# Check 2. The headline margin
# --------------------------------------------------------------------------

def _comparison(systems: dict, cfg: AuditConfig):
    """The comparison itself, or None when there is not exactly one to make."""
    names = list(systems)
    if len(names) != 2:
        return None
    a, b = systems[names[0]], systems[names[1]]
    paired = (a.size == b.size) if cfg.paired is None else cfg.paired
    if paired and a.size != b.size:
        raise ValueError(
            f"a paired comparison needs the same number of items from both "
            f"systems, got {a.size} and {b.size}. Set paired=False in the "
            f"config if the two were run on different items"
        )
    if paired:
        return compare_paired(a, b, confidence=cfg.confidence, seed=cfg.seed)
    return compare_independent(a, b, confidence=cfg.confidence, seed=cfg.seed)


def _compare_finding(comparison, systems: dict, cfg: AuditConfig):
    if comparison is None:
        return _no_comparison_reason(systems)

    names = list(systems)
    if comparison.crosses_zero:
        severity = "critical" if cfg.claims_direction else "warning"
        lead = (
            f"This is the margin between {names[0]} and {names[1]}, which is "
            f"the number the claim rests on."
        )
        action = (
            "Do not report a direction from this data. Either run enough "
            "more items to close the interval, or state the result as "
            "unresolved."
        )
        if not cfg.claims_direction:
            action += (
                " Nothing is claiming a direction here, so this weakens the "
                "result rather than undoing it."
            )
    else:
        severity = "info"
        lead = (
            f"This is the margin between {names[0]} and {names[1]}, and it "
            f"survives the fit."
        )
        action = (
            "Report the interval alongside the difference. The end of the "
            "interval nearer zero is the smallest margin this data supports."
        )

    if cfg.paired is None and comparison.paired:
        action += (
            " Both systems scored the same number of items, so this was read "
            "as a paired comparison. Set paired=False in the config if they "
            "were run on different items."
        )

    return Finding(
        check="compare",
        severity=severity,
        title=(
            "The headline margin does not exclude zero"
            if comparison.crosses_zero
            else "The headline margin clears zero"
        ),
        detail=_detail(lead, comparison, action),
        result=comparison,
    )


def _no_comparison_reason(systems: dict) -> str:
    if not systems:
        return (
            "No scores supplied, so there is no margin to test. Pass scores "
            "for the two systems being compared."
        )
    if len(systems) == 1:
        return (
            "Only one system was scored, so there is no margin to test. Pass "
            "scores for two systems."
        )
    return (
        f"Scores for {len(systems)} systems were supplied and this check "
        f"compares two. Pass the two the claim is about."
    )


# --------------------------------------------------------------------------
# Check 3. What the eval could have detected
# --------------------------------------------------------------------------

def _power_finding(comparison, systems: dict, cfg: AuditConfig):
    effect = cfg.effect_of_interest
    if effect is None and comparison is None:
        return (
            "No effect of interest was stated and there is no comparison "
            "whose reach could be reported. Pass effect_of_interest in the "
            "config, or scores for two systems."
        )

    design = _power_design(comparison, systems, cfg)
    if isinstance(design, str):
        return design
    n, paired, binary = design

    if not binary:
        return (
            "The power arithmetic here is for pass rates and these scores "
            "are not 0/1. Nothing to report rather than a figure that does "
            "not apply."
        )

    result = detectable_effect(
        n,
        baseline=_baseline(systems),
        power=cfg.power,
        alpha=cfg.alpha,
        paired=paired,
        discordance_rate=_discordance(systems, paired, cfg),
    )
    if effect is None:
        return _reach_finding(result)

    lead = (
        f"The effect at issue is {_points(effect)}, as supplied in the "
        f"config."
    )
    if comparison is not None and not comparison.crosses_zero:
        return _established_power_finding(lead, result)

    reached = result.attainable and result.difference <= effect
    if reached:
        return Finding(
            check="power",
            severity="info",
            title="This eval was large enough for the effect at issue",
            detail=_detail(
                lead,
                result,
                "Nothing to do here. A null result from this eval would be "
                "worth something, because the eval could have found the "
                "effect if it were there.",
            ),
            result=result,
        )
    if comparison is not None and _rules_out(comparison, effect):
        return _ruled_out_power_finding(lead, result)
    return Finding(
        check="power",
        severity="critical",
        title=(
            f"This eval had under {cfg.power * 100:g}% power for the effect "
            f"at issue"
        ),
        detail=_detail(
            lead,
            result,
            _null_reading(comparison, systems, effect)
            + " Size the next run off the figure above before grading "
            "anything.",
        ),
        result=result,
    )


def _rules_out(comparison, effect: float) -> bool:
    """True when the margin's interval stops short of the effect both ways."""
    return -effect < comparison.ci_low and comparison.ci_high < effect


def _null_reading(comparison, systems: dict, effect: float) -> str:
    """What the margin's interval says about an effect this size.

    Read one side at a time. A margin whose interval includes zero can still
    rule an effect out in one direction, and a sentence saying nothing can
    be concluded either way would throw that away.
    """
    if comparison is None:
        return (
            "No margin was measured here, so there is no interval to read "
            "against an effect this size."
        )
    first, second = list(systems)
    if comparison.ci_high < effect:
        return (
            f"The interval on the margin rules out {first} ahead by this "
            f"much, and it cannot rule out {second} ahead by this much."
        )
    if comparison.ci_low > -effect:
        return (
            f"The interval on the margin rules out {second} ahead by this "
            f"much, and it cannot rule out {first} ahead by this much."
        )
    return (
        "The interval on the margin runs past an effect this size in both "
        "directions, so the data cannot show an effect this size and cannot "
        "rule one out."
    )


def _ruled_out_power_finding(lead: str, result) -> Finding:
    """Power beside a null margin whose interval rules the effect out.

    The mirror of :func:`_established_power_finding`. A design figure says
    what the eval could be expected to find before it ran. Once the interval
    on the margin stops short of the effect in both directions, the data has
    answered the question power asks, so the finding is info. Only a stated
    effect can land here, since an interval always contains the margin it
    was built around.
    """
    return Finding(
        check="power",
        severity="info",
        title="The interval on the margin rules out the effect at issue",
        detail=_detail(
            lead + " The interval on the margin stops short of it in both "
            "directions, so the figures that follow bear on the design "
            "alone.",
            result,
            "Power is a design figure, and it has no bearing on a question "
            "the interval already answers. The data rules out a difference "
            "this large in either direction, whatever the design promised.",
        ),
        result=result,
    )


def _established_power_finding(lead: str, result) -> Finding:
    """Power beside a margin whose interval already clears zero.

    The smallest difference an eval reaches at a given power is a design
    figure. It says what the eval could be expected to find before the first
    item was graded. Once a margin has been found and its interval clears
    zero, it has nothing left to say about whether the margin is there. What
    it still bears on is precision, so this finding is info whatever the
    figure is, and it is written as the cost of a tighter estimate.

    The branch reads the interval and never the point estimate. A five point
    margin whose interval includes zero is the null case. A five point
    margin below the design's reach whose interval clears zero is this one.
    """
    return Finding(
        check="power",
        severity="info",
        title=(
            "The margin is established, and a tighter estimate would take "
            "more items"
        ),
        detail=_detail(
            lead + " The comparison already clears zero, so the figures that "
            "follow bear on how precisely that margin is measured.",
            result,
            "Nothing here weakens the result. Power is a design figure, and "
            "it has no bearing on a margin whose interval already clears "
            "zero. What more items would buy is a narrower interval on that "
            "margin. The width shrinks with the square root of the item "
            "count, so halving it takes about four times the items.",
        ),
        result=result,
    )


def _reach_finding(result) -> Finding:
    """Power with no stated effect. The design's reach, and nothing else.

    The audit used to make the observed margin the effect at issue when none
    was stated. Power computed on an observed effect is a monotone function
    of the p-value, so on a margin whose interval includes zero it came out
    short almost every time and repeated the comparison, and on one that
    clears zero it had nothing to add. So this reports the smallest
    difference the design could find at the configured power and says
    nothing about the margin. It is a warning because the question power
    exists to answer, whether the eval could find the size of difference
    that matters, has not been asked.
    """
    return Finding(
        check="power",
        severity="warning",
        title=(
            "No effect of interest was stated, so this reports the eval's "
            "reach"
        ),
        detail=_detail(
            "Without a stated effect the audit does not know what size of "
            "difference matters. This reports what the design could find and "
            "makes no claim about the margin it measured.",
            result,
            "Set effect_of_interest in the config to the difference the "
            "decision turns on, and this check will say whether the eval "
            "could find it.",
        ),
        result=result,
    )


def _power_design(comparison, systems: dict, cfg: AuditConfig):
    """Items, pairing and whether the data is binary, or why not."""
    if comparison is not None:
        return comparison.n, comparison.paired, comparison.binary
    if len(systems) == 1:
        values = next(iter(systems.values()))
        paired = True if cfg.paired is None else cfg.paired
        return int(values.size), paired, _is_binary(values)
    if not systems:
        return (
            "No scores supplied, so there is no sample size to check. Pass "
            "scores for the system or systems the eval ran."
        )
    return (
        f"Scores for {len(systems)} systems were supplied, so which "
        f"comparison to size is ambiguous. Pass the two the claim is about."
    )


def _discordance(systems: dict, paired: bool, cfg: AuditConfig):
    """The measured rate when the data carries one, else whatever was set.

    Paired binary power runs on the share of items the two systems disagree
    on, and that share is sitting in the data. Measuring it beats both a
    supplied figure and the conservative default the power module stands in.
    """
    if not paired or len(systems) != 2:
        return cfg.discordance_rate
    a, b = list(systems.values())
    if a.size != b.size or not (_is_binary(a) and _is_binary(b)):
        return cfg.discordance_rate
    return float(np.mean(a != b))


def _baseline(systems: dict) -> float:
    """The rate being compared against, for the independent arithmetic.

    Paired binary power does not use it at all. It still has to be a rate,
    so anything that is not one falls back on a half.
    """
    if not systems:
        return 0.5
    reference = list(systems.values())[-1]
    mean = float(np.mean(reference))
    return mean if 0 < mean < 1 else 0.5


# --------------------------------------------------------------------------
# Check 1. Rater agreement
# --------------------------------------------------------------------------

def _agreement_finding(ratings, cfg: AuditConfig):
    if ratings is None:
        return (
            "No ratings supplied. Pass ratings as a long frame with "
            "item_id, rater_id and rating columns, one row per rating given."
        )

    result = rater_agreement(
        ratings,
        level=cfg.level,
        confidence=cfg.confidence,
        bootstrap_ci=cfg.n_boot > 0,
        n_boot=cfg.n_boot,
        seed=cfg.seed,
    )

    if result.n_overlapping_items == 0:
        return Finding(
            check="agreement",
            severity="critical",
            title="Agreement was never measurable",
            detail=_detail(
                "Every conclusion in the eval rests on these grades, and "
                "this asks whether the people giving them agreed.",
                result,
                "Nothing in the eval can be defended as reproducible until "
                "some items are graded twice. Take a sample, have a second "
                "person grade it blind, and run this again before anything "
                "else in the report is acted on.",
            ),
            result=result,
        )

    alpha = result.alpha
    if alpha != alpha:  # NaN
        return Finding(
            check="agreement",
            severity="warning",
            title="Rater agreement is undefined on this data",
            detail=_detail(
                f"The report holds the grades to "
                f"{cfg.agreement_threshold:.3f}, and this data gives no alpha "
                f"to hold against it.",
                result,
                "Until there is an alpha, the eval has not shown that its "
                "grades reproduce. Grade a larger sample twice and run this "
                "again. If every compared rating came out the same, check "
                "that the rubric leaves graders room to differ.",
            ),
            result=result,
        )

    # The interval decides, the way it does for the margin. Wholly below the
    # line is the warning, wholly above is the pass, and an interval running
    # both sides, or none at all, has not shown which.
    threshold = cfg.agreement_threshold
    if result.has_interval and result.ci_high < threshold:
        return Finding(
            check="agreement",
            severity="warning",
            title="Rater agreement is below the working threshold",
            detail=_detail(
                f"The grades under this eval reproduce less well than the "
                f"{cfg.agreement_threshold:.3f} the report is holding them "
                f"to.",
                result,
                "The eval is measuring the rubric as much as the systems. "
                "Read the item table for the cases the graders split on, "
                "since ambiguous wording is the usual cause and it is "
                "cheaper to fix than more grading.",
            ),
            result=result,
        )

    if not (result.has_interval and result.ci_low > threshold):
        return Finding(
            check="agreement",
            severity="warning",
            title=(
                "The data cannot show that rater agreement clears the "
                "working threshold"
            ),
            detail=_detail(
                f"The report holds the grades to {threshold:.3f}, and "
                + (
                    "the interval on alpha runs both sides of it."
                    if result.has_interval
                    else "without an interval on alpha nothing places the "
                    "estimate against it."
                ),
                result,
                "The eval has not shown that its grades reproduce at the "
                "level this report holds them to. That does not mean they "
                "fall short of it. Grading more items twice narrows the "
                "interval.",
            ),
            result=result,
        )

    return Finding(
        check="agreement",
        severity="info",
        title="Rater agreement holds up",
        detail=_detail(
            f"The grades reproduce at or above the "
            f"{cfg.agreement_threshold:.3f} the report is holding them to.",
            result,
            "Nothing to do here. Keep the double graded sample in the next "
            "round so this stays measurable.",
        ),
        result=result,
    )


# --------------------------------------------------------------------------
# Check 4. The judge against the humans
# --------------------------------------------------------------------------

def _judge_finding(judge_data: dict, cfg: AuditConfig):
    if "human" not in judge_data or "judge" not in judge_data:
        return (
            "No judge labels supplied. Pass judge with human and judge "
            "label sequences for the same items, and slices alongside them "
            "if the eval is cut by difficulty or task type."
        )

    result = judge_validation(
        judge_data["human"],
        judge_data["judge"],
        slices=judge_data.get("slices"),
        level=cfg.level,
        confidence=cfg.confidence,
        n_boot=cfg.n_boot,
        seed=cfg.seed,
    )

    if result.slice_is_distinguishable:
        return Finding(
            check="judge",
            severity="critical",
            title="The judge comes apart on one slice of the data",
            detail=_detail(
                "The judge is standing in for the humans across the whole "
                "eval, so it has to track them everywhere the eval draws a "
                "conclusion.",
                result,
                "Any ranking that leans on this slice is unsupported, "
                "whatever the headline agreement says. Have humans label "
                "that slice, or exclude it and say so, before the "
                "comparison is used to pick anything.",
            ),
            result=result,
        )

    agreement = result.agreement
    if agreement != agreement:  # NaN
        return Finding(
            check="judge",
            severity="warning",
            title="Judge-human agreement is undefined on this data",
            detail=_detail(
                f"The report holds the judge to {cfg.judge_threshold:.3f}, "
                f"and this data gives no alpha to hold against it.",
                result,
                "Until there is an alpha, the eval has not shown that the "
                "judge tracks the humans. Label more items by hand and run "
                "this again. If every compared label came out the same, "
                "check that the rubric leaves the judge and the humans room "
                "to differ.",
            ),
            result=result,
        )

    # The same rule as agreement. The interval decides.
    threshold = cfg.judge_threshold
    if result.has_interval and result.ci_high < threshold:
        return Finding(
            check="judge",
            severity="warning",
            title="Judge-human agreement is below the working threshold",
            detail=_detail(
                f"The judge is standing in for the humans, and it tracks "
                f"them less closely than the {cfg.judge_threshold:.3f} this "
                f"report is holding it to.",
                result,
                "That threshold is a convention rather than a validated "
                "line, so the question is whether this level of agreement "
                "is good enough for the decision being made. Say which "
                "decision, and set judge_threshold from it.",
            ),
            result=result,
        )

    if not (result.has_interval and result.ci_low > threshold):
        return Finding(
            check="judge",
            severity="warning",
            title=(
                "The data cannot show that the judge clears the working "
                "threshold"
            ),
            detail=_detail(
                f"The report holds the judge to {threshold:.3f}, and "
                + (
                    "the interval on its agreement with the humans runs both "
                    "sides of it."
                    if result.has_interval
                    else "without an interval on its agreement with the "
                    "humans nothing places the estimate against it."
                ),
                result,
                "The eval has not shown that the judge tracks the humans at "
                "the level this report holds it to. That does not mean it "
                "falls short of it. Labelling more items by hand narrows the "
                "interval, and the threshold is a convention, so set "
                "judge_threshold from the decision being made.",
            ),
            result=result,
        )

    return Finding(
        check="judge",
        severity="info",
        title="The judge tracks the humans",
        detail=_detail(
            f"The judge agrees with the humans at or above the "
            f"{cfg.judge_threshold:.3f} this report is holding it to.",
            result,
            "Nothing to do here. Keep labelling a slice of items by hand "
            "each round, since a judge that drifts shows up nowhere else.",
        ),
        result=result,
    )


# --------------------------------------------------------------------------
# Check 5. Position and length bias
# --------------------------------------------------------------------------

def _position_finding(comparisons, cfg: AuditConfig):
    if comparisons is None:
        return (
            "No pairwise judgements supplied. Pass comparisons as a frame "
            "with pair_id, option_a, option_b and winner columns, one row "
            "per judgement."
        )

    result = position_bias(comparisons, seed=cfg.seed)

    if result.has_position_effect:
        return Finding(
            check="position",
            severity="warning",
            title="The judge favours whichever answer it reads first",
            detail=_detail(
                "A judge that reads position rather than quality can rank "
                "two systems wrongly while agreeing with humans often "
                "enough to pass a validation.",
                result,
                "Run every pair in both orders and keep only the pairs the "
                "judge called the same way twice. Until then the margin "
                "from these judgements carries an effect that has nothing "
                "to do with the systems.",
            ),
            result=result,
        )

    return Finding(
        check="position",
        severity="info",
        title="The data cannot show a position effect in the pairwise "
        "judgements",
        detail=_detail(
            "This asks whether the judge is reading position rather than "
            "quality.",
            result,
            "The data has not cleared the judge of position bias. Keep the "
            "order shuffled in the next round, since this is a property of "
            "the setup rather than of the model.",
        ),
        result=result,
    )


def _length_finding(judge_data: dict, cfg: AuditConfig):
    if "preferences" not in judge_data or "lengths" not in judge_data:
        return (
            "No judge preferences and lengths supplied. Pass judge with "
            "preferences and lengths for the same pairs, and "
            "human_preferences alongside them to sharpen the fit."
        )

    result = length_bias(
        judge_data["preferences"],
        judge_data["lengths"],
        human_preferences=judge_data.get("human_preferences"),
    )

    if _length_flagged(result):
        _, _, coefficient = _deciding_fit(result)
        toward_long = coefficient > 0
        return Finding(
            check="length",
            severity="warning",
            title=(
                "The judge is pulled by how long the answer is"
                if toward_long
                else "The judge is pulled toward shorter answers"
            ),
            detail=_detail(
                _length_lead(result, toward_long),
                result,
                (
                    "Check whether the winning system is simply the longer "
                    "one. "
                    if toward_long
                    else "Check whether the winning system is simply the "
                    "shorter one. "
                )
                + "Comparing answers trimmed to a common length, or scoring "
                "with length in the rubric rather than in the judge, will "
                "say whether the margin survives.",
            ),
            result=result,
        )

    return Finding(
        check="length",
        severity="info",
        title="The data cannot show a length effect in the judge's choices",
        detail=_detail(
            "This asks whether the judge is rewarding length rather than "
            "quality.",
            result,
            "The data has not cleared the judge of rewarding length.",
        ),
        result=result,
    )


def _length_lead(result, toward_long: bool) -> str:
    """The sentence that frames the numbers, before the reader meets them.

    Three cases, and the third is the one that costs a reader most. When
    the two fits sign differently the paragraph carries a title pointing
    one way and an odds ratio pointing the other, both correct, with
    nothing between them saying why. Every sentence is defensible and the
    paragraph still reads as a contradiction. So the warning goes first,
    where it arrives before the numbers rather than after them.
    """
    if _fits_disagree(result):
        if toward_long:
            return (
                "Two models run here and they point opposite ways. The "
                "judge picks the shorter answer more often, and on the "
                "pairs where it breaks with the humans it breaks toward "
                "the longer one. The verdict below runs on the second, "
                "which is the sharper of the two."
            )
        return (
            "Two models run here and they point opposite ways. The judge "
            "picks the longer answer more often, and on the pairs where it "
            "breaks with the humans it breaks toward the shorter one. The "
            "verdict below runs on the second, which is the sharper of the "
            "two."
        )
    if toward_long:
        return (
            "A judge that rewards length ranks the wordier system higher "
            "whatever it says, and it can do that while agreeing with "
            "humans on most pairs."
        )
    return (
        "A judge that rewards brevity ranks the terser system higher "
        "whatever it says, and it can do that while agreeing with humans "
        "on most pairs."
    )


def _fits_disagree(result) -> bool:
    """True when the verdict runs on the disagreement fit and they differ.

    Two fits that sign differently are only worth warning about when the
    one the reader is being handed is not the one the raw preference number
    will suggest, and when the preference fit's own interval clears zero.
    Otherwise the lead would state a preference the data has not shown.
    """
    if not _uses_disagreement_fit(result):
        return False
    if not (
        np.isfinite(result.coefficient)
        and np.isfinite(result.disagreement_coefficient)
    ):
        return False
    if result.ci_low <= 0.0 <= result.ci_high:
        # The lead says which answer the judge picks more often, a claim
        # about the preference fit. With that fit's interval running over
        # zero the claim has not been shown, so the plain lead runs.
        return False
    return (result.coefficient > 0) != (result.disagreement_coefficient > 0)


def _length_flagged(result) -> bool:
    """Whether the deciding fit's interval clears zero, in either direction.

    Which fit that is lives in :func:`_deciding_fit`.
    """
    low, high, _ = _deciding_fit(result)
    if not (np.isfinite(low) and np.isfinite(high)):
        return False
    return not low <= 0.0 <= high


def _uses_disagreement_fit(result) -> bool:
    """Whether the verdict runs on the disagreement fit.

    The selection rule itself, kept in one place so the predicate, the
    direction and the lead cannot drift apart.
    """
    return bool(
        result.has_human and np.isfinite(result.disagreement_ci_low)
    )


def _deciding_fit(result):
    """The bounds and coefficient the verdict runs on, as one selection.

    Whether the finding fires and which way it points have to come off the
    same fit, so the rule for picking it lives in one place rather than
    twice. Positive means pulled toward the longer answer in both fits,
    since the disagreement model's predictor is the length the humans
    passed over minus the one they picked. The two can still sign
    differently. A judge that prefers long answers at every gap, less
    strongly as the gap widens, breaks with the humans toward the shorter
    answer.
    """
    if _uses_disagreement_fit(result):
        return (
            result.disagreement_ci_low,
            result.disagreement_ci_high,
            result.disagreement_coefficient,
        )
    return result.ci_low, result.ci_high, result.coefficient


# --------------------------------------------------------------------------
# Prose
# --------------------------------------------------------------------------

def _detail(lead: str, result, action: str) -> str:
    """What was measured, what the module said, and what to do about it.

    The middle sentence is the module's own ``summary()``, unedited. Audit
    frames it and never paraphrases it, so there is one set of sentences
    about any given number and it lives next to the arithmetic.
    """
    return f"{lead} {result.summary()} {action}"


def _points(effect: float) -> str:
    """A difference on the scale people say it out loud in."""
    return f"{effect * 100:.1f} points"
