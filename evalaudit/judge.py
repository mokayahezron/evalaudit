"""Is the LLM judge standing in for your humans, or only where it is easy?

Three checks, and the first one is the reason the module exists. A judge
scored against human labels almost always posts a respectable headline
number, because most items in most evals are not close calls. The judge
tracks humans on those and comes apart on the ones that decide the result.
So the per-slice table is the finding and the headline is the setup.

The other two are the failure modes that do not show up as disagreement at
all. A judge that prefers whichever answer it reads first, and a judge that
prefers whichever answer is longer, can both agree with humans often enough
to pass a validation and still rank two systems wrongly.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import special as _special
from scipy import stats as _stats

from ._types import (
    _NOTE_SLICE_NO_VARIANCE,
    _NOTE_SLICE_ONE_ITEM,
    JudgeValidation,
    LengthBias,
    PositionBias,
)
from .agreement import _LEVELS, _Units, _bootstrap
from .scores import _wilson

__all__ = ["judge_validation", "position_bias", "length_bias"]

_COMPARISON_COLUMNS = ("pair_id", "option_a", "option_b", "winner")

# A pair counts as run in both orders when two of its rows are exact
# reverses. The design is read as both-orders when at least this share of
# pairs qualify, so a handful of stray duplicates cannot flip the analysis.
_BOTH_ORDERS_SHARE = 0.5

# Why a logistic fit returns no coefficient. Each is a different thing to go
# and fix, so each gets its own sentence rather than a shared failure.
_NOTE_ONE_OUTCOME = "the outcome never varied, so there is nothing to model"
_NOTE_NO_LENGTH_VARIATION = (
    "every pair had the same length difference, so the predictor is constant"
)
_NOTE_SEPARATED = (
    "length difference splits the outcome perfectly, so the coefficient is unbounded"
)
_NOTE_NO_CONVERGENCE = "the fit did not converge"

_MAX_IRLS_STEPS = 100
_IRLS_TOLERANCE = 1e-10


# --------------------------------------------------------------------------
# judge_validation
# --------------------------------------------------------------------------

def judge_validation(
    human: Sequence,
    judge: Sequence,
    slices: Optional[Sequence] = None,
    level: str = "nominal",
    confidence: float = 0.95,
    n_boot: int = 1000,
    seed: Optional[int] = None,
) -> JudgeValidation:
    """Agreement between an LLM judge's labels and human labels.

    Parameters
    ----------
    human, judge
        Labels for the same items, aligned position by position. Items where
        either side has no label are set aside and counted, since an item the
        judge could not label is not a disagreement.
    slices
        One label per item saying which group it belongs to. Difficulty band,
        task type, language, whatever the eval is cut by. This is the
        argument worth supplying. A judge that agrees with humans at 0.85
        overall and at 0.20 on the close calls is not a judge you can rank
        two systems with, and only the breakdown shows it.
    level
        ``"nominal"``, ``"ordinal"`` or ``"interval"``, read exactly as
        ``rater_agreement`` reads them. A 1-5 quality rubric is ordinal, and
        calling it nominal throws away the fact that 4 and 5 are nearly the
        same judgement.
    confidence
        Nominal coverage of the bootstrap interval, default 0.95.
    n_boot
        Bootstrap resamples over items. Set 0 to skip the interval. Then
        nothing in the result will name a slice, because there is no
        sampling error to judge one against.
    seed
        Seeds the bootstrap. Set it in anything you publish.

    Returns
    -------
    JudgeValidation

    Notes
    -----
    The overall agreement is Krippendorff's alpha between the two label
    series, which is the same statistic and the same scale as
    ``rater_agreement``. A judge is a rater.

    Per-slice alpha is computed on that slice alone, so the value domain is
    the one the slice used. That is the right definition for how well the
    judge did there, and it means two slices can carry alphas that are not on
    quite the same footing when they used different label sets.
    """
    if level not in _LEVELS:
        raise ValueError(f"level must be one of {sorted(_LEVELS)}, got {level!r}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    human_raw = np.asarray(list(human), dtype=object)
    judge_raw = np.asarray(list(judge), dtype=object)
    if human_raw.shape != judge_raw.shape:
        raise ValueError(
            f"human and judge must be the same length, got "
            f"{human_raw.size} and {judge_raw.size}"
        )
    if human_raw.size == 0:
        raise ValueError("human and judge must not be empty")

    if slices is None:
        slice_raw = None
    else:
        slice_raw = np.asarray(list(slices), dtype=object)
        if slice_raw.shape != human_raw.shape:
            raise ValueError(
                f"slices must be the same length as human and judge, got "
                f"{slice_raw.size} and {human_raw.size}"
            )

    keep = np.array(
        [_present(a) and _present(b) for a, b in zip(human_raw, judge_raw)]
    )
    n_dropped = int((~keep).sum())
    human_kept = human_raw[keep]
    judge_kept = judge_raw[keep]
    slice_kept = None if slice_raw is None else slice_raw[keep]

    if human_kept.size == 0:
        raise ValueError("no item carries a label from both sides")

    units = _Units.build(_long_frame(human_kept, judge_kept), level)
    agreement = units.alpha()
    ci_low, ci_high, drawn, usable = _bootstrap(
        units, confidence, n_boot > 0, n_boot, seed
    )

    return JudgeValidation(
        agreement=agreement,
        ci_low=ci_low,
        ci_high=ci_high,
        accuracy=float(np.mean(human_kept == judge_kept)),
        n_items=int(human_kept.size),
        n_dropped=n_dropped,
        level=level,
        by_slice=_slice_table(
            human_kept, judge_kept, slice_kept, level, confidence, n_boot, seed
        ),
        confidence=confidence,
        n_boot=drawn,
        n_boot_usable=usable,
    )


def _present(value) -> bool:
    """True unless the label is missing.

    Written out because None, NaN and pd.NA all turn up in label columns and
    a self-comparison alone does not cover the first two.
    """
    if value is None or value is pd.NA:
        return False
    return not (isinstance(value, float) and value != value)


def _long_frame(human: np.ndarray, judge: np.ndarray) -> pd.DataFrame:
    """The two label series as the long ratings frame agreement.py expects.

    Item ids are positions, so the same item is the same row in both series
    and every item ends up with exactly two ratings.
    """
    n = human.size
    positions = np.arange(n)
    return pd.DataFrame({
        "item_id": np.concatenate([positions, positions]),
        "rater_id": np.array(["human"] * n + ["judge"] * n, dtype=object),
        "rating": np.concatenate([human, judge]),
    })


def _slice_table(human, judge, slices, level, confidence, n_boot, seed):
    """Agreement and accuracy within each slice, worst first, with intervals.

    Sorted ascending because the reader looks at the top row, and undefined
    slices sorted last because a slice where both sides used one label is not
    a failure and must not be presented as the worst one in the eval.

    Each slice gets its own bootstrap interval, and the naming rule needs it.
    A slice is a fraction of the data, so its sampling error is wider than
    the one on the overall figure, often several times wider. Comparing a
    slice's point estimate to the overall interval ignores that and names a
    slice on data that is nothing but noise. Cut three hundred homogeneous
    items into five slices of sixty and the lowest one lands below the
    overall lower bound most of the time.
    """
    columns = ["slice", "n_items", "agreement", "ci_low", "ci_high",
               "accuracy", "note"]
    if slices is None:
        return pd.DataFrame(columns=columns)

    rng = np.random.default_rng(seed)
    rows = []
    for name in dict.fromkeys(slices):
        mask = slices == name
        h, j = human[mask], judge[mask]
        units = _Units.build(_long_frame(h, j), level)

        if units.n_units < 2:
            # The item count is checked before the variance, unlike the
            # dropout table in agreement.py. There, a rater whose removal
            # leaves everyone agreeing is the finding. Here one item is too
            # small to measure whatever else happens to be true of it.
            note = _NOTE_SLICE_ONE_ITEM
        elif units.is_degenerate:
            note = _NOTE_SLICE_NO_VARIANCE
        else:
            note = ""

        lo, hi, _, _ = _bootstrap(
            units, confidence, n_boot > 0, n_boot,
            int(rng.integers(2 ** 32)) if n_boot > 0 else None,
        )

        rows.append({
            "slice": name,
            "n_items": int(h.size),
            "agreement": units.alpha(),
            "ci_low": lo,
            "ci_high": hi,
            "accuracy": float(np.mean(h == j)),
            "note": note,
        })

    table = pd.DataFrame(rows, columns=columns)
    return (
        table.sort_values(
            "agreement", ascending=True, na_position="last", kind="mergesort"
        )
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------
# position_bias
# --------------------------------------------------------------------------

def position_bias(
    comparisons: pd.DataFrame, seed: Optional[int] = None
) -> PositionBias:
    """Does the judge favour whichever output it sees first.

    Parameters
    ----------
    comparisons
        A frame with ``pair_id``, ``option_a``, ``option_b`` and ``winner``.
        One row per judgement. ``option_a`` is the output that was shown
        first and ``option_b`` the one shown second, so the two columns carry
        the presentation order rather than a fixed system label. ``winner``
        names the output the judge picked, so it has to match one of the two.
        Leave it missing to record a tie.
    seed
        Accepted so the signature matches the rest of the package. Nothing
        here is resampled. The rate tested against a half gets the
        Clopper-Pearson interval, which inverts the exact binomial test whose
        p-value is printed beside it, so the two cannot disagree about a
        half. The consistency rate has no test beside it and gets a Wilson
        interval, which beats the bootstrap near the boundaries, where
        consistency rates live.

    Returns
    -------
    PositionBias

    Notes
    -----
    The design is read off the data rather than asked for. A pair counts as
    run in both orders when two of its rows are exact reverses of each other,
    and the whole frame is analysed as a both-orders study when at least half
    the pairs qualify. Otherwise it is analysed as a randomised study.

    A mixed frame is analysed entirely under whichever branch wins that
    threshold, not split between the two. Rows the winning branch cannot use
    are set aside rather than analysed the other way, so a both-orders frame
    with a tail of singly judged pairs reports the consistency rate on the
    pairs that have both orderings and says how many pairs that was. Mixing
    the two rates into one number would average quantities with different
    denominators and different meanings.

    The randomised branch cannot tell whether the order really was
    randomised. If the same system sat in position A every time, a position-A
    win rate above a half is exactly what a better system produces, and the
    summary says so.
    """
    data = _clean_comparisons(comparisons)
    design, n_both_orders = _detect_design(data)

    if design == "both_orders":
        return _both_orders_result(data, n_both_orders)
    return _randomised_result(data, n_both_orders)


def _clean_comparisons(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Validate and copy. Every check here is a data-entry mistake."""
    if not isinstance(comparisons, pd.DataFrame):
        raise ValueError("comparisons must be a pandas DataFrame")

    missing = [c for c in _COMPARISON_COLUMNS if c not in comparisons.columns]
    if missing:
        raise ValueError(
            f"comparisons is missing required column(s): {', '.join(missing)}. "
            f"Expected {', '.join(_COMPARISON_COLUMNS)}, where option_a is the "
            f"output shown first and winner names the one the judge picked."
        )

    data = comparisons.loc[:, list(_COMPARISON_COLUMNS)].copy()
    if data.empty:
        raise ValueError("comparisons must not be empty")

    same = data["option_a"] == data["option_b"]
    if same.any():
        first = data[same].iloc[0]
        raise ValueError(
            f"comparisons pits an output against itself on pair "
            f"{first['pair_id']!r}, where both sides are the same output. "
            f"There is no position to measure on a pair like that."
        )

    decided = data["winner"].notna()
    valid = (data["winner"] == data["option_a"]) | (
        data["winner"] == data["option_b"]
    )
    bad = decided & ~valid
    if bad.any():
        first = data[bad].iloc[0]
        raise ValueError(
            f"comparisons has a winner that was not on offer: pair "
            f"{first['pair_id']!r} names {first['winner']!r}, which is "
            f"neither {first['option_a']!r} nor {first['option_b']!r}. Leave "
            f"winner empty to record a tie."
        )
    return data


def _detect_design(data: pd.DataFrame):
    """Which study this is, and how many pairs were run both ways.

    Returns the design name and the count, because the count belongs in the
    summary either way. Under the randomised branch it says how many stray
    reversals were ignored, and under the both-orders branch it says how much
    of the frame the rate rests on.
    """
    n_both = 0
    for _, rows in data.groupby("pair_id", sort=False):
        if _has_both_orders(rows):
            n_both += 1

    n_pairs = int(data["pair_id"].nunique())
    if n_both and n_both >= _BOTH_ORDERS_SHARE * n_pairs:
        return "both_orders", n_both
    return "randomised", n_both


def _has_both_orders(rows: pd.DataFrame) -> bool:
    """True when two of these rows are exact reverses of each other.

    The same pair judged twice with the same output in front is a repeat
    measurement and says nothing about position, so it does not count.
    """
    seen = set()
    for a, b in zip(rows["option_a"], rows["option_b"]):
        if (b, a) in seen:
            return True
        seen.add((a, b))
    return False


def _reversed_couple(rows: pd.DataFrame):
    """The first two rows of a pair that are reverses, or None."""
    seen = {}
    for position, (a, b) in enumerate(zip(rows["option_a"], rows["option_b"])):
        if (b, a) in seen:
            return rows.iloc[[seen[(b, a)], position]]
        seen.setdefault((a, b), position)
    return None


def _pair_is_consistent(rows: pd.DataFrame) -> bool:
    """True when the judge named the same output under both orderings.

    The same output, not the same position. On a pair the judge flipped, the
    two are opposite answers, and counting positions here would score a judge
    that always grabs whatever is in front of it as perfectly consistent.
    """
    return bool(rows["winner"].nunique() == 1)


def _randomised_result(data: pd.DataFrame, n_both_orders: int) -> PositionBias:
    decided = data[data["winner"].notna()]
    n_decisive = int(len(decided))
    n_a_wins = int((decided["winner"] == decided["option_a"]).sum())
    rate, lo, hi, p = _proportion(n_a_wins, n_decisive)

    return PositionBias(
        design="randomised",
        estimate=rate,
        ci_low=lo,
        ci_high=hi,
        p_value=p,
        position_a_rate=rate,
        n_a_wins=n_a_wins,
        n_decisive=n_decisive,
        consistency_rate=float("nan"),
        n_pairs=int(data["pair_id"].nunique()),
        n_pairs_scored=n_decisive,
        n_judgements=int(len(data)),
        n_both_orders=n_both_orders,
        n_ties=int(len(data) - n_decisive),
        position_a_ci_low=lo,
        position_a_ci_high=hi,
    )


def _both_orders_result(data: pd.DataFrame, n_both_orders: int) -> PositionBias:
    """Consistency over the pairs run both ways, and where the flips went.

    A flip has a direction. If the judge picked whatever was in front of it
    in both presentations, that is position. If the flips split evenly across
    the two positions, the judge is unsteady and that is a different fault.
    """
    n_consistent = 0
    n_scored = 0
    n_flipped = 0
    n_first = 0

    for _, rows in data.groupby("pair_id", sort=False):
        couple = _reversed_couple(rows)
        if couple is None or couple["winner"].isna().any():
            continue

        n_scored += 1
        if _pair_is_consistent(couple):
            n_consistent += 1
            continue

        n_flipped += 1
        # `all` and `any` return the same thing here, and no test can tell
        # them apart, so leave this as it is rather than reaching for the
        # shorter one. The couple is two reversed presentations of one pair
        # and the branch only runs when the winners differ, which leaves two
        # cases. The judge named the first-shown output both times, and both
        # rows have winner == option_a. Or it named the second-shown output
        # both times, and neither does. There is no couple where the rows
        # disagree, so the two functions cannot diverge on reachable input.
        # `all` is kept because it states the intent, which is that the
        # position has to hold across both presentations.
        if bool((couple["winner"] == couple["option_a"]).all()):
            n_first += 1

    consistency, c_lo, c_hi = _consistency(n_consistent, n_scored)
    rate, rate_lo, rate_hi, p = _proportion(n_first, n_flipped)

    return PositionBias(
        design="both_orders",
        estimate=consistency,
        ci_low=c_lo,
        ci_high=c_hi,
        p_value=p,
        position_a_rate=rate,
        n_a_wins=n_first,
        n_decisive=n_flipped,
        consistency_rate=consistency,
        n_pairs=int(data["pair_id"].nunique()),
        n_pairs_scored=n_scored,
        n_judgements=int(len(data)),
        n_both_orders=n_both_orders,
        n_ties=int(data["winner"].isna().sum()),
        position_a_ci_low=rate_lo,
        position_a_ci_high=rate_hi,
    )


def _proportion(k: int, n: int):
    """Rate, Clopper-Pearson interval, and the exact binomial p-value
    against a half.

    The interval inverts that test, so it excludes a half exactly when the
    p-value is below 0.05. Wilson is shorter and sits on the permissive side
    of the test near the line, so a Wilson verdict could contradict the
    p-value printed beside it.
    """
    if n == 0:
        nan = float("nan")
        return nan, nan, nan, nan
    lo, hi = _exact_interval(k, n, 0.95)
    p = float(_stats.binomtest(k, n, 0.5).pvalue)
    return k / n, lo, hi, p


def _exact_interval(k: int, n: int, confidence: float):
    """Clopper-Pearson interval on k of n, from beta quantiles.

    Each bound is the rate at which the exact binomial test would just
    reject on its own side, so each side gets half the leftover probability.
    """
    tail = (1 - confidence) / 2
    lo = 0.0 if k == 0 else float(_stats.beta.ppf(tail, k, n - k + 1))
    hi = 1.0 if k == n else float(_stats.beta.isf(tail, k + 1, n - k))
    return lo, hi


def _consistency(k: int, n: int):
    """Rate and Wilson interval for the consistency rate.

    Nothing is tested against the consistency rate, so no p-value sits
    beside it, and Wilson, which beats the bootstrap near one, stays.
    """
    if n == 0:
        nan = float("nan")
        return nan, nan, nan
    lo, hi = _wilson(k, n, 0.95)
    return k / n, lo, hi


# --------------------------------------------------------------------------
# length_bias
# --------------------------------------------------------------------------

def length_bias(
    preferences: Sequence,
    lengths,
    human_preferences: Optional[Sequence] = None,
) -> LengthBias:
    """Does the judge go for the longer answer.

    Parameters
    ----------
    preferences
        Which option the judge picked, one per pair. Either 0 and 1, or the
        strings ``"A"`` and ``"B"``, where 1 and ``"A"`` mean the first
        option.
    lengths
        An (n, 2) array or two-column frame of the two options' lengths, in
        whatever unit you measure them in. Characters and tokens both work,
        and the summary reports the effect over one standard deviation of the
        observed differences so the unit does not have to be guessed at.
    human_preferences
        The humans' choices on the same pairs, same encoding. Supplying these
        adds the second model, which is the sharper of the two.

    Returns
    -------
    LengthBias

    Notes
    -----
    Two fits on the same length difference, and they answer different
    questions.

    The first regresses judge preference on the signed length difference. A
    positive coefficient says the judge picks the longer answer. That is not
    bias on its own. Longer answers may be better, and on most corpora they
    somewhat are.

    The second regresses judge-human disagreement on the same difference,
    oriented by what the humans picked, so the predictor is the length they
    passed over minus the length they chose. A positive coefficient says the
    judge breaks with the humans on the pairs where the humans went short,
    which is what a length-biased judge does.

    Holding the human verdict fixed removes the part of the length-quality
    link the human labels capture. It does not remove the rest. A binary
    label is a coarse measure of quality and the length difference still
    carries quality information the label missed, so on a corpus where length
    tracks quality closely the second coefficient stays positive for a judge
    with no length preference at all. It is smaller than the first, it is the
    better of the two numbers, and it is not a clean separation. The summary
    says so rather than selling it as one.

    Orienting the second model by the judge's own choice looks equivalent and
    is not. Every disagreement is by definition a pair where the judge took
    what the humans rejected, so the predictor becomes a consequence of the
    outcome and the coefficient comes out negative whether or not the judge
    is biased.
    """
    chose_a = _as_choice(preferences, "preferences")
    len_a, len_b = _as_lengths(lengths, chose_a.size)

    if human_preferences is None:
        human_chose_a = None
    else:
        human_chose_a = _as_choice(human_preferences, "human_preferences")
        if human_chose_a.size != chose_a.size:
            raise ValueError(
                f"human_preferences must be the same length as preferences, "
                f"got {human_chose_a.size} and {chose_a.size}"
            )

    difference = _length_difference(len_a, len_b)
    coefficient, ci_low, ci_high, p_value, note = _fit_logistic(
        difference, chose_a.astype(float)
    )

    picked = np.where(chose_a, len_a, len_b)
    passed = np.where(chose_a, len_b, len_a)
    longer_rate = float(np.mean(picked > passed))

    if human_chose_a is None:
        oriented = np.zeros(0)
        n_disagreements = 0
        d_coef = d_lo = d_hi = d_p = float("nan")
        d_note = ""
    else:
        oriented = _oriented_length_difference(human_chose_a, len_a, len_b)
        disagrees = (chose_a != human_chose_a).astype(float)
        n_disagreements = int(disagrees.sum())
        d_coef, d_lo, d_hi, d_p, d_note = _fit_logistic(oriented, disagrees)

    return LengthBias(
        coefficient=coefficient,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        sd_difference=_sd(difference),
        note=note,
        disagreement_coefficient=d_coef,
        disagreement_ci_low=d_lo,
        disagreement_ci_high=d_hi,
        disagreement_p_value=d_p,
        disagreement_sd_difference=_sd(oriented),
        disagreement_note=d_note,
        n=int(chose_a.size),
        n_disagreements=n_disagreements,
        longer_rate=longer_rate,
        has_human=human_chose_a is not None,
    )


def _length_difference(len_a: np.ndarray, len_b: np.ndarray) -> np.ndarray:
    """The predictor in the preference model, signed.

    Signed rather than absolute, because a judge that favours long answers
    and one that favours short answers are opposite findings and an unsigned
    predictor cannot tell them apart.
    """
    return len_a - len_b


def _oriented_length_difference(
    human_chose_a: np.ndarray, len_a: np.ndarray, len_b: np.ndarray
) -> np.ndarray:
    """The predictor in the disagreement model.

    Length the humans passed over, minus the length they picked, so positive
    means the humans went for the shorter answer on that pair.
    """
    picked = np.where(human_chose_a, len_a, len_b)
    passed = np.where(human_chose_a, len_b, len_a)
    return passed - picked


def _as_choice(values: Sequence, name: str) -> np.ndarray:
    """Preferences as a boolean, True where the first option was picked."""
    array = np.asarray(list(values), dtype=object)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")

    distinct = set(array.tolist())
    letters = {str(v).strip().upper() for v in distinct}
    if letters <= {"A", "B"}:
        return np.array([str(v).strip().upper() == "A" for v in array])

    numeric = {float(v) for v in distinct if isinstance(v, (int, float, np.number))}
    if len(numeric) == len(distinct) and numeric <= {0.0, 1.0}:
        return np.array([float(v) == 1.0 for v in array])

    raise ValueError(
        f"{name} must take exactly two values, either 0 and 1 or 'A' and 'B', "
        f"where 1 and 'A' mean the first option. Got {sorted(map(str, distinct))}."
    )


def _as_lengths(lengths, n: int):
    """The two length columns, validated."""
    if isinstance(lengths, pd.DataFrame):
        array = lengths.to_numpy(dtype=float)
    else:
        array = np.asarray(lengths, dtype=float)

    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(
            f"lengths must have two columns, one per option, got shape "
            f"{array.shape}"
        )
    if array.shape[0] != n:
        raise ValueError(
            f"preferences and lengths must be the same length, got {n} and "
            f"{array.shape[0]}"
        )
    return array[:, 0], array[:, 1]


def _sd(x: np.ndarray) -> float:
    """Standard deviation of the predictor, for reading the coefficient."""
    if x.size < 2:
        return float("nan")
    return float(np.std(x, ddof=1))


def _fit_logistic(x: np.ndarray, y: np.ndarray):
    """Logistic regression of y on x with an intercept, by Newton steps.

    Returns the slope, its Wald interval, its two-sided p-value and a note.
    The note carries the reason when there is no coefficient, because the
    three ways this fails are three different things to go and fix.
    """
    nan = float("nan")
    if y.size == 0 or np.unique(y).size < 2:
        return nan, nan, nan, nan, _NOTE_ONE_OUTCOME
    if np.ptp(x) == 0:
        return nan, nan, nan, nan, _NOTE_NO_LENGTH_VARIATION
    if _is_separated(x, y):
        # The maximum likelihood slope is infinite here. Newton will happily
        # return whatever it reached when the step size fell under tolerance,
        # and printing that number would report the stopping rule as a
        # finding.
        return nan, nan, nan, nan, _NOTE_SEPARATED

    design = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    information = None

    for _ in range(_MAX_IRLS_STEPS):
        mu = _special.expit(design @ beta)
        weights = np.clip(mu * (1 - mu), 1e-12, None)
        weighted = design.T * weights
        information = weighted @ design
        try:
            step = np.linalg.solve(information, design.T @ (y - mu))
        except np.linalg.LinAlgError:
            return nan, nan, nan, nan, _NOTE_NO_CONVERGENCE
        beta = beta + step
        if np.max(np.abs(step)) < _IRLS_TOLERANCE:
            break
    else:
        return nan, nan, nan, nan, _NOTE_NO_CONVERGENCE

    mu = _special.expit(design @ beta)
    weights = np.clip(mu * (1 - mu), 1e-12, None)
    information = (design.T * weights) @ design
    covariance = np.linalg.inv(information)
    se = float(np.sqrt(covariance[1, 1]))

    slope = float(beta[1])
    critical = _stats.norm.ppf(0.975)
    p_value = float(2 * _stats.norm.sf(abs(slope / se)))
    return slope, slope - critical * se, slope + critical * se, p_value, ""


def _is_separated(x: np.ndarray, y: np.ndarray) -> bool:
    """True when the predictor splits the two outcomes with no overlap."""
    low = x[y == np.min(y)]
    high = x[y != np.min(y)]
    if low.size == 0 or high.size == 0:
        return True
    return bool(low.max() < high.min() or high.max() < low.min())
