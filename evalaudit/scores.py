"""Confidence intervals on a single eval score.

The most common missing piece in published eval results: a pass rate with no
interval around it. At n=50 the interval is about +/- 10 points at an 85%
pass rate and +/- 13 points at 50%. That is the uncertainty in one score. A
difference between two systems has its own interval, from
:mod:`evalaudit.compare`, and on paired data it is often much narrower than
either score's.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy import stats as _stats

from ._types import ScoreCI

__all__ = ["score_ci"]

_VALID_METHODS = {"auto", "wilson", "bootstrap", "t"}


def score_ci(
    scores: Sequence[float],
    method: str = "auto",
    confidence: float = 0.95,
    n_boot: int = 10000,
    seed: Optional[int] = None,
) -> ScoreCI:
    """Confidence interval on an eval score.

    Parameters
    ----------
    scores
        Binary (0/1) or continuous values, one per evaluated item.
    method
        ``"auto"`` picks Wilson for binary data and bootstrap for continuous.
        ``"wilson"`` is the score interval for a proportion. Use it rather than
        the normal approximation, which is visibly wrong at small n and near
        the boundaries, and can return bounds outside [0, 1]. It refuses
        scores that are not 0 or 1. Wilson counts passes, and on other scores
        the count it built the interval from did not match the mean printed
        beside it. The bootstrap is the interval for a mean.
        ``"bootstrap"`` is the percentile bootstrap of the mean.
        ``"t"`` is the Student-t interval on the mean. It is for continuous
        scores and refuses 0/1 data, where it can run past [0, 1] and has
        width zero when every score is the same. Wilson is the interval for
        that data. It also refuses a single score, since one score has no
        variance to estimate, and continuous scores that are all the same,
        where its width would be zero. ``_refuse_t`` says which refusal an
        input gets.
    confidence
        Nominal coverage, strictly between 0 and 1, default 0.95.
    n_boot
        Bootstrap resamples. 1000 is enough for a consulting report;
        the default is generous.
    seed
        Seeds the bootstrap so results are reproducible. Always set this in
        anything you publish.

    Returns
    -------
    ScoreCI

    Examples
    --------
    >>> score_ci([1] * 42 + [0] * 8).summary()
    '84.0% pass rate (95% CI: 71.5%-91.7%, n=50). ...'
    """
    x = np.asarray(scores, dtype=float)
    n = len(x)
    if n == 0:
        raise ValueError("scores must not be empty")
    if method not in _VALID_METHODS:
        raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    binary = _is_binary(x)

    if method == "auto":
        method = "wilson" if binary else "bootstrap"

    estimate = float(np.mean(x))

    if method == "wilson":
        if not binary:
            _refuse_wilson(x)
        k = int(np.sum(x))
        ci_low, ci_high = _wilson(k, n, confidence)
    elif method == "bootstrap":
        ci_low, ci_high = _bootstrap_mean(x, confidence, n_boot, seed)
    else:  # "t"
        _refuse_t(x, binary)
        alpha = 1 - confidence
        sem = float(np.std(x, ddof=1) / np.sqrt(n))
        t_crit = _stats.t.ppf(1 - alpha / 2, df=n - 1)
        ci_low = estimate - t_crit * sem
        ci_high = estimate + t_crit * sem

    return ScoreCI(
        estimate=estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        n=n,
        method=method,
        binary=binary,
        confidence=confidence,
    )


def _is_binary(x: np.ndarray) -> bool:
    """True when every value is 0 or 1."""
    return bool(np.all((x == 0) | (x == 1)))


def _refuse_wilson(x: np.ndarray) -> None:
    """Refuse scores that are not 0 or 1.

    Wilson counts passes. On other scores the count came from truncating
    the sum, so ten scores of 0.99 were read as 9 passes in 10 and the
    interval came out 0.596 to 0.982 beside a printed mean of 0.990.
    """
    bad = x[~((x == 0) | (x == 1))][0]
    raise ValueError(
        f"the wilson method is an interval on a pass rate, and these scores "
        f"are not all 0 or 1. The first that is neither is {bad:g}. Wilson "
        f"counts passes, so on scores like these it returns an interval for "
        f"a different number from the mean printed beside it. Use "
        f"method='bootstrap' for the interval on a mean, which 'auto' "
        f"already picks for scores that are not 0/1"
    )


def _refuse_t(x: np.ndarray, binary: bool) -> None:
    """Refuse the three inputs the t interval gets wrong.

    All three are checked before any arithmetic, since each one otherwise
    comes back as numbers. They run in this order, and the order decides
    which message an input gets.

    0/1 data is checked first. That covers all passes and all fails, which
    ``auto`` also reads as 0/1, and a single pass or fail. The message names
    Wilson, which works at one item.

    A single score that is not 0 or 1 is checked second, so ``[3.0]`` is
    told it is one score.

    Continuous scores that are all the same are checked last, so
    ``[3.0] * 10`` is told it has no spread. Checking this before the single
    score would send a caller with one score looking for a constant scorer.
    """
    if binary:
        if np.all(x == x[0]):
            raise ValueError(
                f"the t method is for continuous scores, and these scores "
                f"are all {int(x[0])}. That is 0/1 data with no spread, so "
                f"the t interval would have a width of zero and claim the "
                f"rate is known exactly. Use method='wilson', which 'auto' "
                f"already picks for 0/1 data"
            )
        raise ValueError(
            "the t method is for continuous scores, and these are all 0 or "
            "1. A t interval on a pass rate can run past 0% or 100%, which "
            "is the error the Wilson interval exists to avoid. Use "
            "method='wilson', which 'auto' already picks for 0/1 data"
        )
    if len(x) < 2:
        raise ValueError(
            f"the t method needs at least 2 items to estimate a variance, "
            f"got {len(x)}. One score has no spread to measure, so no "
            f"interval can be put around it, and method='bootstrap' would "
            f"return one of width zero for the same reason. Score more items"
        )
    if np.all(x == x[0]):
        raise ValueError(
            f"the t method needs scores that vary, and all {len(x)} of these "
            f"are {x[0]:g}. With no spread the t interval has width zero, "
            f"which claims {len(x)} observations fixed the mean exactly. "
            f"method='bootstrap' returns width zero for the same reason. "
            f"Check that the scorer is not returning a constant"
        )


def _wilson(k: int, n: int, confidence: float) -> tuple[float, float]:
    """Wilson score interval for a proportion.

        (p + z^2/2n  +/-  z * sqrt( p(1-p)/n + z^2/4n^2 )) / (1 + z^2/n)

    The bounds are clipped to [0, 1]. The algebra keeps them inside that
    range on its own, and the clip catches the rounding at p=0 and p=1.
    """
    p = k / n
    z = _stats.norm.ppf(1 - (1 - confidence) / 2)
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    margin = (z / denom) * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    lo = float(np.clip(centre - margin, 0.0, 1.0))
    hi = float(np.clip(centre + margin, 0.0, 1.0))
    return lo, hi


def _bootstrap_mean(
    x: np.ndarray, confidence: float, n_boot: int, seed: Optional[int]
) -> tuple[float, float]:
    """Percentile bootstrap of the mean.

    Vectorised over resamples. One ``(n_boot, n)`` index matrix draws every
    resample at once, and the means come out as row means of a single array.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = x[idx].mean(axis=1)
    alpha = 1 - confidence
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi
