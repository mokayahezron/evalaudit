"""Confidence intervals on a single eval score.

The most common missing piece in published eval results: a pass rate with no
interval around it. At n=50 the interval is roughly +/- 10 points, which is
usually wider than the difference being claimed.
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
        the boundaries, and can return bounds outside [0, 1].
        ``"bootstrap"`` is the percentile bootstrap of the mean.
        ``"t"`` is the Student-t interval on the mean.
    confidence
        Nominal coverage, default 0.95.
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

    binary = _is_binary(x)

    if method == "auto":
        method = "wilson" if binary else "bootstrap"

    estimate = float(np.mean(x))

    if method == "wilson":
        k = int(np.sum(x))
        ci_low, ci_high = _wilson(k, n, confidence)
    elif method == "bootstrap":
        ci_low, ci_high = _bootstrap_mean(x, confidence, n_boot, seed)
    else:  # "t"
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
