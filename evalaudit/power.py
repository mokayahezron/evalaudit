"""What an eval could have found, and how big the next one has to be.

The question this module exists to answer comes after the result, not
before it. Someone hands you a finished eval, two systems, a difference
that did not reach significance, and asks whether that means the systems
are the same. Usually it does not. It means the eval was never large enough
to tell, and that was fixed before the first item was graded.

For paired binary data the size that matters is not the number of items.
McNemar reads only the pairs where the two systems disagree, and the rest
carry no information about which one is better. So the calculation runs on
the discordance rate. An eval with 500 items and 12 discordant pairs is a
12 item study, and 488 of those items did nothing but cost money.

That rate is a property of the two systems and the item set, so it cannot
be guessed from the pass rates. When it is not supplied these functions
stand in a conservative 0.3 and say so in the output. They never assume it
quietly.

The baseline pass rate has no part in any of this, which is why
:func:`min_sample_size` takes the difference first and leaves the baseline
as a keyword. A paired call reads ``min_sample_size(0.05)`` and passes
nothing it does not use.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import optimize as _optimize
from scipy import stats as _stats

from ._types import PowerResult

__all__ = ["detectable_effect", "min_sample_size"]

# Stands in when no discordance rate is supplied. Conservative in both
# directions, since a higher rate means a larger detectable difference and a
# larger required sample. Real evals usually run well under it, so the
# answers come out pessimistic rather than flattering. That is the safe way
# to be wrong when the question is what an eval could have found.
_DEFAULT_DISCORDANCE = 0.3

# The difference the summary prices out, so the reader has a familiar
# anchor beside the answer. Five points is the size people mean when they
# say a system improved.
_REFERENCE_DIFFERENCE = 0.05

# Room left at the top of the bracket when solving for an independent
# difference. The largest difference a baseline allows is 1 - baseline, and
# brentq needs the bracket to be a genuine interval.
_BRACKET_FLOOR = 1e-12

# brentq's tolerance on the difference. The sample size it feeds back into
# moves about 2n / difference per unit, so at eval sizes this leaves the
# roundtrip accurate to about 1e-8 of an item.
_SOLVER_XTOL = 1e-14


def detectable_effect(
    n: int,
    baseline: float = 0.5,
    power: float = 0.8,
    alpha: float = 0.05,
    paired: bool = True,
    discordance_rate: Optional[float] = None,
) -> PowerResult:
    """The smallest difference an eval of this size could have found.

    The audit direction. You are handed a completed evaluation and you want
    to know what it could ever have shown, whatever it happened to show.

    Parameters
    ----------
    n
        Items the eval ran. Paired comparisons when ``paired`` is True, and
        the total across both groups when it is False, which is the same
        convention :func:`evalaudit.compare_independent` uses for its ``n``.
    baseline
        Pass rate of the system being compared against. Used only when
        ``paired`` is False. A paired binary calculation does not depend on
        it, so it is carried into the result and otherwise ignored. See
        Notes.
    power
        The chance of finding a difference that is really there. 0.8 is the
        convention.
    alpha
        Two sided significance level.
    paired
        True when both systems were run on the same items, which is how
        most evals are built.
    discordance_rate
        Share of items the two systems disagree on. Paired designs only.
        When it is not supplied, 0.3 stands in and ``discordance_assumed``
        comes back True. See Notes.

    Returns
    -------
    PowerResult
        With ``solved_for="difference"``. Check ``attainable`` before
        reading ``difference``.

    Notes
    -----
    Paired binary power runs on the discordance rate, not on the number of
    items and not on the baseline. McNemar's test looks only at the pairs
    where one system passed and the other failed. Items both systems passed
    and items both failed drop out of the statistic entirely, so an eval
    with 500 items and 12 discordant pairs carries the information of a 12
    item study.

    The direction surprises people. A higher discordance rate makes the
    detectable difference larger, not smaller. Disagreement between the two
    systems is noise in the per item comparison, so systems that agree on
    most items are the ones McNemar can separate sharply.

    The arithmetic is Connor's normal approximation, which is the standard
    published formula. evalaudit's own McNemar runs the continuity
    corrected chi-square above 25 discordant pairs and the exact binomial
    below it, and both are a little conservative. Measured over 200,000
    simulated evals the shipped test delivers about 0.784 where this
    formula promises 0.800. So the difference reported here is a floor. An
    eval could not have found anything smaller, and in practice it needs a
    little more.

    Examples
    --------
    >>> detectable_effect(220, discordance_rate=0.3).difference
    0.10289...
    """
    _check_power_and_alpha(power, alpha)
    _check_baseline(baseline)
    n = _check_n(n)
    rate = _resolve_discordance(paired, discordance_rate)

    if paired:
        difference = _paired_difference(n, rate, alpha, power)
        # <= rather than <, matching min_sample_size and _reference_size. A
        # difference equal to the discordance rate is the table where every
        # discordant pair fell the same way, c is zero and b is the whole
        # rate. That is a real point in the parameter space and the
        # arithmetic goes through it, so all three places treat it as
        # reachable.
        #
        # The two comparisons cannot be told apart by a test that goes
        # through this function. Landing on the tie needs the solved
        # difference to equal the rate in float, and a scan of eight rates
        # against every size from 2 to 4000 found no such n. The convention
        # is pinned where it is observable instead, on the min_sample_size
        # and _reference_size side, by
        # test_the_boundary_convention_is_the_same_in_all_three_places.
        attainable = bool(np.isfinite(difference) and difference <= rate)
        n_discordant = int(round(n * rate))
        n_per_group = None
    else:
        n_per_group = n / 2.0
        difference = _independent_difference(n_per_group, baseline, alpha, power)
        attainable = bool(np.isfinite(difference))
        n_discordant = None

    return PowerResult(
        difference=float(difference),
        n=n,
        baseline=baseline,
        power=power,
        alpha=alpha,
        paired=paired,
        solved_for="difference",
        attainable=attainable,
        discordance_rate=rate,
        discordance_assumed=paired and discordance_rate is None,
        n_discordant=n_discordant,
        n_per_group=n_per_group,
        reference_difference=_REFERENCE_DIFFERENCE,
        n_for_reference=_reference_size(baseline, power, alpha, paired, rate),
    )


def min_sample_size(
    mde: float,
    baseline: float = 0.5,
    power: float = 0.8,
    alpha: float = 0.05,
    paired: bool = True,
    discordance_rate: Optional[float] = None,
) -> PowerResult:
    """How many items it takes to find a difference of a given size.

    The design direction, for a team planning a run. It is the same
    relation :func:`detectable_effect` solves, turned around, so composing
    the two returns where it started.

    Parameters
    ----------
    mde
        Minimum difference worth detecting, on the 0 to 1 scale. 0.05 is
        five points of pass rate. First because it is the only argument a
        paired design needs, and a paired design is the default.
    baseline
        Pass rate of the system being compared against. Used only when
        ``paired`` is False, for the same reason given there. A paired call
        should leave it alone rather than pass a number that does nothing.
    power
        The chance of finding that difference when it is really there.
    alpha
        Two sided significance level.
    paired
        True when both systems will be run on the same items. Nearly always
        worth doing, and it cuts the number of items sharply.
    discordance_rate
        Share of items the two systems will disagree on. Paired designs
        only. When it is not supplied, 0.3 stands in and the result says so.

    Returns
    -------
    PowerResult
        With ``solved_for="n"``. ``n`` is paired comparisons for a paired
        design and the total across both groups for an independent one,
        with ``n_per_group`` carrying the split.

    Raises
    ------
    ValueError
        When the difference asked for cannot occur. A paired difference
        cannot exceed the discordance rate, since both come from the same
        discordant pairs. A difference equal to the rate is allowed, since
        that is the table where every discordant pair fell the same way.
        An independent difference cannot push the second pass rate past 1.

    Notes
    -----
    Everything in the Notes on :func:`detectable_effect` applies here. The
    number this returns is a floor on the paired side for the same reason,
    so treat it as the least a run could get away with.

    Examples
    --------
    >>> min_sample_size(0.05, discordance_rate=0.3).n
    940
    """
    _check_power_and_alpha(power, alpha)
    _check_baseline(baseline)
    _check_mde(mde)
    rate = _resolve_discordance(paired, discordance_rate)

    if paired:
        if mde > rate:
            raise ValueError(
                f"a difference of {mde} cannot occur at a discordance rate "
                f"of {rate}. Both come from the same discordant pairs, so "
                f"the difference is at most the discordance rate. Either "
                f"the rate is higher than the one given or the difference "
                f"is not one these two systems can produce"
            )
        exact = _paired_size(rate, mde, alpha, power)
        n = int(np.ceil(exact))
        n_discordant = int(round(n * rate))
        n_per_group = None
    else:
        if baseline + mde > 1.0:
            raise ValueError(
                f"a difference of {mde} on a baseline of {baseline} puts the "
                f"second pass rate at {baseline + mde}, above 1. Lower the "
                f"baseline or the difference"
            )
        per_group = float(np.ceil(_independent_size(baseline, mde, alpha, power)))
        n = int(2 * per_group)
        n_discordant = None
        n_per_group = per_group

    return PowerResult(
        difference=mde,
        n=n,
        baseline=baseline,
        power=power,
        alpha=alpha,
        paired=paired,
        solved_for="n",
        attainable=True,
        discordance_rate=rate,
        discordance_assumed=paired and discordance_rate is None,
        n_discordant=n_discordant,
        n_per_group=n_per_group,
    )


# --------------------------------------------------------------------------
# Paired binary, McNemar
# --------------------------------------------------------------------------

def _power_paired(
    n: int, discordance_rate: float, difference: float, alpha: float
) -> float:
    """Power of McNemar's test, normal approximation.

    Under the hypothesis of no difference the count b - c has variance
    n * pi, and under the alternative it has n * (pi - delta^2). Writing the
    rejection rule in the first and the distribution in the second gives

        power = Phi((sqrt(n) delta - z sqrt(pi)) / sqrt(pi - delta^2))

    The chance of rejecting on the wrong side is dropped. It runs under
    1e-5 at any size a real eval reaches, and dropping it is what lets the
    two entry points be exact inverses of each other rather than
    approximate ones.
    """
    z = _stats.norm.ppf(1 - alpha / 2)
    spread = discordance_rate - difference**2
    if spread <= 0:
        # Every discordant pair falls the same way. Nothing is left to vary.
        return 1.0
    reach = np.sqrt(n) * difference - z * np.sqrt(discordance_rate)
    return float(_stats.norm.cdf(reach / np.sqrt(spread)))


def _paired_size(
    discordance_rate: float, difference: float, alpha: float, power: float
) -> float:
    """Pairs needed, Connor (1987).

        n = [z sqrt(pi) + z_beta sqrt(pi - delta^2)]^2 / delta^2

    Connor writes this with the discordant odds ratio in place of the
    difference. The two are the same formula after substituting
    delta = pi (psi - 1) / (psi + 1), and the test suite checks both routes
    land on the same integer.
    """
    z, z_beta = _critical_values(alpha, power)
    numerator = z * np.sqrt(discordance_rate) + z_beta * np.sqrt(
        discordance_rate - difference**2
    )
    return float(numerator**2 / difference**2)


def _paired_difference(
    n: int, discordance_rate: float, alpha: float, power: float
) -> float:
    """Smallest difference reaching the requested power, in closed form.

    Solving _power_paired for delta squares once and gives a quadratic,

        (n + z_beta^2) delta^2 - 2 sqrt(n) z sqrt(pi) delta
            + pi (z^2 - z_beta^2) = 0

    whose upper root tidies to the expression below. The discriminant is
    pi z_beta^2 (n - z^2 + z_beta^2), so it goes negative on samples too
    small to separate the two critical values at all. There is no
    detectable difference there and the answer is NaN rather than a number.
    """
    z, z_beta = _critical_values(alpha, power)
    inner = n - z**2 + z_beta**2
    if inner < 0:
        return float("nan")
    reach = z * np.sqrt(n) + z_beta * np.sqrt(inner)
    return float(np.sqrt(discordance_rate) * reach / (n + z_beta**2))


# --------------------------------------------------------------------------
# Independent binary, two proportions
# --------------------------------------------------------------------------

def _power_independent(
    n_per_group: float, baseline: float, difference: float, alpha: float
) -> float:
    """Power of the two proportion score test, equal groups.

    The null variance is pooled, which is the maximum likelihood estimate
    when the two rates are equal and the variance the test in
    :func:`evalaudit.compare_independent` actually uses. The alternative
    variance is not pooled, since under the alternative the rates differ.
    Same convention as the far tail in :func:`_power_paired`, dropped for
    the same reason.
    """
    z = _stats.norm.ppf(1 - alpha / 2)
    second = baseline + difference
    pooled = (baseline + second) / 2
    null_var = 2 * pooled * (1 - pooled)
    alt_var = baseline * (1 - baseline) + second * (1 - second)
    reach = abs(difference) - z * np.sqrt(null_var / n_per_group)
    if alt_var <= 0:
        # Both rates pinned at an edge, so the estimate cannot vary.
        return 1.0 if reach > 0 else 0.0
    return float(_stats.norm.cdf(reach / np.sqrt(alt_var / n_per_group)))


def _independent_size(
    baseline: float, difference: float, alpha: float, power: float
) -> float:
    """Items per group needed, the pooled normal approximation.

        n = [z sqrt(2 p q) + z_beta sqrt(p1 q1 + p2 q2)]^2 / delta^2

    The formula behind every published two proportion sample size table,
    and the one statsmodels solves in
    ``samplesize_proportions_2indep_onetail``.
    """
    z, z_beta = _critical_values(alpha, power)
    second = baseline + difference
    pooled = (baseline + second) / 2
    numerator = z * np.sqrt(2 * pooled * (1 - pooled)) + z_beta * np.sqrt(
        baseline * (1 - baseline) + second * (1 - second)
    )
    return float(numerator**2 / difference**2)


def _independent_difference(
    n_per_group: float, baseline: float, alpha: float, power: float
) -> float:
    """Smallest difference reaching the requested power.

    Both variances move with the difference, so there is no closed form and
    this is a root of :func:`_power_independent`. Power rises with the
    difference everywhere it is below 1, so the root is unique.

    The ceiling is 1 - baseline, since a pass rate cannot pass 1. When even
    that difference falls short of the requested power the answer is NaN.
    An eval this size against this baseline could not have found anything.
    """
    ceiling = 1.0 - baseline
    if ceiling <= _BRACKET_FLOOR:
        return float("nan")
    if _power_independent(n_per_group, baseline, ceiling, alpha) < power:
        return float("nan")

    def shortfall(difference: float) -> float:
        return _power_independent(n_per_group, baseline, difference, alpha) - power

    if shortfall(_BRACKET_FLOOR) >= 0:
        return _BRACKET_FLOOR
    return float(
        _optimize.brentq(shortfall, _BRACKET_FLOOR, ceiling, xtol=_SOLVER_XTOL)
    )


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def _critical_values(alpha: float, power: float) -> tuple[float, float]:
    return _stats.norm.ppf(1 - alpha / 2), _stats.norm.ppf(power)


def _reference_size(
    baseline: float,
    power: float,
    alpha: float,
    paired: bool,
    discordance_rate: Optional[float],
) -> Optional[int]:
    """Items needed for the reference difference, for the summary to quote.

    None when that difference is not one the design could produce, drawn at
    exactly the bar :func:`min_sample_size` refuses at. The summary must
    never quote a size that function would decline to return.
    """
    if paired:
        if discordance_rate is None or _REFERENCE_DIFFERENCE > discordance_rate:
            return None
        return int(
            np.ceil(_paired_size(discordance_rate, _REFERENCE_DIFFERENCE, alpha, power))
        )
    if baseline + _REFERENCE_DIFFERENCE > 1.0:
        return None
    per_group = np.ceil(
        _independent_size(baseline, _REFERENCE_DIFFERENCE, alpha, power)
    )
    return int(2 * per_group)


def _resolve_discordance(
    paired: bool, discordance_rate: Optional[float]
) -> Optional[float]:
    if not paired:
        if discordance_rate is not None:
            raise ValueError(
                "discordance_rate applies to paired designs only. An "
                "independent comparison has no discordant pairs, since the "
                "two systems were never run on the same items. Drop the "
                "argument or set paired=True"
            )
        return None
    if discordance_rate is None:
        return _DEFAULT_DISCORDANCE
    if not 0 < discordance_rate <= 1:
        raise ValueError(
            f"discordance_rate must be above 0 and at most 1, got "
            f"{discordance_rate}. It is the share of items the two systems "
            f"disagree on, and at 0 there is nothing for McNemar to read"
        )
    return float(discordance_rate)


def _check_power_and_alpha(power: float, alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    if not 0 < power < 1:
        raise ValueError(f"power must be between 0 and 1, got {power}")
    if power <= alpha:
        raise ValueError(
            f"power must exceed alpha, got power={power} and alpha={alpha}. "
            f"A two sided test rejects at the alpha rate when there is "
            f"nothing to find, so no sample size delivers less than that"
        )


def _check_baseline(baseline: float) -> None:
    if not 0 <= baseline <= 1:
        raise ValueError(
            f"baseline must be a pass rate between 0 and 1, got {baseline}"
        )


def _check_mde(mde: float) -> None:
    if not 0 < mde < 1:
        raise ValueError(
            f"mde must be a difference above 0 and below 1, got {mde}. It is "
            f"points of pass rate on the 0 to 1 scale, so five points is 0.05"
        )


def _check_n(n: int) -> int:
    if n != int(n):
        raise ValueError(f"n must be a whole number of items, got {n}")
    n = int(n)
    if n < 2:
        raise ValueError(f"n must be at least 2 items, got {n}")
    return n
