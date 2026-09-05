"""Why evalaudit uses a score interval for the paired binary difference.

evalaudit reports a difference in pass rates between two systems scored on
the same items. That difference needs an interval around it, and there are
several standard ways to build one. This script measures four of them so
the choice in ``evalaudit/compare.py`` rests on numbers rather than on
taste. The recorded output below came from running this file. Rerun it and
you should get the same, since every setting reseeds.

The four candidates:

    wald             the textbook interval on the paired difference
    newcombe         Newcombe's square-and-add interval, method 10
    newcombe-corr    the same with Newcombe's correlation correction
    score (shipped)  Tango's score interval, what the package uses

The verdict. Wald and plain Newcombe both fall to 0.931 worst case, and
Wald is at its weakest at high pass rates, which is where eval numbers
usually sit. Newcombe with the correction has the best worst case at
0.9505, so on that column alone it wins. It gets there by running wide,
reaching 0.9762 where nominal is 0.95, and an interval that wide is
reporting less than the data support. The score interval stays between
0.9423 and 0.9555 across all eight settings, closest to nominal from both
sides, and it brings two properties the others do not have.

First, it is McNemar's test inverted. At a difference of zero the score
statistic is exactly (b - c) / sqrt(b + c), McNemar's own statistic without
the continuity correction, which the third table below confirms to the last
digit. So the interval clears zero when the test rejects, and the two
numbers in the summary line cannot contradict each other. Wald and Newcombe
are separate constructions that happen to land nearby.

Second, look at the last table. When the two systems agree on every item,
Wald and plain Newcombe return a single point, which reads as proof that
the systems are identical. At n=10 that claim rests on nothing. The score
interval returns plus or minus 0.28, which is the honest width for ten
items and no observed disagreement.

Recorded output:

    Coverage of a nominal 95% interval on the paired difference in
    proportions. 4000 simulated eval runs per setting.

        n  rate A  rate B  corr |            wald        newcombe   newcombe-corr score (shipped)
    ---------------------------------------------------------------------------------------------
      100    0.75    0.60   0.5 |          0.9475          0.9477          0.9523          0.9520
      100    0.75    0.60   0.8 |          0.9433          0.9467          0.9537          0.9490
       50    0.75    0.60   0.7 |          0.9373          0.9363          0.9540          0.9500
       30    0.70    0.50   0.6 |          0.9385          0.9313          0.9575          0.9423
      200    0.90    0.85   0.7 |          0.9480          0.9523          0.9597          0.9545
      100    0.95    0.90   0.8 |          0.9325          0.9643          0.9762          0.9555
      100    0.50    0.50   0.7 |          0.9440          0.9440          0.9505          0.9443
       40    0.60    0.60   0.5 |          0.9315          0.9325          0.9555          0.9555

    worst case across the eight settings:
                 wald: 0.9315
             newcombe: 0.9313
        newcombe-corr: 0.9505
      score (shipped): 0.9423

    Does the score interval invert McNemar's test?

       b    c     n |  score at delta=0   (b-c)/sqrt(b+c)        gap
    --------------------------------------------------------------
      18    7   100 |      2.2000000000      2.2000000000    0.0e+00
      30    5   100 |      4.2257712736      4.2257712736    0.0e+00
      12   11    60 |      0.2085144141      0.2085144141    0.0e+00
       0    5    30 |     -2.2360679775     -2.2360679775    0.0e+00
       7    1    40 |      2.1213203436      2.1213203436    0.0e+00
      45   20   200 |      3.1008683647      3.1008683647    0.0e+00
       3    3    50 |      0.0000000000      0.0000000000    0.0e+00

    What each candidate reports when no item changed:

        n |                  wald              newcombe         newcombe-corr       score (shipped)
    ------------------------------------------------------------------------------------------------
       10 |    (+0.0000, +0.0000)    (+0.0000, +0.0000)    (-0.1666, +0.1666)    (-0.2775, +0.2775)
       30 |    (+0.0000, +0.0000)    (-0.0000, +0.0000)    (-0.0615, +0.0615)    (-0.1135, +0.1135)
      100 |    (+0.0000, +0.0000)    (+0.0000, +0.0000)    (-0.0192, +0.0192)    (-0.0370, +0.0370)

Run it with::

    python examples/interval_method_study.py
"""

from __future__ import annotations

import numpy as np
from scipy import stats

# The study measures what the package actually ships, so it imports the
# private helpers rather than reimplementing them. The three rejected
# candidates are written out below, since they are not in the package.
from evalaudit.compare import _paired_proportion_ci, _tango_score

TRIALS = 4000
CONFIDENCE = 0.95
SEED = 20

# n, pass rate of A, pass rate of B, correlation between the two latent scores
SETTINGS = [
    (100, 0.75, 0.60, 0.5),
    (100, 0.75, 0.60, 0.8),
    (50, 0.75, 0.60, 0.7),
    (30, 0.70, 0.50, 0.6),
    (200, 0.90, 0.85, 0.7),
    (100, 0.95, 0.90, 0.8),
    (100, 0.50, 0.50, 0.7),
    (40, 0.60, 0.60, 0.5),
]


# --------------------------------------------------------------------------
# The rejected candidates
# --------------------------------------------------------------------------

def wilson(k: int, n: int, confidence: float) -> tuple[float, float]:
    """Wilson score interval for a single proportion."""
    p = k / n
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    z2 = z * z
    den = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / den
    margin = (z / den) * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return float(np.clip(centre - margin, 0, 1)), float(np.clip(centre + margin, 0, 1))


def wald(n11: int, b: int, c: int, n22: int, confidence: float) -> tuple[float, float]:
    """The textbook interval on a paired difference in proportions.

        SE = sqrt(b + c - (b - c)^2 / n) / n

    This is what the package used first.
    """
    n = n11 + b + c + n22
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    se = np.sqrt(max(b + c - (b - c) ** 2 / n, 0.0)) / n
    d = (b - c) / n
    return float(np.clip(d - z * se, -1, 1)), float(np.clip(d + z * se, -1, 1))


def newcombe(
    n11: int, b: int, c: int, n22: int, confidence: float, corrected: bool = False
) -> tuple[float, float]:
    """Newcombe's square-and-add interval for paired proportions, method 10.

    Build a Wilson interval for each pass rate, then combine the bounds with
    a term for how correlated the two systems are. ``corrected`` applies
    Newcombe's rule for the correlation estimate, which pulls small positive
    values down to zero.
    """
    n = n11 + b + c + n22
    p1, p2 = (n11 + b) / n, (n11 + c) / n
    l1, u1 = wilson(n11 + b, n, confidence)
    l2, u2 = wilson(n11 + c, n, confidence)

    margins = (n11 + b) * (c + n22) * (n11 + c) * (b + n22)
    cross = n11 * n22 - b * c
    if margins == 0:
        phi = 0.0
    elif not corrected:
        phi = cross / np.sqrt(margins)
    elif cross > n / 2:
        phi = (cross - n / 2) / np.sqrt(margins)
    elif cross >= 0:
        phi = 0.0
    else:
        phi = cross / np.sqrt(margins)
    phi = float(np.clip(phi, -1, 1))

    d = p1 - p2
    low = (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2
    high = (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2
    return (
        float(np.clip(d - np.sqrt(max(low, 0.0)), -1, 1)),
        float(np.clip(d + np.sqrt(max(high, 0.0)), -1, 1)),
    )


CANDIDATES = {
    "wald": lambda n11, b, c, n22: wald(n11, b, c, n22, CONFIDENCE),
    "newcombe": lambda n11, b, c, n22: newcombe(n11, b, c, n22, CONFIDENCE),
    "newcombe-corr": lambda n11, b, c, n22: newcombe(
        n11, b, c, n22, CONFIDENCE, corrected=True
    ),
    "score (shipped)": lambda n11, b, c, n22: _paired_proportion_ci(
        b, c, n11 + b + c + n22, CONFIDENCE
    ),
}


# --------------------------------------------------------------------------
# Simulating correlated binary pairs
# --------------------------------------------------------------------------

def draw(rng, n: int, p_a: float, p_b: float, corr: float):
    """One eval run where both systems score the same n items.

    Each system gets a latent score built from a shared item difficulty term
    and its own noise, then passes an item when that score clears its own
    threshold. The thresholds fix the marginal pass rates at exactly p_a and
    p_b, so the true difference is known, while the shared term makes the
    two systems agree on most items the way real paired evals do.
    """
    shared = rng.normal(size=n)
    latent_a = np.sqrt(corr) * shared + np.sqrt(1 - corr) * rng.normal(size=n)
    latent_b = np.sqrt(corr) * shared + np.sqrt(1 - corr) * rng.normal(size=n)
    a = latent_a < stats.norm.ppf(p_a)
    b = latent_b < stats.norm.ppf(p_b)
    return (
        int(np.sum(a & b)),
        int(np.sum(a & ~b)),
        int(np.sum(~a & b)),
        int(np.sum(~a & ~b)),
    )


def coverage_table() -> dict[str, float]:
    """Coverage of each candidate at every setting. Returns the worst case."""
    names = list(CANDIDATES)
    header = f"{'n':>5} {'rate A':>7} {'rate B':>7} {'corr':>5} | " + " ".join(
        f"{name:>15}" for name in names
    )
    print(header)
    print("-" * len(header))

    worst = {name: 1.0 for name in names}
    for n, p_a, p_b, corr in SETTINGS:
        true_diff = p_a - p_b
        covered = {name: 0 for name in names}
        rng = np.random.default_rng(SEED)
        for _ in range(TRIALS):
            table = draw(rng, n, p_a, p_b, corr)
            for name, interval in CANDIDATES.items():
                low, high = interval(*table)
                if low <= true_diff <= high:
                    covered[name] += 1
        print(
            f"{n:>5} {p_a:>7.2f} {p_b:>7.2f} {corr:>5.1f} | "
            + " ".join(f"{covered[name] / TRIALS:>15.4f}" for name in names)
        )
        for name in names:
            worst[name] = min(worst[name], covered[name] / TRIALS)

    print()
    print("worst case across the eight settings:")
    for name in names:
        print(f"  {name:>15}: {worst[name]:.4f}")
    return worst


def reduction_check() -> None:
    """At a difference of zero the score statistic is McNemar's own.

    This is the property that keeps the interval and the p-value from
    telling different stories. Tango's statistic with delta set to zero
    collapses to (b - c) / sqrt(b + c), which is McNemar's statistic without
    the continuity correction. So the interval clears zero exactly when that
    statistic clears the critical value.
    """
    print(f"{'b':>4} {'c':>4} {'n':>5} | {'score at delta=0':>17} "
          f"{'(b-c)/sqrt(b+c)':>17} {'gap':>10}")
    print("-" * 62)
    for b, c, n in [(18, 7, 100), (30, 5, 100), (12, 11, 60), (0, 5, 30),
                    (7, 1, 40), (45, 20, 200), (3, 3, 50)]:
        theirs = _tango_score(b, c, n, 0.0)
        mcnemar = (b - c) / np.sqrt(b + c) if b + c else float("nan")
        print(f"{b:>4} {c:>4} {n:>5} | {theirs:>17.10f} {mcnemar:>17.10f} "
              f"{abs(theirs - mcnemar):>10.1e}")


def empty_table_check() -> None:
    """What each candidate says when the two systems agree on every item.

    There is no evidence of a difference here and no evidence of equality
    either. An interval that collapses to a single point claims the second
    thing, which is how a package ends up telling someone that 10 items were
    enough to prove two systems identical.
    """
    print(f"{'n':>5} | " + " ".join(f"{name:>21}" for name in CANDIDATES))
    print("-" * 96)
    for n in (10, 30, 100):
        both = n // 2
        cells = (both, 0, 0, n - both)
        row = []
        for interval in CANDIDATES.values():
            low, high = interval(*cells)
            row.append(f"({low:+.4f}, {high:+.4f})")
        print(f"{n:>5} | " + " ".join(f"{cell:>21}" for cell in row))


def main() -> None:
    print("Coverage of a nominal 95% interval on the paired difference in")
    print(f"proportions. {TRIALS} simulated eval runs per setting.")
    print()
    coverage_table()
    print()
    print("Does the score interval invert McNemar's test?")
    print()
    reduction_check()
    print()
    print("What each candidate reports when no item changed:")
    print()
    empty_table_check()


if __name__ == "__main__":
    main()
