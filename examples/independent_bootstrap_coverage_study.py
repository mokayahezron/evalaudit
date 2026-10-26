"""How often the two-sample bootstrap in compare_independent covers.

``compare_independent(method="bootstrap")`` is a percentile bootstrap that
resamples each group on its own. This script measures its coverage, the
share of simulated evals whose 95% interval contains the true difference in
means. The recorded output below came from running this file. Rerun it and
you should get the same, since every setting reseeds.

The settings. Each trial draws two groups and asks compare_independent for a
95% interval with 1,000 resamples. The true difference is 0.5 throughout.

    normal       group A normal with mean 0.5 and standard deviation 1.0,
                 group B normal with mean 0.0 and standard deviation 2.0
    exponential  group A exponential with mean 1.5, group B exponential with
                 mean 1.0, so both groups are skewed to the right

Every setting starts a fresh generator from SEED. Each trial draws group A,
then group B, then the seed for that trial's bootstrap, all from that one
generator. ``independent_bootstrap_coverage`` in ``tests/test_compare.py``
follows the same procedure, so the two 1,000-trial rows are the figures
those tests check. The first 1,000 trials of a 5,000-trial row are the same
trials as the 1,000-trial row for that setting.

Monte Carlo error. A coverage is a share of trials, so it carries binomial
error. The table gives its standard error and the exact Clopper-Pearson 95%
interval on it.

The verdict. On exponential scores the interval covers less than it claims
at small sizes, 92.6% with 25 and 60 items and 93.0% with 40 and 120, and
both intervals on those figures exclude 95%. With 100 and 300 items it
covers 94.5%, and that interval contains 95%. Across the three exponential
sizes the shortfall shrinks as the groups grow. On normal scores with 30 and
90 items it covers 94.1% at 1,000 trials and at 5,000. The interval on the
5,000-trial figure, 93.4% to 94.7%, also sits below 95%, by less than on the
two smaller exponential designs.

The recorded run used the 0.6.0 release tree before its version number
changed, so it printed evalaudit 0.5.0.

Recorded output:

    Python 3.13.14, numpy 2.5.2, scipy 1.18.1, pandas 3.0.5, evalaudit 0.5.0
    seed 44 for every setting, 1000 resamples per interval, nominal 95%, true difference 0.5

    distribution  n_a  n_b trials | covered coverage   s.e.   95% CI on coverage
    ----------------------------------------------------------------------------
          normal   30   90   1000 |     941   0.9410 0.0075   0.9246 to 0.9548
     exponential   40  120   1000 |     929   0.9290 0.0081   0.9113 to 0.9441
          normal   30   90   5000 |    4703   0.9406 0.0033   0.9337 to 0.9470
     exponential   25   60   5000 |    4631   0.9262 0.0037   0.9186 to 0.9333
     exponential   40  120   5000 |    4651   0.9302 0.0036   0.9228 to 0.9371
     exponential  100  300   5000 |    4725   0.9450 0.0032   0.9383 to 0.9512

Run it with::

    python examples/independent_bootstrap_coverage_study.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import scipy
from scipy import stats

import evalaudit
from evalaudit import compare_independent

TRUE_DIFFERENCE = 0.5
SEED = 44
N_BOOT = 1000
CONFIDENCE = 0.95

# (distribution, items in group A, items in group B, trials)
SETTINGS = [
    ("normal", 30, 90, 1000),
    ("exponential", 40, 120, 1000),
    ("normal", 30, 90, 5000),
    ("exponential", 25, 60, 5000),
    ("exponential", 40, 120, 5000),
    ("exponential", 100, 300, 5000),
]


def draw(rng, distribution: str, n_a: int, n_b: int):
    """Group A, then group B, with a true difference in means of 0.5."""
    if distribution == "normal":
        return rng.normal(0.5, 1.0, n_a), rng.normal(0.0, 2.0, n_b)
    return rng.exponential(1.5, n_a), rng.exponential(1.0, n_b)


def covered(distribution: str, n_a: int, n_b: int, trials: int) -> int:
    """How many of the trials' intervals contain the true difference."""
    rng = np.random.default_rng(SEED)
    hits = 0
    for _ in range(trials):
        a, b = draw(rng, distribution, n_a, n_b)
        r = compare_independent(
            a, b, method="bootstrap", confidence=CONFIDENCE, n_boot=N_BOOT,
            seed=int(rng.integers(1_000_000)),
        )
        hits += bool(r.ci_low <= TRUE_DIFFERENCE <= r.ci_high)
    return hits


def main() -> None:
    print(
        f"Python {sys.version.split()[0]}, numpy {np.__version__}, scipy "
        f"{scipy.__version__}, pandas {pd.__version__}, evalaudit "
        f"{evalaudit.__version__}"
    )
    print(
        f"seed {SEED} for every setting, {N_BOOT} resamples per interval, "
        f"nominal {CONFIDENCE:.0%}, true difference {TRUE_DIFFERENCE}"
    )
    print()
    header = (
        f"{'distribution':>12} {'n_a':>4} {'n_b':>4} {'trials':>6} | "
        f"{'covered':>7} {'coverage':>8} {'s.e.':>6}   95% CI on coverage"
    )
    print(header)
    print("-" * len(header))
    for distribution, n_a, n_b, trials in SETTINGS:
        k = covered(distribution, n_a, n_b, trials)
        rate = k / trials
        se = (rate * (1 - rate) / trials) ** 0.5
        ci = stats.binomtest(k, trials).proportion_ci(
            confidence_level=0.95, method="exact"
        )
        print(
            f"{distribution:>12} {n_a:>4} {n_b:>4} {trials:>6} | "
            f"{k:>7} {rate:>8.4f} {se:>6.4f}   {ci.low:.4f} to {ci.high:.4f}"
        )


if __name__ == "__main__":
    main()
