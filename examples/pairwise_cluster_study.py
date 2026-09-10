"""How narrow the comparison bootstrap runs when prompts carry many judgements.

``bradley_terry`` resamples items by default, taking every comparison made
on a prompt together. Before 0.3.0 it resampled single comparisons, and its
docstring said that where one prompt carries many comparisons the interval
"will run slightly narrow". Nobody had measured slightly. This script
measures it on data shaped like MT-Bench's human judgements, six models and
2,575 comparisons over 80 prompts, so about 32 per prompt. The recorded
output below came from running this file. Rerun it and you should get the
same, since every setting reseeds.

How prompts make judgements correlated. Each prompt shifts every model's
strength by its own normal draw, with spread tau on the log-odds scale. A
model that does well on a prompt does well on it in every comparison made
there, so comparisons sharing a prompt move together. At tau = 0 they are
independent, which is what the comparison bootstrap assumes. Nothing here
says what MT-Bench's own tau is. The table shows what each value costs.

For the 15 gaps between six models, at each tau:

    width      average width of the gap interval, comparison bootstrap over
               item bootstrap
    cover      share of gap intervals containing the value the fit converges
               to, with its standard error clustered on the simulated
               benchmark
    separable  average number of the 15 pairs whose gap interval excludes zero

The value the fit converges to is not theta. Averaging over prompt shifts
pulls every win rate toward a half, and the fit settles on the ratings that
reproduce the averaged rates. With every pair equally likely those are the
fit to a credit matrix of averaged rates, computed here by quadrature.

The model spread, the six ratings in THETA, is a choice and not a
measurement. It is wide enough that most pairs separate, which is typical of
a published board. The comparisons carry no ties.

The verdict. With no prompt-level spread the two bootstraps agree. As the
spread grows the comparison bootstrap narrows and stops covering, to 88%
at tau = 0.5 and 80% at tau = 1.0, and it separates pairs the item bootstrap
does not. The item bootstrap stays between 93.7% and 94.6% throughout.

Recorded output:

    400 simulated benchmarks per setting, 500 resamples each, nominal 95%.

      tau |  width |    cover, items  cover, comparisons |  separable, items  separable, comparisons
    ------------------------------------------------------------------------------------------------
     0.00 |  1.008 |    0.946 (0.004)        0.951 (0.004) |             14.76                   14.76
     0.25 |  0.944 |    0.943 (0.004)        0.929 (0.005) |             14.60                   14.69
     0.50 |  0.819 |    0.937 (0.005)        0.881 (0.007) |             14.17                   14.49
     1.00 |  0.635 |    0.946 (0.005)        0.798 (0.008) |             12.80                   13.93

Run it with::

    python examples/pairwise_cluster_study.py
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from evalaudit import bradley_terry

# The study measures what the package ships. The fit it converges to is
# computed with the package's own Newton solver, which the tests check
# against choix and statsmodels.
from evalaudit.pairwise import _fit

MODELS = ["m1", "m2", "m3", "m4", "m5", "m6"]
THETA = np.array([1.0, 0.6, 0.3, 0.0, -0.6, -1.3])
N_PROMPTS = 80
N_COMPARISONS = 2575
TAUS = (0.0, 0.25, 0.5, 1.0)
TRIALS = 400
N_BOOT = 500
SEED = 31
PAIRS = [(i, j) for i in range(len(MODELS)) for j in range(i + 1, len(MODELS))]


def fitted_limit(tau: float) -> dict:
    """The gaps the fit converges to on endless data at this tau."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(80)
    m = len(THETA)
    credit = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i != j:
                shift = THETA[i] - THETA[j] + math.sqrt(2.0) * tau * nodes
                credit[i, j] = np.sum(
                    weights / (1.0 + np.exp(-shift))
                ) / math.sqrt(2.0 * math.pi)
    r = _fit(credit[None])[0]
    return {
        (MODELS[i], MODELS[j]): r[i] - r[j]
        for i in range(m) for j in range(m) if i != j
    }


def benchmark(rng: np.random.Generator, tau: float) -> pd.DataFrame:
    """One simulated MT-Bench. 2,575 comparisons dealt over 80 prompts, 32
    or 33 each, with the pair for every comparison drawn uniformly."""
    per_prompt = np.full(N_PROMPTS, N_COMPARISONS // N_PROMPTS)
    per_prompt[: N_COMPARISONS % N_PROMPTS] += 1
    prompt = np.repeat(np.arange(N_PROMPTS), per_prompt)

    pick = rng.integers(0, len(PAIRS), N_COMPARISONS)
    a = np.array([PAIRS[k][0] for k in pick])
    b = np.array([PAIRS[k][1] for k in pick])
    shift = rng.normal(0.0, tau, size=(N_PROMPTS, len(THETA)))
    strength = THETA[a] + shift[prompt, a] - THETA[b] - shift[prompt, b]
    a_won = rng.random(N_COMPARISONS) < 1.0 / (1.0 + np.exp(-strength))

    names = np.array(MODELS)
    return pd.DataFrame({
        "item_id": prompt,
        "model_a": names[a],
        "model_b": names[b],
        "winner": np.where(a_won, names[a], names[b]),
    })


def measure(tau: float) -> dict:
    target = fitted_limit(tau)
    rng = np.random.default_rng(SEED)
    width = {"items": [], "comparisons": []}
    cover = {"items": [], "comparisons": []}
    separable = {"items": [], "comparisons": []}

    for k in range(TRIALS):
        data = benchmark(rng, tau)
        for scheme in ("items", "comparisons"):
            r = bradley_terry(data, n_boot=N_BOOT, seed=k, resample=scheme)
            p = r.pairs.sort_values(["model_a", "model_b"])
            truth = np.array([
                target[(x, y)] for x, y in zip(p["model_a"], p["model_b"])
            ])
            low, high = p["ci_low"].to_numpy(), p["ci_high"].to_numpy()
            width[scheme].append(np.mean(high - low))
            cover[scheme].append(np.mean((low <= truth) & (truth <= high)))
            separable[scheme].append(r.n_separable)

    out = {"width": np.mean(width["comparisons"]) / np.mean(width["items"])}
    for scheme in ("items", "comparisons"):
        rates = np.array(cover[scheme])
        out[f"cover_{scheme}"] = rates.mean()
        out[f"se_{scheme}"] = rates.std(ddof=1) / math.sqrt(TRIALS)
        out[f"separable_{scheme}"] = np.mean(separable[scheme])
    return out


def main() -> None:
    print(
        f"{TRIALS} simulated benchmarks per setting, {N_BOOT} resamples each, "
        f"nominal 95%."
    )
    print()
    header = (
        f"{'tau':>5} | {'width':>6} | {'cover, items':>15} "
        f"{'cover, comparisons':>19} | {'separable, items':>17} "
        f"{'separable, comparisons':>23}"
    )
    print(header)
    print("-" * len(header))
    for tau in TAUS:
        m = measure(tau)
        print(
            f"{tau:>5.2f} | {m['width']:>6.3f} | "
            f"{m['cover_items']:>8.3f} ({m['se_items']:.3f}) "
            f"{m['cover_comparisons']:>12.3f} ({m['se_comparisons']:.3f}) | "
            f"{m['separable_items']:>17.2f} {m['separable_comparisons']:>23.2f}"
        )


if __name__ == "__main__":
    main()
