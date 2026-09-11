"""Question 1 check. Does resample="comparisons" in 0.4.0 give 0.2.1's intervals?

Fits the same decisive votes with evalaudit 0.2.1 from PyPI, on the same seed
and resample count as q1_leaderboard.py, and compares the rating intervals
with the ones 0.4.0 wrote to output/q1_comparisons_ratings.csv. Also prints
the pairs 0.2.1 itself calls separable, which is its overlap rule, and checks
that the overlap rule applied to the 0.4.0 intervals picks the same pairs.

Run it in a separate environment that has evalaudit 0.2.1, after
q1_leaderboard.py has run:

    python analysis/q1_check_v021.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import evalaudit
from evalaudit import bradley_terry

# These must match q1_leaderboard.py.
SEED = 0
N_BOOT = 2000
REVISION = "f7d2896d2cc5d80f8b55c2bbc722613555233c25"
HERE = Path(__file__).resolve().parent


def load_human() -> pd.DataFrame:
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub = Path(os.environ.get("HF_HUB_CACHE", home / "hub"))
    folder = (
        hub / "datasets--lmsys--mt_bench_human_judgments" / "snapshots"
        / REVISION / "data"
    )
    files = sorted(folder.glob("human-*.parquet"))
    if len(files) != 1:
        sys.exit(f"Could not find the cached human split under {folder}.")
    return pd.read_parquet(
        files[0], columns=["question_id", "model_a", "model_b", "winner", "turn"]
    )


def main() -> None:
    where = Path(evalaudit.__file__).resolve()
    if evalaudit.__version__ != "0.2.1" or HERE.parent in where.parents:
        sys.exit(
            f"This needs evalaudit 0.2.1 installed from PyPI. Found "
            f"{evalaudit.__version__} at {where}."
        )
    print(f"evalaudit {evalaudit.__version__} at {where.parent}")

    saved = HERE / "output" / "q1_comparisons_ratings.csv"
    if not saved.exists():
        sys.exit(f"Run q1_leaderboard.py first. {saved} is missing.")
    new = pd.read_csv(saved).set_index("model")

    human = load_human()
    rows = human[human["winner"].isin(["model_a", "model_b"])]
    winner = np.where(rows["winner"] == "model_a", rows["model_a"], rows["model_b"])
    frame = pd.DataFrame({
        "item_id": rows["question_id"].to_numpy(),
        "model_a": rows["model_a"].to_numpy(),
        "model_b": rows["model_b"].to_numpy(),
        "winner": winner,
    })
    old = bradley_terry(frame, ties="split", n_boot=N_BOOT, seed=SEED)
    r = old.ratings.set_index("model").loc[new.index]

    print(f"decisive votes {len(frame)}, seed {SEED}, n_boot {N_BOOT}")
    for column in ("rating", "ci_low", "ci_high"):
        gap = np.max(np.abs(r[column].to_numpy() - new[column].to_numpy()))
        print(f"largest difference in {column}, 0.2.1 against 0.4.0 comparisons: {gap:.1e}")

    models = list(new.index)
    overlap = set()
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            if new.loc[a, "ci_high"] < new.loc[b, "ci_low"] or new.loc[b, "ci_high"] < new.loc[a, "ci_low"]:
                overlap.add(frozenset((a, b)))
    own = {frozenset((a, b)) for a, b in zip(old.separable_pairs["model_a"], old.separable_pairs["model_b"])}
    print(f"0.2.1 separable pairs: {len(own)} of {old.n_pairs}")
    print(f"overlap rule on the 0.4.0 intervals picks the same pairs: {own == overlap}")
    left_out = [
        f"{a} vs {b}" for i, a in enumerate(models) for b in models[i + 1:]
        if frozenset((a, b)) not in own
    ]
    print(f"0.2.1 does not separate: {', '.join(left_out) if left_out else 'none'}")


if __name__ == "__main__":
    main()
