"""Question 1. The MT-Bench human leaderboard under evalaudit 0.4.0.

Fits ``bradley_terry`` to every decisive human vote in the ``human`` split of
lmsys/mt_bench_human_judgments and runs the bootstrap three ways:

    items, question_id          one cluster per MT-Bench question
    items, (question_id, turn)  one cluster per question and turn
    comparisons                 one row at a time, as 0.2.1 did

Method

    Votes. A vote is decisive when ``winner`` is ``model_a`` or ``model_b``.
    The winner is rewritten as the model's name, because the split lists the
    same pair in both orders on different rows. Ties are passed as
    ``ties="split"``, which has nothing to act on here because no decisive
    vote is a tie. A reading that keeps the 780 tie votes and splits them is
    printed last, as a check on the interpretation.

    Item id. ``question_id`` is the prompt. Each MT-Bench question is one
    two-turn conversation, and both turns of it are answered by the same
    models from the same opening. The second clustering treats the two turns
    as separate prompts.

    Separability. 0.4.0 reads it off the percentile interval on each gap.
    0.2.1 read it off whether the two rating intervals overlapped. Both rules
    are applied to every run, which separates the two 0.3.0 changes.

    Width ratio. For each pair, the width of the gap interval under the item
    bootstrap over its width under the comparison bootstrap. The ratio of
    the mean widths squared is the design effect.

    Reference check. The point ratings are checked against choix, an
    independent Bradley-Terry fit, when it is installed.

    Monte Carlo check. The separable counts are recomputed on 20 more seeds.

Run

    python analysis/q1_leaderboard.py

It writes output/q1_comparisons_ratings.csv, which q1_check_v021.py reads.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import evalaudit
from evalaudit import bradley_terry

SEED = 0
N_BOOT = 2000
SWEEP_SEEDS = range(1, 21)
REVISION = "f7d2896d2cc5d80f8b55c2bbc722613555233c25"
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def check_package() -> None:
    where = Path(evalaudit.__file__).resolve()
    if evalaudit.__version__ != "0.4.0" or HERE.parent in where.parents:
        sys.exit(
            f"This needs evalaudit 0.4.0 installed from PyPI. Found "
            f"{evalaudit.__version__} at {where}."
        )
    print(f"evalaudit {evalaudit.__version__} at {where.parent}")


def load_split(split: str) -> pd.DataFrame:
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub = Path(os.environ.get("HF_HUB_CACHE", home / "hub"))
    folder = (
        hub / "datasets--lmsys--mt_bench_human_judgments" / "snapshots"
        / REVISION / "data"
    )
    files = sorted(folder.glob(f"{split}-*.parquet"))
    if len(files) != 1:
        sys.exit(f"Could not find the cached {split} split under {folder}.")
    columns = ["question_id", "model_a", "model_b", "winner", "judge", "turn"]
    return pd.read_parquet(files[0], columns=columns)


def votes(human: pd.DataFrame, keep_ties: bool) -> pd.DataFrame:
    """One row per vote with the winner as a model name or "tie"."""
    rows = human if keep_ties else human[human["winner"].isin(["model_a", "model_b"])]
    winner = np.where(
        rows["winner"] == "model_a", rows["model_a"],
        np.where(rows["winner"] == "model_b", rows["model_b"], "tie"),
    )
    return pd.DataFrame({
        "question_id": rows["question_id"].to_numpy(),
        "turn": rows["turn"].to_numpy(),
        "model_a": rows["model_a"].to_numpy(),
        "model_b": rows["model_b"].to_numpy(),
        "winner": winner,
    })


def item_ids(frame: pd.DataFrame, run: str) -> pd.Series:
    if run == "items, (question_id, turn)":
        return frame["question_id"].astype(str) + "/t" + frame["turn"].astype(str)
    return frame["question_id"]


RUNS = {
    "items, question_id": "items",
    "items, (question_id, turn)": "items",
    "comparisons": "comparisons",
}


def fit(frame: pd.DataFrame, run: str, seed: int):
    data = frame.assign(item_id=item_ids(frame, run))
    return bradley_terry(
        data[["item_id", "model_a", "model_b", "winner"]],
        ties="split", n_boot=N_BOOT, seed=seed, resample=RUNS[run],
    )


def overlap_rule(result) -> np.ndarray:
    """0.2.1's test. Separable when the two rating intervals do not overlap."""
    r = result.ratings.set_index("model")
    p = result.pairs
    lo_a = r.loc[p["model_a"], "ci_low"].to_numpy()
    hi_a = r.loc[p["model_a"], "ci_high"].to_numpy()
    lo_b = r.loc[p["model_b"], "ci_low"].to_numpy()
    hi_b = r.loc[p["model_b"], "ci_high"].to_numpy()
    return (hi_a < lo_b) | (hi_b < lo_a)


def pair_label(a, b) -> str:
    return f"{a} vs {b}"


def choix_check(frame: pd.DataFrame, result) -> None:
    try:
        import choix
    except ImportError:
        print("choix is not installed, so the reference check is skipped.")
        return
    models = sorted(set(frame["model_a"]) | set(frame["model_b"]))
    index = {m: k for k, m in enumerate(models)}
    loser = np.where(frame["winner"] == frame["model_a"], frame["model_b"], frame["model_a"])
    data = [(index[w], index[l]) for w, l in zip(frame["winner"], loser)]
    params = choix.ilsr_pairwise(len(models), data, alpha=0.0, tol=1e-12, max_iter=10_000)
    p = result.pairs
    ref_gap = np.array([params[index[a]] - params[index[b]] for a, b in zip(p["model_a"], p["model_b"])])
    diff = np.max(np.abs(ref_gap - p["difference"].to_numpy()))
    print(f"choix {choix.__version__ if hasattr(choix, '__version__') else ''} "
          f"ilsr_pairwise on the same votes. Largest difference in any of the "
          f"15 gaps: {diff:.1e}")


def main() -> None:
    check_package()
    human = load_split("human")
    frame = votes(human, keep_ties=False)
    print(f"dataset revision {REVISION}")
    print(f"human rows {len(human)}, decisive votes {len(frame)}, tie votes "
          f"{int((human['winner'] == 'tie').sum())}")
    print(f"seed {SEED}, n_boot {N_BOOT}, confidence 0.95, ties='split'")
    print()

    results = {run: fit(frame, run, SEED) for run in RUNS}

    # ---- a and b: the fits ----------------------------------------------
    print("=" * 78)
    print("1a/1b. Ratings, log-odds scale, reference at zero")
    print("=" * 78)
    first = next(iter(results.values()))
    print(f"comparisons {first.n_comparisons}, ties in them {first.n_ties}, "
          f"models {first.n_models}, reference {first.reference}")
    for run, r in results.items():
        print()
        print(f"--- {run}: {r.n_items} clusters in the bootstrap "
              f"({r.n_boot_usable} of {r.n_boot} resamples usable)"
              if RUNS[run] == "items" else
              f"--- {run}: {r.n_comparisons} rows resampled one at a time "
              f"({r.n_boot_usable} of {r.n_boot} resamples usable)")
        t = r.ratings.copy()
        t["width"] = t["ci_high"] - t["ci_low"]
        print(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    choix_check(frame, results["comparisons"])

    # ---- c: the pairs frames --------------------------------------------
    print()
    print("=" * 78)
    print("1c. Every pair. Gap is model_a minus model_b. 95% percentile interval.")
    print("=" * 78)
    for run, r in results.items():
        p = r.pairs.copy()
        p["width"] = p["ci_high"] - p["ci_low"]
        p["overlap_rule"] = overlap_rule(r)
        print()
        print(f"--- {run}: {int(p['separable'].sum())} of {len(p)} pairs separate")
        print(p[["model_a", "model_b", "difference", "ci_low", "ci_high", "width",
                 "n_head_to_head", "separable"]]
              .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        missing = p[~p["separable"]]
        names = ", ".join(pair_label(a, b) for a, b in zip(missing["model_a"], missing["model_b"]))
        print(f"does not separate: {names if names else 'none'}")

    # ---- d: width ratio and design effect -------------------------------
    print()
    print("=" * 78)
    print("1d. Gap interval width, item bootstrap over comparison bootstrap")
    print("=" * 78)
    base = results["comparisons"].pairs.assign(
        w_cmp=lambda d: d["ci_high"] - d["ci_low"])[["model_a", "model_b", "w_cmp"]]
    base_rating_w = (results["comparisons"].ratings["ci_high"]
                     - results["comparisons"].ratings["ci_low"]).mean()
    for run in ("items, question_id", "items, (question_id, turn)"):
        r = results[run]
        m = base.merge(r.pairs.assign(w_itm=lambda d: d["ci_high"] - d["ci_low"])
                       [["model_a", "model_b", "w_itm"]], on=["model_a", "model_b"])
        ratio = m["w_itm"].mean() / m["w_cmp"].mean()
        per_pair = m["w_itm"] / m["w_cmp"]
        rating_ratio = (r.ratings["ci_high"] - r.ratings["ci_low"]).mean() / base_rating_w
        print(f"--- {run}")
        print(f"mean gap width, items {m['w_itm'].mean():.3f}, comparisons {m['w_cmp'].mean():.3f}")
        print(f"ratio of mean gap widths {ratio:.3f}, squared (design effect) {ratio ** 2:.2f}")
        print(f"comparison width as a share of item width {1 / ratio:.1%}")
        print(f"per-pair width ratio, min {per_pair.min():.3f}, median "
              f"{per_pair.median():.3f}, max {per_pair.max():.3f}")
        print(f"ratio of mean rating-interval widths {rating_ratio:.3f}, squared {rating_ratio ** 2:.2f}")

    # ---- e: which 0.3.0 change dominated --------------------------------
    print()
    print("=" * 78)
    print("1e. Separable pairs of 15, by resampling unit and separability rule")
    print("=" * 78)
    rows = []
    for run, r in results.items():
        rows.append({
            "resample": run,
            "overlap rule (0.2.1)": int(overlap_rule(r).sum()),
            "gap rule (0.3.0 on)": int(r.pairs["separable"].sum()),
        })
    table = pd.DataFrame(rows).set_index("resample")
    print(table.to_string())
    a = table.loc["comparisons", "overlap rule (0.2.1)"]
    b = table.loc["comparisons", "gap rule (0.3.0 on)"]
    for run in ("items, question_id", "items, (question_id, turn)"):
        c = table.loc[run, "overlap rule (0.2.1)"]
        d = table.loc[run, "gap rule (0.3.0 on)"]
        print(f"{run}: gap rule alone {a} -> {b} ({b - a:+d}), item bootstrap "
              f"alone {a} -> {c} ({c - a:+d}), both {a} -> {d} ({d - a:+d})")
    for rule in ("overlap", "gap"):
        for run, r in results.items():
            sep = overlap_rule(r) if rule == "overlap" else r.pairs["separable"].to_numpy()
            p = r.pairs[~sep]
            names = ", ".join(pair_label(x, y) for x, y in zip(p["model_a"], p["model_b"]))
            print(f"not separated, {rule} rule, {run}: {names if names else 'none'}")

    # The bar a gap has to clear. With model_a the higher-rated model, the
    # overlap rule separates when the gap exceeds a's lower half-width plus
    # b's upper half-width. The gap rule separates when it exceeds the lower
    # half-width of the gap's own interval.
    print()
    print("The bar a gap has to clear, mean over the 15 pairs")
    bars = {}
    for run, r in results.items():
        t = r.ratings.set_index("model")
        p = r.pairs
        overlap_bar = (
            t.loc[p["model_a"], "rating"].to_numpy() - t.loc[p["model_a"], "ci_low"].to_numpy()
            + t.loc[p["model_b"], "ci_high"].to_numpy() - t.loc[p["model_b"], "rating"].to_numpy()
        )
        gap_bar = (p["difference"] - p["ci_low"]).to_numpy()
        bars[run] = {"overlap": overlap_bar.mean(), "gap": gap_bar.mean()}
        near = p.index[(p["model_a"] == "gpt-4") & (p["model_b"] == "claude-v1")][0]
        print(f"{run}: overlap rule {overlap_bar.mean():.3f}, gap rule "
              f"{gap_bar.mean():.3f}. For gpt-4 vs claude-v1 the gap is "
              f"{p.loc[near, 'difference']:.3f} against bars of "
              f"{overlap_bar[near]:.3f} and {gap_bar[near]:.3f}")
    c = bars["comparisons"]
    for run in ("items, question_id", "items, (question_id, turn)"):
        i = bars[run]
        print(f"{run}: gap rule alone multiplies the bar by {c['gap'] / c['overlap']:.2f}, "
              f"item bootstrap alone by {i['overlap'] / c['overlap']:.2f}, both by "
              f"{i['gap'] / c['overlap']:.2f}")

    # ---- Monte Carlo stability of the counts ----------------------------
    print()
    print("=" * 78)
    print(f"Seed check. Separable counts on seeds {SWEEP_SEEDS.start} to "
          f"{SWEEP_SEEDS.stop - 1}, n_boot {N_BOOT}")
    print("=" * 78)
    for run in RUNS:
        gap_counts, overlap_counts, misses = [], [], {}
        for s in SWEEP_SEEDS:
            r = fit(frame, run, s)
            gap_counts.append(int(r.pairs["separable"].sum()))
            overlap_counts.append(int(overlap_rule(r).sum()))
            for x, y in zip(r.pairs.loc[~r.pairs["separable"], "model_a"],
                            r.pairs.loc[~r.pairs["separable"], "model_b"]):
                misses[pair_label(x, y)] = misses.get(pair_label(x, y), 0) + 1
        g = pd.Series(gap_counts).value_counts().sort_index().to_dict()
        o = pd.Series(overlap_counts).value_counts().sort_index().to_dict()
        print(f"{run}: gap rule {g}, overlap rule {o} (count: seeds)")
        print(f"    pairs not separated under the gap rule, seeds of 20: {misses if misses else 'none'}")

    # ---- the other reading of "ties split" -------------------------------
    print()
    print("=" * 78)
    print("Reading check. All 3,355 votes, 780 ties split as half a win each")
    print("=" * 78)
    with_ties = votes(human, keep_ties=True)
    for run in RUNS:
        r = fit(with_ties, run, SEED)
        p = r.pairs[~r.pairs["separable"]]
        names = ", ".join(pair_label(x, y) for x, y in zip(p["model_a"], p["model_b"]))
        print(f"{run}: {int(r.pairs['separable'].sum())} of 15 separate by the "
              f"gap rule, {int(overlap_rule(r).sum())} by the overlap rule. "
              f"Not separated (gap rule): {names if names else 'none'}")

    OUT.mkdir(exist_ok=True)
    results["comparisons"].ratings.to_csv(OUT / "q1_comparisons_ratings.csv", index=False)
    print()
    print(f"wrote {OUT / 'q1_comparisons_ratings.csv'}")


if __name__ == "__main__":
    main()
