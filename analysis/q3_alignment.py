"""Question 3. Which alignment of human and judge labels gives which figure?

Method

    Join. Every human row is matched to the gpt4_pair row for the same
    question_id, the same two models in either order, and the same turn.
    Winners are compared by model name, because the human split lists a pair
    in both orders and gpt4_pair lists it in one. The funnel also shows what
    a join on the ordered pair keeps.

    Ties. A human "tie", and a judge "tie" or "tie (inconsistent)", are set
    aside as no label. The funnel prints each drop.

    Individual estimand. Each human judgment against the judge's label for
    its comparison. Rows that share a comparison share one judge label, and
    the bootstrap in judge_validation resamples rows as if they were
    independent.

    Aggregated estimand. One human label per comparison, keyed on
    question_id, unordered pair and turn. The label is the plurality of all
    human votes on it, tie votes included, and a tied plurality is a tie.
    Comparisons where either side is a tie are then set aside.

    Statistics. Raw agreement. Cohen's kappa from evalaudit.cohens_kappa,
    checked against scikit-learn. Krippendorff's alpha from
    evalaudit.judge_validation with its percentile bootstrap, checked against
    the krippendorff package. Kappa and alpha are computed on the same labels.

    Coding. As in q2_human_agreement.py. The label is the winner's
    alphabetical position in the pair. The table also gives it coded by model
    name and by the position gpt4_pair lists the model in.

Run

    python analysis/q3_alignment.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import evalaudit
from evalaudit import cohens_kappa, judge_validation

SEED = 0
N_BOOT = 2000
CONFIDENCE = 0.95
REVISION = "f7d2896d2cc5d80f8b55c2bbc722613555233c25"
HERE = Path(__file__).resolve().parent
CODINGS = ("alphabetical position", "model name", "gpt4_pair position")
UNIT = ["question_id", "lo", "hi", "turn"]
JUDGE_TIES = ("tie", "tie (inconsistent)")

pd.set_option("display.width", 220)


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


def with_names(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["lo"] = d[["model_a", "model_b"]].min(axis=1)
    d["hi"] = d[["model_a", "model_b"]].max(axis=1)
    d["name"] = np.where(
        d["winner"] == "model_a", d["model_a"],
        np.where(d["winner"] == "model_b", d["model_b"], d["winner"]),
    )
    return d


def code(winner: pd.Series, frame: pd.DataFrame, coding: str) -> np.ndarray:
    if coding == "alphabetical position":
        return np.where(winner.to_numpy() == frame["lo"].to_numpy(), "first", "second")
    if coding == "model name":
        return winner.to_numpy()
    return np.where(winner.to_numpy() == frame["gpt4_first"].to_numpy(),
                    "listed first", "listed second")


def statistics(human: np.ndarray, judge: np.ndarray) -> dict:
    import krippendorff
    from sklearn.metrics import cohen_kappa_score

    v = judge_validation(human, judge, level="nominal", confidence=CONFIDENCE,
                         n_boot=N_BOOT, seed=SEED)
    k = cohens_kappa(human, judge)
    cats = {c: i for i, c in enumerate(sorted(set(human) | set(judge)))}
    matrix = np.array([[cats[x] for x in human], [cats[x] for x in judge]], float)
    return {
        "n": v.n_items,
        "agreement": v.accuracy,
        "kappa": k.kappa,
        "kappa sklearn": cohen_kappa_score(human, judge),
        "alpha": v.agreement,
        "alpha ci_low": v.ci_low,
        "alpha ci_high": v.ci_high,
        "alpha krippendorff": krippendorff.alpha(reliability_data=matrix,
                                                 level_of_measurement="nominal"),
    }


def plurality(votes: list) -> str:
    counts = pd.Series(votes).value_counts()
    top = counts[counts == counts.max()].index
    return top[0] if len(top) == 1 else "tie"


def main() -> None:
    check_package()
    human = with_names(load_split("human"))
    gpt4 = with_names(load_split("gpt4_pair"))
    gpt4 = gpt4.rename(columns={"name": "judge_name", "model_a": "gpt4_first"})
    print(f"dataset revision {REVISION}, seed {SEED}, n_boot {N_BOOT}")
    print()

    # ---- a: the funnel ----------------------------------------------------
    print("=" * 78)
    print("3a. Funnel from the human split")
    print("=" * 78)
    print(f"human rows                                                   {len(human):>6}")
    ordered = human.merge(
        gpt4[["question_id", "gpt4_first", "model_b", "turn"]].rename(
            columns={"gpt4_first": "model_a"}),
        on=["question_id", "model_a", "model_b", "turn"])
    print(f"  (a join on the ordered pair would keep                     {len(ordered):>6})")
    joined = human.merge(gpt4[UNIT + ["judge_name", "gpt4_first"]], on=UNIT,
                         how="inner", validate="m:1")
    print(f"matched to gpt4_pair on question, unordered pair, turn       {len(joined):>6}")

    print()
    print("individual estimand, one row per human judgment")
    step = joined
    for label, mask in (
        ("judge label 'tie' set aside", step["judge_name"] != "tie"),
    ):
        step = step[mask]
        print(f"  {label:<59}{len(step):>6}")
    step = step[step["judge_name"] != "tie (inconsistent)"]
    print(f"  {'judge label tie (inconsistent) set aside':<59}{len(step):>6}")
    step = step[step["name"] != "tie"]
    print(f"  {'human tie vote set aside':<59}{len(step):>6}")
    individual = step

    print()
    print("aggregated estimand, one label per comparison")
    units = joined.groupby(UNIT, sort=False).agg(
        votes=("name", list), judge_name=("judge_name", "first"),
        gpt4_first=("gpt4_first", "first")).reset_index()
    units["human_label"] = [plurality(v) for v in units["votes"]]
    print(f"  {'comparisons (question, unordered pair, turn)':<59}{len(units):>6}")
    print(f"  {'  of which one human vote / two or more':<59}"
          f"{int((units['votes'].map(len) == 1).sum()):>6} / "
          f"{int((units['votes'].map(len) >= 2).sum())}")
    step = units[units["judge_name"] != "tie"]
    print(f"  {'judge label tie set aside':<59}{len(step):>6}")
    step = step[step["judge_name"] != "tie (inconsistent)"]
    print(f"  {'judge label tie (inconsistent) set aside':<59}{len(step):>6}")
    step = step[step["human_label"] != "tie"]
    print(f"  {'human plurality is a tie, set aside':<59}{len(step):>6}")
    aggregated = step

    # ---- b and d: both estimands, kappa and alpha ------------------------
    print()
    print("=" * 78)
    print("3b/3d. Judge against humans, ties set aside, 95% bootstrap on alpha")
    print("=" * 78)
    rows = []
    for name, frame, h in (
        ("individual", individual, individual["name"]),
        ("aggregated", aggregated, aggregated["human_label"]),
    ):
        for coding in CODINGS:
            s = statistics(code(h, frame, coding), code(frame["judge_name"], frame, coding))
            rows.append({"estimand": name, "coding": coding, **s})
    table = pd.DataFrame(rows)
    shown = table.drop(columns=["kappa sklearn", "alpha krippendorff"]).copy()
    shown["agreement"] = shown["agreement"].map(lambda x: f"{x:.1%}")
    print(shown.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("largest difference, evalaudit against the reference: kappa "
          f"{np.max(np.abs(table['kappa'] - table['kappa sklearn'])):.1e}, alpha "
          f"{np.max(np.abs(table['alpha'] - table['alpha krippendorff'])):.1e}")

    # ---- c: which estimand gives which figure ----------------------------
    print()
    print("=" * 78)
    print("3c. Against the two quoted figure sets, alphabetical-position coding")
    print("=" * 78)
    prim = table[table["coding"] == "alphabetical position"].set_index("estimand")
    agg = prim.loc["aggregated"]
    ind = prim.loc["individual"]
    print(f"dataset page: 1,814 shared, 1,078 decisive, 88.4%, kappa 0.767")
    print(f"  aggregated: {len(units):,} comparisons, {agg['n']:,} decisive, "
          f"{agg['agreement']:.1%}, kappa {agg['kappa']:.3f}")
    print(f"  individual: {len(joined):,} rows, {ind['n']:,} decisive, "
          f"{ind['agreement']:.1%}, kappa {ind['kappa']:.3f}")
    print(f"earlier run: 744 pairs, 82.0%, alpha 0.640 (0.584 to 0.692)")
    print(f"  aggregated: {agg['n']:,}, {agg['agreement']:.1%}, alpha "
          f"{agg['alpha']:.3f} ({agg['alpha ci_low']:.3f} to {agg['alpha ci_high']:.3f})")
    print(f"  individual: {ind['n']:,}, {ind['agreement']:.1%}, alpha "
          f"{ind['alpha']:.3f} ({ind['alpha ci_low']:.3f} to {ind['alpha ci_high']:.3f})")


if __name__ == "__main__":
    main()
