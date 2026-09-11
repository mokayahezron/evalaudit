"""Question 2. How well do two humans agree on the same comparison?

Method

    Unit. One comparison, which is one question_id, one ordered pair
    (model_a, model_b) and one turn. The pair is kept in the order the split
    records it, so two humans shown the same answers in opposite orders fall
    in different units. A check at the end pools the two orders.

    Label. The model that won. A tie vote is set aside as no label. That is
    the rule q3_alignment.py applies to both the human and the judge labels.
    A unit carries information when at least two distinct human judges gave
    it a decisive label.

    Coding. The label is coded as the winner's alphabetical position in the
    pair, "first" or "second". Alpha and kappa both correct for chance using
    how often each code is used, so the coding moves them.
    q3_alignment.py uses the same coding. Both scripts also print the figures
    coded by model name and by the position gpt4_pair lists the model in.

    Alpha. evalaudit.rater_agreement at level nominal, with its percentile
    bootstrap over units. The reference is krippendorff.alpha on the
    coder-by-unit matrix. That package has no bootstrap, so its interval is
    taken by calling it on the resamples evalaudit draws, rebuilt from the
    same seed.

    Raw agreement. Every pair of distinct judges who labelled the same unit,
    and the share of those pairs that agree. Pooled over pairs, and averaged
    over units.

Run

    python analysis/q2_human_agreement.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import evalaudit
from evalaudit import judge_validation, rater_agreement

SEED = 0
N_BOOT = 2000
CONFIDENCE = 0.95
REVISION = "f7d2896d2cc5d80f8b55c2bbc722613555233c25"
HERE = Path(__file__).resolve().parent
CODINGS = ("alphabetical position", "model name", "gpt4_pair position")


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


def prepare(human: pd.DataFrame, gpt4: pd.DataFrame) -> pd.DataFrame:
    """Human rows with the winner's name and gpt4_pair's listing order."""
    d = human.copy()
    d["lo"] = d[["model_a", "model_b"]].min(axis=1)
    d["hi"] = d[["model_a", "model_b"]].max(axis=1)
    d["winner_name"] = np.where(
        d["winner"] == "model_a", d["model_a"],
        np.where(d["winner"] == "model_b", d["model_b"], "tie"),
    )
    g = gpt4.assign(
        lo=gpt4[["model_a", "model_b"]].min(axis=1),
        hi=gpt4[["model_a", "model_b"]].max(axis=1),
        judge_name=np.where(
            gpt4["winner"] == "model_a", gpt4["model_a"],
            np.where(gpt4["winner"] == "model_b", gpt4["model_b"], "tie"),
        ),
    )[["question_id", "lo", "hi", "turn", "model_a", "judge_name"]].rename(
        columns={"model_a": "gpt4_first"})
    d = d.merge(g, on=["question_id", "lo", "hi", "turn"], how="left", validate="m:1")
    d["ordered_unit"] = (
        d["question_id"].astype(str) + "|" + d["model_a"] + "|" + d["model_b"]
        + "|t" + d["turn"].astype(str)
    )
    d["unordered_unit"] = (
        d["question_id"].astype(str) + "|" + d["lo"] + "|" + d["hi"]
        + "|t" + d["turn"].astype(str)
    )
    return d


def code(d: pd.DataFrame, coding: str) -> np.ndarray:
    if coding == "alphabetical position":
        return np.where(d["winner_name"] == d["lo"], "first", "second")
    if coding == "model name":
        return d["winner_name"].to_numpy()
    return np.where(d["winner_name"] == d["gpt4_first"], "listed first", "listed second")


def reference_alpha(ratings: pd.DataFrame, n_boot: int, seed: int):
    """krippendorff.alpha on the coder-by-unit matrix, and on evalaudit's
    resamples of it.

    rater_agreement keeps the units with two or more ratings, numbers them in
    order of first appearance, and draws one (n_boot, units) index matrix
    from default_rng(seed). The same matrix is drawn here.
    """
    import krippendorff

    per_unit = ratings.groupby("item_id", sort=False)["rating"].transform("size")
    pairable = ratings[per_unit >= 2]
    unit_codes, units = pd.factorize(pairable["item_id"], sort=False)
    coders, coder_codes = np.unique(pairable["rater_id"], return_inverse=True)
    values, value_codes = np.unique(pairable["rating"], return_inverse=True)
    matrix = np.full((len(coders), len(units)), np.nan)
    matrix[coder_codes, unit_codes] = value_codes

    point = krippendorff.alpha(reliability_data=matrix, level_of_measurement="nominal")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(units), size=(n_boot, len(units)))
    draws = np.array([
        krippendorff.alpha(reliability_data=matrix[:, row], level_of_measurement="nominal")
        for row in idx
    ])
    draws = draws[np.isfinite(draws)]
    tail = (1 - CONFIDENCE) / 2
    low, high = np.percentile(draws, [100 * tail, 100 * (1 - tail)])
    return float(point), float(low), float(high), matrix.shape


def raw_agreement(d: pd.DataFrame, unit: str):
    """Share of same-unit judge pairs that agree, pooled and per unit."""
    counts = d.groupby([unit, "winner_name"]).size()
    per_unit = counts.groupby(level=0).agg(
        n=lambda c: int(c.sum()),
        agree=lambda c: float((c * (c - 1) / 2).sum()),
    )
    per_unit["pairs"] = per_unit["n"] * (per_unit["n"] - 1) / 2
    per_unit = per_unit[per_unit["pairs"] > 0]
    pooled = per_unit["agree"].sum() / per_unit["pairs"].sum()
    by_unit = (per_unit["agree"] / per_unit["pairs"]).mean()
    return pooled, by_unit, int(per_unit["pairs"].sum()), len(per_unit)


def main() -> None:
    check_package()
    human = load_split("human")
    gpt4 = load_split("gpt4_pair")
    d = prepare(human, gpt4)
    print(f"dataset revision {REVISION}, seed {SEED}, n_boot {N_BOOT}, "
          f"confidence {CONFIDENCE}")
    print()

    # ---- a: the units ----------------------------------------------------
    print("=" * 78)
    print("2a. Units: question_id, ordered (model_a, model_b), turn")
    print("=" * 78)
    judges_all = d.groupby("ordered_unit")["judge"].nunique()
    print(f"human rows {len(d)}, units {len(judges_all)}")
    print(f"units with at least two distinct judges, any vote: "
          f"{int((judges_all >= 2).sum())}")
    dup = int(d.duplicated(["ordered_unit", "judge"]).sum())
    print(f"rows repeating a (unit, judge) pair: {dup}")

    dec = d[d["winner_name"] != "tie"].copy()
    judges_dec = dec.groupby("ordered_unit")["judge"].nunique()
    keep = judges_dec[judges_dec >= 2].index
    used = dec[dec["ordered_unit"].isin(keep)].copy()
    print(f"tie votes set aside {len(d) - len(dec)}, decisive votes {len(dec)}")
    print(f"units with at least two distinct judges giving a decisive label: "
          f"{len(keep)}")
    print(f"decisive labels in those units {len(used)}, from "
          f"{used['judge'].nunique()} judges")
    print(f"judges per unit: "
          f"{judges_dec[judges_dec >= 2].value_counts().sort_index().to_dict()}")

    # ---- b, c, d: alpha, two ways, same rule ------------------------------
    print()
    print("=" * 78)
    print("2b-2d. Krippendorff's alpha, nominal, ties set aside")
    print("=" * 78)
    rows = []
    for coding in CODINGS:
        ratings = pd.DataFrame({
            "item_id": dec["ordered_unit"].to_numpy(),
            "rater_id": dec["judge"].to_numpy(),
            "rating": code(dec, coding),
        })
        r = rater_agreement(ratings, level="nominal", confidence=CONFIDENCE,
                            n_boot=N_BOOT, seed=SEED)
        ref, ref_lo, ref_hi, shape = reference_alpha(ratings, N_BOOT, SEED)
        rows.append({
            "coding": coding,
            "units": r.n_overlapping_items,
            "alpha": r.alpha,
            "ci_low": r.ci_low,
            "ci_high": r.ci_high,
            "ref alpha": ref,
            "ref ci_low": ref_lo,
            "ref ci_high": ref_hi,
            "matrix": f"{shape[0]} x {shape[1]}",
            "max abs diff": max(abs(r.alpha - ref), abs(r.ci_low - ref_lo),
                                abs(r.ci_high - ref_hi)),
        })
    table = pd.DataFrame(rows)
    print(table.drop(columns="max abs diff").to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))
    print("largest difference between evalaudit and krippendorff, point and "
          "both bounds: " + ", ".join(
              f"{c} {v:.1e}" for c, v in zip(table["coding"], table["max abs diff"])))
    print("rule: a tie vote is no label, a unit needs two distinct judges with "
          "a decisive label, and the label is coded by alphabetical position. "
          "q3_alignment.py sets ties aside the same way on both sides.")

    # ---- e: raw agreement ------------------------------------------------
    print()
    print("=" * 78)
    print("2e. Raw pairwise human-human agreement, same units, ties set aside")
    print("=" * 78)
    pooled, by_unit, n_pairs, n_units = raw_agreement(used, "ordered_unit")
    print(f"judge pairs {n_pairs} over {n_units} units")
    print(f"agreement pooled over pairs {pooled:.1%}, averaged over units {by_unit:.1%}")

    # ---- the judge on the same units -------------------------------------
    print()
    print("=" * 78)
    print("Same units, gpt4_pair against each human label, same rule and coding")
    print("=" * 78)
    same = used[used["judge_name"] != "tie"]
    v = judge_validation(
        code(same, "alphabetical position"),
        np.where(same["judge_name"] == same["lo"], "first", "second"),
        level="nominal", confidence=CONFIDENCE, n_boot=N_BOOT, seed=SEED,
    )
    print(f"human labels {len(used)}, set aside where gpt4_pair says tie "
          f"{len(used) - len(same)}, compared {v.n_items} over "
          f"{same['ordered_unit'].nunique()} units")
    print(f"agreement {v.accuracy:.1%}, alpha {v.agreement:.3f} "
          f"({v.ci_low:.3f} to {v.ci_high:.3f}). The bootstrap resamples labels, "
          f"so labels sharing a unit are treated as independent.")

    # ---- the two orders pooled ------------------------------------------
    print()
    print("=" * 78)
    print("Unit check. question_id, unordered pair, turn. Both orders pooled.")
    print("=" * 78)
    again_all = int(d.duplicated(["unordered_unit", "judge"]).sum())
    again = int(dec.duplicated(["unordered_unit", "judge"]).sum())
    pooled_dec = dec.drop_duplicates(["unordered_unit", "judge"], keep="first")
    j = pooled_dec.groupby("unordered_unit")["judge"].nunique()
    used_u = pooled_dec[pooled_dec["unordered_unit"].isin(j[j >= 2].index)]
    ratings = pd.DataFrame({
        "item_id": pooled_dec["unordered_unit"].to_numpy(),
        "rater_id": pooled_dec["judge"].to_numpy(),
        "rating": code(pooled_dec, "alphabetical position"),
    })
    r = rater_agreement(ratings, level="nominal", confidence=CONFIDENCE,
                        n_boot=N_BOOT, seed=SEED)
    pooled_u, by_unit_u, n_pairs_u, _ = raw_agreement(used_u, "unordered_unit")
    print(f"rows where a judge labelled the same comparison a second time, in "
          f"the other order: {again_all} among all votes, {again} among "
          f"decisive votes. Where it happens the first label is kept.")
    print(f"units with two or more distinct judges {r.n_overlapping_items}, "
          f"judge pairs {n_pairs_u}")
    print(f"alpha {r.alpha:.3f} ({r.ci_low:.3f} to {r.ci_high:.3f}), raw agreement "
          f"pooled {pooled_u:.1%}, averaged over units {by_unit_u:.1%}")


if __name__ == "__main__":
    main()
