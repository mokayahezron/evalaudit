"""Question 5. Does the judge agree with a human more than a second human does?

Method

    Units. The comparisons from q2_human_agreement.py, one question_id, one
    ordered pair and one turn. A unit is used when at least two distinct
    human judges gave it a decisive label and gpt4_pair's label on it is
    decisive. Both figures are computed on exactly these units and these
    human labels. Ties are set aside as no label, as in q2 and q3.

    Figures. Human-human is Krippendorff's alpha over the human labels in
    each unit, as rater_agreement computes it. Judge-human pairs each human
    label with gpt4_pair's label for its unit and computes alpha over those
    two-label items, as judge_validation computes it. Raw agreement is the
    share of same-unit human pairs that agree, and the share of human labels
    that match the judge.

    Resampling unit. question_id. Every comparison on one question shares the
    prompt, and each model's answer to it appears in several of the pairs on
    that question, so comparisons on one question are not independent. The
    bootstrap in q2 took each comparison as independent, and the
    judge_validation bootstrap took each human label as independent although
    the labels in one unit share one judge label. Drawing questions keeps
    every comparison, label and judge label of a drawn question together.
    Both figures and their difference come from the same draws. The same
    computation with the comparison as the resampling unit is printed as a
    check, and there the human-human interval has to equal rater_agreement's.

    Difference. Judge-human minus human-human on each resample, with a
    percentile interval.

    Codings. Alphabetical position, model name and gpt4_pair position, as in
    q2 and q3. Raw agreement does not depend on the coding.

    Sensitivity. The same with units pooled over the two presentation orders,
    keyed on question_id, unordered pair and turn.

    Reference checks. Point figures against rater_agreement, judge_validation
    and the krippendorff package. The first resamples against krippendorff
    run on the resampled data itself.

    Ties as a category. A last section, on ordered units, keeps "tie" as a
    third nominal value on both sides, so every unit with two or more human
    labels enters whatever anyone called. gpt4_pair's "tie" and "tie
    (inconsistent)" both become the value "tie". It reports the same figures
    on the same question resamples, the change against the decisive-only
    result, human-human alpha on the units the decisive-only comparison left
    out, and raw human-human agreement weighted by label count instead of
    pair count. The observed disagreement in alpha already weights each unit
    by its label count on both sides, so that reweighting applies to raw
    agreement only. The section checks that identity on the data.

Run

    python analysis/q5_judge_against_human_baseline.py
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
N_BOOT = 5000
CONFIDENCE = 0.95
CHECK_SEEDS = range(1, 11)
N_REFERENCE_DRAWS = 25
REVISION = "f7d2896d2cc5d80f8b55c2bbc722613555233c25"
HERE = Path(__file__).resolve().parent
CODINGS = ("alphabetical position", "model name", "gpt4_pair position")
UNIT_DEFINITIONS = {
    "ordered units": "ordered_unit",
    "pooled orders": "unordered_unit",
}
SCHEMES = ("question_id", "comparison")

pd.set_option("display.width", 220)
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


def prepare(human: pd.DataFrame, gpt4: pd.DataFrame) -> pd.DataFrame:
    """Human rows with the winner's name, gpt4_pair's label and its listing order."""
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
    )[["question_id", "lo", "hi", "turn", "model_a", "judge_name", "winner"]].rename(
        columns={"model_a": "gpt4_first", "winner": "judge_raw"})
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


def units_for(d: pd.DataFrame, unit: str):
    """Decisive human labels on units with two or more distinct judges, then
    the subset of those units where the judge is decisive too."""
    dec = d[d["winner_name"] != "tie"].drop_duplicates([unit, "judge"], keep="first")
    judges = dec.groupby(unit)["judge"].nunique()
    multi = dec[dec[unit].isin(judges[judges >= 2].index)]
    both = multi[multi["judge_name"] != "tie"].reset_index(drop=True)
    return multi, both


def code(winner: pd.Series, d: pd.DataFrame, coding: str) -> np.ndarray:
    w = np.asarray(winner, dtype=object)
    if coding == "alphabetical position":
        return np.where(w == d["lo"].to_numpy(), "first", "second")
    if coding == "model name":
        return w
    return np.where(w == d["gpt4_first"].to_numpy(), "listed first", "listed second")


def labels(d: pd.DataFrame, coding: str):
    """Human and judge labels as integer codes over one shared value domain."""
    human = code(d["winner_name"], d, coding)
    judge = code(d["judge_name"], d, coding)
    values, inverse = np.unique(np.concatenate([human, judge]).astype(str),
                                return_inverse=True)
    return inverse[: len(d)], inverse[len(d):], len(values)


def cluster_sums(d: pd.DataFrame, unit: str, cluster: str, h, j, n_values):
    """Per-cluster sums that both alphas and both raw agreements are built from.

    Alpha reads a set of units only through each unit's disagreement sum and
    the value marginals, and both add over units. So a resample's figures are
    its cluster multiplicities times these sums.
    """
    unit_codes, _ = pd.factorize(d[unit], sort=False)
    n_units = int(unit_codes.max()) + 1
    counts = np.zeros((n_units, n_values))
    np.add.at(counts, (unit_codes, h), 1.0)
    judge_counts = np.zeros((n_units, n_values))
    np.add.at(judge_counts, (unit_codes, j), 1.0)
    size = counts.sum(axis=1)
    match = np.bincount(unit_codes, weights=(h == j).astype(float), minlength=n_units)
    per_unit = {
        "hh_num": (size ** 2 - (counts ** 2).sum(axis=1)) / (size - 1),
        "hh_marg": counts,
        "hh_pairs": size * (size - 1) / 2,
        "hh_agree": (counts * (counts - 1) / 2).sum(axis=1),
        "hh_agree_by_label": size * (counts * (counts - 1) / 2).sum(axis=1)
        / (size * (size - 1) / 2),
        "jh_num": 2.0 * (size - match),
        "jh_marg": counts + judge_counts,
        "jh_n": size,
        "jh_agree": match,
    }
    first_row = pd.Series(np.arange(len(d))).groupby(unit_codes).first().to_numpy()
    cluster_of_unit, clusters = pd.factorize(d[cluster].to_numpy()[first_row], sort=False)
    to_cluster = np.zeros((len(clusters), n_units))
    to_cluster[cluster_of_unit, np.arange(n_units)] = 1.0
    sums = {k: to_cluster @ v for k, v in per_unit.items()}
    return sums, cluster_of_unit, unit_codes


def figures(sums: dict, weights: np.ndarray) -> dict:
    def alpha(num, marg):
        n = marg.sum(axis=1)
        expected = n ** 2 - (marg ** 2).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            a = 1.0 - (n - 1) * num / expected
        return np.where(expected > 0, a, np.nan)

    w = weights
    return {
        "hh_alpha": alpha(w @ sums["hh_num"], w @ sums["hh_marg"]),
        "jh_alpha": alpha(w @ sums["jh_num"], w @ sums["jh_marg"]),
        "hh_raw": (w @ sums["hh_agree"]) / (w @ sums["hh_pairs"]),
        "hh_raw_by_label": (w @ sums["hh_agree_by_label"]) / (w @ sums["jh_n"]),
        "jh_raw": (w @ sums["jh_agree"]) / (w @ sums["jh_n"]),
    }


def draw(n_clusters: int, n_boot: int, seed: int):
    """One index matrix and its multiplicities. The same draw rater_agreement
    makes from the same seed when the clusters are its units."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    offset = idx + np.arange(n_boot)[:, None] * n_clusters
    times = np.bincount(offset.ravel(), minlength=n_boot * n_clusters)
    return idx, times.reshape(n_boot, n_clusters).astype(float)


def interval(x: np.ndarray):
    tail = (1 - CONFIDENCE) / 2
    return np.percentile(x, [100 * tail, 100 * (1 - tail)])


def bootstrap(sums: dict, seed: int):
    n_clusters = len(sums["jh_n"])
    idx, times = draw(n_clusters, N_BOOT, seed)
    f = figures(sums, times)
    alpha_ok = np.isfinite(f["hh_alpha"]) & np.isfinite(f["jh_alpha"])
    out = {
        "hh_alpha": interval(f["hh_alpha"][alpha_ok]),
        "jh_alpha": interval(f["jh_alpha"][alpha_ok]),
        "d_alpha": interval((f["jh_alpha"] - f["hh_alpha"])[alpha_ok]),
        "hh_raw": interval(f["hh_raw"]),
        "jh_raw": interval(f["jh_raw"]),
        "d_raw": interval(f["jh_raw"] - f["hh_raw"]),
        "hh_raw_by_label": interval(f["hh_raw_by_label"]),
        "d_raw_by_label": interval(f["jh_raw"] - f["hh_raw_by_label"]),
        "usable": int(alpha_ok.sum()),
    }
    return out, idx, times


def reference_resamples(d, unit_codes, cluster_of_unit, idx, times, sums, h, j):
    """krippendorff.alpha on the resampled data, for the first few draws."""
    import krippendorff

    row_cluster = cluster_of_unit[unit_codes]
    members = [np.flatnonzero(row_cluster == c) for c in range(len(sums["jh_n"]))]
    coder_codes, coders = pd.factorize(d["judge"], sort=False)
    n_units = int(unit_codes.max()) + 1
    mine = figures(sums, times[:N_REFERENCE_DRAWS])
    worst = 0.0
    for b, row in enumerate(idx[:N_REFERENCE_DRAWS]):
        rows = np.concatenate([members[c] for c in row])
        copy = np.concatenate([np.full(len(members[c]), p) for p, c in enumerate(row)])
        columns, keys = pd.factorize(copy * n_units + unit_codes[rows], sort=False)
        hh = np.full((len(coders), len(keys)), np.nan)
        hh[coder_codes[rows], columns] = h[rows]
        jh = np.vstack([h[rows], j[rows]]).astype(float)
        ref_hh = krippendorff.alpha(reliability_data=hh, level_of_measurement="nominal")
        ref_jh = krippendorff.alpha(reliability_data=jh, level_of_measurement="nominal")
        worst = max(worst, abs(ref_hh - mine["hh_alpha"][b]), abs(ref_jh - mine["jh_alpha"][b]))
    return worst


def point_checks(d: pd.DataFrame, unit: str, h, j, point: dict) -> float:
    import krippendorff

    ratings = pd.DataFrame({"item_id": d[unit].to_numpy(),
                            "rater_id": d["judge"].to_numpy(), "rating": h})
    hh = rater_agreement(ratings, level="nominal", bootstrap_ci=False).alpha
    jh = judge_validation(h, j, level="nominal", n_boot=0).agreement
    unit_codes, units = pd.factorize(d[unit], sort=False)
    coder_codes, coders = pd.factorize(d["judge"], sort=False)
    matrix = np.full((len(coders), len(units)), np.nan)
    matrix[coder_codes, unit_codes] = h
    ref_hh = krippendorff.alpha(reliability_data=matrix, level_of_measurement="nominal")
    ref_jh = krippendorff.alpha(reliability_data=np.vstack([h, j]).astype(float),
                                level_of_measurement="nominal")
    a, b = point["hh_alpha"][0], point["jh_alpha"][0]
    return max(abs(a - hh), abs(a - ref_hh), abs(b - jh), abs(b - ref_jh))


def fmt(x, ci, pct=False):
    if pct:
        return f"{x:.1%} ({ci[0]:.1%} to {ci[1]:.1%})"
    return f"{x:.3f} ({ci[0]:.3f} to {ci[1]:.3f})"


def fmt_points(x, ci):
    return f"{100 * x:+.1f} ({100 * ci[0]:+.1f} to {100 * ci[1]:+.1f})"


def excludes_zero(ci) -> str:
    return "yes" if ci[0] > 0 or ci[1] < 0 else "no"


def analyse(d: pd.DataFrame, name: str, unit: str) -> None:
    print()
    print("=" * 100)
    print(f"{name}: unit is {unit}")
    print("=" * 100)
    multi, both = units_for(d, unit)
    n_multi = multi[unit].nunique()
    n_units = both[unit].nunique()
    sizes = both.groupby(unit).size()
    print(f"units with two or more distinct judges giving a decisive label   {n_multi}")
    print(f"  of which gpt4_pair's label is a tie, set aside                 {n_multi - n_units}")
    print(f"units used by both figures                                       {n_units}")
    print(f"human labels on them {len(both)}, human pairs "
          f"{int((sizes * (sizes - 1) / 2).sum())}, judge-human pairs {len(both)}, "
          f"questions {both['question_id'].nunique()}")
    print(f"human labels per unit {sizes.value_counts().sort_index().to_dict()}")

    if unit == "ordered_unit":
        h_all = code(multi["winner_name"], multi, "alphabetical position")
        q2 = rater_agreement(pd.DataFrame({
            "item_id": multi[unit].to_numpy(), "rater_id": multi["judge"].to_numpy(),
            "rating": h_all}), level="nominal", bootstrap_ci=False).alpha
        print(f"bridge to q2: human-human alpha on all {n_multi} units, alphabetical "
              f"position, is {q2:.3f}")

    alpha_rows, raw_rows, checks = [], [], []
    for coding in CODINGS:
        h, j, n_values = labels(both, coding)
        for scheme in SCHEMES:
            cluster = "question_id" if scheme == "question_id" else unit
            sums, cluster_of_unit, unit_codes = cluster_sums(both, unit, cluster, h, j, n_values)
            point = figures(sums, np.ones((1, len(sums["jh_n"]))))
            ci, idx, times = bootstrap(sums, SEED)
            p = {k: float(v[0]) for k, v in point.items()}
            alpha_rows.append({
                "coding": coding,
                "resample": f"{scheme} ({len(sums['jh_n'])})",
                "human-human alpha": fmt(p["hh_alpha"], ci["hh_alpha"]),
                "judge-human alpha": fmt(p["jh_alpha"], ci["jh_alpha"]),
                "difference": fmt(p["jh_alpha"] - p["hh_alpha"], ci["d_alpha"]),
                "excludes 0": excludes_zero(ci["d_alpha"]),
                "usable": ci["usable"],
            })
            raw_rows.append({
                "coding": coding,
                "resample": f"{scheme} ({len(sums['jh_n'])})",
                "human-human": fmt(p["hh_raw"], ci["hh_raw"], pct=True),
                "judge-human": fmt(p["jh_raw"], ci["jh_raw"], pct=True),
                "difference, points": fmt_points(p["jh_raw"] - p["hh_raw"], ci["d_raw"]),
                "excludes 0": excludes_zero(ci["d_raw"]),
            })
            if scheme == "question_id":
                checks.append(("point figures against rater_agreement, "
                               "judge_validation and krippendorff", coding,
                               point_checks(both, unit, h, j, point)))
                checks.append((f"first {N_REFERENCE_DRAWS} question resamples against "
                               f"krippendorff", coding,
                               reference_resamples(both, unit_codes, cluster_of_unit,
                                                   idx, times, sums, h, j)))
            else:
                ratings = pd.DataFrame({"item_id": both[unit].to_numpy(),
                                        "rater_id": both["judge"].to_numpy(), "rating": h})
                r = rater_agreement(ratings, level="nominal", confidence=CONFIDENCE,
                                    n_boot=N_BOOT, seed=SEED)
                checks.append(("comparison-unit human-human interval against "
                               "rater_agreement's", coding,
                               max(abs(r.ci_low - ci["hh_alpha"][0]),
                                   abs(r.ci_high - ci["hh_alpha"][1]))))

    print()
    print(f"Krippendorff's alpha, nominal, 95% percentile intervals, {N_BOOT} resamples, "
          f"seed {SEED}. Difference is judge-human minus human-human.")
    print(pd.DataFrame(alpha_rows).to_string(index=False))
    print()
    print("Raw agreement, same resamples")
    print(pd.DataFrame(raw_rows).to_string(index=False))
    print()
    print("Reference checks, largest absolute difference")
    for what, coding, value in checks:
        print(f"  {what}, {coding}: {value:.1e}")

    print()
    print(f"Seed check, resampling question_id, seeds {CHECK_SEEDS.start} to "
          f"{CHECK_SEEDS.stop - 1}: range of each bound on the difference")
    for coding in CODINGS:
        h, j, n_values = labels(both, coding)
        sums, _, _ = cluster_sums(both, unit, "question_id", h, j, n_values)
        lows, highs, raw_lows, raw_highs = [], [], [], []
        for s in CHECK_SEEDS:
            ci, _, _ = bootstrap(sums, s)
            lows.append(ci["d_alpha"][0]); highs.append(ci["d_alpha"][1])
            raw_lows.append(ci["d_raw"][0]); raw_highs.append(ci["d_raw"][1])
        print(f"  {coding}: alpha lower {min(lows):+.3f} to {max(lows):+.3f}, upper "
              f"{min(highs):+.3f} to {max(highs):+.3f}. Raw lower "
              f"{100 * min(raw_lows):+.1f} to {100 * max(raw_lows):+.1f}, upper "
              f"{100 * min(raw_highs):+.1f} to {100 * max(raw_highs):+.1f} points")


def all_labels(d: pd.DataFrame, unit: str) -> pd.DataFrame:
    """Every human label, tie included, on units with two or more distinct judges."""
    votes = d.drop_duplicates([unit, "judge"], keep="first")
    judges = votes.groupby(unit)["judge"].nunique()
    return votes[votes[unit].isin(judges[judges >= 2].index)].reset_index(drop=True)


def labels_with_ties(d: pd.DataFrame, coding: str):
    """As labels, with "tie" kept as its own value on both sides."""
    human = np.where(d["winner_name"] == "tie", "tie", code(d["winner_name"], d, coding))
    judge = np.where(d["judge_name"] == "tie", "tie", code(d["judge_name"], d, coding))
    values, inverse = np.unique(np.concatenate([human, judge]).astype(str),
                                return_inverse=True)
    return inverse[: len(d)], inverse[len(d):], len(values)


def run(frame: pd.DataFrame, unit: str, labeller, coding: str, seed: int = SEED) -> dict:
    """Point figures and question-resampled intervals for one frame and coding."""
    h, j, n_values = labeller(frame, coding)
    sums, cluster_of_unit, unit_codes = cluster_sums(frame, unit, "question_id", h, j, n_values)
    point = figures(sums, np.ones((1, len(sums["jh_n"]))))
    ci, idx, times = bootstrap(sums, seed)
    return {
        "p": {k: float(v[0]) for k, v in point.items()}, "point": point, "ci": ci,
        "h": h, "j": j, "sums": sums, "cluster_of_unit": cluster_of_unit,
        "unit_codes": unit_codes, "idx": idx, "times": times,
        "units": len(sums["hh_n"]) if "hh_n" in sums else int(unit_codes.max()) + 1,
        "questions": len(sums["jh_n"]),
    }


def alpha_weight_check(frame: pd.DataFrame, unit: str, h, n_values: int) -> float:
    """Largest gap between a unit's alpha numerator and n(1 - a).

    n is the unit's label count and a the share of its human pairs that
    agree. When the two are equal, alpha weights each unit by label count.
    """
    unit_codes, _ = pd.factorize(frame[unit], sort=False)
    counts = np.zeros((int(unit_codes.max()) + 1, n_values))
    np.add.at(counts, (unit_codes, h), 1.0)
    n = counts.sum(axis=1)
    a = (counts * (counts - 1) / 2).sum(axis=1) / (n * (n - 1) / 2)
    numerator = (n ** 2 - (counts ** 2).sum(axis=1)) / (n - 1)
    return float(np.max(np.abs(numerator - n * (1 - a))))


def analyse_ties(d: pd.DataFrame) -> None:
    unit = "ordered_unit"
    print()
    print("=" * 100)
    print("Ties as a third category: unit is ordered_unit, resampling question_id")
    print("=" * 100)
    multi, both = units_for(d, unit)
    allv = all_labels(d, unit)
    kept = set(both[unit])
    judged_out = set(multi[unit]) - kept
    human_out = set(allv[unit]) - set(multi[unit])
    sizes = allv.groupby(unit).size()
    judge_raw = allv.groupby(unit)["judge_raw"].first()
    print(f"units with two or more distinct human labels, any label: {len(sizes)}, "
          f"against {len(kept)} in the decisive-only comparison")
    print(f"  in the decisive-only set {len(kept)}, set aside there because gpt4_pair "
          f"tied {len(judged_out)}, set aside there because fewer than two humans "
          f"were decisive {len(human_out)}")
    print(f"human labels {len(allv)}, of which tie {int((allv['winner_name'] == 'tie').sum())}. "
          f"Human pairs {int((sizes * (sizes - 1) / 2).sum())}, judge-human pairs "
          f"{len(allv)}, questions {allv['question_id'].nunique()}")
    print(f"gpt4_pair label per unit: {judge_raw.value_counts().to_dict()}")
    print(f"human labels per unit {sizes.value_counts().sort_index().to_dict()}")

    # ---- b, c, d -----------------------------------------------------------
    alpha_rows, raw_rows, verdict_rows, checks = [], [], [], []
    for coding in CODINGS:
        t = run(allv, unit, labels_with_ties, coding)
        q = run(both, unit, labels, coding)
        p, ci = t["p"], t["ci"]
        diff = p["jh_alpha"] - p["hh_alpha"]
        diff_q = q["p"]["jh_alpha"] - q["p"]["hh_alpha"]
        raw = p["jh_raw"] - p["hh_raw"]
        raw_q = q["p"]["jh_raw"] - q["p"]["hh_raw"]
        alpha_rows.append({
            "coding": coding,
            "human-human alpha": fmt(p["hh_alpha"], ci["hh_alpha"]),
            "judge-human alpha": fmt(p["jh_alpha"], ci["jh_alpha"]),
            "difference": fmt(diff, ci["d_alpha"]),
            "excludes 0": excludes_zero(ci["d_alpha"]),
            "usable": ci["usable"],
        })
        raw_rows.append({
            "coding": coding,
            "human-human": fmt(p["hh_raw"], ci["hh_raw"], pct=True),
            "judge-human": fmt(p["jh_raw"], ci["jh_raw"], pct=True),
            "difference, points": fmt_points(raw, ci["d_raw"]),
            "excludes 0": excludes_zero(ci["d_raw"]),
        })
        verdict_rows.append({
            "coding": coding,
            "alpha, ties as no label": fmt(diff_q, q["ci"]["d_alpha"]),
            "excl 0": excludes_zero(q["ci"]["d_alpha"]),
            "alpha, ties as category": fmt(diff, ci["d_alpha"]),
            "excl 0 ": excludes_zero(ci["d_alpha"]),
            "shift": f"{diff - diff_q:+.3f}",
            "raw, ties as no label": fmt_points(raw_q, q["ci"]["d_raw"]),
            "excl 0  ": excludes_zero(q["ci"]["d_raw"]),
            "raw, ties as category": fmt_points(raw, ci["d_raw"]),
            "excl 0   ": excludes_zero(ci["d_raw"]),
            "raw shift": f"{100 * (raw - raw_q):+.1f}",
        })
        checks.append(("point figures against rater_agreement, judge_validation "
                       "and krippendorff", coding,
                       point_checks(allv, unit, t["h"], t["j"], t["point"])))
        checks.append((f"first {N_REFERENCE_DRAWS} question resamples against krippendorff",
                       coding,
                       reference_resamples(allv, t["unit_codes"], t["cluster_of_unit"],
                                           t["idx"], t["times"], t["sums"], t["h"], t["j"])))

    print()
    print(f"b. Krippendorff's alpha, ties as a category, {N_BOOT} question resamples, "
          f"seed {SEED}. Difference is judge-human minus human-human.")
    print(pd.DataFrame(alpha_rows).to_string(index=False))
    print()
    print("c. Raw agreement, ties as a category, same resamples")
    print(pd.DataFrame(raw_rows).to_string(index=False))
    print()
    print("d. Difference against the decisive-only result. Shift is the change in "
          "the point difference.")
    print(pd.DataFrame(verdict_rows).to_string(index=False))
    print()
    print("Reference checks, largest absolute difference")
    for what, coding, value in checks:
        print(f"  {what}, {coding}: {value:.1e}")

    # ---- e: the units the decisive-only comparison left out ---------------
    print()
    print("e. Human-human alpha by unit set, question resamples. Judge-human "
          "alpha in the last column, alphabetical position.")
    subsets = [
        ("decisive-only set, ties as no label", both, labels),
        ("gpt4_pair tied, ties as no label", multi[multi[unit].isin(judged_out)], labels),
        ("decisive-only set, ties as category", allv[allv[unit].isin(kept)], labels_with_ties),
        ("gpt4_pair tied, ties as category", allv[allv[unit].isin(judged_out)], labels_with_ties),
        ("under two decisive humans, ties as category",
         allv[allv[unit].isin(human_out)], labels_with_ties),
        ("all units, ties as category", allv, labels_with_ties),
    ]
    rows = []
    for name, frame, labeller in subsets:
        row = {"units": name, "n": frame[unit].nunique(),
               "questions": frame["question_id"].nunique()}
        for coding in CODINGS:
            r = run(frame, unit, labeller, coding)
            row[coding] = fmt(r["p"]["hh_alpha"], r["ci"]["hh_alpha"])
            if coding == "alphabetical position":
                judge_side = r["p"]["jh_alpha"], r["ci"]["jh_alpha"]
        judge_tied_without_ties = labeller is labels and frame is not both
        row["judge-human"] = "n/a" if judge_tied_without_ties else fmt(*judge_side)
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))

    # ---- f: weighting ------------------------------------------------------
    print()
    print("f. Weighting. Raw human-human agreement pooled over pairs weights a unit "
          "by n(n-1)/2. Weighted by label count n, as judge-human is:")
    for name, frame, labeller in (("decisive-only set, ties as no label", both, labels),
                                  ("all units, ties as category", allv, labels_with_ties)):
        r = run(frame, unit, labeller, "alphabetical position")
        p, ci = r["p"], r["ci"]
        h, _, n_values = labeller(frame, "alphabetical position")
        print(f"  {name}: human-human by pairs {p['hh_raw']:.1%}, by labels "
              f"{p['hh_raw_by_label']:.1%} ({ci['hh_raw_by_label'][0]:.1%} to "
              f"{ci['hh_raw_by_label'][1]:.1%}), shift "
              f"{100 * (p['hh_raw_by_label'] - p['hh_raw']):+.1f} points. Judge-human "
              f"{p['jh_raw']:.1%}")
        print(f"    difference with pair weighting {fmt_points(p['jh_raw'] - p['hh_raw'], ci['d_raw'])}"
              f" excludes 0 {excludes_zero(ci['d_raw'])}, with label weighting "
              f"{fmt_points(p['jh_raw'] - p['hh_raw_by_label'], ci['d_raw_by_label'])}"
              f" excludes 0 {excludes_zero(ci['d_raw_by_label'])}")
        print(f"    alpha numerator against n(1 - a), largest gap over units: "
              f"{alpha_weight_check(frame, unit, h, n_values):.1e}")

    # ---- seed check ----------------------------------------------------------
    print()
    print(f"Seed check, ties as a category, seeds {CHECK_SEEDS.start} to "
          f"{CHECK_SEEDS.stop - 1}: range of each bound on the difference")
    for coding in CODINGS:
        lows, highs, raw_lows, raw_highs = [], [], [], []
        for s in CHECK_SEEDS:
            ci = run(allv, unit, labels_with_ties, coding, seed=s)["ci"]
            lows.append(ci["d_alpha"][0]); highs.append(ci["d_alpha"][1])
            raw_lows.append(ci["d_raw"][0]); raw_highs.append(ci["d_raw"][1])
        print(f"  {coding}: alpha lower {min(lows):+.3f} to {max(lows):+.3f}, upper "
              f"{min(highs):+.3f} to {max(highs):+.3f}. Raw lower "
              f"{100 * min(raw_lows):+.1f} to {100 * max(raw_lows):+.1f}, upper "
              f"{100 * min(raw_highs):+.1f} to {100 * max(raw_highs):+.1f} points")


def main() -> None:
    check_package()
    d = prepare(load_split("human"), load_split("gpt4_pair"))
    print(f"dataset revision {REVISION}, seed {SEED}, n_boot {N_BOOT}, "
          f"confidence {CONFIDENCE}")
    for name, unit in UNIT_DEFINITIONS.items():
        analyse(d, name, unit)
    analyse_ties(d)


if __name__ == "__main__":
    main()
