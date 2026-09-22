"""Write the two MT-Bench splits the judge tests read, as CSV.

Source. lmsys/mt_bench_human_judgments on Hugging Face at revision
f7d2896d2cc5d80f8b55c2bbc722613555233c25, the human split (3,355 rows) and
the gpt4_pair split (2,400 rows). They are read from the local Hugging Face
cache the way the scripts in analysis/ read them. Nothing is downloaded.

Change. Six columns are kept, question_id, model_a, model_b, winner, judge
and turn. The two conversation columns are left out. Rows, their order and
every value are unchanged. winner keeps its raw strings, "model_a",
"model_b", "tie" and "tie (inconsistent)". Every mapping the tests need
happens in the tests.

The data is CC BY 4.0. tests/data/README.md carries the attribution.

Run from the repository root with pandas and pyarrow installed.

    python tests/data/make_mt_bench_csv.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

REVISION = "f7d2896d2cc5d80f8b55c2bbc722613555233c25"
COLUMNS = ["question_id", "model_a", "model_b", "winner", "judge", "turn"]
SPLITS = {"human": 3355, "gpt4_pair": 2400}
HERE = Path(__file__).resolve().parent


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
    return pd.read_parquet(files[0], columns=COLUMNS)


def read_back(path: Path) -> pd.DataFrame:
    """The CSV as the tests read it. No value is turned into a missing one."""
    return pd.read_csv(path, keep_default_na=False, na_values=[])


def main() -> None:
    for split, rows in SPLITS.items():
        frame = load_split(split)
        if len(frame) != rows:
            sys.exit(f"The {split} split has {len(frame)} rows, expected {rows}.")
        path = HERE / f"mt_bench_{split}.csv"
        frame.to_csv(path, index=False, lineterminator="\n")

        again = read_back(path)
        same = (
            list(again.columns) == COLUMNS
            and len(again) == len(frame)
            and all(
                (again[c].astype(str).to_numpy() == frame[c].astype(str).to_numpy()).all()
                for c in COLUMNS
            )
        )
        if not same:
            sys.exit(f"{path.name} does not read back as the {split} split.")
        print(f"{path.name}: {len(frame)} rows, winner values "
              f"{sorted(frame['winner'].unique())}")


if __name__ == "__main__":
    main()
