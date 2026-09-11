# analysis

Scripts that answer four questions about evalaudit 0.4.0 on the
lmsys/mt_bench_human_judgments dataset. Each script runs on its own, sets its
own seed and prints its own numbers.

## Environment

The scripts need evalaudit 0.4.0 installed from PyPI. Each one checks the
version and exits if evalaudit is imported from this repository's working
tree. The repository's `.venv` holds an editable install whose metadata
reports 0.2.1, so it fails that check.

`q1_check_v021.py` needs a second environment with evalaudit 0.2.1 from PyPI.

Reference packages: `krippendorff` for alpha, `choix` for Bradley-Terry,
`scikit-learn` for Cohen's kappa. `pyarrow` reads the parquet files.

```bash
python -m venv venv040
```

```bash
venv040/Scripts/python -m pip install evalaudit==0.4.0 krippendorff choix scikit-learn pyarrow
```

```bash
python -m venv venv021
```

```bash
venv021/Scripts/python -m pip install evalaudit==0.2.1 pyarrow
```

On Linux and macOS the interpreter is `venv040/bin/python`.

The runs recorded for this analysis used Python 3.13.14, numpy 2.5.3,
scipy 1.18.1, pandas 3.0.5, pyarrow 25.0.1, krippendorff 0.8.2, choix 0.4.1
and scikit-learn 1.9.1.

## Data

The scripts read the two splits from the Hugging Face cache at dataset
revision `f7d2896d2cc5d80f8b55c2bbc722613555233c25`, files
`data/human-*.parquet` (3,355 rows) and `data/gpt4_pair-*.parquet`
(2,400 rows). They look under `HF_HUB_CACHE`, then `HF_HOME/hub`, then
`~/.cache/huggingface/hub`. They do not download anything.

## Scripts

| Script | Question | What it prints | Time |
| --- | --- | --- | --- |
| `q1_leaderboard.py` | 1 | Bradley-Terry ratings and the full pairs frame for three bootstraps: clustered on `question_id`, clustered on `(question_id, turn)`, and `resample="comparisons"`. Width ratio and design effect. Separable counts under the 0.2.1 and 0.3.0 rules. A choix check, a 20-seed check and a check with ties kept. Writes `output/q1_comparisons_ratings.csv`. | about 30 s |
| `q1_check_v021.py` | 1 | Fits the same votes with evalaudit 0.2.1 and compares its rating intervals with the 0.4.0 comparison bootstrap. Prints the pairs 0.2.1 separates. | a few s |
| `q2_human_agreement.py` | 2 | Units with two or more human judges, Krippendorff's alpha from evalaudit and from the krippendorff package with the same bootstrap resamples, raw pairwise human agreement, the judge on the same units, and a check that pools both model orders. | about 20 s |
| `q3_alignment.py` | 3 | The funnel from 3,355 human rows, and judge-human agreement, Cohen's kappa and alpha for individual and aggregated human labels, each under three label codings. | about 10 s |

Settings shared by the scripts: seed 0, 2,000 bootstrap resamples, 95%
percentile intervals. Ties are set aside as no label in questions 2 and 3.
Labels in questions 2 and 3 are coded by the winner's alphabetical position
in the pair, with model-name and gpt4_pair-position codings printed beside
it.

## Running

From the repository root, with the environment's interpreter:

```bash
venv040/Scripts/python analysis/q1_leaderboard.py
```

```bash
venv021/Scripts/python analysis/q1_check_v021.py
```

```bash
venv040/Scripts/python analysis/q2_human_agreement.py
```

```bash
venv040/Scripts/python analysis/q3_alignment.py
```

`q1_check_v021.py` reads the file `q1_leaderboard.py` writes, so it runs
second. The rest run in any order.
