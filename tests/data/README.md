# tests/data

## MT-Bench human judgments

`mt_bench_human.csv` and `mt_bench_gpt4_pair.csv` are taken from the
[MT-Bench human judgments dataset](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments)
by LMSYS, at revision `f7d2896d2cc5d80f8b55c2bbc722613555233c25`. The two
files are its `human` split (3,355 rows) and its `gpt4_pair` split (2,400
rows).

The dataset is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The licence is
stated in the dataset card at that revision.

The dataset is described in this paper.

> Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu,
> Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang,
> Joseph E. Gonzalez and Ion Stoica. Judging LLM-as-a-judge with MT-Bench
> and Chatbot Arena. 2023. arXiv:2306.05685.

### Changes

Six columns are kept, `question_id`, `model_a`, `model_b`, `winner`,
`judge` and `turn`. The `conversation_a` and `conversation_b` columns are
left out. Rows, their order and every value in the kept columns are
unchanged, and `winner` keeps its raw strings. The files are CSV where the
source is parquet.

`make_mt_bench_csv.py` in this folder wrote both files from the local
Hugging Face cache, and checks that each reads back as its split.

The tests in `tests/test_judge.py` read these files. They map the ties,
build the comparison units and keep each human's first vote themselves, the
way `analysis/q5_judge_against_human_baseline.py` does.
