# evalaudit

[![Tests](https://github.com/ondibahezron-glitch/evalaudit/actions/workflows/tests.yml/badge.svg)](https://github.com/ondibahezron-glitch/evalaudit/actions/workflows/tests.yml)

**Statistical validity checks for human-graded AI evaluations.**

Three experts graded 200 items. One of them disagreed with the other two on
almost everything. Nobody noticed, because nobody computed agreement — and
half the items were only graded once, so it could not have been computed.

That evaluation produced a number. The number is now in a slide deck.

`evalaudit` is the set of checks that should have run first.

---

## Start here

```python
from evalaudit import score_ci

wins = [1] * 58 + [0] * 42
print(score_ci(wins).summary())
```

```
58.0% pass rate (95% CI: 48.2%-67.2%, n=100). The interval spans
19 points; treat differences smaller than that as unresolved.
```

The interval crosses 50%. On this data the new model might be worse than the
old one. That evaluation cannot support the decision being made with it.

---

## Why this exists

`scipy` computes McNemar's test. `sklearn` computes Cohen's kappa. Several
libraries now put bootstrapped intervals on automated eval scores.

What is missing is anything built for the **human** side of evaluation: the
graders, the rubric, the LLM judge standing in for both. That is where the
expensive failures happen, and it is where almost no tooling exists.

Every function returns an object with a `.summary()`. The numbers are the
input to the audit. The sentence is the output.

Two rules the library follows throughout:

- **Never report a bare p-value.** Always an effect size and an interval.
- **State what the data cannot support**, not only what it can.

---

## What it checks

| Module | Question |
|---|---|
| `agreement` | Do your graders agree? Which grader is the outlier? Which items are ambiguous rather than hard? |
| `judge` | Does your LLM judge track your humans — and on which slices does it stop? Is it biased by position or length? |
| `pairwise` | In a blind pairwise comparison, which models are actually separable? |
| `power` | Could this many comparisons ever have detected the effect you care about? |
| `scores` | Is there an interval around your headline number, and how wide? |
| `compare` | Does the margin between two systems survive a paired test? |
| `audit` | All of the above, as a report ranked by what changes the conclusion. |

The first four are the point. The last three are the foundation they stand on.

---

## Status

Early. `scores` is implemented. `agreement`, `judge` and `pairwise` are
specified and landing over the coming weeks.

---

## Coverage

A statistics package earns trust by showing its intervals cover at the rate
they claim. Every interval here is tested by simulation: generate 1,000
datasets with a known truth, confirm the 95% interval contains it about 95% of
the time. Those tests run in CI and you can read them in `tests/`.

---

## Related

[`promptstats`](https://pypi.org/project/promptstats/) covers the automated
side well — prompt sensitivity, model comparison across prompt variations,
bootstrapped intervals on benchmark scores. If your evaluation has no human
graders in it, start there.

---

## Install

```bash
git clone https://github.com/ondibahezron-glitch/evalaudit.git
cd evalaudit
pip install -e ".[dev]"
```

Python 3.9+, numpy, scipy, pandas. Nothing else. PyPI package coming soon.

---

## Licence

MIT.
