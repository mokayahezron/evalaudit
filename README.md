# evalaudit

[![Tests](https://github.com/ondibahezron-glitch/evalaudit/actions/workflows/tests.yml/badge.svg)](https://github.com/ondibahezron-glitch/evalaudit/actions/workflows/tests.yml)

**Statistical validity checks for human-graded AI evaluations.**

Three experts graded 200 items. One of them disagreed with the other two on
almost everything. Nobody noticed, because nobody computed agreement. Half the items were
only graded once, so it could not have been computed anyway.

That evaluation produced a number. The number is now in a slide deck.

`evalaudit` is the set of checks that should have run first.

---

## Start here

```python
import numpy as np
import pandas as pd
from evalaudit import audit

rng = np.random.default_rng(0)

new = (rng.random(120) < 0.58).astype(int)
old = (rng.random(120) < 0.51).astype(int)

ratings = pd.DataFrame({
    "item_id": [f"i{i}" for i in range(40) for _ in range(2)],
    "rater_id": ["ana", "raj"] * 40,
    "rating": rng.integers(0, 2, 80),
})

report = audit(
    scores={"new": new, "old": old},
    ratings=ratings,
    config={"claim": "the new model is better", "seed": 0},
)
print(report.to_markdown())
```

```
# Eval audit

Claim under audit: the new model is better. 2 critical findings, 1 warning
and 2 for context. On the critical findings the conclusion this data is
being asked to support does not hold as stated. 3 checks could not run for
lack of data. They are listed at the end.

## Findings

### 1. [critical] The headline margin does not exclude zero

This is the margin between new and old, which is the number the claim rests
on. Difference 2.5% (95% CI: -11.6% to 16.5%, paired, n=120). The interval
crosses zero, so the data cannot confirm that either system is better. 75 of
120 items changed between systems. Do not report a direction from this data.
[...]

### 2. [critical] This eval could not have detected the effect at issue

The effect at issue is 2.5 points, the margin this eval reports. This eval
ran 120 paired comparisons. At 80% power and a 5% significance level the
smallest difference it could have found is 20 points [...] No conclusion
about an effect this size can be drawn from this data, in either direction.
A null result here says the eval was too small and says nothing about the
systems. [...]

### 3. [warning] Rater agreement is below the working threshold

The grades under this eval reproduce less well than the 0.667 the report is
holding them to. Krippendorff's alpha -0.235 (95% CI: -0.539 to 0.057,
nominal, 40 of 40 items graded more than once). That is below 0.667, the
conventional floor for drawing any conclusion from coded data. [...]

### 4. [info] Score interval for new

The headline number for new, with the interval around it. 50.0% pass rate
(95% CI: 41.2%-58.8%, n=120). [...]

### 5. [info] Score interval for old

The headline number for old, with the interval around it. 47.5% pass rate
(95% CI: 38.8%-56.4%, n=120). [...]

## Checks that could not run

- **Judge against humans**: No judge labels supplied. Pass judge with human
  and judge label sequences for the same items [...]
- **Position bias**: No pairwise judgements supplied. Pass comparisons as a
  frame with pair_id, option_a, option_b and winner columns [...]
- **Length bias**: No judge preferences and lengths supplied. Pass judge
  with preferences and lengths for the same pairs [...]
```

Two systems, a 2.5 point margin, and forty items graded twice. The margin
does not survive the fit, the eval was never large enough to find a margin
that size, and the graders underneath it did not agree with each other. The
report opens with whichever of those changes the decision, and it says out
loud which checks it could not run.

Every check runs only on the data you passed. Nothing is inferred from data
that is not there.

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
| `judge` | Does your LLM judge track your humans, and on which slices does it stop? Is it biased by position or length? |
| `pairwise` | In a blind pairwise comparison, which models are actually separable? |
| `power` | Could this many comparisons ever have detected the effect you care about? |
| `scores` | Is there an interval around your headline number, and how wide? |
| `compare` | Does the margin between two systems survive a paired test? |
| `audit` | All of the above, as a report ranked by what changes the conclusion. |

`audit` is the deliverable and the other six are the checks it runs. Each one
also stands alone when you want a single number instead of a report.
Agreement, judge and pairwise are where human graded evaluation goes wrong.
Scores, compare and power are the foundation those three stand on.

---

## Status

Early, and filling in along the build order.

| Module | State |
|---|---|
| `scores` | Implemented |
| `compare` | Implemented |
| `agreement` | Implemented |
| `judge` | Implemented |
| `power` | Implemented |
| `audit` | Implemented |
| `pairwise` | Specified, next |

### What `scores` answers

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

### What `power` answers

```python
from evalaudit import detectable_effect

print(detectable_effect(500, discordance_rate=12 / 500).summary())
```

```
This eval ran 500 paired comparisons. At 80% power and a 5% significance
level the smallest difference it could have found is 1.9 points [...] At a
2.4% discordance rate that is about 12 discordant pairs, and McNemar reads
only those. The rest of the items agree across both systems and carry no
information about which one is better.
```

Paired binary power runs on the discordance rate, not on the item count and
not on the pass rate. An eval with 500 items and 12 discordant pairs carries
the information of a 12 item study. When the rate is not supplied, a
conservative 0.3 stands in and the summary says so rather than assuming it
quietly.

---

## Coverage

A statistics package earns trust by showing its intervals cover at the rate
they claim. Every interval here is tested by simulation: generate 1,000
datasets with a known truth, confirm the 95% interval contains it about 95% of
the time. Those tests run in CI and you can read them in `tests/`.

---

## Related

[`promptstats`](https://pypi.org/project/promptstats/) covers the automated
side well: prompt sensitivity, model comparison across prompt variations,
bootstrapped intervals on benchmark scores. If your evaluation has no human
graders in it, start there.

---

## Install

```bash
pip install evalaudit
```

Python 3.9+, numpy, scipy, pandas. Nothing else.

---

## Licence

MIT.
