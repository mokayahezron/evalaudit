# evalaudit

[![Tests](https://github.com/mokayahezron/evalaudit/actions/workflows/tests.yml/badge.svg)](https://github.com/mokayahezron/evalaudit/actions/workflows/tests.yml)

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

Claim under audit: the new model is better. 1 critical finding, 2 warnings
and 2 for context. On the critical finding this data cannot support the
conclusion as stated. That does not mean the conclusion is wrong. 3 checks
could not run for lack of data. They are listed at the end.

## Findings

### 1. [critical] The headline margin does not exclude zero

This is the margin between new and old, which is the number the claim rests
on. Difference 2.5% (95% CI: -11.6% to 16.5%, paired, n=120). The interval
includes zero, so the data cannot show that either system is better. That
does not mean the two are level. 75 of 120 items changed between systems. Do
not report a direction from this data. [...]

### 2. [warning] No effect of interest was stated, so this reports the eval's reach

Without a stated effect the audit does not know what size of difference
matters. This reports what the design could find and makes no claim about
the margin it measured. This eval ran 120 paired comparisons. At 80% power
and a 5% significance level the smallest difference it could have found is
20 points [...] Set effect_of_interest in the config to the difference the
decision turns on, and this check will say whether the eval could find it.

### 3. [warning] Rater agreement is below the working threshold

The grades under this eval reproduce less well than the 0.667 the report is
holding them to. Krippendorff's alpha -0.235 (95% CI: -0.539 to 0.057,
nominal, 40 of 40 items graded more than once). The whole interval sits
below 0.667, the conventional floor for drawing any conclusion from coded
data. [...]

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
does not survive the fit, the eval could find differences of about 20 points
at 80% power and not reliably less, and the graders underneath it did not
agree with each other. No `effect_of_interest` was set, so the power check
reports what the design could find and says nothing about the margin. The
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
| `pairwise` | On a blind pairwise leaderboard, which pairs of models are actually separable and how many are not? Do the Elo gaps survive their intervals? |
| `power` | Could this many comparisons ever have detected the effect you care about? |
| `scores` | Is there an interval around your headline number, and how wide? |
| `compare` | Does the margin between two systems survive a paired test? |
| `audit` | Everything except `pairwise`, as a report ranked by what changes the conclusion. |

`audit` is the deliverable. It runs seven checks drawn from five of these
modules and orders what it finds by what changes the conclusion. `judge`
supplies three of the seven, since judge-human agreement, position bias and
length bias are separate findings. Each of those five modules also stands
alone when you want a single number instead of a report.

`pairwise` runs on its own. A leaderboard is a different question from a
single eval and arrives as a different frame, one row per comparison rather
than one row per graded item, so it is not folded into the report.

Agreement, judge and pairwise are where human graded evaluation goes wrong.
Scores, compare and power are the foundation those three stand on.

---

## Status

All seven modules are implemented. CI runs the suite on Python 3.9, 3.11 and
3.12, including the checks that pin results against outside implementations.
The examples below are run there too, so what they print here is what they
print for you.

### What `scores` answers

```python
from evalaudit import score_ci

wins = [1] * 58 + [0] * 42
print(score_ci(wins).summary())
```

```
58.0% pass rate (95% CI: 48.2%-67.2%, n=100). The interval spans
19 points. To compare this score with another, read the interval on the
difference, which compare_paired and compare_independent report. Two score
intervals that overlap do not show the systems are level.
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

### What `pairwise` answers

```python
import numpy as np
import pandas as pd
from evalaudit import bradley_terry

rng = np.random.default_rng(0)
models = ["m1", "m2", "m3", "m4", "m5", "m6"]
strength = dict(zip(models, [0.55, 0.35, 0.15, -0.05, -0.30, -0.70]))

rows = []
for i, first in enumerate(models):
    for second in models[i + 1:]:
        p = 1 / (1 + np.exp(-(strength[first] - strength[second])))
        for _ in range(40):
            rows.append((first, second, first if rng.random() < p else second))

comparisons = pd.DataFrame(rows, columns=["model_a", "model_b", "winner"])
comparisons["item_id"] = [f"q{i}" for i in range(len(rows))]

print(bradley_terry(comparisons, seed=0).summary())
```

```
Bradley-Terry ratings for 6 models from 600 comparisons (ties split). 7 of
15 pairs are separable at 95%, meaning the interval on the gap between the
two ratings excludes zero. For the other 8 pairs the interval includes zero,
so this data does not establish an order for those pairs in either
direction. [...] Ratings are anchored on m1 at 0. Only differences between
models mean anything [...]
```

Six models make fifteen pairs. Six hundred comparisons separate seven of
them. Five are the weakest model against each of the others, and the other
two put m1 and m2 above m5. No pair among the top four separates, and the
fit even rates m4 above m3, the reverse of the strengths the data was drawn
from. A published board would still print all six in a line, one through
six, and readers would take that order seriously.

`result.pairs` lists all fifteen pairs with the interval on each gap, so the
eight that do not separate can be read as well as counted.

The bootstrap behind those intervals resamples items, taking every comparison
made on one prompt together, because judgements of the same prompt move
together. Versions before 0.3.0 resampled single comparisons. On simulated
data shaped like MT-Bench, 2,575 comparisons over 80 prompts, that covered
88% where 95% was claimed once prompts shifted model strength by a spread of
0.5 on the log-odds scale, and 80% at a spread of 1.0.
`examples/pairwise_cluster_study.py` has the full table. Pass
`resample="comparisons"` to get the ratings and rating intervals an older
version gave on the same data and seed. Separability still follows the rule
above, so the count of separable pairs can differ.

`to_elo` puts the same fit on the 400 point scale people expect and carries
the intervals across with it. Leaderboards publish Elo without intervals,
which is how a 12 point gap gets read as a ranking.

Two shapes of data get no ratings at all rather than numbers that look like
a ranking. Models that split into groups which never met have no common
scale, and a model that never lost has no finite rating. Both are refused
and named.

---

## Coverage

A statistics package earns trust by showing its intervals cover at the rate
they claim. Generate datasets with a known truth, then confirm the 95%
interval contains it about 95% of the time. Runs per test range from a
hundred or so to a thousand, set by what each simulation costs.

The package builds fifteen intervals and nine of them are tested that way.
Those nine are the Wilson and bootstrap intervals on a single score, the
Tango, bootstrap and Hodges-Lehmann intervals on a paired difference, the
bootstrap intervals on Krippendorff's alpha and on judge-human alpha, and the
Bradley-Terry intervals on each rating and on each gap between two ratings.
The gap interval is tested on data where comparisons made on the same prompt
move together, which is the case its item bootstrap is for.

Five more are pinned against a reference instead, which is the sharper test
where one exists. The score interval for two independent proportions is swept
against its own p-value over every table at two group sizes, since that
interval is the test inverted and the two must never disagree about zero. The
Clopper-Pearson and Wilson intervals in `position_bias` and both logistic
intervals in `length_bias` are matched against `statsmodels`. The
Clopper-Pearson interval is also swept against its exact binomial p-value
over every count up to 30, since it is that test inverted and the two must
never disagree about a half. The Student-t interval in
`score_ci` is matched against `scipy`, and Welch's interval in
`compare_independent` against `statsmodels`.

The last one has nothing pinning its bounds, and this is the place to say so
rather than leave you to find out. It is the two-sample bootstrap in
`compare_independent`.

Those tests run in CI and you can read them in `tests/`.

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

Upgrading from an earlier version? The
[changelog](https://github.com/mokayahezron/evalaudit/blob/main/CHANGELOG.md)
lists every change a caller can see, with what to do about each breaking one.
0.3.0 breaks several, so read it before you upgrade.

[What MT-Bench establishes, and what it does not](analysis/mt-bench.md)
runs these checks over the public MT-Bench human judgements, on the
leaderboard and on the GPT-4 judge against the humans it stands in for.
The scripts behind every number in it are in [`analysis/`](analysis/).

---

## Who wrote this

I'm Hezron Mokaya, a statistician. I have spent years on the task and
QA side of expert AI evaluation. I
graded, and I ran quality control on other people's grading. I did not design
the benchmarks.

This package is the work I do for clients, published in the open. I audit AI
evaluations for statistical validity: rater agreement, pairwise comparison,
judge validation, and whether the numbers in a report support the claim being
made from them.

If you have an eval you are about to publish, act on, or defend, I take that
work.

ondibahezron@gmail.com

---

## Licence

MIT.
