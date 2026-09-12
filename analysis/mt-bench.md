# What MT-Bench establishes, and what it does not

MT-Bench is one of the most cited human evaluation datasets in the field. The
human judgments are public, the GPT-4 judgments sit beside them, and anyone can
download both. I ran two standard checks over them. A Bradley-Terry fit with
intervals on the leaderboard, and chance-corrected agreement between the GPT-4
judge and the humans it stands in for.

I am not arguing that MT-Bench is bad. It is a well-built public dataset, which
is why it is worth reading carefully. The narrower question is what these
observations can actually distinguish, and in two places the answer is less than
the numbers are usually read to support.

Everything below is reproducible. The data is `lmsys/mt_bench_human_judgments`
at revision `f7d2896`. The code is `evalaudit` 0.4.0, which is on PyPI, and the
scripts are in `analysis/` in this repository. Three kinds of figure were checked
against an independent implementation. Bradley-Terry ratings go against `choix`,
Krippendorff's alpha against the `krippendorff` package, and Cohen's kappa
against `scikit-learn`. The width factor and the design effect are not among
them. Both are ratios of two runs of the same bootstrap code, and no second
implementation was run against them.

---

## 1. The pair the votes do not order

I fit a Bradley-Terry model to all 2,575 decisive human votes, ties split, and
put a 95% percentile interval on the gap between every pair of models. The
bootstrap resamples questions rather than single comparisons, because every
judgement made on one prompt moves with that prompt.

Fourteen of the fifteen pairs separate. The interval on their gap excludes zero,
so the data orders them.

The fifteenth is claude-v1 against gpt-3.5-turbo.

| Pair | Gap | 95% CI on the gap |
|---|---|---|
| gpt-4 over gpt-3.5-turbo | 0.569 | 0.298 to 0.833 |
| gpt-4 over claude-v1 | 0.532 | 0.162 to 0.910 |
| claude-v1 over gpt-3.5-turbo | 0.036 | -0.324 to 0.418 |

The gap is 0.036 in log-odds and the interval runs from -0.324 to 0.418. With
2,575 human votes behind the fit, these data do not establish which of those two
models is better.

That result is stable. It holds when the bootstrap clusters on the question,
when it clusters on the question and turn separately, and when it resamples
single comparisons the way older versions did. It holds on twenty-one seeds. The
point estimates match an independent Bradley-Terry fit from `choix` to fifteen
decimal places.

Two things are worth separating here. Overlapping error bars on two models'
ratings are not the test. Two ratings can overlap while the gap between them is
well measured, because both carry the error of the field average they are
measured against and that error cancels in the difference. The test is the
interval on the gap.

The other is clustering. Resampling single comparisons treats 2,575 judgements
spread over 80 prompts as independent. They are not. Correcting for that widens
every gap interval by a factor of 1.58, a design effect of 2.48. An analysis that
skips it reports intervals about a third too narrow.

A leaderboard prints a total order whether or not the data carries one, and it
carries no marker for where the support runs out. That order is partly a
rendering choice. Two models being genuinely close is a normal outcome, and 2,575
votes is a respectable sample. The gap is in the presentation.

---

## 2. The judge against the human baseline

MT-Bench ships GPT-4 judgments alongside the human ones, and GPT-4-as-judge has
since become a standard substitute for human raters. So the question worth asking
is how well the substitute matches what it replaced, and how well the humans
matched each other.

The figure usually quoted for this dataset is 88.4% agreement with Cohen's kappa
0.767, on 1,078 comparisons. I reproduce it exactly. It rests on two conventions.

**Ties are dropped.** A comparison where the judge or a human called it even is
removed before anything is computed.

**The humans are aggregated.** Each comparison gets one human label, taken from
the humans who judged it, and the judge is scored against that.

Both conventions remove disagreement before the measurement runs. Undo them and
the number changes.

Against individual human judgments rather than an aggregated label, the judge
scores 85.6% with alpha 0.712 (0.680 to 0.743).

Those two figures do not run over the same comparisons. The individual estimand
spans 1,232 comparisons and the aggregated one 1,078. The 154 in the gap are
comparisons where the humans have no single winner between them, so the
aggregated label is a tie and the comparison drops out, while the decisive votes
cast on it stay in the individual set. So the 2.8 points and the 0.055 of alpha
carry two things at once. One is aggregation, which helps a judge because it
scores better against a majority vote than against any single rater. The other
is a set of 154 comparisons that only the individual figure sees, and those are
comparisons the humans split on. Nothing here separates the two, and the gap
should not be read as the price of aggregation alone.

Counting ties as a real label, and keeping every comparison that two or more
humans judged:

| | Alpha | 95% CI |
|---|---|---|
| Two humans | 0.478 | 0.421 to 0.530 |
| GPT-4 against a human | 0.479 | 0.420 to 0.534 |
| Difference | +0.001 | -0.055 to +0.054 |

The interval on the difference includes zero. On these comparisons the data
cannot show that the judge agrees with a human any more or less than a second
human does. Raw agreement says the same thing, 64.6% against 65.9%, a difference
of 1.3 points with an interval from -2.7 to +4.9.

The difference is the quantity that matters, and it is computed on the same
resamples of the same units rather than read off two separate intervals. That is
the same mistake as reading a leaderboard for overlapping error bars.

### Why dropping ties changes the answer

Restricting to decisive comparisons does not remove a random subset. It removes
the hard ones, from both sides.

Of the 308 comparisons that the decisive-only rule discards, 83 are ones the
judge called even and 225 are ones where fewer than two humans were decisive. On
that second group, human-human alpha is -0.300. Those are the comparisons people
could not agree about, and the convention drops them before measuring how well
people agree.

So the usual reporting runs on the subset everyone found easy. That is selection
on the thing being measured, and it inflates both figures.

### The number also depends on a coding choice

On the same 1,078 decisive comparisons, Cohen's kappa is:

- 0.767 when the label is the winner's alphabetical position in the pair
- 0.848 when the label is the winning model's name
- 0.468 when the label is the winner's position in the judge split's own ordering

Same judgements, same pairs, three defensible ways to write down who won, and a
spread of 0.38 in the headline reliability statistic. Chance-corrected agreement
depends on how often each category is used, and the coding decides the marginals.

I have not seen this choice stated in any paper that reports a judge-agreement
figure. It should be.

### On the decisive comparisons the judge comes out ahead

Set the ties aside again and the comparison turns over. Take the 453 comparisons
where two or more humans were decisive and the judge was decisive too, code the
winner by alphabetical position, and cluster the bootstrap on the question:

| | Alpha | 95% CI |
|---|---|---|
| Two humans | 0.692 | 0.622 to 0.757 |
| GPT-4 against a human | 0.737 | 0.667 to 0.800 |
| Difference | +0.045 | +0.002 to +0.088 |

The interval on the difference excludes zero. On this set the judge tracks a
human label better than a second human does. Raw agreement says the same, 86.8%
against 83.9%, a difference of 2.9 points from +0.3 to +5.4.

Three things hold that result down. It is the only sample in the run where the
judge separates from the human baseline at all. It carries under one of the
three codings, since the model-name coding gives +0.028 (-0.001 to 0.056) and
the gpt4_pair-position coding gives +0.012 (-0.065 to 0.084). And it is the
sample the judge's own ties select. The 83 comparisons the judge called even are
removed before either figure is computed, so the judge is scored on what it
chose to answer.

Every configuration where nobody picks the sample includes zero. Keep ties as a
label, so that every comparison two humans judged is in, and the three codings
give +0.001 (-0.055 to 0.054), -0.005 (-0.053 to 0.039) and -0.026 (-0.088 to
0.035).

A judge coming out ahead is not a paradox. The judge-human figure sets one fixed
rater against each human in turn, and the human-human figure sets noisy raters
against each other. A rater sitting nearer the middle of the human spread than a
typical human does will beat the human-human figure without being better than a
human at anything. The two are different quantities, and the second can exceed
the first.

### What this means for using a judge

So a judge-human agreement number cannot be read on its own. Put the
human-human number from the same comparisons beside it, or you cannot tell a
judge that tracks people from a set of comparisons people found easy.

Both numbers move with the sample and the coding, and on this dataset they move
a long way. Human-human alpha runs from 0.478 over every comparison two humans
judged up to 0.692 on the decisive ones. On those decisive ones the coding alone
moves it from 0.514 to 0.798. Quote a judge-agreement figure without the sample
definition and the coding beside it and you have not said much.

---

## Method and limits

**Data.** `lmsys/mt_bench_human_judgments` at revision `f7d2896`, both the
`human` split (3,355 rows) and the `gpt4_pair` split (2,400 rows). 2,575 human
votes are decisive.

**Leaderboard.** Bradley-Terry with ties split, 2,000 bootstrap resamples,
seed 0, clustered on `question_id`. Separability is the interval on the gap
between two ratings excluding zero. Point ratings checked against `choix`.

**Agreement.** Krippendorff's alpha, nominal. Every figure that sets the judge
against the human baseline carries a percentile bootstrap over questions, and
both figures and the interval on their difference come from the same resamples
of the same comparisons. That covers the two tables in Section 2 and the alphas
quoted beside them. The one interval that does not is the 0.712 on individual
human judgments. It comes from `q3_alignment.py`, whose bootstrap resamples rows
as if they were independent, and the rows on one comparison share a judge label,
so that interval is narrower than a clustered one would be. Checked against the
reference `krippendorff` package, which agrees to sixteen decimal places, and
Cohen's kappa against `scikit-learn`.

**Weighting.** Alpha weights a comparison by its label count on both sides, so
the two figures are weighted identically. Raw agreement does not, and weighting
it by label count instead of pair count moves the difference by 1.5 points,
from +1.3 to -0.2. Both readings are in the script.

**Ties.** Where ties count as a label they are a third nominal category on both
sides. 128 of the 193 judge ties in this set are labelled "tie (inconsistent)",
meaning the judge named different winners in the two presentation orders.
Treating those as missing rather than as a label would drop the comparisons the
judge handled worst, which is the selection this analysis exists to remove, so
they are kept.

**Intervals.** Bradley-Terry gaps and alpha carry percentile bootstrap
intervals. Every interval in this post is one of those.

**What this is not.** I did not design MT-Bench, I did not collect these votes,
and nothing here says the published ordering is wrong. It says that for one pair
the data does not carry an ordering, that a judge-human agreement figure means
little without the human-human figure beside it, and that the conventions used
to produce that figure inflate it.

**Reproducing it.** `pip install evalaudit==0.4.0 krippendorff choix
scikit-learn pyarrow`, then run the scripts in `analysis/`. The three reference
packages are there for the checks the scripts make against them. Each script
prints its own numbers, and each one that resamples sets its own seed.
`q1_check_v021.py` is the exception to the install line. It needs a second
environment with evalaudit 0.2.1, and exits on 0.4.0.

---

## Who wrote this

I'm Hezron Mokaya, a statistician. I have spent years on the task and QA side of
expert AI evaluation. I graded, and I ran quality control on other people's
grading. I did not design the benchmarks.

This package is the work I do for clients, published in the open. I audit AI
evaluations for statistical validity: rater agreement, pairwise comparison, judge
validation, and whether the numbers in a report support the claim being made from
them.

If you have an eval you are about to publish, act on, or defend, I take that
work.

ondibahezron@gmail.com
