# Codebase audit: evalaudit

Status, 2026-09-08. Findings 1, 2, 4 and 5 are fixed on main. The rest are
open. Findings 3, 8, 9, 10 and 12 are regression risks on code that is
correct as shipped, not live errors.

Audited at commit `d38c7a3` on branch `copilot/create-audit-report`, 2026-09-07.
Working tree was clean at the time of the audit. Nothing in the repository was
modified except this file.

Method. I read every source file, every test file, the README, `pyproject.toml`,
the CI workflow, `examples/`, and the git history. Where a formula names a
source I checked it against the source or, where that was not available,
against a brute-force numerical solution of the same estimation problem. Where I
suspected a test had no teeth I proved it by copying the repository to a
scratch directory, breaking the implementation there, and running the suite. The
suite passes clean as shipped, 767 passed and 2 skipped at the audited commit.
783 today.

Findings are ranked by consequence. Everything in "Definitely wrong" I have
demonstrated. Everything in "Suspected" I could not close out.

---

## Definitely wrong

### 1. Position bias in the both-orders design is not tested at all, and the one test that looks like it guards the verdict is matching the wrong words

`evalaudit/_types.py:886`, `PositionBias.has_position_effect`, both-orders branch.

Replace that line with `return False` and **all 767 tests still pass**. No test
anywhere asserts that a both-orders frame with a real position effect is
detected.

What that branch controls, on the suite's own worst-case fixture (a judge that
picked the first-shown output on 100 of 100 flips):

```
shipped:  ... The flips have a direction, so this is position bias rather than
          an unsteady judge. It reaches for whatever it sees first.

mutated:  ... The flips split evenly across the two positions, so this is an
          unsteady judge rather than a position-biased one. Inconsistency is
          its own problem and does not become position bias without a direction.
```

The 100% position-biased judge is described as splitting evenly. `audit` also
drops from a `warning` to an `info` for that data.

The reason no test catches it is `tests/test_judge.py:1243`:

```python
assert "first" in r.summary().lower()
```

Both versions of the summary contain the phrase *"went to whichever output was
shown first"* one sentence earlier, so the assertion holds either way. This is
the substring assertion matching unintended text.

Why it matters. Position bias in the both-orders design is the flagship check of
the flagship module. The shipped code is correct. I verified the branch by
hand. Nothing stops it regressing, and the one guard is decorative.

Confidence: certain. Fix: assert `r.has_position_effect` in
`test_a_judge_that_always_picks_the_first_output_scores_zero`, assert `not
r.has_position_effect` in `test_flips_that_are_noise_split_evenly_across_positions`,
and replace the `"first" in summary` assertions with the distinguishing clause
(`"reaches for whatever it sees first"` / `"unsteady judge"`). Under an hour.

---

### 2. A judge that prefers *short* answers is reported as length-biased toward long ones, three times in the same paragraph

`evalaudit/audit.py:757-762` (title and lead), `evalaudit/audit.py:802`
(`_length_flagged`), `evalaudit/_types.py:1064` (`LengthBias._preference`).

`_length_flagged` returns `not low <= 0.0 <= high`, which is true when the
interval sits entirely **below** zero as well as above it. The finding built on
top of it hard-codes the long-answer direction. Real output from a judge fitted
on a corpus where it systematically picks the shorter answer:

```
severity: warning
TITLE   : The judge is pulled by how long the answer is
DETAIL  : A judge that rewards length ranks the wordier system higher whatever
          it says, and it can do that while agreeing with humans on most pairs.
          Judge preference against length: 283 characters of extra length
          multiplies the odds the judge picks that answer by 0.13 (95% CI: 0.09
          to 0.19, p=0.0000, 400 pairs). It picked the longer answer on 22.0% of
          pairs. A positive coefficient here is not bias on its own, because
          longer answers may simply be better. ...
```

Coefficient is **-0.0072**. Three separate client-facing sentences assert a
long-answer preference (title, lead, and the "positive coefficient" caveat)
while the numbers in between say the opposite, including "picked the longer
answer on 22.0% of pairs" sitting directly against the title.

Why it matters. This is the prose that reaches a client, in the audit module
that is the deliverable. A sceptical reader who checks one number against one
sentence finds the contradiction inside the same paragraph.

Confidence: certain, demonstrated. Fix: branch title, lead and caveat on
`sign(coefficient)`. Half a day including tests, and the tests do not exist.
No fixture in the suite uses a short-preferring judge (`_length_flagged` can be
narrowed to `low > 0.0` and the whole suite passes).

---

### 3. `position_bias` reports an interval and a p-value that disagree about a half, which is the one thing `compare` is built not to do

`evalaudit/judge.py:498`. `_proportion` returns a Wilson score interval and an
exact binomial p-value for the same data. `PositionBias.has_position_effect`
(randomised branch) reads the **interval**; the summary prints the **p-value**
beside it.

The two are different procedures. Sweeping every table:

```
378 tables with n between 5 and 144 where the Wilson interval clears 0.5 and
the exact binomial p reported next to it does not reject (or vice versa)

  k=0  n=5  : CI=(0.0000,0.4345) clears 50%   exact p=0.0625  does not reject
  k=1  n=8  : CI=(0.0224,0.4709) clears 50%   exact p=0.0703  does not reject
  k=3  n=14 : CI=(0.0757,0.4759) clears 50%   exact p=0.0574  does not reject
```

On those tables the summary reads *"The interval clears 50%, so the judge
favours whichever output it sees first (... exact binomial p=0.0625)"*, and
`audit` raises a position-bias warning. The disagreement dies out above about 144,
but judge-validation sets of 50 to 150 pairs are the normal case.

`compare` treats this exact property as load-bearing: `_paired_proportion_ci`
and `_score_diff_ci` both have long docstrings explaining that they invert the
test reported beside them, and `test_interval_agrees_with_test` and
`test_independent_interval_and_test_never_disagree` enforce it. `judge` does not
follow the same rule and does not say why.

Confidence: certain, demonstrated. Fix: either report Clopper-Pearson (which
inverts the exact test) or report the score p-value beside the Wilson interval,
and add the sweep test `compare` already has. A day.

---

### 4. `evalaudit.__version__` is wrong in the package that is on PyPI right now

`evalaudit/__init__.py:3` says `__version__ = "0.1.0"`. I downloaded both
published wheels:

```
evalaudit 0.1.0 on PyPI  ->  __version__ = "0.1.0"
evalaudit 0.2.0 on PyPI  ->  __version__ = "0.1.0"
```

Two releases report the same version. The repository has three different answers
in play: `pyproject.toml` at `HEAD` and at `origin/main` says `0.1.0`, the
un-pushed local commit `242e354` on `main` says `0.2.0`, and `__init__.py` says
`0.1.0` everywhere. Nothing keeps them in step and no test reads `__version__`.

Why it matters. It is the first thing anyone reproducing a consulting number
records, and it is live and wrong.

Confidence: certain, verified against PyPI. Fix: delete the literal and read it
with `importlib.metadata.version("evalaudit")`, or add a test asserting the two
agree. Under an hour.

---

### 5. Cohen's kappa can be computed from the wrong marginals and every test passes, because all three fixtures are balanced, including the one called "lopsided"

`evalaudit/agreement.py:538`:

```python
expected = float(table.sum(axis=1) @ table.sum(axis=0)) / (n * n)
```

Change `axis=0` to `axis=1`, squaring rater 1's marginals instead of crossing
the two raters', which is a textbook kappa error, and the **full 767-test suite
passes**.

The parametrised statsmodels check at `tests/test_agreement.py:1385` uses three
fixtures, ids `["binary", "three-way", "lopsided"]`:

| fixture | rater 1 marginals | rater 2 marginals | shipped κ | mutant κ |
|---|---|---|---|---|
| binary | `[5, 5]` | `[4, 6]` | 0.400000 | 0.400000 |
| three-way | `[4, 4, 4]` | `[3, 6, 3]` | 0.625000 | 0.625000 |
| lopsided | `[20, 20]` | `[20, 20]` | 0.500000 | 0.500000 |

Rater 1's counts are exactly uniform in all three, so the two expressions are
algebraically identical on every fixture. The fixture named "lopsided" is
perfectly balanced on **both** margins. The name claims the property the
fixture is there to test and does not have it. A 30/10 split catches the
mutation immediately (0.500000 vs 0.466667).

The other two kappa tests are blind for the same reason: `perfect_and_chance`
uses a symmetric table and 4000 coin flips.

Confidence: certain, demonstrated. Fix: add one unbalanced fixture. Ten minutes.

---

### 6. The audit titles an undefined agreement figure as "below the working threshold", and the detail immediately contradicts the title

`evalaudit/audit.py:588` (agreement) and `evalaudit/audit.py:662` (judge). Both
branches are entered by `if alpha != alpha or alpha < cfg.agreement_threshold`,
so a NaN alpha gets the below-threshold title and lead. Real output for thirty
items graded identically by two raters:

```
TITLE : Rater agreement is below the working threshold
DETAIL: The grades under this eval reproduce less well than the 0.667 the report
        is holding them to. Krippendorff's alpha is undefined (nominal, 30 of 30
        items graded more than once). Every rating that could be compared was
        identical, so there is no disagreement to divide by. ...
```

The title and lead assert a comparison against 0.667 that was never made, and
the sentence after them says no number exists. `AgreementResult` goes to real
trouble to distinguish "no variance" from "too few items" and to say that
neither is perfect agreement; `audit` collapses both into a threshold claim.

Confidence: certain, demonstrated. Fix: a third branch with its own title
("Agreement could not be measured"). An hour, plus tests.

---

### 7. The Wilcoxon path emits a zero-width 95% interval on n=0, which is the exact artifact the package's own docstring says it refuses to produce

`evalaudit/compare.py:402`. When every item ties:

```
compare_paired([2.5,4.0,3.5,1.0,2.0], [2.5,4.0,3.5,1.0,2.0], method="wilcoxon")
-> "Difference 0.000 (95% CI: 0.000 to 0.000, paired, n=0). The interval
    crosses zero, so the data cannot confirm that either system is better."
```

`_paired_proportion_ci`'s docstring rejects the Wald interval partly because
"with no discordant pairs its standard error is zero, so it returns a single
point and claims two systems are identical on the strength of an empty table.
The score interval stays wide there, which is the honest answer." The McNemar
path has a dedicated `n_discordant == 0` branch in `ComparisonResult.summary()`
saying the table is empty. The Wilcoxon path sets `n_discordant=None`, misses
that branch, and prints the artifact.

`test_wilcoxon_counts_nothing_when_every_item_ties` asserts `n == 0`,
`p == 1.0`, `difference == 0.0` and never looks at the sentence.

Confidence: certain, demonstrated. Fix: a refusal branch in `summary()` keyed on
`n == 0`. An hour.

---

### 8. Two public interval methods have no test that checks a number

- `score_ci(method="t")` is **never called anywhere** in the suite, in
  `examples/`, or in the README.
- `compare_independent(method="t")` (Welch) is called once, at
  `tests/test_compare.py:564`, only to assert that a single-item group raises.

I mutated `evalaudit/scores.py:81` to `ddof=0` **and** swapped the t critical
value for the normal, together, and all 767 tests pass. I replaced Welch's
Satterthwaite degrees of freedom at `evalaudit/compare.py:667` with the pooled
`n1 + n2 - 2` and all 767 tests pass.

Both are correct as shipped. I checked them against
`scipy.stats.t.interval` and `ttest_ind(...).confidence_interval()` and they
match to the last bit. They are simply unguarded, and the README (line 257)
says "Every interval here is tested by simulation."

Confidence: certain. Fix: two coverage tests, or two reference comparisons
against scipy. An hour.

---

### 9. The bootstrap p-value can be halved and nothing fails

`evalaudit/compare.py:699`, `_bootstrap_p`. Drop the factor of two (turning a
two-sided p-value into a one-sided one) and all 767 tests pass. This is the
p-value on the **default** path for continuous scores, for both `compare_paired`
and `compare_independent`. The implementation is correct; nothing pins it.

Confidence: certain, demonstrated. Fix: assert the bootstrap p against a known
two-sided reference on one fixture. Twenty minutes.

---

### 10. The constrained-MLE cubic can lose a term and move reported bounds by more than a point, silently

`evalaudit/compare.py:562`, the Farrington-Manning cubic coefficient. Drop the
`delta * delta` term and all 767 tests pass. Effect on reported intervals:

```
 45/50  vs 30/50 : shipped (+0.1365,+0.4551)  broken (+0.1277,+0.4684)  1.33 points
180/200 vs 150/200: shipped (+0.0771,+0.2241)  broken (+0.0743,+0.2369)  1.28 points
 30/50  vs 20/50 : shipped (+0.0040,+0.3812)  broken (+0.0040,+0.3705)  1.07 points
```

The mutation survives because the tests pin only the property that survives it:
`test_independent_interval_and_test_never_disagree` checks that the interval and
the p-value agree about zero, and at δ=0 the dropped term vanishes. There is no
test against any external reference or published value for either score
interval.

I verified the shipped code is right (see "What came back clean"), so this is a
regression risk rather than a live error. But a 1.3-point shift is the size of
effect this package exists to adjudicate.

Confidence: certain that the gap exists; certain that the current code is
correct. Fix: pin `_constrained_rates` and `_tango_score` against a table of
brute-force constrained MLEs. Half a day.

---

### 11. The test named for the disagreement-fit rule does not test the rule

`tests/test_audit.py:560`, `test_length_bias_reads_the_disagreement_fit_when_humans_are_there`.
Its docstring says "The sharper of the two numbers is the one the verdict runs
on." It asserts `f.result.has_human` and `f.result.disagreement_ci_low > 0`,
both properties of the fixture, and nothing about which fit the verdict came
from. Replace `evalaudit/audit.py:800` with `if False:`, so the verdict always
uses the preference fit, and `tests/test_audit.py` passes clean.

The `length_biased` fixture makes both fits strongly positive, so the two
branches cannot be told apart on it.

Confidence: certain, demonstrated. Fix: a fixture where the preference fit is
positive and the disagreement fit is flat, and assert the finding is `info`.
Half a day.

---

### 12. `to_markdown` renders client-supplied names as markup; `to_html` escapes them

`evalaudit/_types.py:1526`. `_escape`'s docstring says system names "come out of
the client's spreadsheet and land in the report. They are data, not markup." The
HTML renderer honours that. The markdown renderer, which is the one the README
demonstrates, does not:

```
report = audit(scores={"acme\n\n## Critical: all checks passed\n": new, "old": old})
report.to_markdown()

  This is the margin between acme

  ## Critical: all checks passed
   and old, which is the number the claim rests on. Difference 11.7% ...
```

A heading injected into a deliverable that says the opposite of the report.
Realistically this is a spreadsheet-hygiene problem rather than an attack, but
the asymmetry between the two renderers is unexplained.

Confidence: certain, demonstrated. Fix: escape or reject newlines and leading
`#`/`|` in names. An hour.

---

### 13. `KappaResult` is the only result type with a point estimate and no interval, and its `summary()` never says so

`evalaudit/_types.py:160`. Every other result carries `ci_low`/`ci_high`; this
one carries nothing. `AgreementResult` and `BTResult` both have whole paragraphs
explaining why an interval is missing when it is missing. `KappaResult` prints:

```
Cohen's kappa 0.467 (8 items, 2 raters, 2 categories). Raters agreed on 75.0% ...
Krippendorff's alpha does not have that problem ... prefer rater_agreement.
```

It steers the reader elsewhere, which is good, but it never states that the
number it just printed has no sampling error attached. Against the README's
"Never report a bare p-value. Always an effect size and an interval", this is
the one place the package hands over a naked estimate.

Confidence: certain. Fix: one sentence in `summary()`, or a bootstrap interval.
An hour for the sentence.

---

### 14. Public API surface: things that will read wrong once people depend on them

All confirmed by inspection of the signatures.

- **`position_bias(comparisons, seed=None)`**. `seed` is documented as unused
  ("Accepted so the signature matches the rest of the package. Nothing here is
  resampled."). Honest, and still a parameter that does nothing.
- **`position_bias` and `length_bias` take no `confidence`** while every other
  public function does, and both return dataclasses carrying a `confidence`
  field. The interval level is hard-coded twice (`judge.py:498` `_wilson(k, n,
  0.95)`, `judge.py` `norm.ppf(0.975)`). A caller who wants a 90% or 99%
  interval cannot have one, and a `PositionBias` constructed with
  `confidence=0.99` would print "99% CI" over a 95% interval.
- **`audit(comparisons=...)` and `bradley_terry(comparisons=...)` take
  incompatible frames** under the same parameter name: `pair_id/option_a/
  option_b/winner` versus `item_id/model_a/model_b/winner`. Passing a
  leaderboard frame to `audit` raises a message that names the expected columns
  but never mentions `bradley_terry`.
- **`detectable_effect(n, baseline=0.5, ...)`**. The second positional argument
  does nothing in the default (`paired=True`) mode, and it sits where a reader
  who saw `detectable_effect(500, discordance_rate=12/500)` in the README would
  naturally put the discordance rate. `detectable_effect(500, 0.3)` silently
  means "baseline 0.3, discordance assumed".

Confidence: certain. Fix: add `confidence` to the two judge functions, rename
`audit(comparisons=)` to `judgements=` (or make the error message point at
`bradley_terry`), and make `baseline` keyword-only. Half a day, and the window
for it is closing since 0.1.0 and 0.2.0 are both on PyPI.

---

### 15. Two commit messages do not describe their contents

- `d38c7a3` **"point at the new account, bump to 0.2.0"**. `pyproject.toml` at
  that commit still says `version = "0.1.0"`. The bump landed later, in the
  un-pushed local commit `242e354` ("Version Update"). `origin/main` points at
  `d38c7a3`, so the message is wrong on the published branch.
- `a488aa4` **"add audit module: the report the other five modules exist to
  produce"** contains a single change, deleting `CLAUDE.md`. The commit that
  actually adds the audit module is `0500f53`, which carries the identical
  message.

Nothing secret is committed. I regex-scanned the full history for keys, tokens,
passwords and private-key headers and found none.

Confidence: certain. Fix: nothing to do on published history; worth noting.

---

### 16. Smaller documentation drift

- **README line 257: "Every interval here is tested by simulation."** Not true
  of `score_ci(method="t")`, `compare_independent(method="t")`,
  `compare_independent(method="score")`, `compare_independent(method="bootstrap")`,
  the Wilson interval in `position_bias`, or the Wald interval in `length_bias`.
  Four of those six are validated some other way (exhaustive test/interval
  agreement, or a statsmodels reference), which is arguably better than
  simulation. The sentence as written is still checkable and false. Six of the
  package's intervals do have real simulation coverage tests.
- **README line 139: "It runs five of these checks"** against
  `evalaudit/audit.py:9` "Seven checks run". Both are defensible (five modules,
  seven checks) and they are eleven files apart saying different numbers.
- **`evalaudit/power.py:130`**, docstring example:
  `>>> detectable_effect(220, discordance_rate=0.3).difference` → `0.10289...`.
  The actual value is `0.10290198155585326`. As a doctest with ellipsis it would
  fail. Doctests are not run: `pyproject.toml` sets `testpaths = ["tests"]` with
  no `--doctest-modules`, so the four `Examples` blocks in `scores.py`,
  `compare.py` and `power.py` are never executed. The other three are accurate.
- **`PowerResult._reach` prints half an item** on an odd independent n:
  `detectable_effect(101, paired=False)` → "This eval ran 101 items, 50.5 in
  each group."
- **`pytest-cov` is declared in `[dev]` and never used.** No `--cov` flag in
  `pyproject.toml` or the workflow, no coverage step in CI.

Confidence: certain, all verified. Fix: an hour for the lot.

---

## Suspected, not proven

### A. `_fit` in `pairwise.py` has no convergence check on the primary fit

`evalaudit/pairwise.py`. Newton runs to `_MAX_NEWTON = 100` and breaks on the
gradient tolerance. If it never reaches tolerance it returns whatever it reached,
with no flag and no note. The docstring justifies the cap for resamples ("so a
resample that is nearly separable cannot spin") but the same code path fits the
full data.

Mitigating: Ford's condition is checked before the fit, so a finite maximum is
guaranteed to exist, and the regularised Hessian is well conditioned. I could
not construct a case that fails to converge in 100 steps. Newton without a line
search is not globally convergent on this likelihood in general, so I cannot
rule it out either. `judge.py:_fit_logistic` handles the same risk properly. It
returns `_NOTE_NO_CONVERGENCE` on the `for/else`. `pairwise._fit` does not.

Confidence: low that it bites; certain that the asymmetry with `judge.py` is
real. Fix: a `for/else` and a `converged` flag on `BTResult`. Half a day.

### B. The bootstrap in `pairwise` resamples comparisons, not items

Documented honestly in the docstring ("on a design where one prompt carries many
comparisons the interval will run slightly narrow"). I did not measure how
narrow, and the coverage test uses one comparison per row so it cannot show it.
A leaderboard where 20 model pairs are judged on the same 100 prompts is the
normal shape and would be understated. Whether the understatement is 1% or 20%
of the width, I do not know.

### C. The coverage bands are wider than they need to be

`test_bootstrap_coverage` in `test_scores.py` uses 400 trials and asserts
`0.91 <= rate <= 0.99`. Four standard errors either side. An interval built at
98.5% nominal instead of 95% would pass. `test_wilson_coverage` (1000 trials,
0.92 to 0.98) is about three standard errors and is tighter than it needs to be to
catch the mutations I tried. I killed both Wilson mutations easily. I did not
find a plausible implementation error that squeaks through any of these bands,
so I am listing this as a smell rather than a finding. The `pairwise` and
`compare` bands come with arithmetic in their docstrings justifying the width,
which is the right way to do it; `test_scores.py` does not.

---

## What came back clean

I looked hard at these and found nothing. Saying so is part of the report.

**Krippendorff's alpha (`agreement.py`), the strongest part of the package.**
Checked three independent ways: against the `krippendorff` package on PyPI
across fifteen parametrised settings including gapped domains, continuous
ratings and 50%-missing designs; against Krippendorff's own published worked
examples (the four-observer 0.743/0.815/0.849 and three-observer
0.691358/0.810845 figures both reproduce); and against a slow reference written
from the definition inside the test file, which is itself validated against the
reference package first. I re-derived the nominal, interval and ordinal
coincidence formulas by hand and they match the code, including the ordinal
metric's half-of-each-end term. Every mutation I tried was killed: `n` for
`n-1`, dropping the ordinal end term, `sizes` for `sizes-1`, the usable-share
floor, the two-item floor, the dropout sort direction, the item-table
normalisation. The `_MIN_USABLE_SHARE` refusal and the leave-one-out noise
guard are both tested in both directions.

**The two score intervals are formula-correct, and I verified them the hard
way.** For Tango's paired interval I solved the constrained multinomial
likelihood numerically and compared against the quadratic root the code uses.
They agree to 3e-9 across eight tables. For the independent interval I did the
same for the Farrington-Manning cubic:

```
 case               true (q1,q2)         evalaudit            statsmodels
 45/50 30/50 d=+0.45 (0.930774,0.480774) (0.930774,0.480774) (0.858852,0.408852)
  9/20  3/20 d=+0.40 (0.520234,0.120234) (0.520234,0.120234) (0.488953,0.088953)
```

evalaudit matches the true constrained MLE exactly on every case I tried;
`statsmodels.score_test_proportions_2indep` does not. I also confirmed the
δ=0 collapse in both cases. Tango's statistic reduces to `(b-c)/sqrt(b+c)` and
the independent one to the pooled two-proportion z, so the stated "the interval
inverts the test" property holds in `compare` as claimed.

**Wilcoxon.** `_signed_rank_counts` is a correct subset-sum DP; `_signed_rank_trim`
reproduces the published two-sided α=0.05 critical value table for n=6..18; the
tie-corrected variance matches scipy's; the Hodges-Lehmann interval indexing is
right. The refusal below three distinct non-zero differences, with the 41%
contradiction rate quoted in the error message, is a genuinely good piece of
design. Mutations to the trim and the tie correction were killed.

**`power.py`.** I re-derived the quadratic in `_paired_difference` from
`_power_paired` and it matches the docstring and the code, discriminant
included. `_paired_size` is Connor's published form. `_independent_size` is the
standard pooled two-proportion formula. Every mutation I tried was killed:
dropping the z_β term, dropping z_β² from the denominator, one-sided alpha,
unpooled null variance, changing the default discordance rate, the per-group
doubling. The roundtrip between the two entry points is tested from both
directions.

**`pairwise.py`.** Ford's condition, the disconnected-groups check, the
undefeated/winless refusal, the Elo affine transform, the reference-shifted
intervals, the tie policies, the resample usability filter and the coverage
simulation are all tested with teeth. Mutations killed: Newton step cap, Newton
tolerance, `log(10)` → `log(2)`, tie credit 0.5 → 0.4, dropping the reference
shift, dropping the strong-connectivity check, the usable-share floor. The
`ties="davidson"` `NotImplementedError` is a good call.

**`test_readme.py` is a real test.** It executes every python block that has an
output block, splits the shown output on `[...]`, and requires each fragment to
appear verbatim with whitespace collapsed. It guards its own regex with a block
count. I re-ran the pairwise README example independently and confirmed the
claim "every one of those five is a pair containing the weakest model". All
five separable pairs are `mN` vs `m6`.

**`examples/interval_method_study.py` reproduces its recorded output exactly**,
to the last digit of all four tables, including the coverage figures its
docstring quotes.

**Packaging and CI.** Dependencies are `numpy`, `scipy`, `pandas`, all three
imported, nothing else imported that is not stdlib, nothing declared and unused
except `pytest-cov`. `requires-python = ">=3.9"` is honest: every module has
`from __future__ import annotations`, there is no `match`, no PEP 604 union in a
runtime position. The `choix>=0.4; python_version>='3.10'` marker with the
documented fallback is a careful piece of work. GitHub Actions shows the
`tests` workflow green on `main` across 3.9/3.11/3.12 for the last six runs, so
the 3.9 job really does resolve `krippendorff` (0.8.1 is the last release
supporting 3.9) and really does run. The published sdist contains exactly what
it should: source, tests, examples, README, LICENSE, workflow. No stray data
files, no notebooks, no secrets.

**Prose style.** No em-dashes or en-dashes anywhere in `evalaudit/` or the
README. No "not X, but Y" constructions in the shipped summaries. No leaked
`repr` output in any summary I generated. `!r` is used deliberately and only on
slice labels, which is the right call. The `_escape` HTML path is correct.

**Design invariants.** All thirteen return types are `@dataclass(frozen=True)`.
All of them have `summary()` except `AuditConfig`, which is a config rather than
a result. No public function returns a bare p-value; every p-value reported has
an effect size and (except for `KappaResult`, finding 13) an interval beside it.

---

## What I could not check

1. **Tango (1998) and Mee (1984) against the actual papers.** I do not have
   them. I verified both implementations against a brute-force numerical
   maximisation of the same constrained likelihood, which establishes the
   arithmetic is right, and I verified the δ=0 collapse the docstrings claim.
   What I cannot confirm is that the citation is the right attribution for the
   construction, or that the docstring's claim about
   Miettinen-Nurminen, "Measured over 9213 tables the correction produced 4
   disagreements where this version produces none ... about 0.3 to 1 percent
   wider", reproduces. That measurement is not in the repository, and nothing
   in `examples/` or `tests/` regenerates it. The comparable claim in
   `compare.py`'s Wilcoxon docstring ("Measured over 600 binary samples it
   reported a bound of exactly zero while the p-value rejected on 41% of them")
   is in the same position. `examples/interval_method_study.py` is the model for
   how to make a claim like that checkable, and neither of these two follows it.

2. **Connor (1987) against the paper.** Same situation. The formula matches the
   version reproduced in every secondary source I know, and the algebra of the
   inverse is self-consistent, but I did not read Connor.

3. **The power docstring's simulation claim.** "Measured over 200,000 simulated
   evals the shipped test delivers about 0.784 where this formula promises
   0.800." Not reproduced in the repository and not cheap to rerun. The
   `_floor_sentence()` that reaches the client rests on it.

4. **The `pairwise` bootstrap's behaviour on clustered designs** (suspicion B).
   Measuring it properly needs a simulation with repeated prompts, which I did
   not build.

5. **Real client data.** Every judgement about the prose is on synthetic
   fixtures. I have not seen a real ratings frame go through `audit`.

6. **Whether the `promptstats` link in the README's Related section still
   describes that package accurately.** I did not fetch it.

---

## The three I would fix first

**1. Finding 2, the length-bias direction bug.** It is the only item on this
list that is producing a wrong sentence in a deliverable *right now*, on data a
real client could plausibly have. A judge that prefers terse answers is not
exotic; short-answer preference is a documented LLM-judge failure mode. The
report calls it the opposite, three times, in one paragraph, with its own
numbers sitting next to the contradiction. Everything else here is either a
latent regression risk or a metadata problem. This one is the failure a
sceptical reader actually catches, and catching it costs them nothing. They
just read the paragraph. Fix the sign branch, and add the short-preferring
fixture the suite has never had.

**2. Finding 1, the untestable position-bias verdict.** Not because the code is
wrong (it is not) but because of what the hole says about the rest. The both-
orders branch is the single most distinctive check in the package, the reason
someone would choose `evalaudit` over `scipy`, and the assertion guarding it is
matching the word "first" in a sentence that always contains it. The same
pattern is what let finding 5 hide for three commits. I would fix this one and
then re-read every `assert <string> in summary` in the suite with the same
suspicion. There are enough of them that at least one more is probably
decorative. Two hours of work that changes what the test suite is worth.

**3. Finding 4, the version.** Trivially small and disproportionately
expensive. The pitch is "the numbers hold up when someone checks them", and the
first check anyone runs when reproducing a consulting result is
`evalaudit.__version__`. It answers `0.1.0` from a package that says `0.2.0` on
its label. There is no argument to be had about severity, no statistics to
discuss, and it takes twenty minutes. Read it from `importlib.metadata` and add
the one test that would have caught it.

Finding 3 (the Wilson/exact mismatch) is the next one after these, and it is a
close fourth. It is the only finding where the shipped arithmetic is
genuinely inconsistent with the package's own stated rule rather than merely
untested. I left it out of the three because it needs a design decision
(Clopper-Pearson or a score p-value) rather than a fix, and because the tables
where the two disagree are concentrated at n below 145.
