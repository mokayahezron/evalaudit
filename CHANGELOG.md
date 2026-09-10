# Changelog

Every change to evalaudit that a caller can see is recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
version numbers follow [semantic versioning](https://semver.org/). Before 1.0
a minor version can break things, and 0.3.0 does.

The entries for 0.2.1 and earlier were reconstructed from the git history
after the fact. There were no tags, so they record what the commits say and
not what was published on a given day.

## [0.3.0] - Unreleased

0.3.0 changes what `bradley_terry` reports on data you have already run, and
it turns several calls that used to return numbers into errors. Every
breaking change is under Changed, marked **Breaking**, with what to do about
it.

### Changed

- **Breaking.** `bradley_terry` reads separability off the interval on the
  gap between two ratings. Before, a pair was separable when its two rating
  intervals did not overlap. Now a pair is separable when the percentile
  interval on the difference between the two ratings excludes zero. Two
  rating intervals can overlap while the gap between them is well measured,
  because both ratings carry the error of the field average they are
  measured against and that error cancels in the gap. So the old test
  undercounted the pairs the data orders, and the pairs it left out were
  never shown to be unordered. `n_separable` and `separable_pairs` change on
  existing data, usually upward. On the README example they go from 5 of 15
  pairs to 7. Nothing changes in your code. Rerun, and replace any
  separability count you have published.

- **Breaking.** `separable_pairs` has two new columns, `ci_low` and
  `ci_high`, the interval on `difference`. They sit between `difference` and
  `p_a_beats_b`, so code that reads columns by position will read the wrong
  ones. Code that selects columns by name is unaffected.

- **Breaking.** `bradley_terry` resamples items by default. Before, the
  bootstrap drew single comparisons, which treats judgements of the same
  prompt as independent. Now it draws item ids with replacement and takes
  every comparison made on each drawn item. Where every item carries one
  comparison the default draws the same resamples from the same seed as
  before, and nothing moves. Where items carry several comparisons the
  intervals widen and fewer pairs may separate. On simulated data shaped
  like MT-Bench's human judgements, the old bootstrap covered 88% of the
  time at a nominal 95% once prompts shifted model strength by a spread of
  0.5 on the log-odds scale, and 80% at a spread of 1.0.
  `examples/pairwise_cluster_study.py` has the full table. To reproduce a
  result from an earlier version exactly, pass `resample="comparisons"`.

- **Breaking.** `bradley_terry` raises `ValueError` in two cases that 0.2.1
  accepted. Both apply only when `n_boot` is above zero and `resample` is
  left at its default of `"items"`.

  - Some row has no `item_id`. The item bootstrap has to know which item
    each comparison belongs to.
  - Every row has the same `item_id`. With one item, every resample is the
    full data and every interval would have width zero.

  This is the change most likely to break a working script, so here is what
  to do. Earlier versions required `item_id` and read it for nothing but a
  count, so a constant, a blank or the row number got past the check and
  changed nothing. Look at what yours holds.

  - It names the prompt each comparison was made on. Fill in any gaps and
    change nothing else. Your intervals now account for comparisons sharing
    a prompt, which is the reason for the change.
  - It is the row number, or anything else unique per row. Nothing raises
    and nothing moves.
  - It is a constant or a blank. Record the prompt if you can. If you
    cannot, pass `resample="comparisons"` to keep the intervals you had.
    They treat every judgement as independent and run narrow wherever
    prompts carry several, and the summary says so.

  With `n_boot=0` no resampling happens and any `item_id` is accepted.

- **Breaking.** `score_ci(method="t")` raises `ValueError` on three inputs
  it used to return numbers for. What to do depends on the input.

  - 0/1 data, including all passes and all fails. The interval could run
    past 0% or 100%, or come back with width zero. Use `method="wilson"`,
    which `method="auto"` already picks for 0/1 data.
  - A single score. It came back as NaN bounds behind two RuntimeWarnings.
    One score has no spread, and no method here can put an interval around
    it. Score more items.
  - Continuous scores that are all the same. The interval came back with
    width zero, which claims the observations fixed the mean exactly. Check
    that the scorer is not returning a constant.

- **Breaking.** `score_ci(method="wilson")` raises `ValueError` on scores
  that are not 0 or 1. Wilson counts passes, and on other scores it took the
  count by truncating the sum of the scores. Ten scores of 0.99 printed a
  mean of 0.990 beside an interval of 0.596 to 0.982. Use
  `method="bootstrap"` for the interval on a mean, which `method="auto"`
  already picks.

- **Breaking.** `score_ci`, `compare_paired` and `compare_independent`
  raise `ValueError` for a confidence level that is not strictly between 0
  and 1, with the message `confidence must be in (0, 1), got ...` that
  `bradley_terry`, `rater_agreement`, `judge_validation` and `audit` already
  used. Before, what came back at `confidence=1.5` depended on the method.
  The bootstrap in all three raised from inside numpy. Wilson and both t
  intervals returned NaN bounds. The two score intervals returned bounds
  clipped to -100% and 100%. The signed-rank interval in `compare_paired`
  returned an ordinary-looking interval under a "150% CI" label, which is
  the case nothing would have flagged. Pass a level strictly between 0 and
  1.

- **Breaking.** Several summary strings are reworded, so any test or
  pipeline that matches on them will fail.

  - `BTResult.summary()` defines separable as "the interval on the gap
    between the two ratings excludes zero". For pairs that do not separate
    it says the data does not establish an order for them, that this does
    not mean the models are level, and that more comparisons could separate
    them. The phrases "meaning their intervals do not overlap", "cannot be
    ordered", "an order the data does not carry" and "a ranking of noise"
    are gone.
  - `BTResult.summary()` adds a sentence when `resample="comparisons"` was
    used on data where items carry several comparisons.
  - `AgreementResult.summary()` for a scale nobody varied now ends "This
    comes from a scale nobody varied. It is not perfect agreement." The old
    ending was "That is not perfect agreement, it is a scale nobody varied."
  - `JudgeValidation.summary()` for a rubric with one label now reads "This
    comes from a rubric with one label in it. It is not perfect agreement."
    The old sentence was "That is not perfect agreement, it is a rubric with
    one label in it."

- The Development Status classifier moved from Alpha to Beta.

### Added

- `bradley_terry(..., resample="items")`, with `"comparisons"` as the other
  choice.
- `BTResult.pairs`, one row per pair of models with the gap, the interval on
  the gap and a `separable` column. `separable_pairs` is its separable rows.
- `BTResult.resample`, recording which unit the bootstrap drew. It and
  `pairs` are the last two fields of the dataclass, so building a
  `BTResult` by position still works.
- `to_elo` stretches the gap intervals with the gaps and carries `pairs` and
  `resample` across.
- `examples/pairwise_cluster_study.py`, which measures how narrow the
  comparison bootstrap runs on MT-Bench shaped data.

### Fixed

- Four inputs to `score_ci` and the confidence level in three functions used
  to produce impossible or self-contradicting numbers. They now raise, and
  they are listed under Changed because each one turns a call that returned
  into an error.
- The `scores` module docstring said the interval at n=50 is roughly plus or
  minus 10 points. It is about 10 at an 85% pass rate and about 13 at 50%.
- The README Coverage section said the Student-t interval in `score_ci` and
  Welch's interval in `compare_independent` had nothing pinning their bounds.
  Tests matching them against scipy and statsmodels had already landed. It
  now counts fifteen intervals, nine tested for coverage, five pinned
  against a reference and one with nothing pinning its bounds.
- The README corrected two earlier claims, about simulation coverage and
  about how many checks `audit` runs.

### Removed

- The non-overlap definition of separability. There is no option to bring
  it back, because the summary built on it made a claim its own docstring
  denied.
- `AUDIT.md`, an internal review of the package, is no longer tracked, so
  it is not in the source distribution.

## [0.2.1] - 2026-09-08

Reconstructed from the git history after the fact.

### Added

- Nothing.

### Changed

- `__version__` is read from the installed package metadata. 0.2.0 reported
  `__version__` as 0.1.0, because a literal in `__init__.py` had drifted
  from `pyproject.toml`.

### Fixed

- The length-bias check reported a preference for long answers whatever
  the sign of the fit. The `audit` finding for length now takes its title,
  lead, action and verdict from the fit that decides, with its own lead for
  when the two fits point opposite ways.

### Removed

- Nothing.

## [0.2.0] - 2026-09-07

Reconstructed from the git history after the fact.

### Added

- The `pairwise` module, with `bradley_terry`, `to_elo` and `BTResult`.
  Ratings come with bootstrap intervals and a count of separable pairs.
  Models that fall into groups that never met, and models that never lost
  or never won, get no ratings and a summary naming them. Ties are split or
  dropped, and `ties="davidson"` is refused by name.
- The README code blocks are run by the test suite and checked against the
  output they show.

### Changed

- The project URLs point at github.com/mokayahezron/evalaudit.
- `choix` is a development dependency on Python 3.10 and later, used to
  check the Bradley-Terry fit.

### Fixed

- Nothing recorded.

### Removed

- Nothing.

## [0.1.x] - 2026-09-04 to 2026-09-07

Reconstructed from the git history after the fact. `pyproject.toml` read
0.1.0 at the first commit, 0.0.1 from the third, and 0.1.0 again from a
README update made after the audit module landed. Which of these were
published is not recorded.

### Added

- `score_ci`, with Wilson, bootstrap and Student-t intervals.
- `compare_paired` and `compare_independent`. McNemar with Tango's score
  interval for paired binary data, the paired bootstrap, and the signed-rank
  test with the Hodges-Lehmann interval. The score interval, the bootstrap
  and Welch's t for independent samples.
- `rater_agreement`, Krippendorff's alpha with a bootstrap interval, an item
  disagreement table and a rater dropout table. `cohens_kappa` and
  `fleiss_kappa`.
- `judge_validation`, `position_bias` and `length_bias`.
- `detectable_effect` and `min_sample_size`.
- `audit`, which runs every check the supplied data supports and ranks the
  findings by severity.

### Changed

- Nothing recorded.

### Fixed

- Nothing recorded.

### Removed

- Nothing.
