# Changelog

Every change to evalaudit that a caller can see is recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
version numbers follow [semantic versioning](https://semver.org/). Before 1.0
a minor version can break things, and 0.3.0 does.

An entry marked **Breaking** changes what a call that worked on the previous
release returns, raises or prints. The mark goes on the entry wherever it
sits, under Changed, Fixed or Removed.

0.3.0 is the first release with a tag. The entries below it were
reconstructed after the fact. Their dates are PyPI's upload dates, and their
contents were checked against the published wheels and the commit each wheel
matches.

## [Unreleased]

## [0.3.0] - 2026-09-10

0.3.0 changes what `bradley_terry` reports on data you have already run, and
it turns several calls that used to return numbers into errors. Each
breaking entry says what it was, what it is now and what to do about it.

### Changed

- **Breaking.** `bradley_terry` reads separability off the interval on the
  gap between two ratings. Before, a pair was separable when its two rating
  intervals did not overlap. Now a pair is separable when the percentile
  interval on the difference between the two ratings excludes zero. Two
  rating intervals can overlap while the gap between them is well measured,
  because both ratings carry the error of the field average they are
  measured against and that error cancels in the gap. So the old test
  undercounted the pairs the data orders, and the pairs it left out were
  never shown to be unordered. There is no option to bring the old rule
  back. `n_separable` and `separable_pairs` can change on data you have
  already run. On the README example they go from 5 of 15 pairs to 7. On
  twelve simulated leaderboards where every item carried one comparison,
  the count went up on five and stayed the same on seven. Nothing changes
  in your code. Rerun, and replace any separability count you have
  published.

- **Breaking.** `separable_pairs` has two new columns, `ci_low` and
  `ci_high`, the interval on `difference`. They sit between `difference` and
  `p_a_beats_b`, so code that reads columns by position will read the wrong
  ones. Code that selects columns by name is unaffected.

- **Breaking.** `bradley_terry` resamples items by default. Before, the
  bootstrap drew single comparisons, which treats judgements of the same
  prompt as independent. Now it draws item ids with replacement and takes
  every comparison made on each drawn item.

  Where every item carries one comparison the default draws the same
  resamples from the same seed as 0.2.1, and the ratings, their intervals
  and the win matrix come out identical. Separability can still change,
  for the reason in the entry above.

  Where items carry several comparisons the intervals widen and fewer pairs
  may separate. On a simulated board of 60 items with 20 comparisons each,
  the rating intervals came out 1.1 to 1.5 times as wide as 0.2.1's. On
  simulated data shaped like MT-Bench's human judgements, the rating
  intervals 0.2.1 reported covered 88% of the time at a nominal 95% once
  prompts shifted model strength by a spread of 0.5 on the log-odds scale,
  and 79% at a spread of 1.0. `examples/pairwise_cluster_study.py` has the
  table for 0.3.0.

  To get the ratings and rating intervals an earlier version gave on the
  same data and seed, pass `resample="comparisons"`. They come out
  identical to 0.2.1's. `separable_pairs`, `n_separable` and the summary
  follow the 0.3.0 rules, so those can differ.

- **Breaking.** `bradley_terry` raises `ValueError` in two cases that 0.2.1
  accepted. Both apply only when `n_boot` is above zero and `resample` is
  left at its default of `"items"`.

  - Some row has no `item_id`. The item bootstrap has to know which item
    each comparison belongs to.
  - Every row has the same `item_id`, an empty string included. With one
    item, every resample is the full data and every interval would have
    width zero.

  This is the change most likely to break a working script, so here is what
  to do. Earlier versions required `item_id` and read it for nothing but a
  count, so a constant, a blank or the row number got past the check and
  changed nothing. Look at what yours holds.

  - It names the prompt each comparison was made on. Fill in any gaps and
    change nothing else. Your intervals now account for comparisons sharing
    a prompt, which is the reason for the change.
  - It is the row number, or anything else unique per row. Nothing raises,
    and the ratings and their intervals come out as they did in 0.2.1.
  - It is a constant or a blank. Record the prompt if you can. If you
    cannot, pass `resample="comparisons"` to keep the intervals you had.
    They come out identical to 0.2.1's on the same seed. They treat every
    judgement as independent and run narrow wherever prompts carry several.
    The summary says so when the column holds a constant or an empty
    string. It says nothing when every `item_id` is missing, since then
    nothing records which comparisons share a prompt.

  With `n_boot=0` no resampling happens and any `item_id` is accepted.

- **Breaking.** `score_ci(method="t")` raises `ValueError` on three kinds of
  input it used to return numbers for. What to do depends on the input.

  - 0/1 data, including all passes, all fails and a single pass or fail.
    The interval could run past 100%, as 49 passes in 50 did at 94.0% to
    102.0%, come back with width zero on all passes or all fails, or come
    back as NaN on a single pass. Use `method="wilson"`, which
    `method="auto"` already picks for 0/1 data and which works on a single
    item.
  - A single score that is not 0 or 1. It came back as NaN bounds behind two
    RuntimeWarnings. One score has no spread, and the bootstrap returns
    width zero on it too. Score more items.
  - Continuous scores that are all the same. The interval came back with
    width zero, which claims the observations fixed the mean exactly. Check
    that the scorer is not returning a constant.

- **Breaking.** `score_ci(method="wilson")` raises `ValueError` on scores
  that are not 0 or 1. Wilson counts passes, and on other scores it took the
  count by truncating the sum of the scores. Ten scores of 0.99 printed a
  mean of 0.990 beside an interval of 0.596 to 0.982. For scores that vary,
  use `method="bootstrap"`, the interval on a mean, which `method="auto"`
  already picks. Scores that are all the same, like that example, get width
  zero from the bootstrap too, and 0.3.0 does not refuse that.

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
  - `AgreementResult.summary()` for a scale nobody varied now reads "This
    comes from a scale nobody varied. It is not perfect agreement." where it
    read "That is not perfect agreement, it is a scale nobody varied."
  - `JudgeValidation.summary()` for a rubric with one label now reads "This
    comes from a rubric with one label in it. It is not perfect agreement."
    where it read "That is not perfect agreement, it is a rubric with one
    label in it."

- The Development Status classifier moved from Alpha to Beta.

### Added

- `bradley_terry(..., resample="items")`, with `"comparisons"` as the other
  choice.
- `BTResult.pairs`, one row per pair of models with the gap, the interval on
  the gap and a `separable` column. `separable_pairs` is its separable rows.
- `BTResult.resample`, recording which unit the bootstrap drew. It and
  `pairs` are the last two fields of the dataclass, so building a
  `BTResult` by position with 0.2.1's fields still works.
- `to_elo` stretches the gap intervals with the gaps and carries `pairs` and
  `resample` across.
- `examples/pairwise_cluster_study.py`, in the source distribution, which
  measures how narrow the comparison bootstrap runs on MT-Bench shaped data.

### Fixed

- The `scores` module docstring said the interval at n=50 is roughly plus or
  minus 10 points. It is about 10 at an 85% pass rate and about 13 at 50%.
- The README said every interval in the package is tested by simulation.
  Several were not. It now says which intervals are tested by simulation,
  which are checked against a reference implementation instead, and which
  one is checked by neither.
- The README said `audit` runs five checks. It runs seven, drawn from five
  modules.

### Removed

- `AUDIT.md`, an internal review of the package, is no longer tracked, so
  it is not in the source distribution.

## [0.2.1] - 2026-09-08

Reconstructed after the fact. The published files match commit `809a650`.

### Changed

- `__version__` is read from the installed package metadata. 0.0.1 and
  0.2.0 both reported `__version__` as 0.1.0, because a literal in
  `__init__.py` had drifted from `pyproject.toml`.

### Added

- `AUDIT.md`, an internal review of the package, in the source
  distribution.

### Fixed

- **Breaking.** The length-bias check read every coefficient as a
  preference for long answers. In 0.2.0 a judge that preferred shorter
  answers got the `audit` title "The judge is pulled by how long the answer
  is", and `LengthBias.summary()` printed "A positive coefficient here is
  not bias on its own" beside a negative coefficient. The `audit` finding
  now takes its title, lead, action and verdict from the fit that decides,
  with its own lead for when the two fits point opposite ways, and
  `LengthBias.summary()` has its own sentence for a negative coefficient.
  Anything matching on the old title or sentence will stop matching.

### Removed

- Nothing.

## [0.2.0] - 2026-09-07

Reconstructed after the fact. The wheel matches commit `242e354`. It was
uploaded twelve seconds before that commit was made, so it was built from
the working tree the commit then recorded.

### Changed

- The project URLs point at github.com/mokayahezron/evalaudit. 0.1.0 pointed
  at github.com/ondibahezron-glitch/evalaudit.
- `choix` is a development dependency on Python 3.10 and later, used to
  check the Bradley-Terry fit.

### Added

- The `pairwise` module, with `bradley_terry`, `to_elo` and `BTResult`.
  Ratings come with bootstrap intervals and a count of separable pairs.
  Models that fall into groups that never met, and models that never lost
  or never won, get no ratings and a summary naming them. Ties are split or
  dropped, and `ties="davidson"` is refused by name.
- The README code blocks are run by the test suite and checked against the
  output they show.

### Fixed

- Nothing.

### Removed

- Nothing.

## [0.1.0] - 2026-09-07

Reconstructed after the fact. The wheel matches commit `c1230f1`.

### Changed

- Nothing.

### Added

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
- The result types are exported from the package, `ScoreCI` among them.

### Fixed

- Nothing.

### Removed

- Nothing.

## [0.0.1] - 2026-09-04

Reconstructed after the fact. The wheel's package files match commits
`aa8ad97`, `8a4cab0` and `b090c66`, which do not differ in them. Its version
number first appears in `pyproject.toml` at `b090c66`, committed about an
hour after the upload. It reported `__version__` as 0.1.0.

### Changed

- Nothing.

### Added

- `score_ci`, with Wilson, bootstrap and Student-t intervals, and a default
  that picks Wilson for 0/1 data and the bootstrap otherwise.

### Fixed

- Nothing.

### Removed

- Nothing.

[Unreleased]: https://github.com/mokayahezron/evalaudit/compare/v0.3.0...main
[0.3.0]: https://github.com/mokayahezron/evalaudit/compare/809a650...v0.3.0
[0.2.1]: https://github.com/mokayahezron/evalaudit/compare/242e354...809a650
[0.2.0]: https://github.com/mokayahezron/evalaudit/compare/c1230f1...242e354
[0.1.0]: https://github.com/mokayahezron/evalaudit/compare/b090c66...c1230f1
[0.0.1]: https://github.com/mokayahezron/evalaudit/tree/b090c66
