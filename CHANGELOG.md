# Changelog

Every change to evalaudit that a caller can see is recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
version numbers follow [semantic versioning](https://semver.org/). Before 1.0
a minor version can break things, and 0.3.0 and 0.4.0 both do.

An entry marked **Breaking** changes what a call that worked on the previous
release returns, raises or prints. The mark goes on the entry wherever it
sits, under Changed, Fixed or Removed.

Every release is dated by the day PyPI received its upload, in UTC. 0.3.0 is
the first release with a tag, and 0.4.0 has one too. The entries below 0.3.0
were reconstructed after the fact, and their contents were checked against
the published wheels and the commit each wheel matches.

## [Unreleased]

## [0.4.0] - 2026-09-11

Nearly everything in 0.4.0 changes what a
summary or a report prints. One rule is behind it. A result that does not
clear its threshold has failed to show something, and it has not shown the
opposite. Summaries that described such a result as if the opposite were
shown, and verdicts decided on a point estimate where an interval existed,
now say what the data did and did not show. Several `audit` findings change
severity on data you have already run. Each breaking entry says what it
was, what it is now and what to do about it. Every entry was checked by
running the same calls against 0.3.0 from PyPI and against the code
released as 0.4.0.

### Changed

- **Breaking.** `audit` reads rater agreement and judge agreement off the
  interval on alpha. An interval wholly above the threshold keeps "Rater
  agreement holds up" or "The judge tracks the humans", and one wholly below
  it keeps the "below the working threshold" title. An interval running both
  sides of the threshold, or no interval at all, is now a `warning` titled
  "The data cannot show that rater agreement clears the working threshold"
  or "The data cannot show that the judge clears the working threshold".
  0.3.0 decided on the point estimate. Ratings with alpha 0.730 and an
  interval of 0.466 to 0.933 came back `info` as "Rater agreement holds up",
  and so did any ratings audited with `n_boot=0`. Judge labels with alpha
  0.581 and an interval of 0.467 to 0.691 came back "Judge-human agreement
  is below the working threshold". Rerun any audit you have sent. A finding
  that moved from `info` to `warning` also changes the report's verdict.

- **Breaking.** Without an `effect_of_interest`, `audit` no longer sizes
  power on the margin it measured. 0.3.0 made the observed margin the effect
  at issue, as in "The effect at issue is 5.0 points, the margin this eval
  reports." Power computed on an observed effect is a monotone function of
  the p-value, so on a margin whose interval includes zero the finding came
  out `critical` almost every time and repeated the comparison. The finding
  now reports the design's reach, the smallest difference the eval could
  find at the configured power, and makes no claim about the margin. It is
  a `warning` titled "No effect of interest was stated, so this reports the
  eval's reach", because without a stated effect the audit does not know
  what size of difference matters. Its lead says "This reports what the
  design could find and makes no claim about the margin it measured." On 400
  paired items with a 5.0 point margin and an interval of 1.1% to 9.0%,
  0.3.0 opened the report with a `critical` power finding and the verdict
  said the conclusion did not hold. Two systems with a margin of exactly
  zero, which 0.3.0 skipped, get the same warning.

  With one system and no stated effect the check is still skipped, and its
  reason no longer offers the observed margin as a stand-in. It read
  "Nothing to size the eval against. Pass effect_of_interest in the config,
  or scores for two systems so the observed margin can stand in for it."
  and now reads "No effect of interest was stated and there is no comparison
  whose reach could be reported. Pass effect_of_interest in the config, or
  scores for two systems."

  An audit of two pass-rate systems with no stated effect now always
  carries this warning, so its report never comes back with no warnings.
  Set `effect_of_interest` to the difference the decision turns on.

- **Breaking.** With an `effect_of_interest` stated, `audit` puts a power
  finding at `info` in two cases where 0.3.0 made it `critical`, because the
  margin's interval already answers the question power asks.

  - The margin's interval clears zero. The finding is titled "The margin is
    established, and a tighter estimate would take more items". 0.3.0
    titled it "This eval could not have detected the effect at issue".
  - The margin's interval includes zero and stops short of the stated
    effect in both directions. The finding is titled "The interval on the
    margin rules out the effect at issue".

  Rerun any audit whose report carried a power finding.

- **Breaking.** With an `effect_of_interest` stated, `audit` rewords the
  power finding on a margin whose interval includes zero. Its title is now
  "This eval had under 80% power for the effect at issue", with the
  percentage taken from `power`, because a design
  short of that power can still find the effect, only less often. Its action
  no longer says "No conclusion about an effect this size can be drawn from
  this data, in either direction. A null result here says the eval was too
  small and says nothing about the systems." It reads the margin's interval
  one side at a time and says one of three things. Either "The interval on
  the margin runs past an effect this size in both directions, so the data
  cannot show an effect this size and cannot rule one out.", or "The
  interval on the margin rules out old ahead by this much, and it cannot
  rule out new ahead by this much." with your system names, or for a single
  system "No margin was measured here, so there is no interval to read
  against an effect this size." Anything matching on the old title or action
  will stop matching.

- **Breaking.** `AuditReport.summary()` rewords its verdict, which is the
  first line of every report.

  - With critical findings, "the conclusion this data is being asked to
    support does not hold as stated." is now "this data cannot support the
    conclusion as stated. That does not mean the conclusion is wrong."
  - With warnings and no criticals, "The result stands and the design
    weakens it, so report it with what the warning say attached." is now
    "The checks that ran did not find a problem that leaves the conclusion
    unsupported. That does not mean the eval is sound. Report the result
    with what the warning says attached."
  - With neither, "All 4 checks that ran came back clean, so the eval holds
    up on everything this report could test." is now "The 3 checks that ran
    did not find a problem in what they could test. That does not mean the
    eval is sound." The number changed too. 0.3.0 counted findings, and the
    score check gives one finding per system, so two systems made three
    checks read as four. A single check reads "The one check that ran", and
    a report where nothing could run reads "No check could run, so this
    report has tested nothing."

  Anything matching on the old verdicts will stop matching. The severity
  counts are unchanged by this entry.

- **Breaking.** `audit` rewords several titles and actions.

  - "No position effect in the pairwise judgements" is now "The data cannot
    show a position effect in the pairwise judgements". Its action opens
    "The data has not cleared the judge of position bias." where it opened
    "Nothing to do here."
  - "No length effect in the judge's choices" is now "The data cannot show
    a length effect in the judge's choices". Its action "Nothing to do
    here." is now "The data has not cleared the judge of rewarding length."
  - An alpha that does not exist is titled "Rater agreement is undefined on
    this data" or "Judge-human agreement is undefined on this data", and the
    lead ends "and this data gives no alpha to hold against it." 0.3.0
    titled it "Rater agreement is below the working threshold" or
    "Judge-human agreement is below the working threshold", and said the
    grades "reproduce less well than the 0.667".
  - The action on a margin that clears zero said "The lower bound is the
    size of the win the data actually supports." When the second system
    leads, the lower bound is the far end of the interval. It now says "The
    end of the interval nearer zero is the smallest margin this data
    supports."
  - The length lead "Two models run here and they point opposite ways." now
    also needs the preference fit's interval to clear zero. 0.3.0 printed it
    off the sign of a preference coefficient whose interval ran over zero,
    and said the judge "picks the longer answer more often" beside an odds
    ratio of 0.94 to 1.40. Such data now gets the plain lead for the
    direction the finding reports.

- **Breaking.** Module summaries put every result that falls short of its
  threshold in one form, "so the data cannot show that" followed by "That
  does not mean".

  - `ComparisonResult.summary()`. "The interval crosses zero, so the data
    cannot confirm that either system is better." is now "The interval
    includes zero, so the data cannot show that either system is better.
    That does not mean the two are level."
  - `PositionBias.summary()`, randomised design. "The interval covers 50%,
    so the data cannot show that position moved the judge." is now "The
    interval includes 50%, so the data cannot show that position moved the
    judge. That does not mean the judge ignores position."
  - `PositionBias.summary()`, both orders. "The flips split evenly across
    the two positions, so this is an unsteady judge rather than a
    position-biased one." is now "That share is not clear of 50% at this
    many flips, so the data cannot show that the flips have a direction.
    That does not mean the judge is free of position bias." 0.3.0 printed
    the old sentence at 7 flips of 9 toward the answer shown first.
  - `LengthBias.summary()`. "The interval covers no effect, so the data
    cannot show that length moved the judge." is now "The interval includes
    an odds ratio of 1, so the data cannot show that length moved the
    judge. That does not mean length plays no part in its choices." For the
    disagreement fit, "That interval covers no effect, so the judge does
    not depart from the humans in the direction of length." is now "That
    interval includes an odds ratio of 1, so the data cannot show that the
    judge breaks with the humans toward longer or shorter answers. That does
    not mean it follows them on length."
  - `AgreementResult.summary()`. "so the data cannot distinguish the raters"
    is now "so the data cannot show that any one rater is pulling alpha
    down" where every leave-one-out alpha is undefined. Where there are
    leave-one-out figures the summary no longer judges the raters at all,
    as the entry on rater dropout below says.
  - `JudgeValidation.summary()`. "the data cannot single out a slice" is now
    "the data cannot show that the judge does worse on any one slice".
    Where the intervals overlap it adds "That does not mean it does equally
    well on all of them."

  Anything matching on the old sentences will stop matching.

- **Breaking.** `AgreementResult.summary()` and `JudgeValidation.summary()`
  name one of Krippendorff's bands only when the whole interval sits in it.
  0.3.0 named the band from the point estimate, with "That is at or above
  0.800, the conventional bar for treating coded data as reliable.", "That
  sits between 0.667 and 0.800, which supports tentative conclusions and no
  firm ones." and "That is below 0.667, the conventional floor for drawing
  any conclusion from coded data." These now read "The whole interval sits
  above 0.800", "The whole interval sits between 0.667 and 0.800" and "The
  whole interval sits below 0.667". An interval running over 0.667 reads
  "The interval runs both sides of 0.667, the conventional floor for drawing
  any conclusion from coded data, so the data cannot show that rater
  agreement clears it. That does not mean it falls short of it." In
  `JudgeValidation` the subject is "the judge's agreement with the humans".
  An interval that clears 0.667 and runs over 0.800 gets the same form at
  0.800, and a summary with no interval names no band. Judge labels with
  alpha 0.801 and an interval of 0.601 to 0.951 printed "That is at or
  above 0.800" in 0.3.0. Anything matching on the old sentences will stop
  matching.

- **Breaking.** `AgreementResult.summary()` no longer names a rater. 0.3.0
  named one when dropping them raised alpha by more than half the width of
  the interval on alpha. That compares the shift with the sampling error on
  alpha, and the shift has a sampling error of its own that can be much
  larger. Across simulated evals with three raters who differed only by
  chance, the rule named a rater in 4.2% of 500 when everyone graded every
  item. When the third rater graded a fifth of the items it named one in
  32.0% of 500, always one of the two who graded everything. In one of those
  evals 0.3.0 printed "Dropping r2 raises alpha to 0.878, a shift of 0.326
  against a sampling error of 0.200." about a rater who graded all 60 items.
  The same rule caught a rater wrong three times as often as the others in
  72.4% of 500.

  The summary now lists alpha with each rater left out, as in "Leaving one
  rater out at a time gives alpha of 0.878 without r2 (+0.326), 0.524
  without r3 (-0.028), and 0.483 without r1 (-0.069).", and says "The
  package does not judge whether any of these shifts is larger than noise,
  so it names no rater. A rater near the top of this list is a lead to check
  rather than a finding." It lists five raters at most and points to
  `rater_dropout` for the rest. `top_dropout_rater` always returns None.
  `dropout_is_distinguishable` still computes the old rule, its docstring
  says why the rule was withdrawn, and nothing in the package reads it.

  The `audit` agreement finding carries the new sentence. Its severity never
  came from this rule, so no finding changes severity. If you acted on a
  rater a summary named, check that rater by hand.

- **Breaking.** `PositionBias` puts the Clopper-Pearson interval, the exact
  binomial interval, on the rate it tests against a half, in both designs,
  and decides on it. The exact binomial p-value printed beside it comes from
  the same test, so the two cannot disagree about 50%. 0.3.0 used a Wilson
  interval in the randomised design. Wilson and the exact test disagree at
  50% in 188 of 20,300 (n, k) cells up to 200 judgements, and in every one
  the Wilson interval clears 50% while the p-value does not. At 11 of 14 in
  the randomised design 0.3.0 printed "(95% CI: 52.4% to 92.4%, exact
  binomial p=0.0574)" and then "The interval clears 50%, so the judge
  favours whichever output it sees first." It now prints "(95% CI: 49.2% to
  95.3%, exact binomial p=0.0574)" and "The interval includes 50%, so the
  data cannot show that position moved the judge." Every randomised interval
  moves, and a randomised verdict flips wherever the two used to disagree.

  In the both-orders design 0.3.0 printed the share of flips with a bare
  p-value, "(77.8%, exact binomial p=0.1797).", and decided on that p-value.
  It now prints "(77.8%, 95% CI: 40.0% to 97.2%, exact binomial p=0.1797)."
  and decides on the interval, which gives the same answer as the p-value in
  every case. The consistency rate has no test beside it and keeps its
  Wilson interval. Rerun any position check near the line.

### Added

- `BTResult.n_missing_item_ids`, the number of comparisons with no item id.
  It is the last field of the dataclass and defaults to 0, so building a
  `BTResult` by position with 0.3.0's fields still works. `to_elo` carries
  it across.
- `PositionBias.position_a_ci_low` and `PositionBias.position_a_ci_high`,
  the Clopper-Pearson interval on `position_a_rate`. In the randomised
  design they
  equal `ci_low` and `ci_high`. They are the last two fields and default to
  NaN, so building a `PositionBias` by position with 0.3.0's fields still
  works. A both-orders result built that way has no interval, so its
  `has_position_effect` is False.

### Fixed

- **Breaking.** `KappaResult.summary()` said "Krippendorff's alpha does not
  have that problem and handles missing data, so prefer rater_agreement for
  anything you report." Alpha has the problem too. Two raters who agree on
  90 of 100 items score kappa 0.444 and alpha 0.447 when one category takes
  90% of the ratings, and 0.800 and 0.801 when both are used equally often.
  It now says "Krippendorff's alpha moves the same way, because it also
  corrects for chance using how often each category is used." and gives
  missing ratings, more than two raters and ordered scales as the reasons to
  prefer `rater_agreement`. The `cohens_kappa` docstring made the same claim
  and now quotes the four figures.

- **Breaking.** `ScoreCI.summary()` told the reader to "treat differences
  smaller than that as unresolved." A paired comparison resolves
  differences far smaller than either score's own interval. The sentence
  now reads "To compare this score with another, read the interval on the
  difference, which compare_paired and compare_independent report. Two
  score intervals that overlap do not show the systems are level." The
  `scores` module docstring made the same claim and no longer does.

- **Breaking.** `PowerResult.summary()` from `detectable_effect` said
  "anything smaller was out of reach before the first item was graded". A
  difference below the reach can still reach significance, only less often.
  It now says "A smaller difference could still reach significance here,
  with a chance below 80%." Where no difference reaches the requested power,
  "This eval could not have found anything, whatever the two systems really
  do." and, for independent samples, "and this eval could not have found
  anything." are now "A null result from this eval says little about the
  systems." The `detectable_effect` docstring said "An eval could not have
  found anything smaller" and now says "No smaller difference reaches the
  requested power".

- **Breaking.** `JudgeValidation.summary()` on a single item said "Every
  label that could be compared was identical" whatever the labels were,
  beside a plain accuracy of 0.0% when they differed, and counted "1 items".
  It now reads "(nominal, 1 item). One item cannot carry a reliability
  estimate, so there is no number to report and no interval around it." The
  count of items set aside reads "1 item was set aside" where it read "1
  items were set aside".

- **Breaking.** `BTResult.summary()` under `resample="comparisons"` said
  "these 240 comparisons share 239 items" when one comparison had no item
  id and no two shared anything. Sharing is now counted among the
  comparisons that carry an id. When none share, the sentence is left out.
  When some share and ids are missing, it reads "the 239 comparisons that
  carry an item id share 26 items".

- The README example and the prose under it show the new wording.

### Removed

- Nothing.

### Known limitations

This rule is unchanged since 0.3.0 and is not fixed in this release. The
figures come from simulations run against 0.3.0 and against the code released
as 0.4.0, which give the same results.

- `JudgeValidation` names a slice, and `audit` raises a critical finding on
  it, only when the slice's own interval sits wholly below the interval on
  the overall figure. That reads a difference off two separate intervals
  instead of an interval on the difference, and it is conservative. With
  five slices of 60 items that differ only by chance, it named a slice in
  0.5% of 200 simulated evals. It misses real differences. A slice where the
  judge flipped 25% of the labels, against 10% in the other four, was named
  in 32.5% of 200. When the summary says the data cannot show that the
  judge does worse on any one slice, this rule is what did not show it.

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

[Unreleased]: https://github.com/mokayahezron/evalaudit/compare/v0.4.0...main
[0.4.0]: https://github.com/mokayahezron/evalaudit/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/mokayahezron/evalaudit/compare/809a650...v0.3.0
[0.2.1]: https://github.com/mokayahezron/evalaudit/compare/242e354...809a650
[0.2.0]: https://github.com/mokayahezron/evalaudit/compare/c1230f1...242e354
[0.1.0]: https://github.com/mokayahezron/evalaudit/compare/b090c66...c1230f1
[0.0.1]: https://github.com/mokayahezron/evalaudit/tree/b090c66
