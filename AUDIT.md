# Codebase audit report (`evalaudit`)

## Scope audited
I reviewed:
- `evalaudit/` (all modules)
- `tests/` (all test files)
- `examples/interval_method_study.py`
- `README.md`
- `pyproject.toml`
- `.github/workflows/tests.yml`
- available git history (`git log`, shallow: 2 commits visible)

I also executed the full test suite in this environment:
- `python -m pytest -q`
- Result: **767 passed, 2 skipped**

---

## Findings ranked by consequence (worst first)

## Definitely wrong

### 1) `score_ci(method="wilson")` can return statistically invalid intervals for non-binary data
- **File/line:** `/home/runner/work/evalaudit/evalaudit/evalaudit/scores.py:74-77`
- **What is wrong:** The Wilson branch computes `k = int(np.sum(x))` without requiring binary `0/1` input. For continuous scores, this truncates the sum and feeds a binomial interval formula values that are not binomial counts.
- **Why it matters:** This can produce confident-looking but meaningless intervals on non-binary data, exactly the kind of wrong number that undermines the package’s core trust claim.
- **Evidence:**
  - `score_ci([0.2, 0.7, 0.9], method="wilson")` returns a result instead of refusing.
  - `binary=False` in that result, so the output can look like a valid mean interval while being computed from a proportion formula with truncated counts.
- **Confidence:** High
- **Fix effort:** Small (input validation + tests)

### 2) `compare_*` bootstrap path crashes with internal `IndexError` when `n_boot=0`
- **File/line:**
  - `/home/runner/work/evalaudit/evalaudit/evalaudit/compare.py:192-197`
  - `/home/runner/work/evalaudit/evalaudit/evalaudit/compare.py:356-358`
  - `/home/runner/work/evalaudit/evalaudit/evalaudit/compare.py:692-700`
- **What is wrong:** `n_boot=0` leads to empty bootstrap arrays and then percentile/p-value logic that indexes empty arrays, raising `IndexError`.
- **Why it matters:** This fails with a low-level error message instead of a domain refusal. In client-facing scientific software, this is both usability failure and a reliability signal problem.
- **Evidence:**
  - `compare_paired(..., method="bootstrap", n_boot=0)` -> `IndexError: index -1 is out of bounds for axis 0 with size 0`
  - same for `compare_independent(..., method="bootstrap", n_boot=0)`
- **Confidence:** High
- **Fix effort:** Small (validate `n_boot >= 1` for bootstrap mode or define explicit no-interval behavior)

### 3) Invalid confidence levels are accepted in `scores` and `compare`, producing NaN/Inf outputs
- **File/line:**
  - `/home/runner/work/evalaudit/evalaudit/evalaudit/scores.py:22-85`
  - `/home/runner/work/evalaudit/evalaudit/evalaudit/compare.py:47-211`
- **What is wrong:** Unlike `agreement`, `judge`, `pairwise`, and `audit`, these APIs do not enforce `0 < confidence < 1`.
- **Why it matters:** They can emit NaN/Inf intervals that look like normal results objects instead of refusing invalid requests.
- **Evidence:**
  - `score_ci([1,0,1], confidence=1.0)` returns `ci_low=nan, ci_high=nan`
  - `compare_independent(..., method="t", confidence=1.0)` returns `(-inf, inf)` interval
- **Confidence:** High
- **Fix effort:** Small

### 4) High-value test for summary correctness is weak enough to pass a deliberately broken implementation
- **File/line:** `/home/runner/work/evalaudit/evalaudit/tests/test_scores.py:122-127`
- **What is wrong:** The test checks only a few substrings in `ScoreCI.summary()`, not the semantic correctness of the sentence.
- **Why it matters:** The package explicitly treats prose as product output. Weak prose tests allow materially wrong summaries to pass CI.
- **Proof by breakage (without editing repo files):**
  - I injected a runtime monkeypatch via `/tmp/.../sitecustomize.py` to force `ScoreCI.summary()` to a fixed, deliberately wrong string containing the expected substrings.
  - Ran: `PYTHONPATH=/tmp/evalaudit-audit-mutations python -m pytest -q tests/test_scores.py -k summary_mentions_estimate_and_bounds`
  - Result: **passed**.
- **Confidence:** High
- **Fix effort:** Medium (stronger assertions against structured expectations; include negative checks)

### 5) Commit message/history inconsistency
- **File/line:** git commit `d38c7a3` message vs repository state
- **What is wrong:** Commit message says “bump to 0.2.0”, but package metadata remains `0.1.0`:
  - `/home/runner/work/evalaudit/evalaudit/pyproject.toml:7`
  - `/home/runner/work/evalaudit/evalaudit/evalaudit/__init__.py:3`
- **Why it matters:** This is provenance noise in a repository that positions itself as audit-grade evidence. Inconsistent history text reduces reviewer trust.
- **Confidence:** High
- **Fix effort:** Small (correct version or correct message on next commit)

---

## Suspected / not fully proven

### S1) README output verification can admit false positives
- **File/line:** `/home/runner/work/evalaudit/evalaudit/tests/test_readme.py:8-11, 87-94`
- **Concern:** The test validates that output fragments appear in captured output, with `[...]` elisions and whitespace collapse. This is practical, but containment-based checks can miss some semantic drift if the same fragments appear in unintended context.
- **Why it matters:** README is a public evidence artifact for skeptical readers; partial matching may under-detect drift in client-facing claims.
- **Confidence:** Medium
- **Fix effort:** Medium

### S2) I could not independently verify every cited statistical source equation from primary literature in this run
- **File/line:** multiple docstrings in `compare.py`, `power.py`, `agreement.py`, `judge.py`
- **Concern:** Internal cross-checks are strong (tests against `statsmodels`, `krippendorff`, simulation), but I did not complete source-paper-by-source-paper re-derivation for every formula.
- **Why it matters:** Your stated bar is citation-level validity, not just internal consistency.
- **Confidence:** Medium
- **Fix effort:** Larger (dedicated derivation + citation audit pass)

---

## Clean areas (explicit)

I looked hard at these and did not find concrete breakage in this pass:
- **Agreement/Judge/Pairwise math implementation vs references-in-tests:** substantial cross-check coverage against external libraries (`statsmodels`, `krippendorff`, optional `choix`) and many simulation/property tests in `tests/test_agreement.py`, `tests/test_judge.py`, `tests/test_pairwise.py`, `tests/test_power.py`.
- **Packaging declarations:** runtime deps (`numpy`, `scipy`, `pandas`) are used by runtime modules; test-only deps are under `dev` extras; CI installs `.[dev]` before running tests.
- **Public API return-shape consistency:** public entry points return frozen dataclasses with `summary()` (as promised), based on inspection of `evalaudit/__init__.py` exports and dataclass definitions in `_types.py`.
- **Secrets exposure in current tree:** no obvious tokens/keys found by regex scan.

---

## What I could not check (and why)

1) **Full git history audit**
- Only two commits are available in this clone (`c1230f1`, `d38c7a3`), likely shallow history. I could not inspect older commits for sensitive or misdescribed content.

2) **Primary-literature verification of every formula claim**
- I validated behavior through code/tests and runtime probes, but did not perform a full external paper-by-paper derivation audit in this run.

3) **CI behavior across all matrix interpreters in this environment**
- Local run was Python 3.12 only. I reviewed workflow config but did not execute a full 3.9/3.11/3.12 matrix locally.

---

## The three things I would fix first (if it were my call)

1) **Block invalid statistical requests at API boundaries (`scores`, `compare`)**
- Add strict validation for `confidence` and bootstrap counts, and reject Wilson on non-binary input.
- Reason: these are direct pathways to wrong or meaningless numbers that still look official.

2) **Harden high-value prose tests (starting with `ScoreCI.summary`)**
- Replace brittle substring checks with assertions tied to computed values/semantics.
- Reason: your package sells interpretable conclusions; weak prose tests are a direct trust risk.

3) **Clean provenance mismatch in versioning/history messaging**
- Align commit narrative and package version metadata.
- Reason: in an audit-focused repository, trust in the change record is part of the product.

