# CLAUDE.md — evalaudit project rules

## Design invariants

- Every public function returns a frozen dataclass from `_types.py`.
- Every return type has a `summary()` that produces a plain-English verdict.
- Never return a bare p-value. Always report effect size plus confidence interval.
- Dependencies are numpy, scipy, and pandas only.

## Testing

- Tests are written before implementations and must not be edited to make code pass.
- Coverage tests (simulation-based) verify that intervals cover at the claimed rate.

## Implementation style

- Vectorise bootstraps in numpy. Draw index matrices in one call and compute row statistics — no Python loops over resamples.
- Use `rng.integers(0, n, size=(n_boot, n))` for bootstrap index matrices.

## Build order

scores → compare → agreement → pairwise → judge → power → audit

The core of the package is **agreement**, **judge**, and **pairwise** — these are the point, not extras. scores and compare are the foundation they stand on.

## Commands

- Install dev: `pip install -e ".[dev]"`
- Run tests: `pytest -q`
