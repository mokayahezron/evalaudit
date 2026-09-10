"""Tests for evalaudit.pairwise.

Written before the implementation. Make these pass.

The Bradley-Terry fit is checked four ways. Against ``choix`` on PyPI, a
package that does nothing but fit this model. Against a logistic regression
in statsmodels, since Bradley-Terry with one item fixed at zero is exactly a
logit model on the sign-coded design matrix, and that equivalence is in every
treatment of the model. Against a slow reference in this file that builds the
log-likelihood by looping over comparison rows and hands it to a general
optimiser. And against the score equations, which say that at the maximum
every model's credit equals the credit its fitted rating predicts. The last
check needs no optimiser at all, so it is the one that survives if all three
implementations share a bug.

No published worked example is asserted. The obvious candidate is a league
table, and the fitted values for the ``BASEBALL`` fixture here were computed
from the win matrix rather than copied out of a book, so asserting on them
would test a memory rather than a source. The two-model closed form
``log(wins / losses)`` is in print everywhere and is asserted directly
instead, along with the balanced and cyclic designs whose answers are forced
by symmetry.

Ties are checked with the doubling trick. Splitting a tie gives half credit
to each side, so doubling every count turns the credits into whole
comparisons without moving the maximum. That lets choix and statsmodels,
which only take whole comparisons, check the split policy exactly.

The intervals are on each rating measured against the average of the field,
then displayed against whichever model was named as the reference. That keeps
their widths from turning on whoever the reference is, and the coverage test
targets that quantity rather than the gap to the reference.

Separability is read off a second interval, on the gap between two ratings,
taken from the same resamples. A pair separates when that interval excludes
zero. Two rating intervals that overlap establish nothing about the gap, and
the tests below include a pair built to show it. The gap interval is checked
three ways. Against the binomial quantiles it reduces to with two models,
which is exact. Against Woolf's closed-form interval on a log-odds, which it
should approach. And against a bootstrap that refits every resample with a
general optimiser, on the same draws, which is also exact.

The bootstrap resamples items by default, keeping every comparison made on an
item together. The fixtures built by ``frame`` give every row its own item,
so for them resampling items and resampling comparisons draw the same
resamples, and the tests that check the resampling unit build their own
grouped data.
"""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.optimize import minimize

import statsmodels.api as sm

try:
    import choix
except ImportError:  # choix needs 3.10, this package supports 3.9
    choix = None

import evalaudit
from evalaudit import BTResult
from evalaudit.pairwise import bradley_terry, to_elo


# Where choix cannot be installed the fit is still checked against
# statsmodels, against the slow reference, and against the score equations,
# so the loss is one of four independent checks rather than all of them.
needs_choix = pytest.mark.skipif(choix is None, reason="choix is not installed")


LOG10 = math.log(10.0)

# The share of bootstrap resamples that has to come back usable before an
# interval is reported. Spelled out rather than imported, because a test that
# imported it would pass whatever the number was later changed to.
MIN_USABLE_SHARE = 0.90


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def frame(rows):
    """A comparisons frame from (model_a, model_b, winner) triples.

    Item ids are invented here. Nothing in the fit reads them and the
    library still asks for the column, so every fixture has to carry one.
    """
    return pd.DataFrame(
        [
            {"item_id": f"q{i}", "model_a": a, "model_b": b, "winner": w}
            for i, (a, b, w) in enumerate(rows)
        ]
    )


def from_matrix(wins, names):
    """Long comparisons from a wins matrix. ``wins[i][j]`` is i beating j."""
    rows = []
    size = len(names)
    for i in range(size):
        for j in range(size):
            if i != j:
                rows += [(names[i], names[j], names[i])] * int(wins[i][j])
    return frame(rows)


def simulate(true_ratings, n_per_pair, tie_rate=0.0, seed=0):
    """A round robin generated from the model itself.

    ``true_ratings`` maps model name to a rating on the natural log-odds
    scale, so a gap of 1.0 means the better model takes about 73% of the
    decisive comparisons.
    """
    rng = np.random.default_rng(seed)
    names = list(true_ratings)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            p = 1.0 / (1.0 + math.exp(-(true_ratings[a] - true_ratings[b])))
            for _ in range(n_per_pair):
                if rng.random() < tie_rate:
                    rows.append((a, b, "tie"))
                else:
                    rows.append((a, b, a if rng.random() < p else b))
    rng.shuffle(rows)
    return frame(rows)


def is_tie(winner):
    return not isinstance(winner, str) or winner.strip().lower() == "tie"


def all_models(comparisons):
    return sorted(set(comparisons["model_a"]) | set(comparisons["model_b"]))


def tally(comparisons, ties="split"):
    """Credits and pair counts, built by looping over rows.

    Returns ``(models, credit, n)`` where ``credit[i][j]`` is what model i
    earned against model j and ``n[i][j]`` is how much of a comparison the
    pair carried. A split tie puts half a win in each direction.
    """
    models = all_models(comparisons)
    index = {m: k for k, m in enumerate(models)}
    size = len(models)
    credit = [[0.0] * size for _ in range(size)]
    for a, b, w in zip(
        comparisons["model_a"], comparisons["model_b"], comparisons["winner"]
    ):
        i, j = index[a], index[b]
        if is_tie(w):
            if ties == "drop":
                continue
            credit[i][j] += 0.5
            credit[j][i] += 0.5
        elif w == a:
            credit[i][j] += 1.0
        else:
            credit[j][i] += 1.0
    n = [
        [credit[i][j] + credit[j][i] for j in range(size)]
        for i in range(size)
    ]
    return models, credit, n


def score_residuals(ratings, comparisons, ties="split"):
    """How far the fitted ratings are from solving the score equations.

    At the maximum of the Bradley-Terry likelihood every model's credit
    equals the sum over its opponents of the credit the ratings predict.
    That characterisation is the model, and it goes nowhere near an
    optimiser, so it catches a fit that converged to the wrong place.
    """
    models, credit, n = tally(comparisons, ties)
    out = {}
    for i, m in enumerate(models):
        earned = sum(credit[i])
        expected = 0.0
        for j, other in enumerate(models):
            if i == j or n[i][j] == 0:
                continue
            expected += n[i][j] / (1.0 + math.exp(-(ratings[m] - ratings[other])))
        out[m] = earned - expected
    return out


def used_rows(comparisons, ties):
    rows = []
    for a, b, w in zip(
        comparisons["model_a"], comparisons["model_b"], comparisons["winner"]
    ):
        tie = is_tie(w)
        if tie and ties == "drop":
            continue
        rows.append((a, b, None if tie else w))
    return rows


def slow_fit(comparisons, ties="split", reference=None):
    """Bradley-Terry by handing the log-likelihood to a general optimiser.

    Written from the definition. The likelihood is summed one comparison at
    a time with no matrices anywhere, so it disagrees with the shipped code
    about everything except the answer.
    """
    models = all_models(comparisons)
    reference = reference or models[0]
    free = [m for m in models if m != reference]
    index = {m: k for k, m in enumerate(free)}
    rows = used_rows(comparisons, ties)

    def rating_of(theta, m):
        return 0.0 if m == reference else theta[index[m]]

    def nll(theta):
        total = 0.0
        for a, b, w in rows:
            gap = rating_of(theta, a) - rating_of(theta, b)
            p_a = 1.0 / (1.0 + math.exp(-gap))
            p_a = min(max(p_a, 1e-15), 1 - 1e-15)
            if w is None:
                total -= 0.5 * math.log(p_a) + 0.5 * math.log(1 - p_a)
            elif w == a:
                total -= math.log(p_a)
            else:
                total -= math.log(1 - p_a)
        return total

    fit = minimize(
        nll, np.zeros(len(free)), method="BFGS",
        options={"gtol": 1e-10, "maxiter": 5000},
    )
    out = {reference: 0.0}
    for m in free:
        out[m] = float(fit.x[index[m]])
    return out


def choix_fit(comparisons, ties="split", reference=None):
    """The same fit from choix, an independent Bradley-Terry package.

    choix takes whole (winner, loser) comparisons and cannot express half
    credit, so every decisive comparison goes in twice and every split tie
    once in each direction. That doubles all the credits, which leaves the
    maximum exactly where it was.
    """
    models = all_models(comparisons)
    reference = reference or models[0]
    index = {m: k for k, m in enumerate(models)}

    data = []
    for a, b, w in zip(
        comparisons["model_a"], comparisons["model_b"], comparisons["winner"]
    ):
        if is_tie(w):
            if ties == "drop":
                continue
            data.append((index[a], index[b]))
            data.append((index[b], index[a]))
        elif w == a:
            data += [(index[a], index[b])] * 2
        else:
            data += [(index[b], index[a])] * 2

    params = choix.opt_pairwise(
        len(models), data, alpha=1e-9, method="BFGS", tol=1e-12
    )
    params = params - params[index[reference]]
    return {m: float(params[index[m]]) for m in models}


def statsmodels_fit(comparisons, ties="split", reference=None):
    """The same fit as the logistic regression it is.

    Sign-code the design matrix, drop the reference column, fit a binomial
    logit with no intercept, and the coefficients are the ratings. Ties get
    the same doubling trick choix gets.
    """
    models = all_models(comparisons)
    reference = reference or models[0]
    free = [m for m in models if m != reference]
    index = {m: k for k, m in enumerate(free)}
    design, outcome = [], []

    def add(a, b, y):
        row = [0.0] * len(free)
        if a in index:
            row[index[a]] += 1.0
        if b in index:
            row[index[b]] -= 1.0
        design.append(row)
        outcome.append(y)

    for a, b, w in zip(
        comparisons["model_a"], comparisons["model_b"], comparisons["winner"]
    ):
        if is_tie(w):
            if ties == "drop":
                continue
            add(a, b, 1.0)
            add(a, b, 0.0)
        else:
            y = 1.0 if w == a else 0.0
            add(a, b, y)
            add(a, b, y)

    fit = sm.GLM(
        np.array(outcome), np.array(design), family=sm.families.Binomial()
    ).fit(tol=1e-12, maxiter=200)
    out = {reference: 0.0}
    for m in free:
        out[m] = float(fit.params[index[m]])
    return out


def ratings_dict(result):
    return dict(zip(result.ratings["model"], result.ratings["rating"]))


# Seven models in a full round robin, thirteen comparisons per pair. Shaped
# after a league table so the fit meets the spread and the lopsidedness that
# leaderboard data has, rather than the tidy symmetry of a generated one.
BASEBALL_NAMES = ["mistral", "gemini", "llama", "qwen", "gpt", "phi", "olmo"]
BASEBALL_WINS = [
    [0, 7, 9, 7, 7, 9, 11],
    [6, 0, 7, 9, 8, 9, 9],
    [4, 6, 0, 7, 7, 8, 12],
    [6, 4, 6, 0, 6, 10, 9],
    [6, 5, 6, 7, 0, 7, 12],
    [4, 4, 5, 3, 6, 0, 6],
    [2, 4, 1, 4, 1, 7, 0],
]


def baseball():
    return from_matrix(BASEBALL_WINS, BASEBALL_NAMES)


def thin_four():
    """Four models a hair apart, ten comparisons per pair. Nothing separates."""
    return simulate(
        {"a": 0.3, "b": 0.1, "c": -0.1, "d": -0.3}, n_per_pair=10, seed=5
    )


def two_islands():
    """Two groups of models that never met."""
    return frame(
        [("alpha", "beta", "alpha")] * 5
        + [("beta", "alpha", "beta")] * 3
        + [("gamma", "delta", "gamma")] * 5
        + [("delta", "gamma", "delta")] * 3
    )


def undefeated_alpha():
    """Everyone played everyone. Alpha never lost."""
    return frame(
        [("alpha", "beta", "alpha")] * 5
        + [("alpha", "gamma", "alpha")] * 5
        + [("beta", "gamma", "beta")] * 4
        + [("beta", "gamma", "gamma")] * 3
    )


def winless_gamma():
    """Everyone played everyone. Gamma never won."""
    return frame(
        [("alpha", "gamma", "alpha")] * 5
        + [("beta", "gamma", "beta")] * 5
        + [("alpha", "beta", "alpha")] * 4
        + [("alpha", "beta", "beta")] * 3
    )


def tiny_cycle():
    """Three comparisons in a ring. The smallest data with a maximum."""
    return frame([("a", "b", "a"), ("b", "c", "b"), ("c", "a", "c")])


# --------------------------------------------------------------------------
# The references check out against each other
# --------------------------------------------------------------------------

@needs_choix
def test_slow_reference_matches_choix():
    slow = slow_fit(baseball())
    ref = choix_fit(baseball())
    for m in BASEBALL_NAMES:
        assert slow[m] == pytest.approx(ref[m], abs=1e-4)


def test_slow_reference_matches_statsmodels():
    slow = slow_fit(baseball())
    ref = statsmodels_fit(baseball())
    for m in BASEBALL_NAMES:
        assert slow[m] == pytest.approx(ref[m], abs=1e-4)


def _split_tie_fixture():
    return simulate(
        {"a": 1.0, "b": 0.3, "c": -0.4, "d": -1.1},
        n_per_pair=40, tie_rate=0.4, seed=7,
    )


def test_slow_reference_matches_statsmodels_on_split_ties():
    data = _split_tie_fixture()
    slow = slow_fit(data, ties="split")
    ref = statsmodels_fit(data, ties="split")
    for m in "abcd":
        assert slow[m] == pytest.approx(ref[m], abs=1e-4)


@needs_choix
def test_slow_reference_matches_choix_on_split_ties():
    data = _split_tie_fixture()
    slow = slow_fit(data, ties="split")
    ref = choix_fit(data, ties="split")
    for m in "abcd":
        assert slow[m] == pytest.approx(ref[m], abs=1e-4)


# --------------------------------------------------------------------------
# The fit
# --------------------------------------------------------------------------

@needs_choix
def test_matches_choix():
    got = ratings_dict(bradley_terry(baseball(), n_boot=0, reference="olmo"))
    ref = choix_fit(baseball(), reference="olmo")
    for m in BASEBALL_NAMES:
        assert got[m] == pytest.approx(ref[m], abs=1e-3)


def test_matches_statsmodels():
    got = ratings_dict(bradley_terry(baseball(), n_boot=0, reference="olmo"))
    ref = statsmodels_fit(baseball(), reference="olmo")
    for m in BASEBALL_NAMES:
        assert got[m] == pytest.approx(ref[m], abs=1e-4)


def test_matches_the_slow_reference():
    got = ratings_dict(bradley_terry(baseball(), n_boot=0, reference="olmo"))
    ref = slow_fit(baseball(), reference="olmo")
    for m in BASEBALL_NAMES:
        assert got[m] == pytest.approx(ref[m], abs=1e-4)


@needs_choix
@pytest.mark.parametrize("ties", ["split", "drop"])
def test_matches_choix_on_simulated_data(ties):
    data = simulate(
        {"a": 1.2, "b": 0.5, "c": 0.0, "d": -0.6, "e": -1.3},
        n_per_pair=30, tie_rate=0.25, seed=3,
    )
    got = ratings_dict(
        bradley_terry(data, ties=ties, n_boot=0, reference="c")
    )
    ref = choix_fit(data, ties=ties, reference="c")
    for m in "abcde":
        assert got[m] == pytest.approx(ref[m], abs=1e-3)


@pytest.mark.parametrize("ties", ["split", "drop"])
def test_solves_the_score_equations(ties):
    """The definition of the maximum, checked without an optimiser."""
    data = simulate(
        {"a": 1.2, "b": 0.5, "c": 0.0, "d": -0.6, "e": -1.3},
        n_per_pair=30, tie_rate=0.3, seed=11,
    )
    r = bradley_terry(data, ties=ties, n_boot=0)
    residuals = score_residuals(ratings_dict(r), data, ties)
    assert max(abs(v) for v in residuals.values()) < 1e-6


def test_two_models_match_the_closed_form():
    """With two models the maximum is log(wins / losses), in closed form."""
    data = frame([("a", "b", "a")] * 30 + [("a", "b", "b")] * 10)
    got = ratings_dict(bradley_terry(data, n_boot=0, reference="b"))
    assert got["b"] == 0.0
    assert got["a"] == pytest.approx(math.log(30 / 10), abs=1e-8)


def test_two_models_closed_form_with_split_ties():
    """Half credit each way moves the closed form to log(35 / 15)."""
    data = frame(
        [("a", "b", "a")] * 25
        + [("a", "b", "b")] * 5
        + [("a", "b", "tie")] * 20
    )
    got = ratings_dict(
        bradley_terry(data, ties="split", n_boot=0, reference="b")
    )
    assert got["a"] == pytest.approx(math.log(35 / 15), abs=1e-8)


def test_a_symmetric_cycle_gives_every_model_the_same_rating():
    """Nobody is better. The fit has to say so rather than pick a winner."""
    data = frame(
        [("a", "b", "a")] * 5
        + [("b", "c", "b")] * 5
        + [("c", "a", "c")] * 5
    )
    r = bradley_terry(data, n_boot=0)
    for value in r.ratings["rating"]:
        assert value == pytest.approx(0.0, abs=1e-8)


def test_a_balanced_round_robin_gives_every_model_the_same_rating():
    rows = []
    for a, b in [("a", "b"), ("a", "c"), ("b", "c")]:
        rows += [(a, b, a)] * 10 + [(a, b, b)] * 10
    r = bradley_terry(frame(rows), n_boot=0)
    for value in r.ratings["rating"]:
        assert value == pytest.approx(0.0, abs=1e-8)


def test_the_better_model_gets_the_higher_rating():
    data = simulate(
        {"strong": 1.5, "middle": 0.0, "weak": -1.5}, n_per_pair=200, seed=5
    )
    r = bradley_terry(data, n_boot=0)
    assert list(r.ratings["model"]) == ["strong", "middle", "weak"]


def test_ratings_recover_the_generating_values():
    truth = {"a": 1.4, "b": 0.6, "c": -0.2, "d": -1.0}
    data = simulate(truth, n_per_pair=800, seed=2)
    got = ratings_dict(bradley_terry(data, n_boot=0, reference="d"))
    for m in truth:
        assert got[m] == pytest.approx(truth[m] - truth["d"], abs=0.12)


# --------------------------------------------------------------------------
# Return type
# --------------------------------------------------------------------------

def test_returns_a_btresult():
    assert isinstance(bradley_terry(baseball(), n_boot=0), BTResult)


def test_btresult_is_frozen():
    r = bradley_terry(baseball(), n_boot=0)
    with pytest.raises(Exception):
        r.reference = "gpt"


def test_exported_from_the_package():
    assert evalaudit.bradley_terry is bradley_terry
    assert evalaudit.to_elo is to_elo
    assert evalaudit.BTResult is BTResult


def test_summary_is_a_string():
    r = bradley_terry(baseball(), n_boot=200, seed=0)
    assert isinstance(r.summary(), str)
    assert len(r.summary()) > 50


def test_no_em_dashes_anywhere():
    r = bradley_terry(baseball(), n_boot=200, seed=0)
    assert "—" not in r.summary()
    assert "—" not in to_elo(r).summary()
    assert "—" not in bradley_terry(two_islands(), n_boot=0).summary()
    assert "—" not in bradley_terry(undefeated_alpha(), n_boot=0).summary()


# --------------------------------------------------------------------------
# The ratings table
# --------------------------------------------------------------------------

def test_ratings_has_the_columns_the_docstring_promises():
    r = bradley_terry(baseball(), n_boot=200, seed=0)
    assert list(r.ratings.columns) == [
        "model", "rating", "ci_low", "ci_high", "n_comparisons"
    ]


def test_ratings_lists_every_model_once():
    r = bradley_terry(baseball(), n_boot=0)
    assert sorted(r.ratings["model"]) == sorted(BASEBALL_NAMES)
    assert r.n_models == 7


def test_ratings_sorted_by_rating_descending():
    values = list(bradley_terry(baseball(), n_boot=0).ratings["rating"])
    assert values == sorted(values, reverse=True)


def test_ratings_reports_the_comparison_count_per_model():
    r = bradley_terry(baseball(), n_boot=0)
    counts = dict(zip(r.ratings["model"], r.ratings["n_comparisons"]))
    for m in BASEBALL_NAMES:
        assert counts[m] == 78  # six opponents, thirteen comparisons each


def test_comparison_counts_are_unequal_when_the_design_is():
    data = frame(
        [("a", "b", "a")] * 20
        + [("b", "c", "b")] * 5
        + [("a", "c", "c")] * 5
    )
    r = bradley_terry(data, n_boot=0)
    counts = dict(zip(r.ratings["model"], r.ratings["n_comparisons"]))
    assert counts == {"a": 25, "b": 25, "c": 10}


def test_totals_describe_the_frame():
    data = simulate(
        {"a": 0.5, "b": 0.0, "c": -0.5}, n_per_pair=20, tie_rate=0.5, seed=4
    )
    r = bradley_terry(data, ties="split", n_boot=0)
    assert r.n_comparisons == 60
    assert r.n_ties > 0
    assert r.n_dropped == 0
    assert r.n_items == 60
    assert r.n_pairs == 3


def test_dropping_ties_shrinks_the_comparison_count():
    data = simulate(
        {"a": 0.5, "b": 0.0, "c": -0.5}, n_per_pair=20, tie_rate=0.5, seed=4
    )
    split = bradley_terry(data, ties="split", n_boot=0)
    drop = bradley_terry(data, ties="drop", n_boot=0)
    assert drop.n_comparisons == split.n_comparisons - split.n_ties
    assert drop.n_dropped == split.n_ties
    assert drop.n_ties == split.n_ties


# --------------------------------------------------------------------------
# The win matrix
# --------------------------------------------------------------------------

def test_win_matrix_is_the_model_probability():
    r = bradley_terry(baseball(), n_boot=0)
    got = ratings_dict(r)
    for a in BASEBALL_NAMES:
        for b in BASEBALL_NAMES:
            if a == b:
                continue
            expected = 1.0 / (1.0 + math.exp(-(got[a] - got[b])))
            assert r.win_matrix.loc[a, b] == pytest.approx(expected, abs=1e-9)


def test_win_matrix_rows_and_columns_are_the_models_in_rating_order():
    r = bradley_terry(baseball(), n_boot=0)
    order = list(r.ratings["model"])
    assert list(r.win_matrix.index) == order
    assert list(r.win_matrix.columns) == order


def test_win_matrix_complements_to_one():
    r = bradley_terry(baseball(), n_boot=0)
    for a in BASEBALL_NAMES:
        for b in BASEBALL_NAMES:
            if a != b:
                total = r.win_matrix.loc[a, b] + r.win_matrix.loc[b, a]
                assert total == pytest.approx(1.0, abs=1e-12)


def test_win_matrix_diagonal_is_not_a_number():
    r = bradley_terry(baseball(), n_boot=0)
    assert np.isnan(np.diag(r.win_matrix.to_numpy())).all()


def test_win_matrix_is_half_when_the_models_are_level():
    data = frame(
        [("a", "b", "a")] * 5
        + [("b", "c", "b")] * 5
        + [("c", "a", "c")] * 5
    )
    r = bradley_terry(data, n_boot=0)
    off = ~np.eye(3, dtype=bool)
    assert r.win_matrix.to_numpy()[off] == pytest.approx(0.5, abs=1e-8)


# --------------------------------------------------------------------------
# Separability, which is the point
# --------------------------------------------------------------------------

def test_separable_pairs_has_the_columns_the_docstring_promises():
    r = bradley_terry(baseball(), n_boot=400, seed=0)
    assert list(r.separable_pairs.columns) == PAIR_COLUMNS


# The columns of separable_pairs. ci_low and ci_high are the interval on
# ``difference``, the gap from model_a down to model_b.
PAIR_COLUMNS = [
    "model_a", "model_b", "difference", "ci_low", "ci_high",
    "p_a_beats_b", "n_head_to_head",
]


def test_an_empty_pairs_table_still_has_every_column():
    """No interval, and no fit, both leave the table empty. A reader who
    selects ci_low from it should get an empty column, not a KeyError."""
    for r in (
        bradley_terry(baseball(), n_boot=0),
        bradley_terry(two_islands(), n_boot=200, seed=0),
    ):
        assert r.separable_pairs.empty
        assert list(r.separable_pairs.columns) == PAIR_COLUMNS


# ``pairs`` is every pair of models, and separable_pairs is its separable
# rows. The summary talks about the pairs that did not separate, so their
# intervals have to be somewhere a reader can see them.
PAIRS_COLUMNS = PAIR_COLUMNS + ["separable"]


def test_pairs_lists_every_pair_once_with_the_higher_rated_model_first():
    r = bradley_terry(baseball(), n_boot=400, seed=0)
    assert list(r.pairs.columns) == PAIRS_COLUMNS
    assert len(r.pairs) == r.n_pairs == 21
    assert {
        frozenset(pair) for pair in zip(r.pairs["model_a"], r.pairs["model_b"])
    } == {
        frozenset((x, y))
        for i, x in enumerate(BASEBALL_NAMES) for y in BASEBALL_NAMES[i + 1:]
    }
    got = ratings_dict(r)
    for _, row in r.pairs.iterrows():
        assert got[row["model_a"]] >= got[row["model_b"]]
        assert row["difference"] == pytest.approx(
            got[row["model_a"]] - got[row["model_b"]], abs=1e-12
        )
        assert row["ci_low"] < row["ci_high"]
        assert row["separable"] == (row["ci_low"] > 0 or row["ci_high"] < 0)
    gaps = list(r.pairs["difference"])
    assert gaps == sorted(gaps, reverse=True)


def test_separable_pairs_are_the_separable_rows_of_pairs():
    r = bradley_terry(baseball(), n_boot=400, seed=0)
    expected = (
        r.pairs[r.pairs["separable"]]
        .drop(columns="separable")
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(r.separable_pairs, expected)
    assert r.n_separable == int(r.pairs["separable"].sum())


def test_pairs_without_an_interval_still_carry_the_gaps():
    """No resamples, so no interval and nothing separates. The gaps and the
    head-to-head counts are still the fit's, and a reader can see them."""
    r = bradley_terry(baseball(), n_boot=0)
    assert len(r.pairs) == 21
    assert r.pairs["ci_low"].isna().all()
    assert r.pairs["ci_high"].isna().all()
    assert not r.pairs["separable"].any()
    assert (r.pairs["difference"] >= 0).all()
    assert (r.pairs["n_head_to_head"] == 13).all()


def test_pairs_is_empty_without_a_fit():
    for r in (
        bradley_terry(two_islands(), n_boot=200, seed=0),
        bradley_terry(undefeated_alpha(), n_boot=200, seed=0),
    ):
        assert r.pairs.empty
        assert list(r.pairs.columns) == PAIRS_COLUMNS


def test_n_pairs_is_every_pair_of_models():
    assert bradley_terry(baseball(), n_boot=200, seed=0).n_pairs == 21


def direct_refit(credit, reference):
    """Maximum likelihood on one tallied credit matrix, by a general optimiser.

    The reference is pinned at zero, which is a different parameterisation
    from the package's centred one, and the likelihood goes to BFGS with its
    analytic gradient rather than to Newton's method.
    """
    m = credit.shape[0]
    n = credit + credit.T
    earned = credit.sum(axis=1)
    free = [k for k in range(m) if k != reference]

    def objective(theta):
        r = np.zeros(m)
        r[free] = theta
        gap = r[:, None] - r[None, :]
        p = 1.0 / (1.0 + np.exp(-gap))
        loglik = -np.sum(credit * np.logaddexp(0.0, -gap))
        gradient = earned - (n * p).sum(axis=1)
        return -loglik, -gradient[free]

    fit = minimize(
        objective, np.zeros(m - 1), jac=True, method="BFGS",
        options={"gtol": 1e-11, "maxiter": 2000},
    )
    r = np.zeros(m)
    r[free] = fit.x
    return r


def refit_bootstrap_gaps(comparisons, n_boot, seed, confidence=0.95):
    """Percentile intervals on every gap, refitting each resample from scratch.

    Nothing from the package is used. Rows are tallied with np.add.at, each
    resample is fitted by ``direct_refit``, and the gap on a resample is a
    difference of two pinned ratings. Returns ``{(i, j): (low, high)}`` for
    every pair of models in sorted order, on rating i minus rating j.

    The resamples themselves are the same ones the package draws. Every item
    in these fixtures carries one comparison, so resampling items is
    resampling rows, and the draw follows the house convention for bootstrap
    index matrices, ``rng.integers(0, n, size=(n_boot, n))``. That makes this
    an exact check rather than one within Monte Carlo error. If the package
    ever changes how it consumes its generator, this fails first, and the fix
    belongs in the draw here and not in the tolerance.
    """
    models = all_models(comparisons)
    index = {name: k for k, name in enumerate(models)}
    m = len(models)
    a = comparisons["model_a"].map(index).to_numpy()
    b = comparisons["model_b"].map(index).to_numpy()
    tie = np.array([is_tie(w) for w in comparisons["winner"]])
    a_won = (comparisons["winner"] == comparisons["model_a"]).to_numpy()
    credit_a = np.where(tie, 0.5, a_won.astype(float))
    n_rows = len(comparisons)

    draws = np.random.default_rng(seed).integers(
        0, n_rows, size=(n_boot, n_rows)
    )
    fits = []
    for rows in draws:
        credit = np.zeros((m, m))
        np.add.at(credit, (a[rows], b[rows]), credit_a[rows])
        np.add.at(credit, (b[rows], a[rows]), 1.0 - credit_a[rows])
        fits.append(direct_refit(credit, 0))
    fits = np.array(fits)

    tail = 100 * (1 - confidence) / 2
    out = {}
    for i in range(m):
        for j in range(i + 1, m):
            gap = fits[:, i] - fits[:, j]
            out[(models[i], models[j])] = (
                float(np.percentile(gap, tail)),
                float(np.percentile(gap, 100 - tail)),
            )
    return out


def oriented(gaps, top, bottom):
    """The reference interval on top minus bottom, whichever way it is keyed."""
    if (top, bottom) in gaps:
        return gaps[(top, bottom)]
    low, high = gaps[(bottom, top)]
    return -high, -low


def clear_of_zero(gaps):
    """The pairs whose reference interval excludes zero, as frozensets."""
    return {
        frozenset(pair) for pair, (low, high) in gaps.items()
        if low > 0 or high < 0
    }


def listed(result):
    return {
        frozenset((row["model_a"], row["model_b"]))
        for _, row in result.separable_pairs.iterrows()
    }


def overlapping_rating_intervals_rule(result):
    """The pairs the old definition called separable. Used only to show a
    fixture tells the two definitions apart."""
    lo = dict(zip(result.ratings["model"], result.ratings["ci_low"]))
    hi = dict(zip(result.ratings["model"], result.ratings["ci_high"]))
    models = list(result.ratings["model"])
    return {
        frozenset((x, y))
        for i, x in enumerate(models) for y in models[i + 1:]
        if hi[x] < lo[y] or hi[y] < lo[x]
    }


def test_separable_pairs_are_exactly_the_gaps_a_refit_puts_clear_of_zero():
    """The definition, against a bootstrap that shares nothing with the
    package but the draws.

    The fixture also has to tell the new definition from the old one, or a
    return to non-overlapping rating intervals would pass. On this data the
    two disagree, and the last assertion says so.
    """
    r = bradley_terry(baseball(), n_boot=400, seed=0)
    assert r.n_boot_usable == 400
    gaps = refit_bootstrap_gaps(baseball(), n_boot=400, seed=0)

    expected = clear_of_zero(gaps)
    assert listed(r) == expected
    assert r.n_separable == len(expected)
    assert 0 < len(expected) < r.n_pairs
    assert len(r.pairs) == len(gaps)
    for _, row in r.pairs.iterrows():
        low, high = oriented(gaps, row["model_a"], row["model_b"])
        assert row["ci_low"] == pytest.approx(low, abs=1e-6)
        assert row["ci_high"] == pytest.approx(high, abs=1e-6)

    assert overlapping_rating_intervals_rule(r) != expected


def small_trio():
    """Three models, twenty comparisons per pair. a takes 12 of 20 from b, b
    takes 12 of 20 from c, and a takes 16 of 20 from c."""
    return frame(
        [("a", "b", "a")] * 12 + [("a", "b", "b")] * 8
        + [("b", "c", "b")] * 12 + [("b", "c", "c")] * 8
        + [("a", "c", "a")] * 16 + [("a", "c", "c")] * 4
    )


def test_gap_intervals_match_a_direct_refit_on_a_small_case():
    """Sixty comparisons, three hundred resamples, every one refitted by
    BFGS. a over c clears zero and the two adjacent pairs do not. All three
    intervals have to be the reference's to six places, the two that
    include zero as well as the one that does not."""
    r = bradley_terry(small_trio(), n_boot=300, seed=2)
    assert r.n_boot_usable == 300
    gaps = refit_bootstrap_gaps(small_trio(), n_boot=300, seed=2)

    assert listed(r) == clear_of_zero(gaps) == {frozenset(("a", "c"))}
    assert len(r.pairs) == 3
    for _, row in r.pairs.iterrows():
        low, high = oriented(gaps, row["model_a"], row["model_b"])
        assert row["ci_low"] == pytest.approx(low, abs=1e-6)
        assert row["ci_high"] == pytest.approx(high, abs=1e-6)


@pytest.mark.parametrize("resample", ["items", "comparisons"])
def test_two_model_gap_interval_is_the_binomial_quantile(resample):
    """With two models the gap is log(W / (n - W)) in closed form, where W
    is a's wins. Resampling n comparisons, or n items holding one each, makes
    W binomial at the observed rate. The logit is monotone, so the
    percentiles of the gap are the logits of the binomial quantiles.

    Resamples with W at 0 or n have no maximum and are dropped, so the
    quantiles are of the binomial with both ends cut off. At 22 of 26 that
    removes about 1.3% of them, which this exercises.

    22 of 26 is chosen so both quantiles sit well inside one step of that
    binomial's CDF. The nearest step edge is 4.9 standard errors of an
    empirical quantile away at 4000 resamples, so each percentile lands on
    the lattice point itself and the comparison is exact. The design is
    lopsided on purpose. At 8 of 16 the interval is symmetric about zero and
    a gap reported the wrong way round would pass.
    """
    n, w = 26, 22
    data = frame([("a", "b", "a")] * w + [("a", "b", "b")] * (n - w))
    r = bradley_terry(data, n_boot=4000, seed=3, resample=resample)

    k = np.arange(1, n)
    pmf = stats.binom.pmf(k, n, w / n)
    cdf = np.cumsum(pmf / pmf.sum())
    k_low = int(k[np.searchsorted(cdf, 0.025)])
    k_high = int(k[np.searchsorted(cdf, 0.975)])

    assert r.n_boot_usable / r.n_boot == pytest.approx(
        1 - stats.binom.pmf(n, n, w / n) - stats.binom.pmf(0, n, w / n),
        abs=0.005,
    )
    row = r.separable_pairs.set_index(["model_a", "model_b"]).loc[("a", "b")]
    assert row["difference"] == pytest.approx(math.log(w / (n - w)), abs=1e-8)
    assert row["ci_low"] == pytest.approx(
        math.log(k_low / (n - k_low)), abs=1e-9
    )
    assert row["ci_high"] == pytest.approx(
        math.log(k_high / (n - k_high)), abs=1e-9
    )


def test_two_model_gap_interval_agrees_with_woolfs_interval():
    """Woolf's closed-form interval on a log-odds, log(w / l) plus or minus
    z * sqrt(1/w + 1/l), is what the percentile interval approaches as the
    comparisons grow.

    At 1000 comparisons with 700 wins Woolf's half-width is 0.135. Over
    forty seeds at 2000 resamples each end of the bootstrap interval sat
    within 0.012 of Woolf's, with a spread of 0.004. At 3000 resamples the
    spread is smaller again, so 0.02 is about six spreads. This is here to
    catch a wrong scale, a factor like sqrt(2) or a z for the wrong level,
    which moves an end by 0.05 or more. The binomial test above is the exact
    one.
    """
    n, w = 1000, 700
    data = frame([("a", "b", "a")] * w + [("a", "b", "b")] * (n - w))
    r = bradley_terry(data, n_boot=3000, seed=4)
    row = r.separable_pairs.set_index(["model_a", "model_b"]).loc[("a", "b")]

    centre = math.log(w / (n - w))
    half = stats.norm.ppf(0.975) * math.sqrt(1 / w + 1 / (n - w))
    assert row["ci_low"] == pytest.approx(centre - half, abs=0.02)
    assert row["ci_high"] == pytest.approx(centre + half, abs=0.02)


def test_a_pair_whose_rating_intervals_overlap_can_still_separate():
    """The case the old definition got wrong.

    a and b met 300 times and a took 180 of them. Each met c only eight
    times. The gap between a and b is measured well, with an interval near
    0.18 to 0.63. Each rating on its own is measured against the average of
    the field, and that average moves with c, which is barely measured. So
    both rating intervals are wide, and they overlap.

    Overlapping rating intervals establish nothing about the gap. Reading
    them as a pair that cannot be ordered was the old error, and a return to
    that rule fails here.
    """
    data = frame(
        [("a", "b", "a")] * 180 + [("a", "b", "b")] * 120
        + [("a", "c", "a")] * 4 + [("a", "c", "c")] * 4
        + [("b", "c", "b")] * 4 + [("b", "c", "c")] * 4
    )
    r = bradley_terry(data, n_boot=1000, seed=0)
    table = r.ratings.set_index("model")
    assert table.loc["b", "ci_high"] > table.loc["a", "ci_low"], (
        "the fixture no longer has overlapping rating intervals for a and b"
    )

    pairs = r.separable_pairs.set_index(["model_a", "model_b"])
    assert ("a", "b") in pairs.index
    row = pairs.loc[("a", "b")]
    assert 0 < row["ci_low"] < row["difference"] < row["ci_high"]


def test_every_separable_pair_carries_a_gap_interval_clear_of_zero():
    """Higher-rated model first, so every interval in the table sits above
    zero and brackets the gap it is on."""
    r = bradley_terry(baseball(), n_boot=600, seed=1)
    assert not r.separable_pairs.empty
    for _, row in r.separable_pairs.iterrows():
        assert 0 < row["ci_low"] <= row["difference"] <= row["ci_high"]


def test_separable_pairs_names_the_higher_rated_model_first():
    r = bradley_terry(baseball(), n_boot=600, seed=1)
    got = ratings_dict(r)
    for _, row in r.separable_pairs.iterrows():
        assert got[row["model_a"]] > got[row["model_b"]]
        assert row["difference"] > 0


def test_separable_pairs_reports_the_gap_and_the_head_to_head_count():
    r = bradley_terry(baseball(), n_boot=600, seed=1)
    got = ratings_dict(r)
    for _, row in r.separable_pairs.iterrows():
        a, b = row["model_a"], row["model_b"]
        assert row["difference"] == pytest.approx(got[a] - got[b], abs=1e-12)
        assert row["p_a_beats_b"] == pytest.approx(
            r.win_matrix.loc[a, b], abs=1e-12
        )
        assert row["n_head_to_head"] == 13


def test_separable_pairs_sorted_by_gap_descending():
    gaps = list(
        bradley_terry(baseball(), n_boot=600, seed=1).separable_pairs["difference"]
    )
    assert gaps == sorted(gaps, reverse=True)


def test_thin_data_separates_nothing():
    r = bradley_terry(thin_four(), n_boot=400, seed=0)
    assert r.has_interval is True
    assert r.n_separable == 0
    assert r.separable_pairs.empty


def test_wide_gaps_and_plenty_of_data_separate_everything():
    data = simulate(
        {"a": 3.0, "b": 1.0, "c": -1.0, "d": -3.0}, n_per_pair=600, seed=8
    )
    r = bradley_terry(data, n_boot=400, seed=0)
    assert r.n_separable == r.n_pairs == 6


def test_a_six_model_leaderboard_reports_how_many_pairs_are_distinguishable():
    """The finding. Fifteen pairs, and most of them cannot be ordered."""
    truth = {"a": 0.55, "b": 0.35, "c": 0.15, "d": -0.05, "e": -0.3, "f": -0.7}
    r = bradley_terry(simulate(truth, n_per_pair=40, seed=12), n_boot=600, seed=0)
    assert r.n_pairs == 15
    assert 0 < r.n_separable < 15
    assert f"{r.n_separable} of 15 pairs" in r.summary()
    assert "separable" in r.summary().lower()


def test_summary_says_when_nothing_separates():
    r = bradley_terry(thin_four(), n_boot=400, seed=0)
    assert "0 of 6 pairs" in r.summary()


# What separable means, and what not separating means, as the summary puts
# them. Copied out by hand. The old summary said pairs that did not separate
# "cannot be ordered", and that a board ranking them reports "an order the
# data does not carry". Both say the data rules an order out. A gap interval
# that includes zero rules nothing out. What the data fails to do is
# establish the order, so the summary says that and says the models are not
# thereby shown level.
SEPARABLE_MEANS = (
    "meaning the interval on the gap between the two ratings excludes zero."
)
NOT_SHOWN_LEVEL = "That does not mean the models are level."
OLD_OVERREACH = ("do not overlap", "cannot be ordered", "does not carry", "noise")


def test_summary_says_separable_means_the_gap_interval_excludes_zero():
    truth = {"a": 0.55, "b": 0.35, "c": 0.15, "d": -0.05, "e": -0.3, "f": -0.7}
    r = bradley_terry(simulate(truth, n_per_pair=40, seed=12), n_boot=600, seed=0)
    text = r.summary()
    assert f"{r.n_separable} of 15 pairs are separable at 95%, {SEPARABLE_MEANS}" in text
    for phrase in OLD_OVERREACH:
        assert phrase not in text


def test_summary_says_what_the_unseparated_pairs_lack():
    r = bradley_terry(baseball(), n_boot=600, seed=1)
    rest = r.n_pairs - r.n_separable
    assert rest > 1
    text = r.summary()
    assert (
        f"For the other {rest} pairs the interval includes zero, so this data "
        f"does not establish an order for those pairs in either direction. "
        f"{NOT_SHOWN_LEVEL} More comparisons could separate them. A "
        f"leaderboard that puts those models in a line is showing an order "
        f"this data has not established."
    ) in text
    for phrase in OLD_OVERREACH:
        assert phrase not in text


def test_summary_for_one_unseparated_pair_is_singular():
    """Two wide gaps and one narrow one. The narrow pair is the only one
    left, and the sentence has to be about that pair."""
    data = simulate({"a": 2.0, "b": 0.0, "c": -0.05}, n_per_pair=200, seed=3)
    r = bradley_terry(data, n_boot=600, seed=0)
    assert r.n_pairs - r.n_separable == 1, "the fixture stopped leaving one pair"
    assert (
        "For the other 1 pair the interval includes zero, so this data does "
        "not establish an order for that pair in either direction."
    ) in r.summary()


def test_summary_when_nothing_separates_does_not_call_the_order_noise():
    """The old sentence called a ranking on this data "a ranking of noise".
    The data does not show the order is noise. It fails to establish it."""
    text = bradley_terry(thin_four(), n_boot=400, seed=0).summary()
    assert (
        "0 of 6 pairs are separable at 95%, " + SEPARABLE_MEANS
        + " This data does not establish an order for any pair. A ranking "
        "built on it puts the models in a line this data has not established. "
        + NOT_SHOWN_LEVEL + " More comparisons could separate them."
    ) in text
    for phrase in OLD_OVERREACH:
        assert phrase not in text


def test_more_comparisons_separate_more_pairs():
    truth = {"a": 0.6, "b": 0.3, "c": 0.0, "d": -0.3, "e": -0.6}
    thin = bradley_terry(
        simulate(truth, n_per_pair=25, seed=13), n_boot=500, seed=0
    )
    thick = bradley_terry(
        simulate(truth, n_per_pair=400, seed=13), n_boot=500, seed=0
    )
    assert thick.n_separable > thin.n_separable


def test_no_pair_is_separable_without_an_interval():
    r = bradley_terry(baseball(), n_boot=0)
    assert r.has_interval is False
    assert r.n_separable == 0
    assert r.separable_pairs.empty
    assert "no interval" in r.summary().lower()


# --------------------------------------------------------------------------
# The reference model
# --------------------------------------------------------------------------

def test_the_reference_model_reads_zero():
    r = bradley_terry(baseball(), n_boot=0, reference="qwen")
    assert r.reference == "qwen"
    assert ratings_dict(r)["qwen"] == 0.0


def test_default_reference_is_the_most_compared_model():
    data = frame(
        [("a", "b", "a")] * 3
        + [("b", "c", "b")] * 3
        + [("c", "a", "c")] * 3
        + [("b", "c", "c")] * 20
    )
    assert bradley_terry(data, n_boot=0).reference == "b"


def test_default_reference_breaks_ties_alphabetically():
    data = frame([("z", "a", "z")] * 5 + [("a", "z", "a")] * 5)
    assert bradley_terry(data, n_boot=0).reference == "a"


def test_changing_the_reference_shifts_every_rating_by_one_constant():
    first = ratings_dict(bradley_terry(baseball(), n_boot=0, reference="olmo"))
    second = ratings_dict(bradley_terry(baseball(), n_boot=0, reference="gpt"))
    shifts = [first[m] - second[m] for m in BASEBALL_NAMES]
    for s in shifts:
        assert s == pytest.approx(shifts[0], abs=1e-12)
    assert shifts[0] != 0.0


def test_changing_the_reference_leaves_the_win_matrix_alone():
    first = bradley_terry(baseball(), n_boot=0, reference="olmo")
    second = bradley_terry(baseball(), n_boot=0, reference="gpt")
    pd.testing.assert_frame_equal(first.win_matrix, second.win_matrix, atol=1e-12)


def test_changing_the_reference_leaves_separability_alone():
    first = bradley_terry(baseball(), n_boot=600, seed=1, reference="olmo")
    second = bradley_terry(baseball(), n_boot=600, seed=1, reference="gpt")
    assert first.n_separable == second.n_separable > 0
    pd.testing.assert_frame_equal(
        first.separable_pairs, second.separable_pairs, atol=1e-12
    )


def test_changing_the_reference_leaves_every_gap_interval_alone():
    """A gap is a difference of two ratings, so the constant a new reference
    adds to both cancels on every resample. The interval on it cannot move,
    for the pairs that separate and for the ones that do not."""
    first = bradley_terry(baseball(), n_boot=600, seed=1, reference="olmo")
    second = bradley_terry(baseball(), n_boot=600, seed=1, reference="gpt")
    assert len(first.pairs) == 21
    pd.testing.assert_frame_equal(first.pairs, second.pairs, atol=1e-12)


def test_changing_the_reference_shifts_the_intervals_by_the_same_constant():
    """Otherwise the separable count would turn on an arbitrary choice."""
    a = bradley_terry(
        baseball(), n_boot=600, seed=1, reference="olmo"
    ).ratings.set_index("model")
    b = bradley_terry(
        baseball(), n_boot=600, seed=1, reference="gpt"
    ).ratings.set_index("model")
    shift = a.loc["gpt", "rating"] - b.loc["gpt", "rating"]
    for m in BASEBALL_NAMES:
        for column in ("rating", "ci_low", "ci_high"):
            assert a.loc[m, column] - b.loc[m, column] == pytest.approx(
                shift, abs=1e-12
            )


def test_the_reference_model_still_gets_an_interval():
    """Its rating is fixed by convention. Its standing is not known exactly."""
    row = bradley_terry(
        baseball(), n_boot=600, seed=1, reference="olmo"
    ).ratings.set_index("model").loc["olmo"]
    assert row["rating"] == 0.0
    assert row["ci_low"] < 0.0 < row["ci_high"]


def test_summary_says_the_ratings_are_only_differences():
    text = bradley_terry(
        baseball(), n_boot=200, seed=0, reference="olmo"
    ).summary().lower()
    assert "olmo" in text
    assert "difference" in text


def test_docstring_says_the_ratings_are_only_differences():
    assert "difference" in bradley_terry.__doc__.lower()


def test_unknown_reference_is_refused():
    with pytest.raises(ValueError, match="reference"):
        bradley_terry(baseball(), reference="claude", n_boot=0)


# --------------------------------------------------------------------------
# Ties
# --------------------------------------------------------------------------

def test_split_gives_half_credit_to_each_side():
    """Two wins each way plus four ties is a dead heat under split."""
    data = frame(
        [("a", "b", "tie")] * 4
        + [("a", "b", "a")] * 2
        + [("a", "b", "b")] * 2
    )
    r = bradley_terry(data, ties="split", n_boot=0)
    assert list(r.ratings["rating"]) == pytest.approx([0.0, 0.0], abs=1e-9)


def test_drop_discards_ties():
    with_ties = frame(
        [("a", "b", "a")] * 20
        + [("a", "b", "b")] * 10
        + [("a", "b", "tie")] * 40
    )
    without = frame([("a", "b", "a")] * 20 + [("a", "b", "b")] * 10)
    dropped = bradley_terry(with_ties, ties="drop", n_boot=0, reference="b")
    clean = bradley_terry(without, ties="drop", n_boot=0, reference="b")
    assert ratings_dict(dropped)["a"] == pytest.approx(
        ratings_dict(clean)["a"], abs=1e-9
    )


def test_the_tie_policy_materially_changes_the_ratings():
    """Not a cosmetic switch. Splitting ties pulls every model toward level.

    The bar is the sampling error rather than a chosen constant. With 60% of
    the comparisons drawn as ties, the two policies have to put the top
    model further apart than the width of both intervals put together,
    which is to say further apart than the data can explain as noise.
    """
    truth = {"a": 1.5, "b": 0.5, "c": -0.5, "d": -1.5}
    data = simulate(truth, n_per_pair=120, tie_rate=0.6, seed=17)
    split = bradley_terry(data, ties="split", n_boot=400, seed=0, reference="d")
    drop = bradley_terry(data, ties="drop", n_boot=400, seed=0, reference="d")

    s = split.ratings.set_index("model").loc["a"]
    d = drop.ratings.set_index("model").loc["a"]
    spans = (s["ci_high"] - s["ci_low"]) + (d["ci_high"] - d["ci_low"])
    assert abs(d["rating"] - s["rating"]) > spans


def test_split_ratings_are_closer_to_level_than_dropped_ones():
    truth = {"a": 1.5, "b": 0.5, "c": -0.5, "d": -1.5}
    data = simulate(truth, n_per_pair=120, tie_rate=0.6, seed=17)
    split = bradley_terry(data, ties="split", n_boot=0)
    drop = bradley_terry(data, ties="drop", n_boot=0)
    assert np.ptp(split.ratings["rating"].to_numpy()) < np.ptp(
        drop.ratings["rating"].to_numpy()
    )


def test_ties_reported_on_the_result():
    data = simulate(
        {"a": 0.5, "b": 0.0, "c": -0.5}, n_per_pair=30, tie_rate=0.4, seed=6
    )
    assert bradley_terry(data, ties="split", n_boot=0).ties == "split"
    assert bradley_terry(data, ties="drop", n_boot=0).ties == "drop"


def test_summary_names_the_tie_policy():
    data = simulate(
        {"a": 0.5, "b": 0.0, "c": -0.5}, n_per_pair=30, tie_rate=0.4, seed=6
    )
    assert "split" in bradley_terry(data, n_boot=100, seed=0).summary().lower()
    assert "drop" in bradley_terry(
        data, ties="drop", n_boot=100, seed=0
    ).summary().lower()


def test_a_missing_winner_counts_as_a_tie():
    """Spreadsheets record a tie by leaving the cell empty."""
    literal = frame([("a", "b", "tie")] * 4 + [("a", "b", "a")] * 6)
    blank = frame([("a", "b", None)] * 4 + [("a", "b", "a")] * 6)
    first = bradley_terry(literal, n_boot=0, reference="b")
    second = bradley_terry(blank, n_boot=0, reference="b")
    assert first.n_ties == second.n_ties == 4
    assert ratings_dict(first)["a"] == pytest.approx(
        ratings_dict(second)["a"], abs=1e-12
    )


def test_tie_is_matched_without_regard_to_case():
    data = frame([("a", "b", "TIE")] * 4 + [("a", "b", "a")] * 6)
    assert bradley_terry(data, n_boot=0).n_ties == 4


def test_davidson_is_named_as_the_principled_treatment_and_not_built():
    with pytest.raises(NotImplementedError, match="Davidson"):
        bradley_terry(baseball(), ties="davidson", n_boot=0)


def test_the_docstring_points_at_davidson():
    assert "davidson" in bradley_terry.__doc__.lower()


def test_unknown_tie_policy_is_refused():
    with pytest.raises(ValueError, match="ties"):
        bradley_terry(baseball(), ties="ignore", n_boot=0)


def test_dropping_every_row_is_refused():
    with pytest.raises(ValueError, match="tie"):
        bradley_terry(frame([("a", "b", "tie")] * 10), ties="drop", n_boot=0)


# --------------------------------------------------------------------------
# Disconnected comparison graphs
# --------------------------------------------------------------------------

def test_disconnected_graph_yields_no_ratings():
    r = bradley_terry(two_islands(), n_boot=200, seed=0)
    assert r.connected is False
    assert r.has_fit is False
    assert r.ratings["rating"].isna().all()
    assert r.ratings["ci_low"].isna().all()
    assert r.ratings["ci_high"].isna().all()


def test_disconnected_graph_yields_no_win_matrix_and_no_pairs():
    r = bradley_terry(two_islands(), n_boot=200, seed=0)
    assert r.win_matrix.isna().to_numpy().all()
    assert r.separable_pairs.empty
    assert r.n_separable == 0


def test_disconnected_graph_names_the_groups():
    r = bradley_terry(two_islands(), n_boot=0)
    groups = {frozenset(g) for g in r.comparable_groups}
    assert groups == {
        frozenset({"alpha", "beta"}), frozenset({"gamma", "delta"})
    }


def test_disconnected_summary_refuses_to_rank():
    text = bradley_terry(two_islands(), n_boot=200, seed=0).summary().lower()
    assert "never met" in text
    for m in ("alpha", "beta", "gamma", "delta"):
        assert m in text
    assert "rank" in text


def test_disconnected_graph_still_reports_the_comparison_counts():
    r = bradley_terry(two_islands(), n_boot=0)
    counts = dict(zip(r.ratings["model"], r.ratings["n_comparisons"]))
    assert counts == {"alpha": 8, "beta": 8, "gamma": 8, "delta": 8}


def test_a_connected_graph_has_one_group():
    r = bradley_terry(baseball(), n_boot=0)
    assert r.connected is True
    assert len(r.comparable_groups) == 1
    assert set(r.comparable_groups[0]) == set(BASEBALL_NAMES)


def test_three_islands_are_all_named():
    data = frame(
        [
            ("alpha", "beta", "alpha"), ("beta", "alpha", "beta"),
            ("gamma", "delta", "gamma"), ("delta", "gamma", "delta"),
            ("epsilon", "zeta", "epsilon"), ("zeta", "epsilon", "zeta"),
        ]
    )
    assert len(bradley_terry(data, n_boot=0).comparable_groups) == 3


def test_no_resamples_are_drawn_when_there_is_no_fit():
    r = bradley_terry(two_islands(), n_boot=500, seed=0)
    assert r.n_boot_usable == 0
    assert r.has_interval is False


# --------------------------------------------------------------------------
# Connected, and still no maximum. Ford's condition.
# --------------------------------------------------------------------------

def test_an_undefeated_model_has_no_finite_rating():
    r = bradley_terry(undefeated_alpha(), n_boot=200, seed=0)
    assert r.connected is True
    assert r.has_fit is False
    assert r.undefeated == ("alpha",)
    assert r.ratings["rating"].isna().all()


def test_an_undefeated_model_is_named_in_the_summary():
    text = bradley_terry(undefeated_alpha(), n_boot=0).summary().lower()
    assert "never lost" in text
    assert "alpha" in text


def test_a_winless_model_has_no_finite_rating():
    r = bradley_terry(winless_gamma(), n_boot=0)
    assert r.connected is True
    assert r.has_fit is False
    assert r.winless == ("gamma",)
    text = r.summary().lower()
    assert "never won" in text
    assert "gamma" in text


def test_a_perfectly_transitive_round_robin_has_no_maximum():
    """The case people hit first. A clean sweep is not a rating."""
    order = ["a", "b", "c", "d"]
    rows = []
    for i, better in enumerate(order):
        for worse in order[i + 1:]:
            rows += [(better, worse, better)] * 3
    assert bradley_terry(frame(rows), n_boot=0).has_fit is False


def test_a_split_tie_can_restore_the_maximum():
    """Half credit runs both ways, so a tie connects the graph both ways."""
    data = frame(
        list(zip(
            undefeated_alpha()["model_a"],
            undefeated_alpha()["model_b"],
            undefeated_alpha()["winner"],
        ))
        + [("alpha", "beta", "tie")] * 2
    )
    assert bradley_terry(data, ties="split", n_boot=0).has_fit is True
    assert bradley_terry(data, ties="drop", n_boot=0).has_fit is False


def test_a_healthy_fit_reports_no_separation():
    r = bradley_terry(baseball(), n_boot=0)
    assert r.has_fit is True
    assert r.undefeated == ()
    assert r.winless == ()


# --------------------------------------------------------------------------
# The bootstrap
# --------------------------------------------------------------------------

def test_bootstrap_is_deterministic_under_a_seed():
    first = bradley_terry(baseball(), n_boot=200, seed=42)
    second = bradley_terry(baseball(), n_boot=200, seed=42)
    pd.testing.assert_frame_equal(first.ratings, second.ratings)


def test_different_seeds_give_different_intervals():
    first = bradley_terry(baseball(), n_boot=200, seed=1)
    second = bradley_terry(baseball(), n_boot=200, seed=2)
    assert not np.allclose(
        first.ratings["ci_low"].to_numpy(), second.ratings["ci_low"].to_numpy()
    )


def test_bootstrap_brackets_the_estimate():
    r = bradley_terry(baseball(), n_boot=500, seed=0)
    assert (r.ratings["ci_low"] <= r.ratings["rating"]).all()
    assert (r.ratings["rating"] <= r.ratings["ci_high"]).all()


def test_intervals_narrow_with_more_comparisons():
    truth = {"a": 1.0, "b": 0.0, "c": -1.0}
    thin = bradley_terry(
        simulate(truth, n_per_pair=30, seed=21), n_boot=400, seed=0
    )
    thick = bradley_terry(
        simulate(truth, n_per_pair=600, seed=21), n_boot=400, seed=0
    )
    thin_span = (thin.ratings["ci_high"] - thin.ratings["ci_low"]).mean()
    thick_span = (thick.ratings["ci_high"] - thick.ratings["ci_low"]).mean()
    assert thick_span < thin_span / 2


def test_confidence_level_widens_the_interval():
    narrow = bradley_terry(baseball(), n_boot=800, seed=0, confidence=0.80)
    wide = bradley_terry(baseball(), n_boot=800, seed=0, confidence=0.99)
    n = (narrow.ratings["ci_high"] - narrow.ratings["ci_low"]).mean()
    w = (wide.ratings["ci_high"] - wide.ratings["ci_low"]).mean()
    assert w > n
    assert wide.confidence == 0.99


def test_every_model_gets_a_positive_width():
    data = frame([("a", "b", "a")] * 20 + [("a", "b", "b")] * 20)
    r = bradley_terry(data, n_boot=300, seed=0)
    assert ((r.ratings["ci_high"] - r.ratings["ci_low"]) > 0).all()


def test_n_boot_zero_means_no_interval():
    r = bradley_terry(baseball(), n_boot=0)
    assert r.n_boot == 0
    assert r.n_boot_usable == 0
    assert r.ratings["ci_low"].isna().all()
    assert r.ratings["ci_high"].isna().all()
    assert r.has_interval is False


def test_an_interval_never_asked_for_reads_differently_from_a_refused_one():
    never = bradley_terry(baseball(), n_boot=0).summary().lower()
    refused = bradley_terry(tiny_cycle(), n_boot=500, seed=0).summary().lower()
    assert "no interval" in never and "requested" in never
    assert "no interval" in refused and "resample" in refused
    assert "requested" not in refused


def test_undefined_resamples_are_counted_and_the_interval_refused():
    """Three comparisons in a ring. A resample missing one has no maximum.

    All three rows have to be drawn for the ring to close, which happens on
    6 of the 27 equally likely draws, so about a fifth of the resamples
    survive. That is far below the share this reports through.
    """
    r = bradley_terry(tiny_cycle(), n_boot=2000, seed=0)
    assert r.has_fit is True
    assert r.n_boot == 2000
    assert r.n_boot_usable / r.n_boot == pytest.approx(6 / 27, abs=0.05)
    assert r.has_interval is False
    assert r.ratings["ci_low"].isna().all()
    assert r.n_separable == 0


def test_a_thin_share_of_undefined_resamples_passes_through():
    """The same ring with three comparisons per edge.

    An edge goes missing from a resample with probability (2/3) ** 9 and
    there are three edges, so about 92% of resamples survive. That clears
    the 90% floor, the interval is reported, and the shortfall is still
    recorded.
    """
    data = frame(
        [("a", "b", "a")] * 3
        + [("b", "c", "b")] * 3
        + [("c", "a", "c")] * 3
    )
    r = bradley_terry(data, n_boot=4000, seed=0)
    share = r.n_boot_usable / r.n_boot
    assert share == pytest.approx(0.922, abs=0.02)
    assert MIN_USABLE_SHARE < share < 1.0
    assert r.has_interval is True
    assert r.ratings["ci_low"].notna().all()


def test_healthy_data_uses_every_resample():
    r = bradley_terry(baseball(), n_boot=300, seed=0)
    assert r.n_boot_usable == 300
    assert r.has_interval is True


def test_a_refused_interval_says_how_many_resamples_failed():
    r = bradley_terry(tiny_cycle(), n_boot=2000, seed=0)
    assert "2000" in r.summary()
    assert str(r.n_boot - r.n_boot_usable) in r.summary()


def test_negative_n_boot_is_refused():
    with pytest.raises(ValueError, match="n_boot"):
        bradley_terry(baseball(), n_boot=-1)


def test_confidence_outside_the_unit_interval_is_refused():
    with pytest.raises(ValueError, match="confidence"):
        bradley_terry(baseball(), confidence=1.5, n_boot=0)


def test_bootstrap_covers_at_the_claimed_rate():
    """Simulation, against the quantity the interval is actually about.

    Each interval is on a rating measured against the average of the field,
    displayed against the reference. So the thing that should land inside it
    95% of the time is the true rating less the true average, moved onto the
    reported scale by the mean of the reported ratings.

    The band is wide on purpose and the low end is the reason. A percentile
    bootstrap undercovers at this size, and it does so here: about 0.92 at
    120 comparisons per pair, climbing to about 0.94 at 200 and 300, which
    is the finite-sample shortfall behaving as it should rather than a bug.
    Trials are independent and the five ratings inside a trial are not, so
    the standard error is clustered on the trial. At 120 trials it is about
    0.012, which puts 0.88 roughly 3.7 errors below the expected 0.923 and
    0.98 roughly 4.9 above. Both edges clear the noise by a wide margin, and
    an implementation with intervals half the right width lands near 0.70.
    """
    truth = {"a": 1.0, "b": 0.5, "c": 0.0, "d": -0.5, "e": -1.0}
    centre = np.mean(list(truth.values()))
    covered = total = 0
    for k in range(120):
        data = simulate(truth, n_per_pair=120, seed=7000 + k)
        r = bradley_terry(data, n_boot=200, seed=k, reference="c")
        table = r.ratings.set_index("model")
        shift = table["rating"].mean()
        for m in truth:
            total += 1
            target = truth[m] - centre + shift
            if table.loc[m, "ci_low"] <= target <= table.loc[m, "ci_high"]:
                covered += 1
    assert 0.88 <= covered / total <= 0.98


# --------------------------------------------------------------------------
# Resampling items
#
# Supplying item_id says which comparisons were made on the same prompt, and
# judgements of one prompt are correlated. Resampling single comparisons
# treats them as independent. The default resamples whole items, which is the
# cluster bootstrap, and the old behaviour is resample="comparisons".
# --------------------------------------------------------------------------

def spread_over_items(comparisons, n_items):
    """The same comparisons, dealt round-robin onto ``n_items`` item ids."""
    out = comparisons.copy()
    out["item_id"] = [f"p{k % n_items}" for k in range(len(out))]
    return out


def beta_binomial_items(n_items, per_item, icc, rate, seed):
    """Two models, and every item has its own rate for a beating b.

    Item rates are drawn from a beta with mean ``rate`` and with alpha plus
    beta equal to 1 / icc - 1, which makes ``icc`` the intraclass correlation
    between two judgements of the same item. Each item then carries
    ``per_item`` judgements at its own rate. Returns the frame and a's wins
    per item.
    """
    rng = np.random.default_rng(seed)
    total = 1.0 / icc - 1.0
    rates = rng.beta(rate * total, (1 - rate) * total, size=n_items)
    wins = rng.binomial(per_item, rates)
    rows = []
    for k, won in enumerate(wins):
        rows += [(f"p{k}", "a", "b", "a")] * int(won)
        rows += [(f"p{k}", "a", "b", "b")] * int(per_item - won)
    data = pd.DataFrame(rows, columns=["item_id", "model_a", "model_b", "winner"])
    return data, wins


def gap_row(result, top="a", bottom="b"):
    pairs = result.pairs.set_index(["model_a", "model_b"])
    assert (top, bottom) in pairs.index, f"{top} is not rated above {bottom}"
    return pairs.loc[(top, bottom)]


def gap_width(result, top="a", bottom="b"):
    row = gap_row(result, top, bottom)
    return row["ci_high"] - row["ci_low"]


def test_the_default_resamples_items():
    assert bradley_terry(baseball(), n_boot=0).resample == "items"
    assert bradley_terry(
        baseball(), n_boot=0, resample="comparisons"
    ).resample == "comparisons"


def test_unknown_resample_is_refused():
    with pytest.raises(ValueError, match="resample"):
        bradley_terry(baseball(), resample="rows", n_boot=0)


def test_the_item_interval_widens_by_the_square_root_of_the_design_effect():
    """How much wider, and why.

    Eighty items with 32 judgements each, which is the shape of MT-Bench's
    human judgements, and an intraclass correlation of 0.1 between two
    judgements of the same item. The design effect for a mean over clusters
    is 1 + (m - 1) * icc, here 1 + 31 * 0.1 = 4.1. Resampling single
    comparisons treats it as 1. The gap between two models is the logit of
    their win rate, so its standard error scales with the square root of
    that factor, and the item interval should come out about
    sqrt(4.1) = 2.02 times as wide as the comparison interval.

    The design effect this sample realised differs from 4.1 by how the
    eighty item rates happened to fall. So the test measures it from the
    data, as the item count times the spread of the item win rates over the
    binomial variance, and holds the width ratio to within 10% of its square
    root. Across thirty seeds of this design the ratio over that square root
    averaged 0.997 with a spread of 0.025, so 10% is four spreads.
    Weighting rows one at a time instead of by item puts the ratio near 1,
    and drawing items without replacement gives the item interval width
    zero.
    """
    per_item = 32
    data, wins = beta_binomial_items(80, per_item, icc=0.1, rate=0.6, seed=1003)
    items = bradley_terry(data, n_boot=2000, seed=0)
    single = bradley_terry(data, n_boot=2000, seed=1, resample="comparisons")

    rate = wins.sum() / (len(wins) * per_item)
    item_rates = wins / per_item
    design_effect = (
        per_item * np.mean((item_rates - rate) ** 2) / (rate * (1 - rate))
    )
    ratio = gap_width(items) / gap_width(single)

    assert ratio > 1.5
    assert ratio == pytest.approx(math.sqrt(design_effect), rel=0.10)


def test_one_comparison_per_item_gives_the_same_interval_either_way():
    """With one comparison on every item, resampling items is resampling
    comparisons, so the two intervals differ by Monte Carlo error alone.

    The two runs take different seeds. Under one seed they would draw the
    same resamples and the check would say nothing.

    The bounds come from measuring that error. Over fifty data sets of this
    shape at 3000 resamples, two runs on different seeds put the average
    relative difference in gap width across the ten pairs at a spread of
    0.010, never past 0.024, and no single pair differed by more than 0.072.
    The test allows 0.05 on the average, about five spreads, and 0.12 on any
    one pair. Drawing items without replacement gives every resample the
    full data and every interval width zero, a relative difference of 1.
    """
    truth = {"a": 1.2, "b": 0.6, "c": 0.0, "d": -0.6, "e": -1.2}
    data = simulate(truth, n_per_pair=150, seed=41)
    items = bradley_terry(data, n_boot=3000, seed=10)
    single = bradley_terry(data, n_boot=3000, seed=11, resample="comparisons")

    def widths(result):
        table = result.pairs.set_index(["model_a", "model_b"])
        return (table["ci_high"] - table["ci_low"]).sort_index()

    relative = widths(items) / widths(single) - 1.0
    assert len(relative) == 10
    assert abs(relative.mean()) < 0.05
    assert relative.abs().max() < 0.12


def clustered_benchmark(theta, n_items, per_item, tau, seed):
    """Comparisons where every item shifts each model's strength.

    Each item draws one shift per model from a normal with spread ``tau`` on
    the log-odds scale, and every comparison made on that item uses those
    shifts. So comparisons sharing an item are correlated, and more so as
    ``tau`` grows. The pair for each comparison is drawn uniformly.
    """
    rng = np.random.default_rng(seed)
    m = len(theta)
    names = np.array([f"m{k}" for k in range(m)])
    pairs = np.array([(i, j) for i in range(m) for j in range(i + 1, m)])
    items = np.repeat(np.arange(n_items), per_item)
    pick = rng.integers(0, len(pairs), n_items * per_item)
    a, b = pairs[pick, 0], pairs[pick, 1]
    shift = rng.normal(0.0, tau, size=(n_items, m))
    strength = theta[a] + shift[items, a] - theta[b] - shift[items, b]
    a_won = rng.random(len(a)) < 1.0 / (1.0 + np.exp(-strength))
    return pd.DataFrame({
        "item_id": [f"p{k}" for k in items],
        "model_a": names[a],
        "model_b": names[b],
        "winner": np.where(a_won, names[a], names[b]),
    })


def fitted_limit_gaps(theta, tau):
    """Where the fit settles on endless data from ``clustered_benchmark``.

    That is not theta. The chance a beats b on a random item is the logistic
    of the true gap plus a normal with variance 2 * tau ** 2, averaged over
    that normal, and the averaging pulls every win rate toward a half. The
    fit converges to the ratings that reproduce the averaged rates. With
    every pair equally likely, those are the fit to a credit matrix of the
    averaged rates, which Gauss-Hermite quadrature computes and
    ``direct_refit`` fits.
    """
    nodes, weights = np.polynomial.hermite_e.hermegauss(80)
    m = len(theta)
    credit = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i != j:
                shift = theta[i] - theta[j] + math.sqrt(2.0) * tau * nodes
                credit[i, j] = np.sum(
                    weights / (1.0 + np.exp(-shift))
                ) / math.sqrt(2.0 * math.pi)
    r = direct_refit(credit, 0)
    names = [f"m{k}" for k in range(m)]
    return {
        (names[i], names[j]): r[i] - r[j]
        for i in range(m) for j in range(m) if i != j
    }


def test_item_gap_intervals_cover_at_the_claimed_rate_on_clustered_data():
    """Simulation, on data where items make comparisons correlated.

    Five models, 60 items, 20 comparisons per item, and every item shifts
    each model by a normal draw with spread 0.7 on the log-odds scale. The
    target is ``fitted_limit_gaps``, where the fit settles on endless data
    from that process.

    Measured over 150 trials on these seeds the item interval covered 0.930,
    with a standard error of 0.009 clustered on the trial, since the ten
    gaps in a trial are not independent. A percentile bootstrap on 60
    clusters runs a little short of 95%, the way the rating coverage test
    above does on its own design. The band is 0.89 to 0.97, about 4.4 errors
    either side. On the same data the comparison bootstrap covered 0.848,
    with intervals about 78% as wide. That shortfall is what the item
    default exists to fix, and it falls outside the band.
    """
    theta = np.array([0.8, 0.4, 0.0, -0.4, -0.8])
    target = fitted_limit_gaps(theta, 0.7)
    covered = total = 0
    for k in range(150):
        data = clustered_benchmark(theta, 60, 20, 0.7, seed=7000 + k)
        r = bradley_terry(data, n_boot=200, seed=k)
        for low, high, a, b in zip(
            r.pairs["ci_low"], r.pairs["ci_high"],
            r.pairs["model_a"], r.pairs["model_b"],
        ):
            total += 1
            if low <= target[(a, b)] <= high:
                covered += 1
    assert total == 150 * 10
    assert 0.89 <= covered / total <= 0.97


def test_copies_of_one_comparison_add_nothing_when_items_are_resampled():
    """The limit of within-item correlation, where every judgement of an item
    is the same judgement.

    Each of 400 items carries nine identical copies of one comparison.
    Resampling items draws the copies together, so it sees 400 comparisons
    with weight nine each, and Bradley-Terry does not move when every credit
    is scaled by the same amount. The item interval on the copies is then the
    comparison interval on the 400 originals. Both draw 400 indices from the
    same seed, so they agree as far as the fit resolves.

    Resampling single comparisons treats the 3600 copies as independent and
    comes out sqrt(9) = 3 times too narrow. Across thirty seeds of this
    design the ratio averaged 2.99 with a spread of 0.08 and stayed inside
    2.82 to 3.15, so the test holds it within 10% of 3. Weighting rows one at
    a time instead of by item makes the item interval the narrow one, a ratio
    of 1.
    """
    won = np.random.default_rng(5).random(400) < 0.62
    originals = pd.DataFrame({
        "item_id": [f"p{k}" for k in range(400)],
        "model_a": "a",
        "model_b": "b",
        "winner": np.where(won, "a", "b"),
    })
    copies = originals.loc[originals.index.repeat(9)].reset_index(drop=True)

    items = bradley_terry(copies, n_boot=2000, seed=7)
    once = bradley_terry(originals, n_boot=2000, seed=7, resample="comparisons")
    single = bradley_terry(copies, n_boot=2000, seed=8, resample="comparisons")

    for column in ("difference", "ci_low", "ci_high"):
        assert gap_row(items)[column] == pytest.approx(
            gap_row(once)[column], abs=1e-8
        )
    assert gap_width(items) / gap_width(single) == pytest.approx(3.0, rel=0.10)


def test_one_item_cannot_carry_an_item_bootstrap():
    """Every comparison was made on the same prompt. Resampling items would
    draw that one item every time, every resample would be the full data,
    and the interval would have width zero. That claims certainty from a
    design with nothing to resample, so it is refused.

    Without resamples the unit does not matter and the fit goes ahead. Asking
    for comparisons to be resampled goes ahead too, since that is a choice
    the caller made with the item id in front of them.
    """
    data = baseball().assign(item_id="only-prompt")
    with pytest.raises(ValueError, match="one item"):
        bradley_terry(data, n_boot=200, seed=0)
    assert bradley_terry(data, n_boot=0).has_fit
    assert bradley_terry(
        data, n_boot=200, seed=0, resample="comparisons"
    ).has_interval


def test_a_missing_item_id_is_refused_by_the_item_bootstrap():
    """A row with no item cannot be kept with the rest of its item, so the
    item bootstrap refuses rather than guess which item it belongs to."""
    data = baseball()
    data.loc[3, "item_id"] = None
    with pytest.raises(ValueError, match="item_id"):
        bradley_terry(data, n_boot=200, seed=0)
    assert bradley_terry(data, n_boot=0).has_fit
    assert bradley_terry(
        data, n_boot=200, seed=0, resample="comparisons"
    ).has_interval


# The sentence the summary adds when comparisons were resampled one at a time
# on data where items carry several. Copied out by hand.
COMPARISONS_RESAMPLED = (
    "The intervals resample single comparisons, and these 273 comparisons "
    "share 26 items."
)


def test_summary_says_when_comparisons_were_resampled_over_shared_items():
    grouped = spread_over_items(baseball(), 26)
    flagged = bradley_terry(grouped, n_boot=300, seed=0, resample="comparisons")
    assert COMPARISONS_RESAMPLED in flagged.summary()
    assert "can run narrow" in flagged.summary()

    by_item = bradley_terry(grouped, n_boot=300, seed=0)
    assert "resample single comparisons" not in by_item.summary()

    one_each = bradley_terry(baseball(), n_boot=300, seed=0, resample="comparisons")
    assert "resample single comparisons" not in one_each.summary()


def test_summary_says_nothing_about_shared_items_when_there_are_no_item_ids():
    """With every item id missing there is nothing to say which comparisons
    share a prompt, and "these 273 comparisons share 0 items" is not a
    sentence anyone should read."""
    data = baseball().assign(item_id=None)
    r = bradley_terry(data, n_boot=300, seed=0, resample="comparisons")
    assert r.has_interval
    assert "resample single comparisons" not in r.summary()
    assert "share 0 items" not in r.summary()


def test_to_elo_carries_the_resampling_unit():
    r = bradley_terry(baseball(), n_boot=0, resample="comparisons")
    assert to_elo(r).resample == "comparisons"
    assert to_elo(bradley_terry(baseball(), n_boot=0)).resample == "items"


# --------------------------------------------------------------------------
# Elo
# --------------------------------------------------------------------------

def test_to_elo_rescales_the_ratings():
    r = bradley_terry(baseball(), n_boot=400, seed=0, reference="olmo")
    a = r.ratings.set_index("model")
    b = to_elo(r).ratings.set_index("model")
    for m in BASEBALL_NAMES:
        assert b.loc[m, "rating"] == pytest.approx(
            1500 + a.loc[m, "rating"] * 400 / LOG10, abs=1e-9
        )


def test_to_elo_carries_the_intervals_through():
    r = bradley_terry(baseball(), n_boot=400, seed=0, reference="olmo")
    a = r.ratings.set_index("model")
    b = to_elo(r).ratings.set_index("model")
    for m in BASEBALL_NAMES:
        for column in ("ci_low", "ci_high"):
            assert b.loc[m, column] == pytest.approx(
                1500 + a.loc[m, column] * 400 / LOG10, abs=1e-9
            )


def test_a_four_hundred_point_elo_gap_is_a_ten_to_one_favourite():
    """The definition of the scale, and the reason 400 is the default."""
    data = simulate({"a": LOG10, "b": 0.0}, n_per_pair=4000, seed=31)
    e = to_elo(bradley_terry(data, n_boot=0, reference="b"))
    gap = e.ratings.set_index("model").loc["a", "rating"] - 1500
    assert gap == pytest.approx(400, abs=40)


def test_to_elo_accepts_another_scale_and_base():
    r = bradley_terry(baseball(), n_boot=200, seed=0, reference="olmo")
    e = to_elo(r, scale=200, base=1000)
    assert e.elo_scale == 200
    assert e.elo_base == 1000
    assert e.ratings.set_index("model").loc["gpt", "rating"] == pytest.approx(
        1000 + r.ratings.set_index("model").loc["gpt", "rating"] * 200 / LOG10,
        abs=1e-9,
    )


def test_the_reference_model_reads_the_base():
    e = to_elo(bradley_terry(baseball(), n_boot=200, seed=0, reference="olmo"))
    assert e.ratings.set_index("model").loc["olmo", "rating"] == 1500


def test_to_elo_leaves_the_win_matrix_alone():
    r = bradley_terry(baseball(), n_boot=200, seed=0)
    pd.testing.assert_frame_equal(to_elo(r).win_matrix, r.win_matrix)


def test_to_elo_leaves_separability_alone():
    r = bradley_terry(baseball(), n_boot=600, seed=1)
    e = to_elo(r)
    assert e.n_separable == r.n_separable > 0
    assert list(e.separable_pairs["model_a"]) == list(r.separable_pairs["model_a"])
    assert list(e.separable_pairs["model_b"]) == list(r.separable_pairs["model_b"])


def test_to_elo_rescales_the_pair_gaps():
    r = bradley_terry(baseball(), n_boot=600, seed=1)
    e = to_elo(r)
    for before, after in zip(
        r.separable_pairs["difference"], e.separable_pairs["difference"]
    ):
        assert after == pytest.approx(before * 400 / LOG10, abs=1e-9)


def test_to_elo_rescales_the_gap_intervals():
    """The interval on a gap is in the gap's units, so it stretches with it.
    A positive factor keeps every interval on the side of zero it was on."""
    r = bradley_terry(baseball(), n_boot=600, seed=1)
    e = to_elo(r)
    for table in ("separable_pairs", "pairs"):
        for column in ("difference", "ci_low", "ci_high"):
            for before, after in zip(
                getattr(r, table)[column], getattr(e, table)[column]
            ):
                assert after == pytest.approx(before * 400 / LOG10, abs=1e-9)
    assert (e.separable_pairs["ci_low"] > 0).all()
    assert list(e.pairs["separable"]) == list(r.pairs["separable"])


def test_to_elo_keeps_the_head_to_head_numbers():
    r = bradley_terry(baseball(), n_boot=600, seed=1)
    e = to_elo(r)
    assert list(e.separable_pairs["p_a_beats_b"]) == pytest.approx(
        list(r.separable_pairs["p_a_beats_b"])
    )
    assert list(e.ratings["n_comparisons"]) == list(r.ratings["n_comparisons"])


def test_natural_ratings_are_not_marked_as_elo():
    r = bradley_terry(baseball(), n_boot=0)
    assert r.elo_scale is None
    assert r.elo_base is None
    assert r.is_elo is False
    assert to_elo(r).is_elo is True


def test_elo_summary_names_the_scale_and_the_anchor():
    text = to_elo(
        bradley_terry(baseball(), n_boot=400, seed=0, reference="olmo")
    ).summary()
    assert "Elo" in text
    assert "400" in text
    assert "olmo" in text
    assert "difference" in text.lower()


def test_elo_summary_still_reports_separability():
    r = bradley_terry(baseball(), n_boot=600, seed=1)
    assert f"{r.n_separable} of {r.n_pairs} pairs" in to_elo(r).summary()


def test_converting_twice_is_refused():
    e = to_elo(bradley_terry(baseball(), n_boot=0))
    with pytest.raises(ValueError, match="Elo"):
        to_elo(e)


def test_a_nonpositive_scale_is_refused():
    with pytest.raises(ValueError, match="scale"):
        to_elo(bradley_terry(baseball(), n_boot=0), scale=0)


def test_to_elo_carries_a_failed_fit_through_without_inventing_numbers():
    e = to_elo(bradley_terry(two_islands(), n_boot=0))
    assert e.has_fit is False
    assert e.ratings["rating"].isna().all()
    assert "never met" in e.summary().lower()


def test_to_elo_wants_a_btresult():
    with pytest.raises(ValueError, match="BTResult"):
        to_elo({"a": 1.0})


# --------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------

def test_comparisons_must_be_a_frame():
    with pytest.raises(ValueError, match="DataFrame"):
        bradley_terry([("a", "b", "a")], n_boot=0)


def test_missing_columns_are_named():
    with pytest.raises(ValueError, match="winner"):
        bradley_terry(baseball().drop(columns=["winner"]), n_boot=0)


def test_empty_frame_is_refused():
    data = pd.DataFrame(columns=["item_id", "model_a", "model_b", "winner"])
    with pytest.raises(ValueError, match="empty"):
        bradley_terry(data, n_boot=0)


def test_a_model_against_itself_is_refused():
    with pytest.raises(ValueError, match="itself"):
        bradley_terry(frame([("a", "a", "a"), ("a", "b", "a")]), n_boot=0)


def test_a_winner_that_was_not_on_offer_is_refused():
    with pytest.raises(ValueError, match="not on offer"):
        bradley_terry(frame([("a", "b", "c"), ("a", "b", "a")]), n_boot=0)


def test_a_model_called_tie_is_refused():
    """The word is reserved, so no model can answer to it."""
    with pytest.raises(ValueError, match="tie"):
        bradley_terry(frame([("tie", "b", "tie")] * 5), n_boot=0)


def test_extra_columns_are_ignored():
    data = baseball()
    data["judge"] = "gpt"
    assert bradley_terry(data, n_boot=0).n_models == 7


def test_the_input_frame_is_not_modified():
    data = baseball()
    before = data.copy()
    bradley_terry(data, n_boot=200, seed=0)
    pd.testing.assert_frame_equal(data, before)


def test_row_order_does_not_change_the_fit():
    data = baseball()
    shuffled = data.sample(frac=1.0, random_state=3).reset_index(drop=True)
    a = bradley_terry(data, n_boot=0, reference="olmo")
    b = bradley_terry(shuffled, n_boot=0, reference="olmo")
    pd.testing.assert_frame_equal(a.ratings, b.ratings, atol=1e-10)
