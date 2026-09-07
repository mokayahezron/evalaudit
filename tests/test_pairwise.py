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
then displayed against whichever model was named as the reference. That is
what makes non-overlap mean the same thing whoever the reference is, and the
coverage test targets that quantity rather than the gap to the reference.
"""

import math

import numpy as np
import pandas as pd
import pytest
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
    assert list(r.separable_pairs.columns) == [
        "model_a", "model_b", "difference", "p_a_beats_b", "n_head_to_head"
    ]


def test_n_pairs_is_every_pair_of_models():
    assert bradley_terry(baseball(), n_boot=200, seed=0).n_pairs == 21


def test_separable_pairs_are_exactly_the_non_overlapping_intervals():
    r = bradley_terry(baseball(), n_boot=600, seed=1)
    lo = dict(zip(r.ratings["model"], r.ratings["ci_low"]))
    hi = dict(zip(r.ratings["model"], r.ratings["ci_high"]))
    expected = set()
    for i, a in enumerate(BASEBALL_NAMES):
        for b in BASEBALL_NAMES[i + 1:]:
            if hi[a] < lo[b] or hi[b] < lo[a]:
                expected.add(frozenset((a, b)))
    got = {
        frozenset((row["model_a"], row["model_b"]))
        for _, row in r.separable_pairs.iterrows()
    }
    assert got == expected
    assert r.n_separable == len(expected)
    assert len(r.separable_pairs) == len(expected)
    assert 0 < len(expected) < r.n_pairs


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
