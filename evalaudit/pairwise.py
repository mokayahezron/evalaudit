"""Which models on your leaderboard are actually in a different position?

A blind pairwise comparison produces a ranking, and the ranking gets
published. Almost nobody publishes what the ranking rests on. Six models make
fifteen pairs, and on the volumes people usually run the data establishes an
order for a few of those pairs. The rest get printed in a line anyway, in an
order the data has not established.

Bradley-Terry is the model underneath every Elo leaderboard. Each model gets
a rating, and the chance one beats another is the logistic of the gap between
their ratings. Fit by maximum likelihood, with one rating pinned at zero
because only differences are identified.

Two things get in the way of a fit, and both show up in real leaderboard
data. Models that split into groups which never played each other have no
common scale at all. And a model that never lost has no finite rating, which
is what a clean sweep against a weak baseline produces. Neither returns
numbers here.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from ._types import _MIN_USABLE_SHARE, BTResult

__all__ = ["bradley_terry", "to_elo"]

_COLUMNS = ("item_id", "model_a", "model_b", "winner")

# The string a tie is recorded as, matched without regard to case. A missing
# winner counts too, because that is how a spreadsheet records one.
_TIE = "tie"

# Tie policies this builds. "davidson" is recognised and refused on purpose,
# so asking for it gets an explanation rather than a list of valid options
# that quietly omits the treatment a reader came looking for.
_TIE_POLICIES = ("split", "drop")
_TIE_HOOK = "davidson"

# What one bootstrap draw picks up. See bradley_terry for why items is first.
_RESAMPLE_UNITS = ("items", "comparisons")

# Newton on a concave log-likelihood from a sensible start converges in a few
# steps. The cap is here so a resample that comes close to leaving a model
# unbeaten cannot spin.
_MAX_NEWTON = 100
_NEWTON_TOL = 1e-11

# Ceiling on the working array the bootstrap allocates, in floats. Resamples
# run in blocks sized to fit under it. A block is not a resample: every
# resample inside a block is still fitted in the same pass as the others.
_BLOCK_BUDGET = 4_000_000


def bradley_terry(
    comparisons: pd.DataFrame,
    ties: str = "split",
    reference: Optional[str] = None,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: Optional[int] = None,
    resample: str = "items",
) -> BTResult:
    """Fit Bradley-Terry ratings and report which pairs they actually order.

    Parameters
    ----------
    comparisons
        A frame with ``item_id``, ``model_a``, ``model_b`` and ``winner``.
        One row per judgement. ``winner`` names one of the two models, or is
        the string ``"tie"``, or is left empty to mean the same thing.
        ``item_id`` labels the prompt the pair was judged on. The fit does
        not read it. The bootstrap does, since comparisons made on the same
        prompt are resampled together. See ``resample``.
    ties
        ``"split"`` gives half a win to each side. ``"drop"`` discards the
        tied comparisons. The choice is not cosmetic. Splitting pulls every
        rating toward level, and on data where most comparisons end tied the
        two policies produce ratings that are nowhere near each other, so
        pick the one that matches what a tie meant to your judges.

        Davidson's model is the principled third option. It fits an explicit
        tie parameter instead of assuming a tie is half a win, which is the
        assumption split makes and cannot check. It is not built here.
        Passing ``"davidson"`` raises rather than silently falling back.
    reference
        The model pinned at zero. Defaults to the one with the most
        comparisons, with ties broken alphabetically. The choice changes
        which model reads zero and nothing else. Ratings, the win matrix,
        the interval on every gap and so the separable pairs are all
        unchanged by it.
    n_boot
        Bootstrap resamples for the intervals. Items are resampled with
        replacement and the model is refitted on each draw. Zero skips the
        intervals, and then no pair can be called separable, which is the
        finding this function exists to produce.
    confidence
        Nominal coverage of the percentile intervals, default 0.95.
    seed
        Seeds the bootstrap. Set it in anything you publish.
    resample
        What one bootstrap draw picks up. ``"items"``, the default, draws
        item ids with replacement and takes every comparison made on each
        one, so judgements of the same prompt stay together. That is the
        cluster bootstrap. ``"comparisons"`` draws single rows, which treats
        every judgement as independent of every other.

        Changed in 0.3.0, and the change breaks results. Earlier versions
        always resampled comparisons and read ``item_id`` for nothing but a
        count. Supplying item ids says which comparisons share a prompt, so
        they are now used. Where every item carries one comparison the two
        settings draw the same resamples from the same seed and nothing
        moves. Where items carry several, the intervals widen by an amount
        set by how strongly judgements of one prompt agree with each other,
        and fewer pairs may separate. Pass ``"comparisons"`` to reproduce a
        result from an earlier version.

        The item bootstrap needs an item id on every row and at least two
        items, and refuses otherwise.

    Returns
    -------
    BTResult

    Notes
    -----
    Ratings are differences. One is pinned at zero to make the fit
    identifiable and the pick is arbitrary, so the level of a single rating
    says nothing and only gaps between models do.

    A pair is separable when the percentile interval on its gap, the
    difference between the two ratings, excludes zero. Both ratings on a
    resample come from one fit, so whatever that fit shares between them
    cancels in the gap, and no choice of reference can move the interval.
    A pair that does not separate is one the data has not established an
    order for. That is a statement about how much was measured, and it does
    not say the two models are level.

    Changed in 0.3.0. Earlier versions called a pair separable when the two
    rating intervals did not overlap. Two rating intervals can overlap while
    the gap between them is well measured, because both ratings carry the
    error of the field average they are measured against. So that test
    undercounted the pairs the data orders, and it established nothing about
    the pairs it left out. In every case checked in the tests, the new test
    separated every pair the old one did, and on some data it separated more.

    The rating intervals are on each rating measured against the average of
    the field, then shifted onto the reference. That keeps their widths from
    turning on which model was pinned.

    Resampling single comparisons treats judgements of one prompt as
    independent, and where prompts carry many comparisons that makes the
    intervals too narrow. This was measured on simulated data shaped like
    MT-Bench's human judgements, six models and 2,575 comparisons over 80
    prompts, where every prompt shifts each model's strength by its own
    normal draw. With a prompt-level spread of 0.5 on the log-odds scale,
    the comparison bootstrap's gap intervals came out 82% as wide as the item
    bootstrap's and covered 88% of the time where 95% was claimed. With a
    spread of 1.0 they were 63.5% as wide, covered 80%, and separated about
    one more pair of the fifteen on average. With no prompt-level spread the
    two agree, and the item bootstrap covered between 93.7% and 94.6% at
    every spread. The spread on MT-Bench itself was not measured.
    ``examples/pairwise_cluster_study.py`` reproduces these figures.

    A resample can be undefined the same two ways the full data can, by
    splitting the models into groups or by leaving one unbeaten. Those
    resamples are dropped, and if too many of them are the interval is
    refused rather than taken from the survivors.
    """
    if ties == _TIE_HOOK:
        raise NotImplementedError(
            "ties='davidson' is not built. Davidson's model treats a tie as "
            "its own outcome with its own parameter, rather than assuming a "
            "tie is worth half a win, and it is the more principled "
            "treatment when ties are common. Use ties='split' or "
            "ties='drop' and say in your write-up which you picked."
        )
    if ties not in _TIE_POLICIES:
        raise ValueError(
            f"ties must be one of {sorted(_TIE_POLICIES)}, got {ties!r}"
        )
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n_boot < 0:
        raise ValueError(f"n_boot must not be negative, got {n_boot}")
    if resample not in _RESAMPLE_UNITS:
        raise ValueError(
            f"resample must be one of {sorted(_RESAMPLE_UNITS)}, got "
            f"{resample!r}"
        )

    data = _clean(comparisons)
    models = sorted(set(data["model_a"]) | set(data["model_b"]))
    index = {m: k for k, m in enumerate(models)}

    used = _apply_tie_policy(data, ties)
    if used.empty:
        raise ValueError(
            "every comparison was a tie and ties='drop' discarded all of "
            "them, so there is nothing left to fit. Use ties='split' to give "
            "each side half a win."
        )

    rows = _Rows.build(used, index)
    credit = rows.credit_matrix()
    counts = credit + credit.T

    reference = _pick_reference(models, counts, reference)
    ref = index[reference]

    groups = _components(counts > 0, models)
    connected = len(groups) == 1
    undefeated, winless = ((), ())
    if connected:
        undefeated, winless = _separated_sets(credit > 0, models)

    totals = counts.sum(axis=1)
    shared = dict(
        reference=reference,
        ties=ties,
        n_models=len(models),
        n_comparisons=int(len(used)),
        n_ties=int(_is_tie(data["winner"]).sum()),
        n_dropped=int(len(data) - len(used)),
        n_items=int(used["item_id"].nunique()),
        n_missing_item_ids=int(used["item_id"].isna().sum()),
        n_pairs=len(models) * (len(models) - 1) // 2,
        connected=connected,
        comparable_groups=groups,
        undefeated=undefeated,
        winless=winless,
        confidence=confidence,
        n_boot=n_boot,
        resample=resample,
    )

    if not connected or undefeated:
        return _refused(models, totals, shared)

    centred = _fit(credit[None])[0]
    rating = centred - centred[ref]

    draws = None
    if n_boot:
        clusters, n_clusters = _clusters(used, resample)
        draws = _bootstrap(rows, clusters, n_clusters, n_boot, seed)
    usable = 0 if draws is None else len(draws)
    shared["n_boot_usable"] = usable
    if usable < _MIN_USABLE_SHARE * n_boot or usable < 2:
        draws = None

    lo, hi = _rating_intervals(draws, centred[ref], confidence)

    order = np.argsort(-rating, kind="stable")
    names = [models[k] for k in order]
    ratings = pd.DataFrame({
        "model": names,
        "rating": rating[order],
        "ci_low": np.full(len(models), np.nan) if lo is None else lo[order],
        "ci_high": np.full(len(models), np.nan) if hi is None else hi[order],
        "n_comparisons": totals[order].round().astype(int),
    })
    pairs = _pairs(rating, draws, counts, models, confidence)

    return BTResult(
        ratings=ratings,
        win_matrix=_win_matrix(rating[order], names),
        separable_pairs=_separable(pairs),
        pairs=pairs,
        **shared,
    )


def to_elo(
    bt_result: BTResult, scale: float = 400, base: float = 1500
) -> BTResult:
    """Put Bradley-Terry ratings on the Elo scale, intervals included.

    Elo is what a leaderboard audience expects to read, and it is the same
    numbers stretched. A gap of ``scale`` points means the better model takes
    ten decisive comparisons out of eleven, which is where 400 comes from.

    Leaderboards publish Elo without intervals. Publishing them with
    intervals is the reason this function exists, so the interval on every
    rating is carried across the same affine change of units the rating is,
    and the interval on every gap is stretched with the gap. That leaves the
    win matrix and the separable pairs exactly as they were, since
    stretching a scale by a positive factor cannot move an interval across
    zero.

    ``base`` lands on the reference model. It is a display choice with no
    content, the same way the reference is.
    """
    if not isinstance(bt_result, BTResult):
        raise ValueError("to_elo takes a BTResult from bradley_terry")
    if bt_result.is_elo:
        raise ValueError(
            "this result is already on the Elo scale. Convert the "
            "Bradley-Terry result instead of converting twice."
        )
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")

    factor = scale / math.log(10.0)

    ratings = bt_result.ratings.copy()
    for column in ("rating", "ci_low", "ci_high"):
        ratings[column] = base + ratings[column] * factor

    separable = bt_result.separable_pairs.copy()
    every = None if bt_result.pairs is None else bt_result.pairs.copy()
    for table in (separable, every):
        if table is None:
            continue
        for column in ("difference", "ci_low", "ci_high"):
            table[column] = table[column] * factor

    return BTResult(
        ratings=ratings,
        win_matrix=bt_result.win_matrix.copy(),
        separable_pairs=separable,
        pairs=every,
        resample=bt_result.resample,
        reference=bt_result.reference,
        ties=bt_result.ties,
        n_models=bt_result.n_models,
        n_comparisons=bt_result.n_comparisons,
        n_ties=bt_result.n_ties,
        n_dropped=bt_result.n_dropped,
        n_items=bt_result.n_items,
        n_missing_item_ids=bt_result.n_missing_item_ids,
        n_pairs=bt_result.n_pairs,
        connected=bt_result.connected,
        comparable_groups=bt_result.comparable_groups,
        undefeated=bt_result.undefeated,
        winless=bt_result.winless,
        confidence=bt_result.confidence,
        n_boot=bt_result.n_boot,
        n_boot_usable=bt_result.n_boot_usable,
        elo_scale=float(scale),
        elo_base=float(base),
    )


# --------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------

def _is_tie(winner: pd.Series) -> pd.Series:
    """True where the row records a tie.

    A missing winner and the literal word both count. Case is ignored, since
    the column is typed by hand as often as it is generated.
    """
    text = winner.astype("object")
    missing = text.isna()
    words = text.where(~missing, "").astype(str).str.strip().str.lower()
    return missing | (words == _TIE)


def _clean(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Validate and copy. Every check here is a data-entry mistake."""
    if not isinstance(comparisons, pd.DataFrame):
        raise ValueError("comparisons must be a pandas DataFrame")

    missing = [c for c in _COLUMNS if c not in comparisons.columns]
    if missing:
        raise ValueError(
            f"comparisons is missing required column(s): {', '.join(missing)}. "
            f"Expected {', '.join(_COLUMNS)}, where winner names one of the "
            f"two models or is {_TIE!r}."
        )

    data = comparisons.loc[:, list(_COLUMNS)].copy()
    if data.empty:
        raise ValueError("comparisons must not be empty")

    same = data["model_a"] == data["model_b"]
    if same.any():
        first = data[same].iloc[0]
        raise ValueError(
            f"comparisons pits a model against itself on item "
            f"{first['item_id']!r}, where both sides are "
            f"{first['model_a']!r}. There is no rating to learn from that."
        )

    models = set(data["model_a"]) | set(data["model_b"])
    reserved = sorted(
        m for m in models if str(m).strip().lower() == _TIE
    )
    if reserved:
        raise ValueError(
            f"a model is called {reserved[0]!r}, and {_TIE!r} is how this "
            f"reads a tie in the winner column. Rename the model."
        )

    tie = _is_tie(data["winner"])
    on_offer = (data["winner"] == data["model_a"]) | (
        data["winner"] == data["model_b"]
    )
    bad = ~tie & ~on_offer
    if bad.any():
        first = data[bad].iloc[0]
        raise ValueError(
            f"comparisons has a winner that was not on offer: item "
            f"{first['item_id']!r} names {first['winner']!r}, which is "
            f"neither {first['model_a']!r} nor {first['model_b']!r}. Use "
            f"{_TIE!r} or leave it empty to record a tie."
        )
    return data


def _apply_tie_policy(data: pd.DataFrame, ties: str) -> pd.DataFrame:
    if ties == "drop":
        return data[~_is_tie(data["winner"])]
    return data


def _pick_reference(models, counts, reference) -> str:
    """The model pinned at zero.

    Defaults to the most compared one, which is the best measured, with ties
    broken alphabetically so the same frame always gives the same answer.
    The choice moves nothing except which model reads zero.
    """
    if reference is not None:
        if reference not in models:
            raise ValueError(
                f"reference {reference!r} is not one of the models compared: "
                f"{', '.join(map(str, models))}"
            )
        return reference
    totals = counts.sum(axis=1)
    return models[int(np.lexsort((models, -totals))[0])]


class _Rows:
    """The comparisons as index arrays, ready to be resampled.

    One entry per row of the frame, carrying which models met and how much
    credit each took. A split tie is half a win each way, which is why the
    credits are floats rather than a winner index.
    """

    def __init__(self, a, b, credit_a, credit_b, n_models):
        self.a = a
        self.b = b
        self.credit_a = credit_a
        self.credit_b = credit_b
        self.n_models = n_models

    @classmethod
    def build(cls, used: pd.DataFrame, index) -> "_Rows":
        a = used["model_a"].map(index).to_numpy(dtype=np.intp)
        b = used["model_b"].map(index).to_numpy(dtype=np.intp)
        tie = _is_tie(used["winner"]).to_numpy()
        a_won = (used["winner"] == used["model_a"]).to_numpy()

        credit_a = np.where(tie, 0.5, a_won.astype(float))
        credit_b = np.where(tie, 0.5, 1.0 - a_won.astype(float))
        return cls(a, b, credit_a, credit_b, len(index))

    def __len__(self) -> int:
        return len(self.a)

    def credit_matrix(self) -> np.ndarray:
        """The full-data credits, as a (models, models) array."""
        return self.credits(np.ones((1, len(self))))[0]

    def credits(self, weights: np.ndarray) -> np.ndarray:
        """Credits for a block of resamples, in one pass.

        ``weights`` is (blocks, rows), how many times each row was drawn into
        each resample. Holding the shape at one column per row is what lets a
        cluster resample, which takes a different number of rows every time,
        go through the same pass as a row resample. Every resample in the
        block is tallied by a single bincount over a flattened index, so the
        cost of a block does not depend on how many resamples are in it.
        """
        blocks, _ = weights.shape
        m = self.n_models
        cell = m * m
        offset = (np.arange(blocks) * cell)[:, None]

        forward = offset + (self.a * m + self.b)[None, :]
        backward = offset + (self.b * m + self.a)[None, :]

        flat = np.bincount(
            forward.ravel(),
            weights=(weights * self.credit_a).ravel(),
            minlength=blocks * cell,
        )
        flat += np.bincount(
            backward.ravel(),
            weights=(weights * self.credit_b).ravel(),
            minlength=blocks * cell,
        )
        return flat.reshape(blocks, m, m)


def _clusters(used: pd.DataFrame, resample: str):
    """Which rows a bootstrap draw takes together, as (codes, count).

    For ``"comparisons"`` every row is its own cluster. For ``"items"`` the
    rows sharing an item id are one cluster, numbered in order of first
    appearance, so a frame whose ids are all distinct numbers its rows
    0, 1, 2 and draws exactly what the comparison bootstrap draws.
    """
    if resample == "comparisons":
        return np.arange(len(used)), len(used)

    ids = used["item_id"]
    missing = ids.isna()
    if missing.any():
        first = used[missing].iloc[0]
        raise ValueError(
            f"item_id is missing on {int(missing.sum())} of {len(used)} "
            f"comparisons, the first between {first['model_a']!r} and "
            f"{first['model_b']!r}. The bootstrap resamples whole items, so "
            f"it needs to know which item every comparison belongs to. Fill "
            f"in the item ids, or pass resample='comparisons' to resample "
            f"comparisons one at a time, which treats every judgement as "
            f"independent."
        )

    codes, uniques = pd.factorize(ids, sort=False)
    if len(uniques) < 2:
        raise ValueError(
            f"every comparison was made on one item, {uniques[0]!r}. The "
            f"bootstrap resamples whole items, so it would draw that item "
            f"every time and return intervals of width zero, which would "
            f"claim the ratings are known exactly. Pass "
            f"resample='comparisons' to resample comparisons one at a time, "
            f"and read the intervals knowing they treat judgements of one "
            f"prompt as independent."
        )
    return codes.astype(np.intp), len(uniques)


# --------------------------------------------------------------------------
# The fit
# --------------------------------------------------------------------------

def _fit(credit: np.ndarray) -> np.ndarray:
    """Maximum likelihood ratings for a stack of credit matrices.

    Newton's method on the mean-centred parameterisation. The likelihood is
    concave and flat along the direction that adds a constant to every
    rating, so the Hessian is singular by exactly that one direction. Adding
    1/m to every entry fills it in without touching the rest.

    The two centring lines are defensive and nothing turns on them. The
    gradient already sums to zero at every point, because a pair is counted
    once in each direction and the two probabilities add to one, so the
    projection and the recentring only stop rounding error accumulating
    across steps. Deleting either one changes no answer this package
    produces, which is to say no test can tell the difference and none
    should try.

    The loop is over Newton steps, not over resamples. Every resample in
    ``credit`` takes each step at the same time.
    """
    blocks, m, _ = credit.shape
    n = credit + np.transpose(credit, (0, 2, 1))
    earned = credit.sum(axis=2)
    r = np.zeros((blocks, m))

    for _ in range(_MAX_NEWTON):
        p = _sigmoid(r[:, :, None] - r[:, None, :])
        _fill_diagonal(p, 0.0)

        gradient = (n * p).sum(axis=2) - earned
        gradient -= gradient.mean(axis=1, keepdims=True)
        if np.max(np.abs(gradient)) < _NEWTON_TOL:
            break

        weight = n * p * (1.0 - p)
        _fill_diagonal(weight, 0.0)
        hessian = -weight
        _fill_diagonal(hessian, weight.sum(axis=2))
        hessian += 1.0 / m

        r -= np.linalg.solve(hessian, gradient[:, :, None])[:, :, 0]
        r -= r.mean(axis=1, keepdims=True)

    return r


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _fill_diagonal(stack: np.ndarray, value) -> None:
    """Set the diagonal of every matrix in a (blocks, m, m) stack."""
    np.einsum("bii->bi", stack)[...] = value


# --------------------------------------------------------------------------
# Whether a fit exists at all
# --------------------------------------------------------------------------

def _reachable(adjacency: np.ndarray) -> np.ndarray:
    """Transitive closure of a stack of boolean adjacency matrices.

    Repeated squaring, so the number of passes is the log of the model count
    rather than the model count. Every matrix in the stack closes at once.
    """
    m = adjacency.shape[-1]
    reach = adjacency | np.eye(m, dtype=bool)
    for _ in range(max(1, int(np.ceil(np.log2(max(m, 2)))))):
        reach = reach | (reach.astype(np.uint8) @ reach.astype(np.uint8) > 0)
    return reach


def _components(played: np.ndarray, models) -> tuple:
    """The groups of models joined by chains of comparisons.

    One group means every model can be reached from every other, which is
    what putting them on a single scale requires.
    """
    reach = _reachable(played[None])[0]
    groups, seen = [], set()
    for i, name in enumerate(models):
        if name in seen:
            continue
        members = tuple(models[j] for j in np.flatnonzero(reach[i]))
        seen.update(members)
        groups.append(members)
    return tuple(groups)


def _separated_sets(beat: np.ndarray, models):
    """A set that never lost outside itself, and one that never won.

    Ford's condition: the maximum exists and is unique when every model is
    reachable from every other through a chain of wins. When it is not, some
    group of models never lost to anything outside it, and its ratings run
    off to infinity. Both ends are returned because they read as different
    findings. One model that never lost and one that never won are the two
    shapes this takes in practice, and they call for different sentences.
    """
    reach = _reachable(beat[None])[0]
    strong = reach & reach.T
    if strong.all():
        return (), ()

    def pick(mask_out):
        """The first group with no arrow coming in, or going out."""
        for i in range(len(models)):
            group = np.flatnonzero(strong[i])
            outside = np.setdiff1d(np.arange(len(models)), group)
            if not mask_out[np.ix_(outside, group)].any():
                return tuple(models[j] for j in group)
        return ()

    return pick(beat), pick(beat.T)


# --------------------------------------------------------------------------
# The bootstrap
# --------------------------------------------------------------------------

def _bootstrap(rows, clusters, n_clusters, n_boot, seed) -> np.ndarray:
    """Refitted ratings on every usable resample, one row per resample.

    Each resample draws ``n_clusters`` cluster numbers with replacement and
    weights every row by how many times its cluster came up. That is the
    cluster bootstrap, taking every row of each drawn cluster, with the
    shape held at one column per row. When every row is its own cluster the
    weights are the row counts of an ordinary row bootstrap, and the credits
    are the same ones summing the drawn rows would give.

    Every draw is fitted on the mean-centred scale. The matrix is kept whole,
    because the interval on each rating and the interval on each gap are
    both read off it.
    """
    rng = np.random.default_rng(seed)
    n_models = rows.n_models
    block = max(1, int(_BLOCK_BUDGET // max(n_models * n_models, 1)))

    kept = []
    drawn = 0
    while drawn < n_boot:
        size = min(block, n_boot - drawn)
        draws = rng.integers(0, n_clusters, size=(size, n_clusters))
        tally = (np.arange(size)[:, None] * n_clusters + draws).ravel()
        times = np.bincount(tally, minlength=size * n_clusters).reshape(
            size, n_clusters
        )
        credit = rows.credits(times[:, clusters])

        usable = _usable(credit)
        if usable.any():
            kept.append(_fit(credit[usable]))
        drawn += size

    if not kept:
        return np.empty((0, n_models))
    return np.concatenate(kept, axis=0)


def _rating_intervals(draws, ref_centred, confidence):
    """Percentile intervals on each rating, on the reported scale.

    The draws are on each rating against the average of the field. Shifting
    by the reference's own centred rating at the end puts them on the
    reported scale without letting the choice of reference change their
    widths.
    """
    if draws is None:
        return None, None
    tail = (1 - confidence) / 2
    lo = np.percentile(draws, 100 * tail, axis=0) - ref_centred
    hi = np.percentile(draws, 100 * (1 - tail), axis=0) - ref_centred
    return lo, hi


def _usable(credit: np.ndarray) -> np.ndarray:
    """Which resamples in a block have a maximum at all.

    The same two conditions the full data has to meet. Every model reachable
    from every other by having played, and every model reachable from every
    other by having won.
    """
    beat = credit > 0
    played = beat | np.transpose(beat, (0, 2, 1))
    connected = _reachable(played).all(axis=(1, 2))
    reach = _reachable(beat)
    strong = (reach & np.transpose(reach, (0, 2, 1))).all(axis=(1, 2))
    return connected & strong


# --------------------------------------------------------------------------
# Output tables
# --------------------------------------------------------------------------

def _win_matrix(rating: np.ndarray, names) -> pd.DataFrame:
    """Modelled P(row beats column), in rating order.

    The diagonal is NaN. A model's chance against itself is a half by
    construction and means nothing, and printing 0.5 down the diagonal
    invites a reader to treat it as a measurement.
    """
    matrix = _sigmoid(rating[:, None] - rating[None, :])
    np.fill_diagonal(matrix, np.nan)
    return pd.DataFrame(matrix, index=list(names), columns=list(names))


_PAIR_COLUMNS = [
    "model_a", "model_b", "difference", "ci_low", "ci_high",
    "p_a_beats_b", "n_head_to_head",
]
_ALL_PAIR_COLUMNS = _PAIR_COLUMNS + ["separable"]


def _pairs(rating, draws, counts, models, confidence) -> pd.DataFrame:
    """Every pair, higher-rated model first, with an interval on the gap.

    The gap on each resample is a difference of two ratings from one fit, so
    anything that fit adds to both cancels. The interval is taken on that
    difference directly. Differencing two rating intervals instead would
    count the error the ratings share twice.

    A pair is separable when the interval excludes zero. Without an interval
    no pair is.
    """
    i, j = np.triu_indices(len(models), 1)
    ahead = rating[i] > rating[j]
    top = np.where(ahead, i, j)
    bottom = np.where(ahead, j, i)
    gap = rating[top] - rating[bottom]

    if draws is None:
        low = np.full(len(top), np.nan)
        high = np.full(len(top), np.nan)
        separable = np.zeros(len(top), dtype=bool)
    else:
        tail = (1 - confidence) / 2
        spread = draws[:, top] - draws[:, bottom]
        low = np.percentile(spread, 100 * tail, axis=0)
        high = np.percentile(spread, 100 * (1 - tail), axis=0)
        separable = (low > 0) | (high < 0)

    table = pd.DataFrame({
        "model_a": [models[k] for k in top],
        "model_b": [models[k] for k in bottom],
        "difference": gap,
        "ci_low": low,
        "ci_high": high,
        "p_a_beats_b": _sigmoid(gap),
        "n_head_to_head": np.round(counts[top, bottom]).astype(int),
        "separable": separable,
    })
    return (
        table.sort_values("difference", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )


def _separable(pairs: pd.DataFrame) -> pd.DataFrame:
    """The rows of ``pairs`` whose gap interval excludes zero."""
    return (
        pairs[pairs["separable"]]
        .drop(columns="separable")
        .reset_index(drop=True)
    )


def _refused(models, totals, shared) -> BTResult:
    """A result with no ratings in it.

    The models, their comparison counts and the reason are all still here.
    What is missing is the ranking, because the data does not carry one.
    """
    order = np.argsort(-totals, kind="stable")
    names = [models[k] for k in order]
    nan = np.full(len(models), np.nan)
    ratings = pd.DataFrame({
        "model": names,
        "rating": nan,
        "ci_low": nan.copy(),
        "ci_high": nan.copy(),
        "n_comparisons": totals[order].round().astype(int),
    })
    empty = pd.DataFrame(
        np.full((len(models), len(models)), np.nan),
        index=names, columns=names,
    )
    return BTResult(
        ratings=ratings,
        win_matrix=empty,
        separable_pairs=pd.DataFrame(columns=_PAIR_COLUMNS),
        pairs=pd.DataFrame(columns=_ALL_PAIR_COLUMNS),
        n_boot_usable=0,
        **shared,
    )
