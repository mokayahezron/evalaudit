"""Which models on your leaderboard are actually in a different position?

A blind pairwise comparison produces a ranking, and the ranking gets
published. Almost nobody publishes what the ranking rests on. Six models make
fifteen pairs, and on the volumes people usually run, two or three of those
pairs are distinguishable and the rest are a coin flip dressed as an order.

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

# Newton on a concave log-likelihood from a sensible start converges in a few
# steps. The cap is here so a resample that is nearly separable cannot spin.
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
) -> BTResult:
    """Fit Bradley-Terry ratings and report which pairs they actually order.

    Parameters
    ----------
    comparisons
        A frame with ``item_id``, ``model_a``, ``model_b`` and ``winner``.
        One row per judgement. ``winner`` names one of the two models, or is
        the string ``"tie"``, or is left empty to mean the same thing.
        ``item_id`` labels the prompt the pair was judged on. Nothing in the
        fit reads it and it is required anyway, so the frame says what was
        compared as well as who won.
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
        which model reads zero and nothing else. Ratings, the win matrix and
        the separable pairs are all unchanged by it.
    n_boot
        Bootstrap resamples for the intervals. Comparisons are resampled with
        replacement and the model is refitted on each draw. Zero skips the
        interval, and then no pair can be called separable, which is the
        finding this function exists to produce.
    confidence
        Nominal coverage of the percentile interval, default 0.95.
    seed
        Seeds the bootstrap. Set it in anything you publish.

    Returns
    -------
    BTResult

    Notes
    -----
    Ratings are differences. One is pinned at zero to make the fit
    identifiable and the pick is arbitrary, so the level of a single rating
    says nothing and only gaps between models do.

    The intervals are on each rating measured against the average of the
    field, then shifted onto the reference. That keeps them, and so the
    count of separable pairs, from turning on which model was pinned.

    The bootstrap resamples comparisons, not items. Rows that share an
    ``item_id`` are correlated, since they are judgements of the same prompt,
    so on a design where one prompt carries many comparisons the interval
    will run slightly narrow.

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
        n_pairs=len(models) * (len(models) - 1) // 2,
        connected=connected,
        comparable_groups=groups,
        undefeated=undefeated,
        winless=winless,
        confidence=confidence,
        n_boot=n_boot,
    )

    if not connected or undefeated:
        return _refused(models, totals, shared)

    centred = _fit(credit[None])[0]
    rating = centred - centred[ref]

    lo, hi, usable = _bootstrap(
        rows, len(models), centred[ref], n_boot, confidence, seed
    )
    shared["n_boot_usable"] = usable

    order = np.argsort(-rating, kind="stable")
    names = [models[k] for k in order]
    ratings = pd.DataFrame({
        "model": names,
        "rating": rating[order],
        "ci_low": np.full(len(models), np.nan) if lo is None else lo[order],
        "ci_high": np.full(len(models), np.nan) if hi is None else hi[order],
        "n_comparisons": totals[order].round().astype(int),
    })

    return BTResult(
        ratings=ratings,
        win_matrix=_win_matrix(rating[order], names),
        separable_pairs=_separable(rating, lo, hi, counts, models, index),
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
    rating is carried across the same affine change of units the rating is.
    That leaves the win matrix and the separable pairs exactly as they were,
    since stretching a scale cannot make two intervals stop overlapping.

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

    pairs = bt_result.separable_pairs.copy()
    pairs["difference"] = pairs["difference"] * factor

    return BTResult(
        ratings=ratings,
        win_matrix=bt_result.win_matrix.copy(),
        separable_pairs=pairs,
        reference=bt_result.reference,
        ties=bt_result.ties,
        n_models=bt_result.n_models,
        n_comparisons=bt_result.n_comparisons,
        n_ties=bt_result.n_ties,
        n_dropped=bt_result.n_dropped,
        n_items=bt_result.n_items,
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
        return self.credits(np.arange(len(self))[None])[0]

    def credits(self, draws: np.ndarray) -> np.ndarray:
        """Credits for a block of resamples, in one pass.

        ``draws`` is (blocks, rows) of row indices. Every resample in the
        block is tallied by a single bincount over a flattened index, so the
        cost of a block does not depend on how many resamples are in it.
        """
        blocks, _ = draws.shape
        m = self.n_models
        cell = m * m
        offset = (np.arange(blocks) * cell)[:, None]

        forward = offset + self.a[draws] * m + self.b[draws]
        backward = offset + self.b[draws] * m + self.a[draws]

        flat = np.bincount(
            forward.ravel(),
            weights=self.credit_a[draws].ravel(),
            minlength=blocks * cell,
        )
        flat += np.bincount(
            backward.ravel(),
            weights=self.credit_b[draws].ravel(),
            minlength=blocks * cell,
        )
        return flat.reshape(blocks, m, m)


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

def _bootstrap(rows, n_models, ref_centred, n_boot, confidence, seed):
    """Percentile intervals from refitting on resampled comparisons.

    Every draw is fitted on the mean-centred scale, so the percentiles are
    on each rating against the average of the field. Shifting by the
    reference's own centred rating at the end puts them on the reported
    scale without letting the choice of reference change their widths.
    """
    if n_boot == 0:
        return None, None, 0

    rng = np.random.default_rng(seed)
    n_rows = len(rows)
    block = max(1, int(_BLOCK_BUDGET // max(n_models * n_models, 1)))

    kept = []
    drawn = 0
    while drawn < n_boot:
        size = min(block, n_boot - drawn)
        draws = rng.integers(0, n_rows, size=(size, n_rows))
        credit = rows.credits(draws)

        usable = _usable(credit)
        if usable.any():
            kept.append(_fit(credit[usable]))
        drawn += size

    n_usable = sum(len(part) for part in kept)
    if n_usable < _MIN_USABLE_SHARE * n_boot or n_usable < 2:
        return None, None, n_usable

    draws = np.concatenate(kept, axis=0)
    tail = (1 - confidence) / 2
    lo = np.percentile(draws, 100 * tail, axis=0) - ref_centred
    hi = np.percentile(draws, 100 * (1 - tail), axis=0) - ref_centred
    return lo, hi, n_usable


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
    "model_a", "model_b", "difference", "p_a_beats_b", "n_head_to_head"
]


def _separable(rating, lo, hi, counts, models, index) -> pd.DataFrame:
    """The pairs whose intervals do not overlap.

    Conservative on purpose. Two intervals that overlap a little can still
    describe a difference the data supports, so this undercounts rather than
    overcounts what the leaderboard orders.
    """
    if lo is None:
        return pd.DataFrame(columns=_PAIR_COLUMNS)

    rows = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            if not (hi[i] < lo[j] or hi[j] < lo[i]):
                continue
            top, bottom = (i, j) if rating[i] > rating[j] else (j, i)
            gap = rating[top] - rating[bottom]
            rows.append({
                "model_a": models[top],
                "model_b": models[bottom],
                "difference": gap,
                "p_a_beats_b": float(_sigmoid(gap)),
                "n_head_to_head": int(round(counts[i, j])),
            })

    table = pd.DataFrame(rows, columns=_PAIR_COLUMNS)
    return (
        table.sort_values("difference", ascending=False, kind="mergesort")
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
        n_boot_usable=0,
        **shared,
    )
