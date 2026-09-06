"""Tests for evalaudit.agreement.

Written before the implementation. Make these pass.

Krippendorff's alpha is checked three ways. Against the ``krippendorff``
package on PyPI as an independent implementation, against Krippendorff's own
published worked examples where the answers are in print, and against a slow
reference written straight from the definition in this file. The slow
reference is the one that catches vectorisation bugs, since the shipped code
builds coincidence matrices in one pass and the definition builds them by
looping over pairs.

Cohen's and Fleiss' kappa are checked against statsmodels, the same way the
McNemar tests in test_compare.py are.
"""

import numpy as np
import pandas as pd
import pytest

import krippendorff as kref
from statsmodels.stats.inter_rater import cohens_kappa as sm_cohens_kappa
from statsmodels.stats.inter_rater import fleiss_kappa as sm_fleiss_kappa

from evalaudit import AgreementResult, KappaResult
from evalaudit.agreement import cohens_kappa, fleiss_kappa, rater_agreement


LEVELS = ["nominal", "ordinal", "interval"]

# The three reasons a leave-one-out alpha can come back undefined. They are
# spelled out here rather than imported, because the point of the note column
# is that a reader learns which of the three happened, and a test that
# imported the strings would pass however they were assigned.
NOTE_SINGLE_RATER = "only one rater left, nothing to compare"
NOTE_NO_OVERLAP = "no item left with two or more ratings"
NOTE_NO_VARIANCE = "remaining raters agreed everywhere, so alpha has no denominator"
NOTE_ONE_ITEM = "only one item left with two or more ratings"

# The phrase summary() opens with when it does name a rater. Tests assert on
# this rather than on a bare rater id, because "r0" is a substring of plenty
# of things a summary legitimately contains, and a test that fails because an
# item got called "essay-r05" is a test failing for the wrong reason.
NAMES_A_RATER = "Dropping "


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def to_long(matrix, rater_names=None, item_names=None):
    """Turn a (raters x items) matrix with NaN holes into a long frame.

    The reference package wants the wide matrix, this library wants long
    format, so every cross-check goes through here.
    """
    arr = np.asarray(matrix, dtype=float)
    n_raters, n_items = arr.shape
    raters = rater_names or [f"r{i}" for i in range(n_raters)]
    items = item_names or [f"u{j}" for j in range(n_items)]
    rows = []
    for i in range(n_raters):
        for j in range(n_items):
            if not np.isnan(arr[i, j]):
                rows.append((items[j], raters[i], arr[i, j]))
    return pd.DataFrame(rows, columns=["item_id", "rater_id", "rating"])


def mostly_unanimous(n_items, n_split, n_raters=3):
    """Items everyone agreed on, plus a few they split over.

    Built so the share of undefined bootstrap resamples is known in advance.
    A resample is undefined when it draws no split item, because then every
    pairable rating is identical and alpha has no denominator. That happens
    with probability ((n_items - n_split) / n_items) ** n_items, which is
    near e**-n_split and barely moves with n_items.
    """
    split = [1.0, 5.0, 3.0, 2.0, 4.0][:n_raters]
    rows = []
    for j in range(n_items):
        vals = split if j < n_split else [3.0] * n_raters
        for i, v in enumerate(vals):
            rows.append((f"u{j}", f"r{i}", v))
    return pd.DataFrame(rows, columns=["item_id", "rater_id", "rating"])


def with_solo_items(df, n_solo, rater="r0"):
    """Bolt ungraded-by-anyone-else items onto a frame.

    The shape real grading arrives in. A few hundred items graded once, a
    handful graded twice, and every claim about agreement resting on the
    handful.
    """
    extra = pd.DataFrame(
        [(f"solo{i}", rater, 1.0) for i in range(n_solo)],
        columns=["item_id", "rater_id", "rating"],
    )
    return pd.concat([df, extra], ignore_index=True)


def random_matrix(rng, n_raters, n_items, values, missing=0.25):
    """A ragged rating matrix. Real grading is never fully crossed."""
    arr = rng.choice(values, size=(n_raters, n_items)).astype(float)
    holes = rng.random((n_raters, n_items)) < missing
    arr[holes] = np.nan
    # every unit keeps at least two raters, so the design stays connected
    for j in range(n_items):
        if np.sum(~np.isnan(arr[:, j])) < 2:
            fill = rng.choice(n_raters, size=2, replace=False)
            for i in fill:
                arr[i, j] = rng.choice(values)
    return arr


# --------------------------------------------------------------------------
# A slow reference implementation, written straight from the definition
# --------------------------------------------------------------------------

def slow_alpha(df, level):
    """Krippendorff's alpha the obvious way, looping over pairs.

    Kept deliberately naive. It is the control for the vectorised version,
    so it must not share any of its shortcuts.
    """
    units = _slow_units(df)
    if not units:
        return float("nan")

    coincidence, domain = _slow_coincidence(units)
    marginals = coincidence.sum(axis=1)
    n = marginals.sum()
    dist = _slow_distances(domain, marginals, level)
    size = len(domain)

    d_o = sum(
        coincidence[c, k] * dist[c][k] for c in range(size) for k in range(size)
    )
    d_e = sum(
        marginals[c] * marginals[k] * dist[c][k]
        for c in range(size)
        for k in range(size)
    ) / (n - 1)

    if d_e == 0:
        return float("nan")
    return 1.0 - d_o / d_e


def _slow_units(df):
    """Ratings per item, keeping only items graded twice or more."""
    units = {}
    for item, group in df.groupby("item_id", sort=True):
        vals = list(group["rating"])
        if len(vals) >= 2:
            units[item] = vals
    return units


def _slow_coincidence(units):
    """The coincidence matrix, built one ordered pair at a time."""
    domain = sorted({v for vals in units.values() for v in vals})
    index = {v: i for i, v in enumerate(domain)}
    coincidence = np.zeros((len(domain), len(domain)))
    for vals in units.values():
        m = len(vals)
        for i in range(m):
            for j in range(m):
                if i != j:
                    coincidence[index[vals[i]], index[vals[j]]] += 1.0 / (m - 1)
    return coincidence, domain


def _slow_distances(domain, marginals, level):
    """The three metrics, written out as nested lists."""
    size = len(domain)
    dist = [[0.0] * size for _ in range(size)]
    for c in range(size):
        for k in range(size):
            if level == "nominal":
                dist[c][k] = 0.0 if c == k else 1.0
            elif level == "interval":
                dist[c][k] = (domain[c] - domain[k]) ** 2
            else:  # ordinal
                lo, hi = min(c, k), max(c, k)
                run = sum(marginals[g] for g in range(lo, hi + 1))
                dist[c][k] = (run - (marginals[c] + marginals[k]) / 2) ** 2
    return dist


def slow_item_disagreement(df, level):
    """Mean squared distance between two ratings of the same item.

    The normalisation the table promises. An item graded four times and an
    item graded twice with the same spread of ratings land on the same
    number, because both average over ordered pairs rather than summing.
    """
    units = _slow_units(df)
    coincidence, domain = _slow_coincidence(units)
    index = {v: i for i, v in enumerate(domain)}
    dist = _slow_distances(domain, coincidence.sum(axis=1), level)

    out = {}
    for item, vals in units.items():
        m = len(vals)
        total = sum(
            dist[index[vals[i]]][index[vals[j]]]
            for i in range(m)
            for j in range(m)
            if i != j
        )
        out[item] = total / (m * (m - 1))
    return out


def slow_bootstrap_ci(df, level, n_boot, seed, confidence=0.95):
    """The bootstrap written as a loop, one resample at a time.

    Draws the index matrix the way every bootstrap in this package does,
    then rebuilds a frame per resample and calls ``slow_alpha`` on it. Each
    drawn copy of an item becomes its own unit, which is what sampling items
    with replacement means.

    For ordinal this is the test that matters. The ordinal distance depends
    on the marginals, so it has to be rebuilt from each resample rather than
    computed once from the full sample. Here that happens for free, because
    ``slow_alpha`` derives the distances from whatever frame it is handed.
    """
    items = [i for i, n in df["item_id"].value_counts().items() if n >= 2]
    items = [i for i in dict.fromkeys(df["item_id"]) if i in set(items)]
    blocks = {i: df[df["item_id"] == i] for i in items}

    n_units = len(items)
    idx = np.random.default_rng(seed).integers(0, n_units, size=(n_boot, n_units))

    alphas = []
    for row in idx:
        parts = []
        for copy, pick in enumerate(row):
            block = blocks[items[pick]].copy()
            block["item_id"] = f"{items[pick]}#copy{copy}"
            parts.append(block)
        alphas.append(slow_alpha(pd.concat(parts, ignore_index=True), level))

    good = np.array(alphas)[np.isfinite(alphas)]
    tail = (1 - confidence) / 2
    return (
        float(np.percentile(good, 100 * tail)),
        float(np.percentile(good, 100 * (1 - tail))),
        int(good.size),
    )


def varied_items(n_items=12, n_raters=3, seed=0):
    """A ragged frame where every item carries real disagreement.

    Every item keeps at least two ratings and at least two distinct values,
    so no resample can come back undefined and the usable count is always
    the full one. That keeps the reference comparison about the arithmetic
    rather than about which resamples got dropped.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for j in range(n_items):
        # zero-padded so first-appearance order and sorted order agree
        item = f"u{j:02d}"
        keep = sorted(rng.choice(n_raters, size=rng.integers(2, n_raters + 1),
                                 replace=False))
        vals = rng.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=len(keep), replace=False)
        for i, v in zip(keep, vals):
            rows.append((item, f"r{i}", float(v)))
    return pd.DataFrame(rows, columns=["item_id", "rater_id", "rating"])


def test_slow_reference_matches_the_package():
    """The control has to be right before it can control anything."""
    rng = np.random.default_rng(0)
    arr = random_matrix(rng, 4, 25, [1, 2, 3, 4, 5])
    df = to_long(arr)
    for level in LEVELS:
        assert slow_alpha(df, level) == pytest.approx(
            kref.alpha(reliability_data=arr, level_of_measurement=level), abs=1e-9
        )


# --------------------------------------------------------------------------
# Krippendorff's published worked examples
# --------------------------------------------------------------------------

# Krippendorff (2011), "Computing Krippendorff's Alpha-Reliability".
# Four observers, twelve units, missing values throughout. The paper prints
# alpha for each level of measurement.
PUBLISHED_4_OBSERVER = [
    [1, 2, 3, 3, 2, 1, 4, 1, 2, np.nan, np.nan, np.nan],
    [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, np.nan, 3],
    [np.nan, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, np.nan],
    [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, np.nan],
]

# The three-observer, fifteen-unit example that runs through Krippendorff's
# Content Analysis and the alpha entry it is quoted in.
PUBLISHED_3_OBSERVER = [
    [np.nan, np.nan, np.nan, np.nan, np.nan, 3, 4, 1, 2, 1, 1, 3, 3, np.nan, 3],
    [1, np.nan, 2, 1, 3, 3, 4, 3, np.nan, np.nan, np.nan, np.nan, np.nan,
     np.nan, np.nan],
    [np.nan, np.nan, 2, 1, 3, 4, 4, np.nan, 2, 1, 1, 3, 3, np.nan, 4],
]


@pytest.mark.parametrize(
    "level, expected",
    [("nominal", 0.743), ("ordinal", 0.815), ("interval", 0.849)],
)
def test_published_four_observer_example(level, expected):
    """The printed answers, to the three decimals the paper prints."""
    df = to_long(PUBLISHED_4_OBSERVER)
    r = rater_agreement(df, level=level, bootstrap_ci=False)
    assert r.alpha == pytest.approx(expected, abs=5e-4)


@pytest.mark.parametrize(
    "level, expected", [("nominal", 0.691358), ("interval", 0.810845)]
)
def test_published_three_observer_example(level, expected):
    df = to_long(PUBLISHED_3_OBSERVER)
    r = rater_agreement(df, level=level, bootstrap_ci=False)
    assert r.alpha == pytest.approx(expected, abs=1e-6)


def test_published_examples_have_missing_data():
    """Guard the guard. These fixtures are only worth anything ragged."""
    for fixture in (PUBLISHED_4_OBSERVER, PUBLISHED_3_OBSERVER):
        assert np.isnan(np.asarray(fixture, dtype=float)).any()


# --------------------------------------------------------------------------
# External check against the krippendorff package
# --------------------------------------------------------------------------

@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize(
    "n_raters, n_items, values, missing",
    [
        (3, 40, [1, 2, 3, 4, 5], 0.3),
        (5, 60, [1, 2, 3], 0.4),
        (2, 80, [0, 1], 0.2),
        (4, 30, [1, 2, 4, 5], 0.35),      # a domain with a gap in it
        (6, 50, [1, 2, 3, 4, 5, 6, 7], 0.5),
    ],
    ids=["3x40", "5x60", "2x80-binary", "gapped-domain", "6x50-sparse"],
)
def test_alpha_matches_reference_package(level, n_raters, n_items, values, missing):
    """Same numbers as the reference implementation, missing data included."""
    rng = np.random.default_rng(n_raters * 1000 + n_items)
    arr = random_matrix(rng, n_raters, n_items, values, missing)
    df = to_long(arr)

    r = rater_agreement(df, level=level, bootstrap_ci=False)
    expected = kref.alpha(reliability_data=arr, level_of_measurement=level)
    assert r.alpha == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("level", LEVELS)
def test_alpha_matches_reference_on_continuous_ratings(level):
    """Non-integer ratings, where the value domain is nearly all distinct."""
    rng = np.random.default_rng(7)
    arr = np.round(rng.normal(3.0, 1.0, size=(3, 40)), 2)
    arr[rng.random(arr.shape) < 0.2] = np.nan
    for j in range(arr.shape[1]):
        if np.sum(~np.isnan(arr[:, j])) < 2:
            arr[:2, j] = np.round(rng.normal(3.0, 1.0, 2), 2)
    df = to_long(arr)

    r = rater_agreement(df, level=level, bootstrap_ci=False)
    expected = kref.alpha(reliability_data=arr, level_of_measurement=level)
    assert r.alpha == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("level", LEVELS)
def test_alpha_matches_slow_reference(level):
    """And the definition, looped over pairs."""
    rng = np.random.default_rng(11)
    arr = random_matrix(rng, 4, 35, [1, 2, 3, 4], missing=0.3)
    df = to_long(arr)
    r = rater_agreement(df, level=level, bootstrap_ci=False)
    assert r.alpha == pytest.approx(slow_alpha(df, level), abs=1e-9)


def test_levels_are_not_interchangeable():
    """A test suite that passes with the distance function ignored is no
    test suite. Ordered data has to move when the level changes."""
    rng = np.random.default_rng(5)
    arr = random_matrix(rng, 4, 40, [1, 2, 3, 4, 5], missing=0.3)
    df = to_long(arr)
    alphas = {
        level: rater_agreement(df, level=level, bootstrap_ci=False).alpha
        for level in LEVELS
    }
    assert len({round(a, 6) for a in alphas.values()}) == 3


def test_perfect_agreement_is_one():
    arr = np.array([[1, 2, 3, 4, 5, 1, 2], [1, 2, 3, 4, 5, 1, 2]], dtype=float)
    for level in LEVELS:
        r = rater_agreement(to_long(arr), level=level, bootstrap_ci=False)
        assert r.alpha == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Counts
# --------------------------------------------------------------------------

def test_counts_report_the_design():
    arr = np.array([
        [1, 2, 3, np.nan, np.nan],
        [1, 2, np.nan, 4, np.nan],
        [np.nan, np.nan, np.nan, np.nan, 5],
    ], dtype=float)
    r = rater_agreement(to_long(arr), bootstrap_ci=False)
    assert r.n_items == 5
    assert r.n_raters == 3
    # u0 and u1 have two raters each, u2 u3 u4 have one
    assert r.n_overlapping_items == 2


def test_overlapping_count_ignores_singly_graded_items():
    """Five hundred items graded once are five hundred items of nothing."""
    rows = [(f"solo{i}", "r0", 1.0) for i in range(500)]
    rows += [("shared", "r0", 1.0), ("shared", "r1", 2.0)]
    df = pd.DataFrame(rows, columns=["item_id", "rater_id", "rating"])
    r = rater_agreement(df, bootstrap_ci=False)
    assert r.n_items == 501
    assert r.n_overlapping_items == 1


# --------------------------------------------------------------------------
# item_disagreement
# --------------------------------------------------------------------------

@pytest.mark.parametrize("level", LEVELS)
def test_item_disagreement_matches_the_definition(level):
    rng = np.random.default_rng(13)
    arr = random_matrix(rng, 4, 30, [1, 2, 3, 4, 5], missing=0.3)
    df = to_long(arr)
    r = rater_agreement(df, level=level, bootstrap_ci=False)

    expected = slow_item_disagreement(df, level)
    table = r.item_disagreement.set_index("item_id")
    assert set(table.index) == set(expected)
    for item, value in expected.items():
        assert table.loc[item, "disagreement"] == pytest.approx(value, abs=1e-9)


def test_item_disagreement_has_the_columns_the_docstring_promises():
    rng = np.random.default_rng(2)
    df = to_long(random_matrix(rng, 3, 20, [1, 2, 3]))
    table = rater_agreement(df, bootstrap_ci=False).item_disagreement
    assert list(table.columns) == [
        "item_id", "n_ratings", "disagreement", "vs_chance"
    ]


def test_item_disagreement_sorted_descending():
    rng = np.random.default_rng(4)
    df = to_long(random_matrix(rng, 4, 40, [1, 2, 3, 4, 5]))
    table = rater_agreement(df, bootstrap_ci=False).item_disagreement
    values = table["disagreement"].to_numpy()
    assert np.all(np.diff(values) <= 1e-12)


def test_item_disagreement_reports_the_grading_count():
    """So a reader can tell a score built on ten ratings from one built on
    two."""
    rows = [
        ("thin", "r0", 1.0), ("thin", "r1", 2.0),
        ("thick", "r0", 1.0), ("thick", "r1", 2.0),
        ("thick", "r2", 1.0), ("thick", "r3", 2.0),
    ]
    df = pd.DataFrame(rows, columns=["item_id", "rater_id", "rating"])
    table = rater_agreement(df, bootstrap_ci=False).item_disagreement
    counts = dict(zip(table["item_id"], table["n_ratings"]))
    assert counts == {"thin": 2, "thick": 4}


def test_item_disagreement_does_not_punish_being_graded_more():
    """The normalisation is the whole point of this table.

    Both items are split half and half between two ratings. One was graded
    twice, the other eight times. An unnormalised contribution would put the
    eight-rating item four times higher for the same amount of confusion.
    """
    rows = [("thin", "r0", 1.0), ("thin", "r1", 2.0)]
    for i in range(8):
        rows.append(("thick", f"g{i}", float(1 + i % 2)))
    df = pd.DataFrame(rows, columns=["item_id", "rater_id", "rating"])
    table = rater_agreement(
        df, bootstrap_ci=False
    ).item_disagreement.set_index("item_id")

    thin = table.loc["thin", "disagreement"]
    thick = table.loc["thick", "disagreement"]
    # 2 ratings, always different: every ordered pair disagrees, so 1.0
    assert thin == pytest.approx(1.0)
    # 8 ratings, 4 and 4: 32 of 56 ordered pairs disagree
    assert thick == pytest.approx(32 / 56)
    assert thick < thin


def test_item_disagreement_is_zero_when_raters_agree():
    rows = [
        ("clean", "r0", 3.0), ("clean", "r1", 3.0), ("clean", "r2", 3.0),
        ("messy", "r0", 1.0), ("messy", "r1", 5.0), ("messy", "r2", 3.0),
    ]
    df = pd.DataFrame(rows, columns=["item_id", "rater_id", "rating"])
    table = rater_agreement(
        df, bootstrap_ci=False
    ).item_disagreement.set_index("item_id")
    assert table.loc["clean", "disagreement"] == pytest.approx(0.0)
    assert table.loc["messy", "disagreement"] > 0


def test_item_disagreement_excludes_singly_graded_items():
    rows = [("shared", "r0", 1.0), ("shared", "r1", 2.0), ("solo", "r0", 1.0)]
    df = pd.DataFrame(rows, columns=["item_id", "rater_id", "rating"])
    table = rater_agreement(df, bootstrap_ci=False).item_disagreement
    assert list(table["item_id"]) == ["shared"]


def test_vs_chance_is_one_at_chance_level():
    """vs_chance divides by the expected disagreement, so the average item
    in data with alpha near zero sits near one."""
    rng = np.random.default_rng(17)
    arr = rng.choice([1, 2, 3, 4, 5], size=(4, 400)).astype(float)
    df = to_long(arr)
    r = rater_agreement(df, bootstrap_ci=False)
    assert abs(r.alpha) < 0.05
    assert r.item_disagreement["vs_chance"].mean() == pytest.approx(1.0, abs=0.05)


# --------------------------------------------------------------------------
# rater_dropout
# --------------------------------------------------------------------------

def test_rater_dropout_has_the_columns_the_docstring_promises():
    rng = np.random.default_rng(6)
    df = to_long(random_matrix(rng, 4, 30, [1, 2, 3]))
    table = rater_agreement(df, bootstrap_ci=False).rater_dropout
    assert list(table.columns) == [
        "rater_id", "n_ratings", "alpha_without", "delta", "note"
    ]


@pytest.mark.parametrize("level", LEVELS)
def test_rater_dropout_recomputes_alpha_without_each_rater(level):
    rng = np.random.default_rng(19)
    df = to_long(random_matrix(rng, 5, 40, [1, 2, 3, 4], missing=0.3))
    r = rater_agreement(df, level=level, bootstrap_ci=False)

    for row in r.rater_dropout.itertuples():
        kept = df[df["rater_id"] != row.rater_id]
        expected = rater_agreement(kept, level=level, bootstrap_ci=False).alpha
        assert row.alpha_without == pytest.approx(expected, abs=1e-9)
        assert row.delta == pytest.approx(expected - r.alpha, abs=1e-9)


def test_rater_dropout_sorted_by_improvement():
    """The rater whose removal helps most comes first."""
    rng = np.random.default_rng(23)
    arr = np.zeros((5, 60))
    truth = rng.choice([1, 2, 3, 4, 5], size=60)
    for i in range(4):
        arr[i] = truth
    arr[4] = rng.choice([1, 2, 3, 4, 5], size=60)   # the odd one out
    df = to_long(arr)
    table = rater_agreement(df, bootstrap_ci=False).rater_dropout

    assert table.iloc[0]["rater_id"] == "r4"
    deltas = table["delta"].to_numpy()
    assert np.all(np.diff(deltas) <= 1e-12)


def test_rater_dropout_reports_the_grading_count():
    """A rater with three items out of five hundred must not top the table
    on noise alone, so the count is there to be read."""
    rows = []
    truth = np.random.default_rng(29).choice([1, 2, 3], size=500)
    for j, v in enumerate(truth):
        rows.append((f"u{j}", "heavy_a", float(v)))
        rows.append((f"u{j}", "heavy_b", float(v)))
    for j in range(3):
        rows.append((f"u{j}", "tourist", 5.0))
    df = pd.DataFrame(rows, columns=["item_id", "rater_id", "rating"])

    table = rater_agreement(
        df, bootstrap_ci=False
    ).rater_dropout.set_index("rater_id")
    assert table.loc["tourist", "n_ratings"] == 3
    assert table.loc["heavy_a", "n_ratings"] == 500
    assert table.loc["heavy_b", "n_ratings"] == 500


def test_rater_dropout_with_two_raters_is_undefined_not_perfect():
    """Removing one of two raters leaves nobody to disagree with.

    That is undefined, not perfect agreement, and the table has to say so
    rather than hand back a number somebody will quote.
    """
    rng = np.random.default_rng(31)
    df = to_long(random_matrix(rng, 2, 40, [1, 2, 3]))
    table = rater_agreement(df, bootstrap_ci=False).rater_dropout

    assert len(table) == 2
    assert table["alpha_without"].isna().all()
    assert table["delta"].isna().all()
    assert (table["note"] == NOTE_SINGLE_RATER).all()
    assert not table["note"].str.contains("1.0").any()


def test_rater_dropout_note_is_empty_when_alpha_is_computable():
    rng = np.random.default_rng(37)
    df = to_long(random_matrix(rng, 4, 30, [1, 2, 3]))
    table = rater_agreement(df, bootstrap_ci=False).rater_dropout
    assert (table["note"] == "").all()


def test_rater_dropout_handles_a_rater_whose_removal_breaks_the_overlap():
    """Three raters, but one of them is the only one who ever pairs up with
    anyone. Remove them and nothing is left to compare.

    Removing either of the other two leaves one shared item on which the two
    survivors happened to agree, which is undefined for the other reason.
    All three rows are NaN and no two of them are NaN for the same cause.
    """
    rows = [
        ("u0", "hub", 1.0), ("u0", "a", 1.0),
        ("u1", "hub", 2.0), ("u1", "b", 2.0),
    ]
    df = pd.DataFrame(rows, columns=["item_id", "rater_id", "rating"])
    table = rater_agreement(
        df, bootstrap_ci=False
    ).rater_dropout.set_index("rater_id")

    assert table["alpha_without"].isna().all()
    assert table.loc["hub", "note"] == NOTE_NO_OVERLAP
    assert table.loc["a", "note"] == NOTE_NO_VARIANCE
    assert table.loc["b", "note"] == NOTE_NO_VARIANCE


def test_rater_dropout_sorts_undefined_rows_last():
    """A rater with a number ranks above raters with no number at all.

    Dropping c leaves a and b on three shared items. Dropping either a or b
    leaves nothing comparable behind, so those two rows are undefined and
    belong at the bottom where nobody reads them as a ranking.
    """
    rows = [
        ("u0", "a", 1.0), ("u0", "b", 2.0),
        ("u1", "a", 2.0), ("u1", "b", 1.0),
        ("u2", "a", 1.0), ("u2", "b", 1.0),
        ("u3", "a", 3.0), ("u3", "c", 3.0),
    ]
    df = pd.DataFrame(rows, columns=["item_id", "rater_id", "rating"])
    table = rater_agreement(df, bootstrap_ci=False).rater_dropout

    assert table.iloc[0]["rater_id"] == "c"
    assert not np.isnan(table.iloc[0]["alpha_without"])
    assert table["alpha_without"].isna().sum() == 2
    assert table["alpha_without"].iloc[1:].isna().all()


def test_the_two_reasons_for_an_undefined_dropout_get_different_notes():
    """One NaN means there was nothing left to compare. The other means the
    comparison happened and came back with no variance to normalise by.

    They are the same value in the alpha column and opposite findings. The
    first says the grading design fell apart when that rater left. The
    second says the graders who remain agreed, which is the outcome a client
    is paying to hear about. A note rule keyed on the overlapping-item count
    gets the first right and the second exactly backwards.

    Same frame as the sort test above. Dropping "a" leaves b and c sharing
    no item at all. Dropping "b" leaves a and c sharing u3, where both said
    3, so alpha divides by zero.
    """
    rows = [
        ("u0", "a", 1.0), ("u0", "b", 2.0),
        ("u1", "a", 2.0), ("u1", "b", 1.0),
        ("u2", "a", 1.0), ("u2", "b", 1.0),
        ("u3", "a", 3.0), ("u3", "c", 3.0),
    ]
    df = pd.DataFrame(rows, columns=["item_id", "rater_id", "rating"])
    table = rater_agreement(
        df, bootstrap_ci=False
    ).rater_dropout.set_index("rater_id")

    assert np.isnan(table.loc["a", "alpha_without"])
    assert np.isnan(table.loc["b", "alpha_without"])

    assert table.loc["a", "note"] == NOTE_NO_OVERLAP
    assert table.loc["b", "note"] == NOTE_NO_VARIANCE
    assert table.loc["c", "note"] == ""

    # the wrong rule, spelled out. "fewer than two overlapping items" is
    # true of both rows and would collapse them onto one message.
    assert table.loc["a", "note"] != table.loc["b", "note"]


def test_the_three_undefined_notes_are_distinct_strings():
    """Three causes, three sentences. If any two coincide the column has
    stopped carrying the one thing it is for."""
    notes = {NOTE_SINGLE_RATER, NOTE_NO_OVERLAP, NOTE_NO_VARIANCE}
    assert len(notes) == 3
    assert "" not in notes

    # and each is reachable, so none is dead text
    two_raters = rater_agreement(
        pd.DataFrame(
            [("u0", "a", 1.0), ("u0", "b", 2.0), ("u1", "a", 2.0), ("u1", "b", 1.0)],
            columns=["item_id", "rater_id", "rating"],
        ),
        bootstrap_ci=False,
    ).rater_dropout
    reached = set(two_raters["note"])

    three_raters = rater_agreement(
        pd.DataFrame(
            [
                ("u0", "a", 1.0), ("u0", "b", 2.0),
                ("u1", "a", 2.0), ("u1", "b", 1.0),
                ("u2", "a", 1.0), ("u2", "b", 1.0),
                ("u3", "a", 3.0), ("u3", "c", 3.0),
            ],
            columns=["item_id", "rater_id", "rating"],
        ),
        bootstrap_ci=False,
    ).rater_dropout
    reached |= set(three_raters["note"])

    assert notes <= reached


# --------------------------------------------------------------------------
# The bootstrap interval
# --------------------------------------------------------------------------

def test_bootstrap_is_deterministic_under_a_seed():
    rng = np.random.default_rng(43)
    df = to_long(random_matrix(rng, 4, 40, [1, 2, 3, 4]))
    a = rater_agreement(df, n_boot=300, seed=1)
    b = rater_agreement(df, n_boot=300, seed=1)
    c = rater_agreement(df, n_boot=300, seed=2)
    assert a.ci_low == b.ci_low and a.ci_high == b.ci_high
    assert (a.ci_low, a.ci_high) != (c.ci_low, c.ci_high)


def test_bootstrap_can_be_switched_off():
    rng = np.random.default_rng(47)
    df = to_long(random_matrix(rng, 3, 30, [1, 2, 3]))
    r = rater_agreement(df, bootstrap_ci=False)
    assert np.isnan(r.ci_low) and np.isnan(r.ci_high)
    assert not np.isnan(r.alpha)


def test_bootstrap_brackets_the_estimate():
    rng = np.random.default_rng(53)
    df = to_long(random_matrix(rng, 4, 60, [1, 2, 3, 4]))
    r = rater_agreement(df, n_boot=500, seed=3)
    assert r.ci_low <= r.alpha <= r.ci_high


def test_bootstrap_interval_narrows_with_more_items():
    rng = np.random.default_rng(59)
    small = to_long(random_matrix(rng, 3, 25, [1, 2, 3, 4, 5]))
    large = to_long(random_matrix(rng, 3, 800, [1, 2, 3, 4, 5]))
    r_small = rater_agreement(small, n_boot=500, seed=4)
    r_large = rater_agreement(large, n_boot=500, seed=4)
    assert (r_large.ci_high - r_large.ci_low) < (r_small.ci_high - r_small.ci_low)


def test_bootstrap_respects_the_confidence_level():
    rng = np.random.default_rng(61)
    df = to_long(random_matrix(rng, 4, 60, [1, 2, 3, 4]))
    narrow = rater_agreement(df, confidence=0.80, n_boot=800, seed=5)
    wide = rater_agreement(df, confidence=0.99, n_boot=800, seed=5)
    assert (wide.ci_high - wide.ci_low) > (narrow.ci_high - narrow.ci_low)
    assert narrow.confidence == 0.80


def test_bootstrap_resamples_items_not_ratings():
    """Resampling ratings would break items apart and destroy the very
    structure alpha measures. Resampling items keeps each item whole.

    With perfect agreement every resample is still perfect, so the interval
    collapses onto 1.0. Resampling individual ratings would not do that.
    """
    truth = np.random.default_rng(67).choice([1, 2, 3, 4, 5], size=50)
    arr = np.vstack([truth, truth]).astype(float)
    r = rater_agreement(to_long(arr), n_boot=300, seed=6)
    assert r.ci_low == pytest.approx(1.0)
    assert r.ci_high == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Designs where the bootstrap runs out of overlap
#
# The interval resamples the items that were graded twice or more, not every
# item in the frame. Solo items contribute nothing to alpha and nothing to
# its sampling error, so resampling them would let the double-grading rate,
# which is a scheduling decision, leak into a statistic about raters.
#
# That choice does not make the problem go away, it moves it. A resample can
# still come back undefined, because if it happens to draw only items every
# rater agreed on then the expected disagreement is zero and alpha is 0/0.
# Undefined resamples are not missing at random. They are exactly the
# unanimous ones, so dropping them shaves the top off the distribution and
# drags the upper bound down. That is tolerable in small doses and dishonest
# in large ones, so there is a floor on how many may be dropped.
# --------------------------------------------------------------------------

def test_bootstrap_resamples_the_overlapping_items():
    """Five hundred solo items must not move the interval by a hair.

    They carry no information about agreement. If they change the bounds,
    the bootstrap is resampling the wrong thing.
    """
    rng = np.random.default_rng(211)
    df = to_long(random_matrix(rng, 3, 30, [1, 2, 3]))
    padded = with_solo_items(df, 500)

    lean = rater_agreement(df, n_boot=500, seed=20)
    fat = rater_agreement(padded, n_boot=500, seed=20)

    assert fat.n_items == lean.n_items + 500
    assert fat.n_overlapping_items == lean.n_overlapping_items
    assert fat.alpha == pytest.approx(lean.alpha, abs=1e-12)
    assert fat.ci_low == pytest.approx(lean.ci_low, abs=1e-12)
    assert fat.ci_high == pytest.approx(lean.ci_high, abs=1e-12)


def test_one_overlapping_item_yields_no_alpha_and_no_interval():
    """The frame the README opens with. Five hundred graded once, one graded
    twice.

    The arithmetic does return a value at this size, and that is the trap.
    One item cannot carry a reliability estimate, and it cannot be resampled
    into a distribution either, so both the number and the width around it
    would be artefacts. A reader skimming a report quotes whatever number is
    in front of them, so there must not be one.
    """
    df = with_solo_items(
        pd.DataFrame(
            [("shared", "r0", 1.0), ("shared", "r1", 2.0)],
            columns=["item_id", "rater_id", "rating"],
        ),
        500,
    )
    r = rater_agreement(df, n_boot=1000, seed=21)

    assert r.n_overlapping_items == 1
    assert np.isnan(r.alpha)
    assert np.isnan(r.ci_low) and np.isnan(r.ci_high)

    text = r.summary().lower()
    assert "1 of 501 items" in r.summary()
    assert "undefined" in text
    assert "no interval" in text
    assert "cannot distinguish" in text
    # the withheld value must not appear anywhere in the prose
    assert "0.000" not in r.summary()


def test_the_two_reasons_for_no_alpha_read_differently():
    """One item is a scheduling problem. No variance is a finding.

    Both come back NaN and the summary has to separate them, the same way
    the dropout notes do, and in the same order.
    """
    thin = rater_agreement(
        pd.DataFrame(
            [("shared", "r0", 1.0), ("shared", "r1", 2.0)],
            columns=["item_id", "rater_id", "rating"],
        ),
        bootstrap_ci=False,
    )
    agreed = rater_agreement(
        pd.DataFrame(
            [(f"u{i}", r, 1.0) for i in range(20) for r in ("r0", "r1")],
            columns=["item_id", "rater_id", "rating"],
        ),
        bootstrap_ci=False,
    )

    assert np.isnan(thin.alpha) and np.isnan(agreed.alpha)
    assert thin.summary() != agreed.summary()
    assert "one item cannot carry" in thin.summary().lower()
    assert "nobody varied" in agreed.summary().lower()


def test_two_overlapping_items_is_the_floor_not_one():
    """Two is where a number starts being reported. Pin both sides of it so
    the threshold cannot drift."""
    rows = [("a", "r0", 1.0), ("a", "r1", 2.0)]
    one = rater_agreement(
        pd.DataFrame(rows, columns=["item_id", "rater_id", "rating"]),
        bootstrap_ci=False,
    )
    two = rater_agreement(
        pd.DataFrame(
            rows + [("b", "r0", 2.0), ("b", "r1", 1.0)],
            columns=["item_id", "rater_id", "rating"],
        ),
        bootstrap_ci=False,
    )
    assert one.n_overlapping_items == 1 and np.isnan(one.alpha)
    assert two.n_overlapping_items == 2 and not np.isnan(two.alpha)


def test_dropout_says_when_a_removal_leaves_one_item_rather_than_none():
    """A fourth reason a leave-one-out alpha goes missing, distinct from the
    other three. Dropping c leaves a and b sharing only u0, which has real
    disagreement on it, so this is neither an empty overlap nor a scale
    nobody varied. It is one item, and one item is not enough.
    """
    rows = [
        ("u0", "a", 1.0), ("u0", "b", 2.0),
        ("u1", "a", 2.0), ("u1", "c", 1.0),
        ("u2", "b", 1.0), ("u2", "c", 3.0),
    ]
    df = pd.DataFrame(rows, columns=["item_id", "rater_id", "rating"])
    table = rater_agreement(
        df, bootstrap_ci=False
    ).rater_dropout.set_index("rater_id")

    assert table.loc["c", "note"] == NOTE_ONE_ITEM
    assert np.isnan(table.loc["c", "alpha_without"])
    assert NOTE_ONE_ITEM not in (NOTE_SINGLE_RATER, NOTE_NO_OVERLAP, NOTE_NO_VARIANCE)


def test_bootstrap_drops_a_thin_share_of_undefined_resamples():
    """Four split items in forty. About 1.5% of resamples draw none of them
    and come back undefined. Drop those, keep the interval, say how many
    survived."""
    df = mostly_unanimous(n_items=40, n_split=4)
    r = rater_agreement(df, n_boot=1000, seed=22)

    assert not np.isnan(r.ci_low) and not np.isnan(r.ci_high)
    assert r.n_boot == 1000
    assert r.n_boot_usable < 1000            # some really were dropped
    assert r.n_boot_usable >= 900            # but nowhere near the floor
    assert r.ci_low <= r.alpha <= r.ci_high


def test_bootstrap_refuses_when_too_many_resamples_are_undefined():
    """One split item in six. A third of resamples miss it entirely.

    Percentiles of the two thirds that survive are percentiles of a
    conditional distribution, conditioned on the resample having enough
    disagreement to define alpha at all. That is not the interval anybody
    thinks they are reading.
    """
    df = mostly_unanimous(n_items=6, n_split=1)
    r = rater_agreement(df, n_boot=1000, seed=23)

    assert not np.isnan(r.alpha)
    assert r.n_boot_usable < 900
    assert np.isnan(r.ci_low) and np.isnan(r.ci_high)

    text = r.summary().lower()
    assert "no interval" in text
    assert "undefined" in text
    assert "cannot distinguish" in text


def test_the_undefined_resample_floor_is_ninety_percent():
    """The threshold, pinned. Nine hundred usable resamples in a thousand.

    Where the boundary itself falls is checked by the test below. This one
    only asserts that the interval appears and disappears with the count.
    """
    seen = []
    for n_items, n_split in [(40, 4), (40, 3), (40, 2), (6, 1), (30, 5)]:
        r = rater_agreement(
            mostly_unanimous(n_items, n_split), n_boot=1000, seed=24
        )
        has_interval = not np.isnan(r.ci_low)
        assert has_interval is (r.n_boot_usable >= 900), (
            f"{n_items}/{n_split}: usable={r.n_boot_usable} "
            f"interval={has_interval}"
        )
        seen.append(has_interval)

    # the sweep has to straddle the floor, or it pins nothing
    assert True in seen and False in seen


def test_the_floor_is_inclusive_at_exactly_nine_hundred():
    """Land on the boundary and check which side it belongs to.

    The other floor test sweeps 993, 980, 948, 845 and 671 usable resamples.
    Nothing there lands near 900, so ">= 900" and "> 900" behave identically
    across all of it and the boundary goes unchecked.

    Partial splits do not help. An item either carries two distinct ratings
    or it does not, and a resample is undefined only when every item in it
    is unanimous on the same value, so splitting two of three raters instead
    of all three moves nothing. The knob that does move in fine steps is the
    item count: the undefined share is ((n - k) / n) ** n, which for k=2
    walks 0.0878, 0.0949, 0.1001, 0.1042 as n goes 6, 7, 8, 9. Eight items
    put it at 0.100113, so the usable count lands within a few of 900 and
    both sides of the boundary come up across seeds.
    """
    df = mostly_unanimous(n_items=8, n_split=2)
    landed = {}

    for seed in range(400):
        r = rater_agreement(df, n_boot=1000, seed=seed)
        usable = r.n_boot_usable
        has_interval = not np.isnan(r.ci_low)
        assert has_interval is (usable >= 900), f"seed {seed}: usable={usable}"
        landed.setdefault(usable, has_interval)
        if 900 in landed and 899 in landed:
            break

    assert 900 in landed, "no seed landed on the floor, boundary unchecked"
    assert 899 in landed, "no seed landed just under the floor"
    assert landed[900] is True, "exactly 900 usable must keep the interval"
    assert landed[899] is False, "899 usable must lose it"


def test_n_boot_usable_is_the_full_count_on_healthy_data():
    rng = np.random.default_rng(227)
    df = to_long(random_matrix(rng, 4, 60, [1, 2, 3, 4, 5]))
    r = rater_agreement(df, n_boot=800, seed=25)
    assert r.n_boot_usable == 800
    assert r.n_boot == 800


def test_no_resamples_are_drawn_when_the_bootstrap_is_off():
    rng = np.random.default_rng(229)
    df = to_long(random_matrix(rng, 3, 30, [1, 2, 3]))
    r = rater_agreement(df, bootstrap_ci=False, n_boot=1000)
    assert r.n_boot == 0
    assert r.n_boot_usable == 0
    assert np.isnan(r.ci_low) and np.isnan(r.ci_high)


def test_a_refused_interval_reads_differently_from_one_never_asked_for():
    """Turning the bootstrap off and having it refuse are different events
    and must not produce the same sentence."""
    refused = rater_agreement(
        mostly_unanimous(n_items=6, n_split=1), n_boot=1000, seed=26
    ).summary()
    never_asked = rater_agreement(
        mostly_unanimous(n_items=6, n_split=1), bootstrap_ci=False
    ).summary()
    assert refused != never_asked
    assert "undefined" in refused.lower()
    assert "not requested" in never_asked.lower()


# --------------------------------------------------------------------------
# The bootstrap at every level of measurement
#
# Nominal and interval reach the denominator through closed forms and never
# build a distance matrix. Ordinal cannot, because its distances are read off
# the marginals, so they have to be rebuilt for every resample. That is a
# whole code path, and a suite that only ever bootstraps at the default level
# never runs it.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("level", LEVELS)
def test_bootstrap_runs_at_every_level(level):
    df = varied_items(n_items=40, seed=1)
    r = rater_agreement(df, level=level, n_boot=600, seed=30)

    assert not np.isnan(r.ci_low) and not np.isnan(r.ci_high)
    assert r.ci_low <= r.alpha <= r.ci_high
    assert r.n_boot_usable == 600           # nothing degenerate to drop
    assert r.level == level


@pytest.mark.parametrize("level", LEVELS)
def test_bootstrap_is_deterministic_at_every_level(level):
    df = varied_items(n_items=25, seed=2)
    a = rater_agreement(df, level=level, n_boot=400, seed=31)
    b = rater_agreement(df, level=level, n_boot=400, seed=31)
    c = rater_agreement(df, level=level, n_boot=400, seed=32)
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)
    assert (a.ci_low, a.ci_high) != (c.ci_low, c.ci_high)


@pytest.mark.parametrize("level", LEVELS)
def test_bootstrap_matches_a_loop_written_from_the_definition(level):
    """The vectorised bootstrap against one resample at a time.

    Every shortcut the fast path takes has to survive this. The closed forms
    that replace the distance matrix for nominal and interval, the single
    bincount that builds the multiplicity matrix, and for ordinal the
    per-resample rebuild of the metric.
    """
    df = varied_items(n_items=12, seed=3)
    r = rater_agreement(df, level=level, n_boot=120, seed=33)
    lo, hi, usable = slow_bootstrap_ci(df, level, n_boot=120, seed=33)

    assert r.n_boot_usable == usable
    assert r.ci_low == pytest.approx(lo, abs=1e-9)
    assert r.ci_high == pytest.approx(hi, abs=1e-9)


def test_the_ordinal_metric_moves_between_resamples():
    """The reference test above only has teeth for ordinal if the distances
    genuinely differ from one resample to the next.

    If they happened to be constant, an implementation that computed them
    once from the full sample would pass and still be wrong on other data.
    Show they are not constant.
    """
    df = varied_items(n_items=12, seed=3)
    items = list(dict.fromkeys(df["item_id"]))
    blocks = {i: df[df["item_id"] == i] for i in items}
    idx = np.random.default_rng(33).integers(0, len(items), size=(6, len(items)))

    seen = []
    for row in idx:
        frame = pd.concat(
            [
                blocks[items[p]].assign(item_id=f"{items[p]}#copy{c}")
                for c, p in enumerate(row)
            ],
            ignore_index=True,
        )
        units = _slow_units(frame)
        coincidence, domain = _slow_coincidence(units)
        seen.append(np.asarray(_slow_distances(domain, coincidence.sum(axis=1), "ordinal")))

    full_units = _slow_units(df)
    full_coincidence, full_domain = _slow_coincidence(full_units)
    full = np.asarray(
        _slow_distances(full_domain, full_coincidence.sum(axis=1), "ordinal")
    )

    assert any(d.shape != full.shape or not np.allclose(d, full) for d in seen)
    assert any(
        seen[i].shape != seen[j].shape or not np.allclose(seen[i], seen[j])
        for i in range(len(seen))
        for j in range(i + 1, len(seen))
    )


@pytest.mark.parametrize("level", LEVELS)
def test_the_block_loop_does_not_change_the_answer(level):
    """The bootstrap runs in memory-bounded blocks. Blocks are an allocation
    detail and must not move a single bound, however small they get."""
    import evalaudit.agreement as agreement

    df = varied_items(n_items=20, seed=4)
    one_block = rater_agreement(df, level=level, n_boot=500, seed=34)

    original = agreement._BLOCK_BUDGET
    agreement._BLOCK_BUDGET = 50      # forces many small blocks
    try:
        many_blocks = rater_agreement(df, level=level, n_boot=500, seed=34)
    finally:
        agreement._BLOCK_BUDGET = original

    assert many_blocks.ci_low == pytest.approx(one_block.ci_low, abs=1e-12)
    assert many_blocks.ci_high == pytest.approx(one_block.ci_high, abs=1e-12)
    assert many_blocks.n_boot_usable == one_block.n_boot_usable


def test_bootstrap_coverage():
    """The 95% interval should cover the population alpha about 95% of the
    time when items are drawn from a fixed population.

    There is no closed form for the alpha of a generating process, so the
    target is alpha computed on one large fixed pool of items. Each trial
    draws a sample of items from that pool. The bounds are wider than the
    ones in test_compare.py because alpha's sampling distribution is skewed
    and the percentile bootstrap does not fully correct for that.
    """
    rng = np.random.default_rng(71)
    pool_size = 6000
    n_raters = 3
    truth = rng.choice([1, 2, 3, 4, 5], size=pool_size)
    pool = np.empty((n_raters, pool_size))
    for i in range(n_raters):
        noise = rng.random(pool_size) < 0.3
        pool[i] = np.where(noise, rng.choice([1, 2, 3, 4, 5], size=pool_size), truth)

    population_alpha = kref.alpha(
        reliability_data=pool, level_of_measurement="nominal"
    )

    trials = 200
    n = 120
    covered = 0
    for _ in range(trials):
        take = rng.choice(pool_size, size=n, replace=False)
        df = to_long(pool[:, take])
        r = rater_agreement(df, n_boot=400, seed=int(rng.integers(1_000_000)))
        if r.ci_low <= population_alpha <= r.ci_high:
            covered += 1
    rate = covered / trials
    assert 0.88 <= rate <= 0.99, f"coverage was {rate:.3f}"


# --------------------------------------------------------------------------
# summary(), and refusing to read noise as signal
# --------------------------------------------------------------------------

def test_summary_refuses_when_nothing_was_graded_twice():
    rows = [(f"u{i}", "r0", float(i % 3)) for i in range(40)]
    rows += [(f"v{i}", "r1", float(i % 3)) for i in range(40)]
    df = pd.DataFrame(rows, columns=["item_id", "rater_id", "rating"])
    r = rater_agreement(df)

    assert r.n_overlapping_items == 0
    assert np.isnan(r.alpha)
    text = r.summary().lower()
    assert "never measurable" in text
    assert "graded" in text
    assert "alpha 0." not in text and "alpha of 0." not in text


def test_summary_names_an_outlier_when_the_data_can_carry_it():
    rng = np.random.default_rng(73)
    truth = rng.choice([1, 2, 3, 4, 5], size=300)
    arr = np.empty((4, 300))
    for i in range(3):
        noise = rng.random(300) < 0.08
        arr[i] = np.where(noise, rng.choice([1, 2, 3, 4, 5], size=300), truth)
    arr[3] = rng.choice([1, 2, 3, 4, 5], size=300)
    df = to_long(arr)

    r = rater_agreement(df, n_boot=500, seed=8)
    assert r.dropout_is_distinguishable
    assert r.top_dropout_rater == "r3"
    assert NAMES_A_RATER + "r3" in r.summary()


def test_summary_refuses_to_name_an_outlier_on_thin_data():
    """Six overlapping items cannot separate three raters, and a table that
    still ranks somebody first invites the reader to act on noise."""
    rng = np.random.default_rng(79)
    arr = rng.choice([1, 2, 3, 4, 5], size=(3, 6)).astype(float)
    df = to_long(arr)

    r = rater_agreement(df, n_boot=500, seed=9)
    assert not r.dropout_is_distinguishable
    assert r.top_dropout_rater is None
    assert NAMES_A_RATER not in r.summary()
    assert "cannot distinguish" in r.summary().lower()


def test_the_naming_rule_is_the_shift_against_the_sampling_error():
    """The rule, asserted directly. A rater is named only when removing them
    shifts alpha by more than the sampling error on alpha.

    The obvious alternative, comparing the leave-one-out alpha to ci_high,
    fails on data that is nothing but noise. When alpha sits near zero the
    interval runs far below it and barely above, so ci_high is a low bar and
    a rater picked out of six random items clears it. The test above pins
    that case; this one pins the rule that makes it come out right.
    """
    for seed, n_items, n_noisy in [(83, 300, 1), (89, 8, 1), (97, 40, 0)]:
        rng = np.random.default_rng(seed)
        truth = rng.choice([1, 2, 3, 4, 5], size=n_items)
        arr = np.empty((4, n_items))
        for i in range(4 - n_noisy):
            noise = rng.random(n_items) < 0.1
            arr[i] = np.where(
                noise, rng.choice([1, 2, 3, 4, 5], size=n_items), truth
            )
        for i in range(4 - n_noisy, 4):
            arr[i] = rng.choice([1, 2, 3, 4, 5], size=n_items)

        r = rater_agreement(to_long(arr), n_boot=500, seed=seed)
        shift = r.rater_dropout["delta"].max()
        sampling_error = (r.ci_high - r.ci_low) / 2
        expected = bool(np.isfinite(shift) and shift > sampling_error)
        assert r.dropout_is_distinguishable is expected

    # and the rule the implementation must not be using instead
    rng = np.random.default_rng(79)
    noise = rater_agreement(
        to_long(rng.choice([1, 2, 3, 4, 5], size=(3, 6)).astype(float)),
        n_boot=500,
        seed=9,
    )
    assert noise.rater_dropout["alpha_without"].max() > noise.ci_high
    assert not noise.dropout_is_distinguishable


def test_summary_will_not_name_anyone_without_an_interval():
    """No bootstrap means no way to tell signal from noise, so no name."""
    rng = np.random.default_rng(101)
    truth = rng.choice([1, 2, 3, 4, 5], size=300)
    arr = np.empty((4, 300))
    for i in range(3):
        arr[i] = truth
    arr[3] = rng.choice([1, 2, 3, 4, 5], size=300)
    r = rater_agreement(to_long(arr), bootstrap_ci=False)
    assert not r.dropout_is_distinguishable
    assert r.top_dropout_rater is None
    assert NAMES_A_RATER not in r.summary()
    assert "cannot distinguish" in r.summary().lower()


def test_summary_will_not_name_anyone_with_only_two_raters():
    """Every leave-one-out alpha is undefined here, so there is nobody to
    name and no sentence that names one."""
    rng = np.random.default_rng(103)
    arr = rng.choice([1, 2, 3], size=(2, 50)).astype(float)
    r = rater_agreement(to_long(arr), n_boot=300, seed=10)
    assert r.top_dropout_rater is None
    assert NAMES_A_RATER not in r.summary()


def test_summary_reports_alpha_and_the_interval():
    rng = np.random.default_rng(107)
    df = to_long(random_matrix(rng, 4, 80, [1, 2, 3]))
    text = rater_agreement(df, n_boot=400, seed=11).summary()
    assert "95%" in text
    assert "alpha" in text.lower()


def test_summary_flags_alpha_below_the_conventional_floor():
    """Alpha under 0.667 is the line Krippendorff draws for data nobody
    should draw conclusions from, and the summary has to say it."""
    rng = np.random.default_rng(109)
    arr = rng.choice([1, 2, 3, 4, 5], size=(3, 200)).astype(float)
    r = rater_agreement(to_long(arr), n_boot=300, seed=12)
    assert r.alpha < 0.667
    assert "0.667" in r.summary()


def test_summary_mentions_how_many_items_carried_the_estimate():
    rows = [(f"u{i}", "r0", float(i % 3)) for i in range(100)]
    rows += [(f"u{i}", "r1", float(i % 3)) for i in range(10)]
    df = pd.DataFrame(rows, columns=["item_id", "rater_id", "rating"])
    r = rater_agreement(df, n_boot=300, seed=13)
    assert r.n_overlapping_items == 10
    assert "10 of 100 items" in r.summary()


def test_summary_is_a_string_in_every_branch():
    """Whatever the shape of the data, summary() returns prose."""
    frames = [
        pd.DataFrame(
            [("a", "r0", 1.0), ("a", "r1", 1.0)],
            columns=["item_id", "rater_id", "rating"],
        ),
        pd.DataFrame(
            [("a", "r0", 1.0), ("b", "r1", 2.0)],
            columns=["item_id", "rater_id", "rating"],
        ),
        to_long(np.random.default_rng(113).choice([1, 2], size=(3, 20)).astype(float)),
    ]
    for df in frames:
        text = rater_agreement(df, n_boot=200, seed=14).summary()
        assert isinstance(text, str) and len(text) > 20


def test_alpha_is_undefined_when_every_comparable_rating_is_identical():
    """No variation means no denominator. That is undefined, not perfect."""
    rows = [(f"u{i}", r, 1.0) for i in range(20) for r in ("r0", "r1")]
    df = pd.DataFrame(rows, columns=["item_id", "rater_id", "rating"])
    r = rater_agreement(df, bootstrap_ci=False)
    assert np.isnan(r.alpha)
    assert "undefined" in r.summary().lower()


# --------------------------------------------------------------------------
# Cohen's kappa, checked against statsmodels
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "r1, r2",
    [
        ([1, 1, 1, 0, 0, 0, 1, 0, 1, 0], [1, 0, 1, 1, 0, 0, 1, 1, 1, 0]),
        ([2, 1, 3, 2, 1, 3, 3, 2, 1, 1, 2, 3], [2, 2, 3, 2, 1, 1, 3, 2, 1, 2, 2, 3]),
        ([1] * 20 + [0] * 20, [1] * 15 + [0] * 5 + [1] * 5 + [0] * 15),
    ],
    ids=["binary", "three-way", "lopsided"],
)
def test_cohens_kappa_matches_statsmodels(r1, r2):
    result = cohens_kappa(r1, r2)

    cats = sorted(set(r1) | set(r2))
    idx = {c: i for i, c in enumerate(cats)}
    table = np.zeros((len(cats), len(cats)))
    for a, b in zip(r1, r2):
        table[idx[a], idx[b]] += 1
    expected = sm_cohens_kappa(table).kappa

    assert result.kappa == pytest.approx(expected, abs=1e-9)
    assert result.n_items == len(r1)
    assert result.n_raters == 2
    assert result.method == "cohen"


def test_cohens_kappa_perfect_and_chance():
    perfect = cohens_kappa([1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3])
    assert perfect.kappa == pytest.approx(1.0)
    assert perfect.p_observed == pytest.approx(1.0)

    rng = np.random.default_rng(127)
    a = rng.choice([0, 1], size=4000)
    b = rng.choice([0, 1], size=4000)
    assert abs(cohens_kappa(a, b).kappa) < 0.06


def test_cohens_kappa_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        cohens_kappa([1, 2, 3], [1, 2])


def test_cohens_kappa_rejects_empty():
    with pytest.raises(ValueError):
        cohens_kappa([], [])


# --------------------------------------------------------------------------
# Fleiss' kappa, checked against statsmodels
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n_raters, n_items, values", [
    (3, 30, [1, 2, 3]),
    (5, 25, [0, 1]),
    (4, 40, [1, 2, 3, 4, 5]),
])
def test_fleiss_kappa_matches_statsmodels(n_raters, n_items, values):
    rng = np.random.default_rng(131)
    arr = rng.choice(values, size=(n_raters, n_items)).astype(float)
    df = to_long(arr)

    result = fleiss_kappa(df)

    cats = sorted(set(values))
    table = np.zeros((n_items, len(cats)))
    for j in range(n_items):
        for i in range(n_raters):
            table[j, cats.index(arr[i, j])] += 1
    expected = sm_fleiss_kappa(table)

    assert result.kappa == pytest.approx(expected, abs=1e-9)
    assert result.n_items == n_items
    assert result.n_raters == n_raters
    assert result.method == "fleiss"


def test_fleiss_kappa_needs_the_same_number_of_raters_per_item():
    """Fleiss is only defined on a fixed rater count. Refuse rather than
    quietly compute something else, and point at the function that does
    handle ragged data."""
    rows = [
        ("u0", "r0", 1.0), ("u0", "r1", 1.0), ("u0", "r2", 2.0),
        ("u1", "r0", 1.0), ("u1", "r1", 2.0),
    ]
    df = pd.DataFrame(rows, columns=["item_id", "rater_id", "rating"])
    with pytest.raises(ValueError, match="rater_agreement"):
        fleiss_kappa(df)


def test_fleiss_kappa_perfect_agreement():
    rng = np.random.default_rng(137)
    truth = rng.choice([1, 2, 3], size=30)
    arr = np.vstack([truth, truth, truth]).astype(float)
    assert fleiss_kappa(to_long(arr)).kappa == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------

def test_rejects_missing_columns():
    df = pd.DataFrame({"item": ["a"], "rater_id": ["r0"], "rating": [1.0]})
    with pytest.raises(ValueError, match="item_id"):
        rater_agreement(df)


def test_rejects_empty_frame():
    df = pd.DataFrame(columns=["item_id", "rater_id", "rating"])
    with pytest.raises(ValueError):
        rater_agreement(df)


def test_rejects_unknown_level():
    rng = np.random.default_rng(139)
    df = to_long(random_matrix(rng, 3, 10, [1, 2, 3]))
    with pytest.raises(ValueError, match="level"):
        rater_agreement(df, level="ratio")


def test_rejects_duplicate_rater_item_pairs():
    """Two rows for the same rater on the same item double-count that
    rater's opinion and push alpha up. Refuse rather than inflate."""
    rows = [("u0", "r0", 1.0), ("u0", "r0", 2.0), ("u0", "r1", 1.0)]
    df = pd.DataFrame(rows, columns=["item_id", "rater_id", "rating"])
    with pytest.raises(ValueError, match="duplicate"):
        rater_agreement(df)


def test_drops_rows_with_a_missing_rating():
    """A blank rating is an ungraded item, not a rating of zero."""
    rows = [
        ("u0", "r0", 1.0), ("u0", "r1", 1.0),
        ("u1", "r0", 2.0), ("u1", "r1", np.nan),
    ]
    df = pd.DataFrame(rows, columns=["item_id", "rater_id", "rating"])
    r = rater_agreement(df, bootstrap_ci=False)
    assert r.n_overlapping_items == 1
    assert r.n_items == 2


def test_does_not_mutate_the_input_frame():
    rng = np.random.default_rng(149)
    df = to_long(random_matrix(rng, 3, 20, [1, 2, 3]))
    before = df.copy()
    rater_agreement(df, n_boot=200, seed=15)
    pd.testing.assert_frame_equal(df, before)


def test_column_order_does_not_matter():
    rng = np.random.default_rng(151)
    df = to_long(random_matrix(rng, 3, 20, [1, 2, 3]))
    shuffled = df[["rating", "rater_id", "item_id"]]
    assert rater_agreement(df, bootstrap_ci=False).alpha == pytest.approx(
        rater_agreement(shuffled, bootstrap_ci=False).alpha
    )


def test_string_rater_and_item_ids_survive():
    rows = [
        ("essay-01", "alice", 3.0), ("essay-01", "bob", 4.0),
        ("essay-02", "alice", 2.0), ("essay-02", "bob", 2.0),
    ]
    df = pd.DataFrame(rows, columns=["item_id", "rater_id", "rating"])
    r = rater_agreement(df, bootstrap_ci=False)
    assert set(r.rater_dropout["rater_id"]) == {"alice", "bob"}
    assert set(r.item_disagreement["item_id"]) == {"essay-01", "essay-02"}


def test_string_ratings_survive_at_nominal():
    """Labels, not numbers, in the rating column.

    The common shape for a categorical rubric. Graders tick "pass" or
    "fail" and nobody maps them to integers on the way in. Nominal is
    defined on unordered labels, so this has to work without the caller
    doing that mapping.

    Checked by relabelling rather than against a fixed number. Alpha does
    not care what the categories are called, only which ratings match, so
    the string version and the same data mapped to integers have to agree
    exactly, and both have to match the reference package on the integers.

    This was broken for the life of the module. _coded_values built the
    value domain from the row count and returned the unique labels where
    the per-row codes belonged, so a non-numeric rating column raised
    IndexError out of the coincidence matrix. It surfaced through
    judge_validation, where string labels are the normal case, and the
    regression belongs here where the code lives.
    """
    rng = np.random.default_rng(163)
    labels = np.array(["pass", "fail", "borderline"])
    codes = rng.integers(0, 3, size=(4, 30))

    lettered = to_long(codes.astype(float)).assign(
        rating=lambda d: labels[d["rating"].astype(int)]
    )
    numeric = to_long(codes.astype(float))

    from_labels = rater_agreement(lettered, bootstrap_ci=False).alpha
    from_numbers = rater_agreement(numeric, bootstrap_ci=False).alpha

    assert from_labels == pytest.approx(from_numbers, abs=1e-12)
    assert from_labels == pytest.approx(
        kref.alpha(reliability_data=codes, level_of_measurement="nominal"),
        abs=1e-9,
    )
    assert 0.0 < from_labels < 1.0, "the fixture stopped exercising the metric"


def test_string_ratings_are_refused_at_the_ordered_levels():
    """The other half. Ordinal and interval read the values as positions, so
    labels there are a question the data cannot answer.
    """
    rows = [
        ("i1", "alice", "pass"), ("i1", "bob", "fail"),
        ("i2", "alice", "fail"), ("i2", "bob", "fail"),
    ]
    df = pd.DataFrame(rows, columns=["item_id", "rater_id", "rating"])

    for level in ["ordinal", "interval"]:
        with pytest.raises(ValueError, match="numeric"):
            rater_agreement(df, level=level, bootstrap_ci=False)


# --------------------------------------------------------------------------
# Public surface
# --------------------------------------------------------------------------

def test_returns_a_frozen_agreement_result():
    rng = np.random.default_rng(157)
    df = to_long(random_matrix(rng, 3, 20, [1, 2, 3]))
    r = rater_agreement(df, bootstrap_ci=False)
    assert isinstance(r, AgreementResult)
    with pytest.raises(Exception):
        r.alpha = 0.5


def test_kappas_return_a_frozen_kappa_result():
    r = cohens_kappa([1, 0, 1], [1, 1, 1])
    assert isinstance(r, KappaResult)
    assert isinstance(r.summary(), str)
    with pytest.raises(Exception):
        r.kappa = 0.5


def test_result_records_the_level_it_used():
    rng = np.random.default_rng(163)
    df = to_long(random_matrix(rng, 3, 20, [1, 2, 3]))
    for level in LEVELS:
        assert rater_agreement(df, level=level, bootstrap_ci=False).level == level


def test_agreement_is_importable_from_the_package_root():
    import evalaudit

    assert hasattr(evalaudit, "rater_agreement")
    assert hasattr(evalaudit, "cohens_kappa")
    assert hasattr(evalaudit, "fleiss_kappa")
    assert hasattr(evalaudit, "AgreementResult")
    assert hasattr(evalaudit, "KappaResult")
