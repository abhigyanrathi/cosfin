"""Replication tests: the mentor items anchored to published O&G values.

Anchor inventory (all printed values from pages supplied by Abhi):
Table 6.4 references (eq 6.51), the Table 6.5 reference 0.273306496 with the
pays-K cash convention of Example 6.3.1 / eq (6.52), Table 6.7 (CGMYB) and
Table 6.8 (Variance Gamma). Figure-level items assert the parameter-robust
qualitative results; exact figure parameters that the book does not print are
recorded as ASSUMED in ``PARAMETER_PROVENANCE`` and deliberately not asserted.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from cosfin.models.black_scholes import black_scholes_price, digital_price
from cosfin.replications import (
    FIG_5_8_PANELS,
    PARAMETER_PROVENANCE,
    TABLE_6_4_REF,
    TABLE_6_5_REF,
    TABLE_6_7_REF,
    TABLE_6_8_REF,
    exercise_6_3,
    figure_5_7_data,
    figure_5_8_data,
    figure_6_4_data,
    lognormal_density_recovery,
    normal_density_recovery,
    table_6_4,
    table_6_5,
    table_6_6,
    table_6_7,
    table_6_8,
)


# --------------------------------------------------------------------------- #
# Table 6.4
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("strike", sorted(TABLE_6_4_REF))
def test_table_6_4_closed_form_matches_printed_references(strike: float) -> None:
    # The printed reference values are Black-Scholes prices at eq (6.51);
    # our closed form reproduces all three to the printed precision.
    price = float(black_scholes_price(100.0, strike, 0.10, 0.25, 0.1))
    assert price == pytest.approx(TABLE_6_4_REF[strike], abs=1e-8)


def test_table_6_4_cos_convergence() -> None:
    rows = table_6_4()
    for strike, series in rows.items():
        errors = [row.error for row in series]
        assert series[-1].price == pytest.approx(TABLE_6_4_REF[strike], abs=1e-8)
        # The printed references are rounded to 9 dp, so the error against
        # them floors at ~3e-10; against our own closed form it is ~2e-14.
        assert errors[-1] < 1e-9
        assert errors[0] > errors[-1]  # N=16 worse than N=256


def test_table_6_4_book_range_variant_converges_to_same_references() -> None:
    rows = table_6_4(ns=(16, 256), book_range=True)
    per_strike = table_6_4(ns=(16,))
    for strike, series in rows.items():
        assert series[-1].price == pytest.approx(TABLE_6_4_REF[strike], abs=1e-7)
        # The shared interval is strictly less accurate at low N than the
        # per-strike interval wherever x = ln(S0/K) != 0; at K = 100 the two
        # intervals coincide exactly, so no comparison there.
        if strike != 100.0:
            assert series[0].error > per_strike[strike][0].error


# --------------------------------------------------------------------------- #
# Table 6.5 -- the resolved cash-or-nothing anchor
# --------------------------------------------------------------------------- #
def test_table_6_5_closed_form_reproduces_printed_reference() -> None:
    # Pays-K CONVENTION (Example 6.3.1): 120 * unit-cash digital.
    price = 120.0 * float(digital_price(100.0, 120.0, 0.05, 0.2, 0.1))
    assert price == pytest.approx(TABLE_6_5_REF, abs=2e-9)


def test_table_6_5_cos_convergence() -> None:
    rows = table_6_5()
    assert rows[-1].n == 140
    # Signed error against the 9-dp printed reference floors at ~5e-10.
    assert abs(rows[-1].error) < 1e-9
    assert rows[-1].price == pytest.approx(TABLE_6_5_REF, abs=1e-8)


# --------------------------------------------------------------------------- #
# Tables 6.7 / 6.8
# --------------------------------------------------------------------------- #
def test_table_6_7_matches_published_values() -> None:
    rows = table_6_7()
    for y, series in rows.items():
        assert series[-1].n == 512
        assert series[-1].price == pytest.approx(TABLE_6_7_REF[y], abs=1e-6)


def test_table_6_8_matches_published_values() -> None:
    rows = table_6_8()
    for t, series in rows.items():
        assert series[-1].n == 4096
        assert series[-1].price == pytest.approx(TABLE_6_8_REF[t], abs=1e-6)
        # Short maturity is the hard one: N=64 must be visibly worse than
        # N=4096 for T=0.1 (the point of the book running the table so far).
        if t == 0.1:
            assert series[0].error > 100.0 * series[-1].error


# --------------------------------------------------------------------------- #
# Table 6.6 / Figure 6.4 -- CGMYB density recovery
# --------------------------------------------------------------------------- #
def test_table_6_6_density_recovery_errors_decay() -> None:
    rows = table_6_6()
    for _, series in rows.items():
        errors = [err for _, err in series]
        assert all(late <= early for early, late in pairwise(errors))
        assert errors[-1] < 1e-4
        assert errors[0] > errors[-1]


def test_figure_6_4_data_shapes_and_convergence() -> None:
    x, curves, refs = figure_6_4_data()
    assert x.shape == (400,)
    for y in (0.5, 1.5):
        assert set(curves[y]) == {4, 8, 16, 32}
        err_4 = float(np.max(np.abs(curves[y][4] - refs[y])))
        err_32 = float(np.max(np.abs(curves[y][32] - refs[y])))
        assert err_32 < err_4


# --------------------------------------------------------------------------- #
# Exercise 6.3
# --------------------------------------------------------------------------- #
def test_exercise_6_3_two_pricers_agree() -> None:
    rows = exercise_6_3()
    assert [row.strike for row in rows] == [80.0, 90.0, 100.0, 110.0, 120.0]
    assert max(row.abs_diff for row in rows) < 1e-10


# --------------------------------------------------------------------------- #
# Item 3 -- normal / lognormal density recovery (p.169 examples)
# --------------------------------------------------------------------------- #
def test_normal_density_recovery_is_exponentially_convergent() -> None:
    rows = dict(normal_density_recovery())
    assert rows[4] > rows[16] > rows[64]
    assert rows[64] < 1e-12  # machine precision by N=64 on [-10, 10]


def test_lognormal_density_recovery_via_change_of_variables() -> None:
    rows = dict(lognormal_density_recovery())
    assert rows[64] < 1e-12
    assert rows[8] > rows[64]


# --------------------------------------------------------------------------- #
# Figure 5.7 -- hedging frequency under jumps
# --------------------------------------------------------------------------- #
def test_figure_5_7_jump_risk_survives_frequent_hedging() -> None:
    data = figure_5_7_data()
    assert set(data) == {10, 2000}
    coarse, fine = data[10], data[2000]
    assert coarse.shape == fine.shape == (1000,)
    # Under GBM a 200x finer grid would shrink the std ~14x; under jumps the
    # dispersion floors at the unhedgeable jump risk (Section 5.2.3 RESULT).
    assert fine.std() > 0.4 * coarse.std()
    # Large losses present in both histograms, mean negative in both (the
    # seller charged the diffusion value for jump risk).
    for pnl in (coarse, fine):
        assert pnl.min() < -0.3
        assert pnl.mean() < 0.0


def test_figure_5_7_reproducible() -> None:
    a = figure_5_7_data(n_paths=50, seed=3)
    b = figure_5_7_data(n_paths=50, seed=3)
    np.testing.assert_array_equal(a[10], b[10])
    np.testing.assert_array_equal(a[2000], b[2000])


# --------------------------------------------------------------------------- #
# Figure 5.8 -- CGMYB implied-volatility panels
# --------------------------------------------------------------------------- #
def test_figure_5_8_panel_orderings_match_book() -> None:
    strikes = np.linspace(60.0, 160.0, 9)
    ks, panels = figure_5_8_data(strikes=strikes)
    assert set(panels) == set(FIG_5_8_PANELS)
    mid, last = 4, 8  # K = 110 and K = 160
    for curves in panels.values():
        for values in curves.values():
            assert values.shape == ks.shape
            assert np.all(np.isfinite(values))
            assert np.all((values > 0.05) & (values < 2.0))
    # Directional orderings read off the printed panels:
    assert panels["c"][1.0][mid] > panels["c"][0.1][mid]  # more jump activity
    assert panels["g"][0.5][mid] > panels["g"][3.0][mid]  # fatter left tail
    assert panels["m"][2.0][mid] > panels["m"][10.0][mid]  # fatter right tail
    assert panels["y"][0.9][last] > panels["y"][0.2][last]  # heavier tails overall


# --------------------------------------------------------------------------- #
# Provenance bookkeeping
# --------------------------------------------------------------------------- #
def test_every_harness_has_provenance() -> None:
    for key in (
        "table_6_4",
        "table_6_5",
        "table_6_6",
        "table_6_7",
        "table_6_8",
        "figure_6_4",
        "exercise_6_3",
        "figure_5_7",
        "figure_5_8",
    ):
        assert key in PARAMETER_PROVENANCE
        text = PARAMETER_PROVENANCE[key]
        assert any(tag in text for tag in ("CONFIRMED", "ASSUMED", "PENDING"))


def test_table_6_5_book_range_variant_hits_same_reference() -> None:
    rows = table_6_5(ns=(140,), book_range=True)
    assert rows[0].price == pytest.approx(TABLE_6_5_REF, abs=1e-8)


def test_figure_5_8_wing_guard_returns_nan(monkeypatch: pytest.MonkeyPatch) -> None:
    # Unit-test the except->nan guard directly: strikes whose quotes fall
    # outside the invertible region must come back as gaps, not crashes.
    from cosfin import replications
    from cosfin.models.characteristic_functions import CGMY
    from cosfin.replications import cgmyb_implied_volatility_curve

    def unsolvable(*args: object, **kwargs: object) -> float:
        raise ValueError("outside the no-arbitrage bounds")

    monkeypatch.setattr(replications, "implied_volatility_bracketed", unsolvable)
    model = CGMY(r=0.1, q=0.0, C=1.0, G=1.0, M=5.0, Y=0.5, sigma=0.2)
    curve = cgmyb_implied_volatility_curve(model, np.array([100.0]))
    assert np.isnan(curve[0])
