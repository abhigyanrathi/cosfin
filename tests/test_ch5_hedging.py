"""Chapter 5 tests: discrete delta hedging under GBM and Merton dynamics.

All experiments are seeded, so the assertions are deterministic; tolerances
sit at roughly twice the observed run-to-run spread. The economic content:
under GBM the hedge error is pure discretisation noise (std ~ 1/sqrt(n));
under Merton the seller who prices and hedges at the diffusion value both
loses money on average (the true Merton value 12.76 exceeds the BS 10.45
they charged) and keeps a P&L dispersion no rebalancing frequency removes.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyfinlib_practice.pricing.hedging import delta_hedge_pnl

ARGS = (100.0, 100.0, 0.05, 0.2, 1.0)


def test_gbm_error_shrinks_like_inverse_sqrt_steps() -> None:
    pnl_13 = delta_hedge_pnl(*ARGS, 13, 20_000, rng=np.random.default_rng(7))
    pnl_52 = delta_hedge_pnl(*ARGS, 52, 20_000, rng=np.random.default_rng(7))
    pnl_208 = delta_hedge_pnl(*ARGS, 208, 20_000, rng=np.random.default_rng(7))
    assert 3.0 < pnl_13.std() / pnl_208.std() < 5.0  # theory: sqrt(16) = 4
    assert 1.6 < pnl_13.std() / pnl_52.std() < 2.4  # theory: sqrt(4) = 2


@pytest.mark.parametrize("n_steps", [13, 208])
def test_gbm_mean_pnl_is_zero(n_steps: int) -> None:
    pnl = delta_hedge_pnl(*ARGS, n_steps, 20_000, rng=np.random.default_rng(7))
    assert np.abs(pnl.mean()) < 4.0 * pnl.std(ddof=1) / np.sqrt(pnl.size)


def test_merton_jump_risk_is_not_hedged_away() -> None:
    gbm = delta_hedge_pnl(*ARGS, 208, 20_000, rng=np.random.default_rng(7))
    merton = delta_hedge_pnl(*ARGS, 208, 20_000, lam=1.0, mu_j=-0.1, sigma_j=0.15,
                             rng=np.random.default_rng(7))
    assert merton.std() > 4.0 * gbm.std()  # observed ratio ~ 8.8
    # Selling jump risk at the diffusion price is systematically unprofitable:
    # the BS premium collected (10.45) is below the true Merton value (12.76).
    assert merton.mean() < -1.0


def test_dividend_yield_accounting_is_self_financing() -> None:
    pnl = delta_hedge_pnl(*ARGS, 208, 20_000, q=0.03, rng=np.random.default_rng(11))
    assert np.abs(pnl.mean()) < 4.0 * pnl.std(ddof=1) / np.sqrt(pnl.size)
    assert pnl.std() < 1.0  # same discretisation-noise scale as q = 0


def test_put_hedging_symmetric() -> None:
    pnl = delta_hedge_pnl(*ARGS, 208, 20_000, option_type="put",
                          rng=np.random.default_rng(13))
    assert np.abs(pnl.mean()) < 4.0 * pnl.std(ddof=1) / np.sqrt(pnl.size)


def test_selling_at_higher_vol_is_profitable_on_average() -> None:
    # Hedge/charge at 30% while the world realises 20%: mean P&L ~ the
    # premium gap price(0.3) - price(0.2) ~ +3.8 (RESULT: vol mismeasurement
    # shows up as first-order P&L drift).
    pnl = delta_hedge_pnl(*ARGS, 208, 20_000, sigma_hedge=0.3,
                          rng=np.random.default_rng(17))
    assert pnl.mean() > 2.0


def test_reproducible_and_validated() -> None:
    a = delta_hedge_pnl(*ARGS, 13, 100, rng=np.random.default_rng(1))
    b = delta_hedge_pnl(*ARGS, 13, 100, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError):
        delta_hedge_pnl(100.0, 100.0, 0.05, 0.2, 0.0, 13, 100)
    with pytest.raises(ValueError):
        delta_hedge_pnl(100.0, 100.0, 0.05, 0.2, 1.0, 0, 100)
    with pytest.raises(ValueError):
        delta_hedge_pnl(100.0, 100.0, 0.05, 0.2, 1.0, 13, 100, lam=-1.0)
    with pytest.raises(ValueError):
        delta_hedge_pnl(100.0, 100.0, 0.05, 0.2, 1.0, 13, 100,
                        option_type="straddle")  # type: ignore[arg-type]


def test_validation_edges() -> None:
    with pytest.raises(ValueError):  # spot <= 0
        delta_hedge_pnl(-1.0, 100.0, 0.05, 0.2, 1.0, 13, 100)
    with pytest.raises(ValueError):  # true sigma < 0
        delta_hedge_pnl(100.0, 100.0, 0.05, -0.2, 1.0, 13, 100)
    with pytest.raises(ValueError):  # hedger sigma < 0
        delta_hedge_pnl(100.0, 100.0, 0.05, 0.2, 1.0, 13, 100, sigma_hedge=-0.1)
