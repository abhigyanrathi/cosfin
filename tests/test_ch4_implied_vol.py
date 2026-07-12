"""Chapter 4 (Section 4.1) tests: implied volatility solvers.

The recovery grid prices with our own Black-Scholes and inverts — a
round-trip that checks the solvers but not the pricer. The independent
anchor (invert the *published* 10.4506 and demand sigma ~ 0.20) closes that
gap: if the pricer itself were wrong, this test fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyfinlib_practice.models.black_scholes import OptionType, black_scholes_price
from pyfinlib_practice.models.local_vol import (
    Method,
    implied_volatility,
    implied_volatility_bracketed,
    implied_volatility_newton,
    implied_volatility_secant,
)


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("sigma", [0.15, 0.4])
@pytest.mark.parametrize("t", [0.25, 1.0, 2.0])
@pytest.mark.parametrize("moneyness", [0.7, 1.0, 1.4])
def test_bracketed_recovers_sigma_across_grid(
    option_type: OptionType, sigma: float, t: float, moneyness: float
) -> None:
    strike = 100.0 * moneyness
    price = float(
        black_scholes_price(100.0, strike, 0.05, sigma, t, option_type=option_type)
    )
    recovered = implied_volatility_bracketed(
        price, 100.0, strike, 0.05, t, option_type=option_type
    )
    assert recovered == pytest.approx(sigma, abs=1e-7)


@pytest.mark.parametrize("sigma", [0.15, 0.3])
@pytest.mark.parametrize("moneyness", [0.9, 1.0, 1.1])
def test_newton_recovers_sigma_near_the_money(sigma: float, moneyness: float) -> None:
    strike = 100.0 * moneyness
    price = float(black_scholes_price(100.0, strike, 0.05, sigma, 1.0))
    assert implied_volatility_newton(price, 100.0, strike, 0.05, 1.0) == pytest.approx(
        sigma, abs=1e-7
    )


@pytest.mark.parametrize("sigma", [0.15, 0.3])
@pytest.mark.parametrize("moneyness", [0.9, 1.0, 1.1])
def test_secant_recovers_sigma_near_the_money(sigma: float, moneyness: float) -> None:
    strike = 100.0 * moneyness
    price = float(black_scholes_price(100.0, strike, 0.05, sigma, 1.0))
    assert implied_volatility_secant(price, 100.0, strike, 0.05, 1.0) == pytest.approx(
        sigma, abs=1e-6
    )


def test_independent_published_anchor() -> None:
    # 10.4506 is the published ATM value: inverting it must give ~20% vol.
    assert implied_volatility(10.4506, 100.0, 100.0, 0.05, 1.0) == pytest.approx(
        0.20, abs=1e-4
    )


def test_deep_otm_sigma_step_criterion() -> None:
    # K=150, T=0.25, sigma=0.25: vega ~ 0.17, price ~ 3e-3. A price-residual
    # stopping rule would accept sigma errors ~ tol/vega; the sigma-step
    # criterion recovers the volatility itself to 1e-6.
    price = float(black_scholes_price(100.0, 150.0, 0.05, 0.25, 0.25))
    recovered = implied_volatility_bracketed(price, 100.0, 150.0, 0.05, 0.25)
    assert recovered == pytest.approx(0.25, abs=1e-6)


def test_deep_itm_quote_is_rejected_as_ill_posed() -> None:
    # 16 standard deviations ITM (K=60, T=0.1, sigma=0.1): the time value is
    # ~1e-58, so the double-precision quote is bit-identical to the intrinsic
    # lower bound and carries no volatility information. The bounds check
    # rejects it rather than returning a meaningless number.
    price = float(black_scholes_price(100.0, 60.0, 0.05, 0.1, 0.1))
    lower_bound = 100.0 - 60.0 * float(np.exp(-0.05 * 0.1))
    assert price == lower_bound  # the quote *is* the bound in double precision
    with pytest.raises(ValueError):
        implied_volatility_bracketed(price, 100.0, 60.0, 0.05, 0.1)


def test_dispatcher_methods_agree_and_reject_unknown() -> None:
    price = float(black_scholes_price(100.0, 105.0, 0.05, 0.22, 1.0))
    methods: tuple[Method, ...] = ("newton", "secant", "bracketed")
    results = [
        implied_volatility(price, 100.0, 105.0, 0.05, 1.0, method=m) for m in methods
    ]
    for value in results:
        assert value == pytest.approx(0.22, abs=1e-6)
    with pytest.raises(ValueError):
        implied_volatility(price, 100.0, 105.0, 0.05, 1.0, method="brent")  # type: ignore[arg-type]


def test_arbitrage_bounds_rejected() -> None:
    with pytest.raises(ValueError):  # call above S e^{-qT}
        implied_volatility_bracketed(101.0, 100.0, 100.0, 0.05, 1.0)
    with pytest.raises(ValueError):  # call below forward intrinsic
        implied_volatility_bracketed(4.0, 100.0, 100.0, 0.05, 1.0)
    with pytest.raises(ValueError):  # put above K e^{-rT}
        implied_volatility_bracketed(
            96.0, 100.0, 100.0, 0.05, 1.0, option_type="put"
        )


def test_newton_failure_paths() -> None:
    price = float(black_scholes_price(100.0, 100.0, 0.05, 0.2, 1.0))
    with pytest.raises(RuntimeError):  # far seed, one iteration: step >> xtol
        implied_volatility_newton(price, 100.0, 100.0, 0.05, 1.0,
                                  initial_guess=2.5, max_iter=1)


def test_solver_input_validation() -> None:
    with pytest.raises(ValueError):
        implied_volatility_bracketed(1.0, 100.0, 100.0, 0.05, 0.0)
    with pytest.raises(ValueError):
        implied_volatility_bracketed(1.0, -100.0, 100.0, 0.05, 1.0)
    with pytest.raises(ValueError):
        implied_volatility_bracketed(
            1.0, 100.0, 100.0, 0.05, 1.0, option_type="straddle"  # type: ignore[arg-type]
        )


def test_solver_failure_guards() -> None:
    otm = float(black_scholes_price(100.0, 150.0, 0.05, 0.25, 0.25))
    atm = float(black_scholes_price(100.0, 100.0, 0.05, 0.2, 1.0))
    with pytest.raises(ValueError):  # non-finite quote
        implied_volatility_bracketed(float("inf"), 100.0, 100.0, 0.05, 1.0)
    with pytest.raises(RuntimeError):  # Newton: vega underflow at the sigma floor
        implied_volatility_newton(otm, 100.0, 150.0, 0.05, 0.25, initial_guess=1e-9)
    with pytest.raises(RuntimeError):  # secant: price flat in sigma at huge vols
        implied_volatility_secant(atm, 100.0, 100.0, 0.05, 1.0, initial_guess=30.0)
    with pytest.raises(RuntimeError):  # secant: max_iter exhausted
        implied_volatility_secant(atm, 100.0, 100.0, 0.05, 1.0,
                                  initial_guess=0.9, max_iter=1)
    with pytest.raises(RuntimeError):  # bracketed: max_iter with unreachable xtol
        implied_volatility_bracketed(atm, 100.0, 100.0, 0.05, 1.0,
                                     xtol=1e-18, max_iter=1)


def test_secant_negative_overshoot_is_halved_back() -> None:
    # From a far seed the secant step lands negative; the halving guard keeps
    # the iterate positive and the solver still recovers the true vol.
    atm = float(black_scholes_price(100.0, 100.0, 0.05, 0.2, 1.0))
    recovered = implied_volatility_secant(atm, 100.0, 100.0, 0.05, 1.0, initial_guess=5.0)
    assert recovered == pytest.approx(0.2, abs=1e-6)
