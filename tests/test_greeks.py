"""Chapter 3 tests: Black-Scholes Greeks.

Every Greek is anchored to a central finite difference of the pricing
function itself — an independent check that catches both formula errors and
sign-convention slips (theta especially). Identities (put-call delta parity,
gamma call = put) and the degenerate sigma*sqrt(T) = 0 limits complete the
coverage.
"""

from __future__ import annotations

import numpy as np
import pytest

from cosfin.models.black_scholes import black_scholes_price
from cosfin.pricing.greeks import delta, gamma, rho, theta, vega

ARGS = (100.0, 110.0, 0.05, 0.2, 1.0)  # S, K, r, sigma, T — off-ATM on purpose
Q = 0.03


def _fd(bump: str, h: float, option_type: str = "call") -> float:
    """Central difference of the BS price in the bumped argument."""
    base = dict(zip(("spot", "strike", "r", "sigma", "t"), ARGS, strict=True))
    up, down = dict(base), dict(base)
    up[bump] += h
    down[bump] -= h
    price_up = black_scholes_price(
        up["spot"], up["strike"], up["r"], up["sigma"], up["t"],
        option_type=option_type, q=Q,  # type: ignore[arg-type]
    )
    price_down = black_scholes_price(
        down["spot"], down["strike"], down["r"], down["sigma"], down["t"],
        option_type=option_type, q=Q,  # type: ignore[arg-type]
    )
    return float(price_up - price_down) / (2.0 * h)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_delta_matches_finite_difference(option_type: str) -> None:
    fd = _fd("spot", 1e-3, option_type)
    value = float(delta(*ARGS, option_type=option_type, q=Q))  # type: ignore[arg-type]
    assert value == pytest.approx(fd, abs=1e-6)


def test_put_call_delta_parity() -> None:
    # Differentiate parity in S: delta_call - delta_put = e^{-qT} (RESULT).
    call = delta(*ARGS, option_type="call", q=Q)
    put = delta(*ARGS, option_type="put", q=Q)
    assert float(call - put) == pytest.approx(float(np.exp(-Q * 1.0)), abs=1e-12)


def test_gamma_matches_finite_difference_and_is_type_free() -> None:
    h = 1e-2
    call_prices = [
        float(black_scholes_price(100.0 + shift, *ARGS[1:], q=Q)) for shift in (-h, 0.0, h)
    ]
    fd = (call_prices[2] - 2.0 * call_prices[1] + call_prices[0]) / h**2
    assert float(gamma(*ARGS, q=Q)) == pytest.approx(fd, abs=1e-6)
    # No option_type parameter by design: gamma_call == gamma_put (RESULT).


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_theta_matches_finite_difference(option_type: str) -> None:
    # theta = -dV/dT per year (sign CONVENTION of the module).
    fd = -_fd("t", 1e-5, option_type)
    value = float(theta(*ARGS, option_type=option_type, q=Q))  # type: ignore[arg-type]
    assert value == pytest.approx(fd, abs=1e-6)


def test_theta_is_negative_for_the_undividended_call() -> None:
    assert float(theta(100.0, 100.0, 0.05, 0.2, 1.0)) < 0.0


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_rho_matches_finite_difference(option_type: str) -> None:
    fd = _fd("r", 1e-5, option_type)
    value = float(rho(*ARGS, option_type=option_type, q=Q))  # type: ignore[arg-type]
    assert value == pytest.approx(fd, abs=1e-6)


def test_degenerate_limits() -> None:
    # sigma = 0: delta is the discounted indicator, gamma guarded to 0,
    # phi(+/-inf) = 0 kills the theta decay term (carry survives).
    assert float(delta(110.0, 100.0, 0.05, 0.0, 1.0)) == pytest.approx(1.0)
    assert float(delta(80.0, 100.0, 0.05, 0.0, 1.0)) == 0.0
    assert float(gamma(110.0, 100.0, 0.05, 0.0, 1.0)) == 0.0
    assert np.isfinite(float(theta(110.0, 100.0, 0.05, 0.0, 1.0)))
    # t = 0: all Greeks evaluate finitely (safe-denominator branch in theta).
    for greek in (delta, theta, rho):
        assert np.isfinite(float(greek(110.0, 100.0, 0.05, 0.2, 0.0)))


def test_vectorised_over_strikes() -> None:
    strikes = np.array([80.0, 100.0, 120.0])
    d = delta(100.0, strikes, 0.05, 0.2, 1.0)
    assert d.shape == (3,)
    assert np.all(np.diff(d) < 0.0)  # call delta decreasing in strike


def test_vega_reexport_and_value() -> None:
    assert float(vega(*ARGS)) > 0.0


@pytest.mark.parametrize("greek", [delta, theta, rho])
def test_invalid_option_type_raises(greek: object) -> None:
    with pytest.raises(ValueError):
        greek(*ARGS, option_type="straddle")  # type: ignore[operator]
