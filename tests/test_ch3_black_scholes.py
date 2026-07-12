"""Chapter 3 tests: Black-Scholes prices, digitals, butterfly, degenerate limits.

Anchor policy: external published values (the classic ATM 10.4506 and Hull's
S=42 example) pin correctness against the literature; a full-precision
internal regression value then locks the implementation against silent
drift. Structural identities (parity, digital = -dC/dK, butterfly = call
combo) are independent of both.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm  # type: ignore[import-untyped]

from pyfinlib_practice.models.black_scholes import (
    black_scholes_price,
    butterfly_price,
    d1_d2,
    digital_price,
    norm_cdf,
    norm_pdf,
    put_call_parity_residual,
    vega,
)

ATM_CALL_REFERENCE = 10.450583572185565  # internal full-precision regression value


def test_published_atm_anchor() -> None:
    # S=K=100, r=5%, sigma=20%, T=1: the widely tabulated 10.4506.
    price = float(black_scholes_price(100.0, 100.0, 0.05, 0.2, 1.0))
    assert price == pytest.approx(10.4506, abs=2e-4)
    assert price == pytest.approx(ATM_CALL_REFERENCE, rel=1e-12)


def test_published_hull_anchor() -> None:
    # Hull's example: S=42, K=40, r=10%, sigma=20%, T=0.5 -> C=4.76, P=0.81.
    call = float(black_scholes_price(42.0, 40.0, 0.1, 0.2, 0.5))
    put = float(black_scholes_price(42.0, 40.0, 0.1, 0.2, 0.5, option_type="put"))
    assert call == pytest.approx(4.76, abs=5e-3)
    assert put == pytest.approx(0.81, abs=5e-3)


@pytest.mark.parametrize("strike", [60.0, 80.0, 100.0, 120.0, 160.0])
@pytest.mark.parametrize("t", [0.1, 1.0, 3.0])
@pytest.mark.parametrize("q", [0.0, 0.03])
def test_put_call_parity(strike: float, t: float, q: float) -> None:
    call = black_scholes_price(100.0, strike, 0.05, 0.25, t, q=q)
    put = black_scholes_price(100.0, strike, 0.05, 0.25, t, option_type="put", q=q)
    residual = put_call_parity_residual(call, put, 100.0, strike, 0.05, t, q=q)
    assert float(np.abs(residual)) < 1e-12


def test_vectorised_over_strikes() -> None:
    strikes = np.array([80.0, 100.0, 120.0])
    prices = black_scholes_price(100.0, strikes, 0.05, 0.2, 1.0)
    assert prices.shape == (3,)
    assert np.all(np.diff(prices) < 0.0)  # call price decreasing in strike


def test_monotone_in_sigma_and_bounds() -> None:
    sigmas = np.array([0.05, 0.1, 0.2, 0.4, 0.8])
    prices = black_scholes_price(100.0, 110.0, 0.05, sigmas, 1.0)
    assert np.all(np.diff(prices) > 0.0)
    lower = max(100.0 - 110.0 * np.exp(-0.05), 0.0)
    assert np.all(prices > lower) and np.all(prices < 100.0)


def test_degenerate_zero_volatility_is_forward_intrinsic() -> None:
    # ITM-forward call, OTM-forward call, exact forward-ATM.
    itm = float(black_scholes_price(110.0, 100.0, 0.05, 0.0, 1.0))
    assert itm == pytest.approx(110.0 - 100.0 * np.exp(-0.05), rel=1e-14)
    otm = float(black_scholes_price(80.0, 100.0, 0.05, 0.0, 1.0))
    assert otm == 0.0
    k_atm = 100.0 * np.exp(0.05)
    atm = float(black_scholes_price(100.0, k_atm, 0.05, 0.0, 1.0))
    assert atm == pytest.approx(0.0, abs=1e-12)


def test_degenerate_zero_maturity_is_intrinsic() -> None:
    assert float(black_scholes_price(110.0, 100.0, 0.05, 0.2, 0.0)) == pytest.approx(10.0)
    assert float(
        black_scholes_price(90.0, 100.0, 0.05, 0.2, 0.0, option_type="put")
    ) == pytest.approx(10.0)


def test_d1_d2_degenerate_signs() -> None:
    d1, d2 = d1_d2(110.0, 100.0, 0.05, 0.0, 1.0)
    assert np.isposinf(d1) and np.isposinf(d2)
    d1, d2 = d1_d2(80.0, 100.0, 0.05, 0.0, 1.0)
    assert np.isneginf(d1) and np.isneginf(d2)
    # Exact forward-ATM in floating point requires S == K and r == q == 0;
    # with r*t compensation the log leaves a ~1e-17 residue and the limit is
    # correctly the signed infinity of that residue.
    d1, _ = d1_d2(100.0, 100.0, 0.0, 0.0, 1.0)
    assert float(d1) == 0.0


def test_digital_degenerate_is_discounted_indicator() -> None:
    # The bug class caught here: an unguarded d2 at sigma=0 silently prices
    # the digital off N(log-moneyness) instead of the indicator.
    itm = float(digital_price(110.0, 100.0, 0.05, 0.0, 1.0))
    assert itm == pytest.approx(np.exp(-0.05), rel=1e-14)
    assert float(digital_price(80.0, 100.0, 0.05, 0.0, 1.0)) == 0.0


def test_digital_matches_strike_derivative() -> None:
    # Cash digital = -dC/dK (RESULT); central difference, error O(h^2).
    h = 1e-4
    up = black_scholes_price(100.0, 100.0 + h, 0.05, 0.2, 1.0)
    down = black_scholes_price(100.0, 100.0 - h, 0.05, 0.2, 1.0)
    fd = -float(up - down) / (2.0 * h)
    assert float(digital_price(100.0, 100.0, 0.05, 0.2, 1.0)) == pytest.approx(fd, abs=1e-6)


def test_digital_call_plus_put_is_discount_factor() -> None:
    call = digital_price(100.0, 90.0, 0.05, 0.2, 2.0)
    put = digital_price(100.0, 90.0, 0.05, 0.2, 2.0, option_type="put")
    assert float(call + put) == pytest.approx(np.exp(-0.1), rel=1e-12)


def test_vega_matches_finite_difference() -> None:
    h = 1e-5
    up = black_scholes_price(100.0, 110.0, 0.05, 0.2 + h, 1.0)
    down = black_scholes_price(100.0, 110.0, 0.05, 0.2 - h, 1.0)
    fd = float(up - down) / (2.0 * h)
    assert float(vega(100.0, 110.0, 0.05, 0.2, 1.0)) == pytest.approx(fd, abs=1e-6)


def test_butterfly_is_call_combination_and_positive() -> None:
    combo = (
        black_scholes_price(100.0, 95.0, 0.05, 0.2, 1.0)
        - 2.0 * black_scholes_price(100.0, 100.0, 0.05, 0.2, 1.0)
        + black_scholes_price(100.0, 105.0, 0.05, 0.2, 1.0)
    )
    fly = butterfly_price(100.0, 100.0, 0.05, 0.2, 1.0, 5.0)
    assert float(fly) == pytest.approx(float(combo), abs=1e-12)
    strikes = np.array([70.0, 90.0, 100.0, 110.0, 130.0])
    assert np.all(butterfly_price(100.0, strikes, 0.05, 0.2, 1.0, 5.0) > 0.0)


def test_butterfly_guards() -> None:
    with pytest.raises(ValueError):
        butterfly_price(100.0, 100.0, 0.05, 0.2, 1.0, 0.0)
    with pytest.raises(ValueError):
        butterfly_price(100.0, 100.0, 0.05, 0.2, 1.0, -1.0)
    with pytest.raises(ValueError):
        butterfly_price(100.0, 100.0, 0.05, 0.2, 1.0, 100.0)  # spread >= strike
    with pytest.raises(ValueError):
        butterfly_price(100.0, np.array([50.0, 100.0]), 0.05, 0.2, 1.0, 60.0)


def test_norm_helpers_match_scipy() -> None:
    x = np.linspace(-5.0, 5.0, 41)
    np.testing.assert_allclose(norm_pdf(x), norm.pdf(x), atol=1e-12)
    np.testing.assert_allclose(norm_cdf(x) + norm_cdf(-x), 1.0, atol=1e-12)


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        black_scholes_price(100.0, 100.0, 0.05, 0.2, 1.0, option_type="straddle")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        black_scholes_price(-1.0, 100.0, 0.05, 0.2, 1.0)
    with pytest.raises(ValueError):
        black_scholes_price(100.0, 100.0, 0.05, -0.2, 1.0)
    with pytest.raises(ValueError):
        black_scholes_price(100.0, 100.0, 0.05, 0.2, -1.0)
