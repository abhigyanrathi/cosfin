"""Chapter 6 tests: the COS method.

Anchor strategy (documented deliberately): (1) closed-form Black-Scholes is
the exact external reference for the GBM leg; (2) the Merton COS price is
cross-validated against the *independent* Merton series of Chapter 5 — two
unrelated algorithms agreeing to ~1e-12; (3) the pure-CGMY Y=0.5 price
reproduces the published Fang & Oosterlee (2008) reference 19.812948843118
to ~6e-9 without any tuning — a genuinely external anchor. The O&G table
anchors (6.4, 6.5, 6.7, 6.8, from the supplied book pages) live in
tests/test_replications.py; this file keeps the model-free and cross-method
checks.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from cosfin.models.black_scholes import black_scholes_price, digital_price
from cosfin.models.characteristic_functions import (
    CGMY,
    GBM,
    Merton,
    VarianceGamma,
)
from cosfin.models.jump_diffusion import merton_jump_price
from cosfin.numerical.cos import (
    cos_density,
    cos_digital_price,
    cos_european_price,
    truncation_range,
)

GBM_MODEL = GBM(r=0.05, q=0.0, sigma=0.2)
STRIKES = np.array([80.0, 100.0, 120.0])


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_cos_matches_black_scholes_at_n256(option_type: str) -> None:
    cos_prices = cos_european_price(
        GBM_MODEL.cf, GBM_MODEL.cumulants, 100.0, STRIKES, 0.05, 1.0, 256,
        option_type=option_type,  # type: ignore[arg-type]
    )
    closed = black_scholes_price(
        100.0, STRIKES, 0.05, 0.2, 1.0,
        option_type=option_type,  # type: ignore[arg-type]
    )
    assert np.max(np.abs(cos_prices - closed)) < 1e-12


def test_cos_short_maturity_regime() -> None:
    model = GBM(r=0.1, q=0.0, sigma=0.25)
    strikes = np.array([90.0, 100.0, 110.0])
    cos_prices = cos_european_price(model.cf, model.cumulants, 100.0, strikes,
                                    0.1, 0.1, 256)
    closed = black_scholes_price(100.0, strikes, 0.1, 0.25, 0.1)
    assert np.max(np.abs(cos_prices - closed)) < 1e-12


def test_cos_convergence_in_n() -> None:
    closed = float(black_scholes_price(100.0, 100.0, 0.05, 0.2, 1.0))
    errors = [
        abs(float(cos_european_price(GBM_MODEL.cf, GBM_MODEL.cumulants,
                                     100.0, 100.0, 0.05, 1.0, n)[0]) - closed)
        for n in (16, 64, 256)
    ]
    assert errors[0] > errors[2]
    assert errors[2] < 1e-12


def test_cos_put_call_parity() -> None:
    call = cos_european_price(GBM_MODEL.cf, GBM_MODEL.cumulants,
                              100.0, STRIKES, 0.05, 1.0, 256)
    put = cos_european_price(GBM_MODEL.cf, GBM_MODEL.cumulants,
                             100.0, STRIKES, 0.05, 1.0, 256, option_type="put")
    forward_leg = 100.0 - STRIKES * np.exp(-0.05)
    assert np.max(np.abs(call - put - forward_leg)) < 1e-10


def test_cos_digital_matches_closed_form() -> None:
    cos_dig = cos_digital_price(GBM_MODEL.cf, GBM_MODEL.cumulants,
                                100.0, STRIKES, 0.05, 1.0, 512)
    closed = digital_price(100.0, STRIKES, 0.05, 0.2, 1.0)
    assert np.max(np.abs(cos_dig - closed)) < 1e-12
    cos_dig_put = cos_digital_price(GBM_MODEL.cf, GBM_MODEL.cumulants,
                                    100.0, STRIKES, 0.05, 1.0, 512,
                                    option_type="put")
    assert np.max(np.abs(cos_dig + cos_dig_put - np.exp(-0.05))) < 1e-10


def test_merton_cos_cross_validates_series() -> None:
    # Exercise 6.3 in spirit: two independent algorithms, one model.
    model = Merton(r=0.05, q=0.0, sigma=0.2, lam=1.0, mu_j=-0.1, sigma_j=0.15)
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    cos_prices = cos_european_price(model.cf, model.cumulants, 100.0, strikes,
                                    0.05, 1.0, 1024, l_trunc=12.0)
    series = merton_jump_price(100.0, strikes, 0.05, 0.2, 1.0,
                               lam=1.0, mu_j=-0.1, sigma_j=0.15)
    assert np.max(np.abs(cos_prices - series)) < 1e-10


def test_cgmy_reproduces_fang_oosterlee_reference() -> None:
    # Fang & Oosterlee (2008), pure CGMY (C, G, M, Y) = (1, 5, 5, 0.5),
    # S0 = K = 100, r = 0.1, T = 1: published 19.812948843118.
    # Reproduced here to ~6e-9 with L = 10 — an external, untuned anchor.
    model = CGMY(r=0.1, q=0.0, C=1.0, G=5.0, M=5.0, Y=0.5, sigma=0.0)
    price = float(cos_european_price(model.cf, model.cumulants,
                                     100.0, 100.0, 0.1, 1.0, 1024)[0])
    assert price == pytest.approx(19.812948843118, abs=1e-6)


def test_cgmy_heavy_tail_regime_is_consistent() -> None:
    model = CGMY(r=0.1, q=0.0, C=1.0, G=5.0, M=5.0, Y=1.5, sigma=0.0)
    strikes = np.array([80.0, 100.0, 120.0])
    call = cos_european_price(model.cf, model.cumulants, 100.0, strikes,
                              0.1, 1.0, 2048)
    put = cos_european_price(model.cf, model.cumulants, 100.0, strikes,
                             0.1, 1.0, 2048, option_type="put")
    assert np.all(call > 0.0) and np.all(np.diff(call) < 0.0)
    forward_leg = 100.0 - strikes * np.exp(-0.1)
    assert np.max(np.abs(call - put - forward_leg)) < 1e-6


def test_variance_gamma_parity_and_sanity() -> None:
    model = VarianceGamma(r=0.1, q=0.0, sigma=0.12, theta=-0.14, beta=0.2)
    call = float(cos_european_price(model.cf, model.cumulants,
                                    100.0, 90.0, 0.1, 1.0, 512)[0])
    put = float(cos_european_price(model.cf, model.cumulants,
                                   100.0, 90.0, 0.1, 1.0, 512,
                                   option_type="put")[0])
    assert 0.0 < call < 100.0
    assert call - put == pytest.approx(100.0 - 90.0 * np.exp(-0.1), abs=1e-8)


def test_cos_density_recovers_gaussian() -> None:
    a, b = truncation_range(GBM_MODEL.cumulants, 1.0)
    y = np.linspace(a, b, 801)
    density = cos_density(GBM_MODEL.cf, y, a, b, 1024, 1.0)
    c1, c2, _ = GBM_MODEL.cumulants(1.0)
    analytic = np.exp(-0.5 * (y - c1) ** 2 / c2) / np.sqrt(2.0 * np.pi * c2)
    assert np.max(np.abs(density - analytic)) < 1e-12
    assert np.trapezoid(density, y) == pytest.approx(1.0, abs=1e-8)
    assert np.trapezoid(y * density, y) == pytest.approx(c1, abs=1e-8)


def test_cos_density_rejects_points_outside_interval() -> None:
    a, b = truncation_range(GBM_MODEL.cumulants, 1.0)
    with pytest.raises(ValueError):
        cos_density(GBM_MODEL.cf, np.array([b + 1.0]), a, b, 256, 1.0)
    with pytest.raises(ValueError):
        cos_density(GBM_MODEL.cf, np.array([0.0]), 1.0, -1.0, 256, 1.0)


def test_truncation_range_properties() -> None:
    a, b = truncation_range(GBM_MODEL.cumulants, 1.0)
    c1, _, _ = GBM_MODEL.cumulants(1.0)
    assert a < c1 < b
    assert 0.5 * (a + b) == pytest.approx(c1, abs=1e-12)
    with pytest.raises(ValueError):
        truncation_range(GBM_MODEL.cumulants, 1.0, l_trunc=0.0)
    with pytest.raises(ValueError):
        truncation_range(lambda t: (0.0, 0.0, 0.0), 1.0)
    with pytest.raises(ValueError):
        truncation_range(lambda t: (0.0, 1.0, -1.0), 1.0)


def test_cos_price_insensitive_to_l_trunc() -> None:
    tight = float(cos_european_price(GBM_MODEL.cf, GBM_MODEL.cumulants,
                                     100.0, 100.0, 0.05, 1.0, 512, l_trunc=8.0)[0])
    wide = float(cos_european_price(GBM_MODEL.cf, GBM_MODEL.cumulants,
                                    100.0, 100.0, 0.05, 1.0, 512, l_trunc=12.0)[0])
    assert tight == pytest.approx(wide, abs=1e-10)


def test_cos_scalar_strike_returns_1d() -> None:
    price = cos_european_price(GBM_MODEL.cf, GBM_MODEL.cumulants,
                               100.0, 100.0, 0.05, 1.0, 64)
    assert price.shape == (1,)


def test_cos_validation() -> None:
    with pytest.raises(ValueError):
        cos_european_price(GBM_MODEL.cf, GBM_MODEL.cumulants,
                           100.0, 100.0, 0.05, 1.0, 1)
    with pytest.raises(ValueError):
        cos_european_price(GBM_MODEL.cf, GBM_MODEL.cumulants,
                           -1.0, 100.0, 0.05, 1.0, 64)
    with pytest.raises(ValueError):
        cos_european_price(GBM_MODEL.cf, GBM_MODEL.cumulants,
                           100.0, -100.0, 0.05, 1.0, 64)
    with pytest.raises(ValueError):
        cos_european_price(GBM_MODEL.cf, GBM_MODEL.cumulants,
                           100.0, 100.0, 0.05, 1.0, 64,
                           option_type="straddle")  # type: ignore[arg-type]

    def bad_cf(u: npt.ArrayLike, t: float) -> npt.NDArray[np.complex128]:
        return np.array([1.0 + 0.0j], dtype=np.complex128)

    with pytest.raises(ValueError):
        cos_european_price(bad_cf, GBM_MODEL.cumulants, 100.0, 100.0, 0.05, 1.0, 64)


def test_setup_and_density_validation_edges() -> None:
    with pytest.raises(ValueError):  # t <= 0
        cos_european_price(GBM_MODEL.cf, GBM_MODEL.cumulants, 100.0, 100.0, 0.05, 0.0, 64)
    with pytest.raises(ValueError):  # strike must be scalar or 1-D
        cos_european_price(GBM_MODEL.cf, GBM_MODEL.cumulants, 100.0,
                           np.array([[100.0]]), 0.05, 1.0, 64)
    with pytest.raises(ValueError):  # digital: option_type
        cos_digital_price(GBM_MODEL.cf, GBM_MODEL.cumulants, 100.0, 100.0, 0.05, 1.0,
                          64, option_type="straddle")  # type: ignore[arg-type]
    a, b = truncation_range(GBM_MODEL.cumulants, 1.0)
    with pytest.raises(ValueError):  # density: n < 2
        cos_density(GBM_MODEL.cf, np.array([0.0]), a, b, 1, 1.0)
    with pytest.raises(ValueError):  # density: t <= 0
        cos_density(GBM_MODEL.cf, np.array([0.0]), a, b, 64, 0.0)
