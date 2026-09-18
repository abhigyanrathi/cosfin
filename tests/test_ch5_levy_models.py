"""Chapter 5 tests: Levy characteristic functions, Merton series, jump paths.

Two independent verification paths for every model: (1) the martingale probe
cf(-i, t) = e^{(r-q)t}, exact analytically, evaluated at a genuinely complex
argument; (2) cumulants against numerical derivatives of log cf at u = 0 —
if either the cf or the cumulant formula were wrong they would not agree.
"""

from __future__ import annotations

import numpy as np
import pytest

from cosfin.core.gbm import gbm_terminal_moments
from cosfin.core.jump_diffusion import simulate_merton_paths
from cosfin.models.black_scholes import black_scholes_price
from cosfin.models.characteristic_functions import (
    CGMY,
    GBM,
    CharacteristicModel,
    Merton,
    VarianceGamma,
)
from cosfin.models.jump_diffusion import merton_jump_price

MODELS: list[tuple[str, CharacteristicModel, float, float]] = [
    ("gbm", GBM(r=0.05, q=0.0, sigma=0.2), 0.05, 0.0),
    ("merton", Merton(r=0.05, q=0.0, sigma=0.2, lam=1.0, mu_j=-0.1, sigma_j=0.15), 0.05, 0.0),
    ("cgmy", CGMY(r=0.1, q=0.02, C=1.0, G=5.0, M=5.0, Y=1.5, sigma=0.2), 0.1, 0.02),
    ("vg", VarianceGamma(r=0.1, q=0.0, sigma=0.12, theta=-0.14, beta=0.2), 0.1, 0.0),
]


@pytest.mark.parametrize(("name", "model", "r", "q"), MODELS, ids=[m[0] for m in MODELS])
def test_cf_at_zero_is_one(name: str, model: CharacteristicModel, r: float, q: float) -> None:
    value = complex(model.cf(np.array([0.0]), 1.0)[0])
    assert value == pytest.approx(1.0 + 0.0j, abs=1e-14)


@pytest.mark.parametrize("t", [0.5, 1.0])
@pytest.mark.parametrize(("name", "model", "r", "q"), MODELS, ids=[m[0] for m in MODELS])
def test_martingale_probe(
    name: str, model: CharacteristicModel, r: float, q: float, t: float
) -> None:
    # cf(-i, t) = E[S_t / S_0] = e^{(r-q)t} (RESULT); complex evaluation.
    value = complex(model.cf(np.array([-1j]), t)[0])
    assert value.imag == pytest.approx(0.0, abs=1e-12)
    assert value.real == pytest.approx(float(np.exp((r - q) * t)), abs=1e-10)


@pytest.mark.parametrize(("name", "model", "r", "q"), MODELS, ids=[m[0] for m in MODELS])
def test_cumulants_match_log_cf_derivatives(
    name: str, model: CharacteristicModel, r: float, q: float
) -> None:
    # c1 = Im log cf(h)/h, c2 = -2 Re log cf(h)/h^2 at u -> 0 (RESULT:
    # cumulant generating function), finite-difference with h = 1e-3.
    h = 1e-3
    log_cf = complex(np.log(model.cf(np.array([h]), 1.0)[0]))
    c1, c2, c4 = model.cumulants(1.0)
    assert c1 == pytest.approx(log_cf.imag / h, abs=1e-7)
    assert c2 == pytest.approx(-2.0 * log_cf.real / h**2, abs=1e-6)
    assert c4 >= 0.0


def test_model_validation() -> None:
    with pytest.raises(ValueError):
        GBM(r=0.05, q=0.0, sigma=0.0)
    with pytest.raises(ValueError):
        Merton(r=0.05, q=0.0, sigma=0.2, lam=-1.0, mu_j=0.0, sigma_j=0.1)
    with pytest.raises(ValueError):
        Merton(r=0.05, q=0.0, sigma=0.2, lam=1.0, mu_j=0.0, sigma_j=-0.1)
    with pytest.raises(ValueError):
        VarianceGamma(r=0.1, q=0.0, sigma=-0.12, theta=-0.14, beta=0.2)
    with pytest.raises(ValueError):
        VarianceGamma(r=0.1, q=0.0, sigma=0.12, theta=-0.14, beta=0.0)
    with pytest.raises(ValueError):  # 1 - theta*beta - sigma^2 beta/2 <= 0
        VarianceGamma(r=0.1, q=0.0, sigma=0.12, theta=5.0, beta=1.0)
    for bad in (
        {"C": 0.0}, {"G": 0.0}, {"M": 1.0}, {"Y": 1.0}, {"Y": 2.5}, {"Y": -0.1},
        {"sigma": -0.1},
    ):
        params = {"r": 0.1, "q": 0.0, "C": 1.0, "G": 5.0, "M": 5.0, "Y": 0.5, "sigma": 0.0}
        params.update(bad)
        with pytest.raises(ValueError):
            CGMY(**params)
    with pytest.raises(ValueError):
        GBM(r=0.05, q=0.0, sigma=0.2).cf(np.array([0.0]), 0.0)


def test_merton_series_reduces_to_black_scholes() -> None:
    strikes = np.array([80.0, 100.0, 120.0])
    series = merton_jump_price(100.0, strikes, 0.05, 0.2, 1.0,
                               lam=0.0, mu_j=-0.1, sigma_j=0.15)
    np.testing.assert_allclose(
        series, black_scholes_price(100.0, strikes, 0.05, 0.2, 1.0), atol=1e-12
    )


def test_merton_series_parity_and_regression() -> None:
    call = merton_jump_price(100.0, 100.0, 0.05, 0.2, 1.0,
                             lam=1.0, mu_j=-0.1, sigma_j=0.15)
    put = merton_jump_price(100.0, 100.0, 0.05, 0.2, 1.0,
                            lam=1.0, mu_j=-0.1, sigma_j=0.15, option_type="put")
    forward_leg = 100.0 - 100.0 * np.exp(-0.05)
    assert float(call - put) == pytest.approx(forward_leg, abs=1e-10)
    # Internal full-precision regression (independently cross-validated
    # against the COS engine in test_ch6_cos to 5e-13).
    assert float(call) == pytest.approx(12.761288593628754, rel=1e-10)


def test_merton_series_non_convergence_raises() -> None:
    with pytest.raises(RuntimeError):
        merton_jump_price(100.0, 100.0, 0.05, 0.2, 1.0,
                          lam=5.0, mu_j=-0.1, sigma_j=0.15, max_terms=2)


def test_merton_series_validation() -> None:
    with pytest.raises(ValueError):
        merton_jump_price(100.0, 100.0, 0.05, 0.2, 0.0,
                          lam=1.0, mu_j=0.0, sigma_j=0.1)
    with pytest.raises(ValueError):
        merton_jump_price(100.0, 100.0, 0.05, 0.2, 1.0,
                          lam=-1.0, mu_j=0.0, sigma_j=0.1)


def test_merton_paths_martingale_and_shapes() -> None:
    times, paths = simulate_merton_paths(
        100.0, 0.05, 0.2, 1.0, -0.1, 0.15, 1.0, 200_000, 2,
        rng=np.random.default_rng(5),
    )
    assert times.shape == (3,) and paths.shape == (200_000, 3)
    assert np.all(paths[:, 0] == 100.0)
    terminal = paths[:, -1]
    standard_error = terminal.std(ddof=1) / np.sqrt(terminal.size)
    assert np.abs(terminal.mean() - 100.0 * np.exp(0.05)) < 4.0 * standard_error


def test_merton_paths_zero_intensity_matches_gbm_law() -> None:
    _, paths = simulate_merton_paths(
        100.0, 0.05, 0.25, 0.0, -0.1, 0.15, 1.0, 300_000, 1,
        rng=np.random.default_rng(6),
    )
    terminal = paths[:, -1]
    mean, variance = gbm_terminal_moments(100.0, 0.05, 0.25, 1.0)
    assert terminal.mean() == pytest.approx(mean, rel=3e-3)
    assert terminal.var() == pytest.approx(variance, rel=0.03)


def test_merton_paths_validation_and_reproducibility() -> None:
    with pytest.raises(ValueError):
        simulate_merton_paths(100.0, 0.05, 0.2, -1.0, 0.0, 0.1, 1.0, 5, 5)
    _, a = simulate_merton_paths(100.0, 0.05, 0.2, 1.0, -0.1, 0.15, 1.0, 4, 6,
                                 rng=np.random.default_rng(7))
    _, b = simulate_merton_paths(100.0, 0.05, 0.2, 1.0, -0.1, 0.15, 1.0, 4, 6,
                                 rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_validation_edges_paths_series_and_cf() -> None:
    with pytest.raises(ValueError):  # Merton cf: diffusive sigma < 0
        Merton(r=0.05, q=0.0, sigma=-0.1, lam=1.0, mu_j=0.0, sigma_j=0.1)
    with pytest.raises(ValueError):  # paths: s0
        simulate_merton_paths(-1.0, 0.05, 0.2, 1.0, 0.0, 0.1, 1.0, 5, 5)
    with pytest.raises(ValueError):  # paths: sigma_j
        simulate_merton_paths(100.0, 0.05, 0.2, 1.0, 0.0, -0.1, 1.0, 5, 5)
    with pytest.raises(ValueError):  # paths: t_end
        simulate_merton_paths(100.0, 0.05, 0.2, 1.0, 0.0, 0.1, 0.0, 5, 5)
    with pytest.raises(ValueError):  # series: sigma < 0
        merton_jump_price(100.0, 100.0, 0.05, -0.2, 1.0, lam=1.0, mu_j=0.0, sigma_j=0.1)
    with pytest.raises(ValueError):  # series: option_type
        merton_jump_price(100.0, 100.0, 0.05, 0.2, 1.0, lam=1.0, mu_j=0.0,
                          sigma_j=0.1, option_type="straddle")  # type: ignore[arg-type]
