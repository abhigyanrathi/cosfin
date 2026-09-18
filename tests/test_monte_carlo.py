"""Monte Carlo estimator tests: unbiasedness within CLT bands, SE scaling."""

from __future__ import annotations

import numpy as np
import pytest

from cosfin.core.gbm import simulate_gbm_paths
from cosfin.core.jump_diffusion import simulate_merton_paths
from cosfin.models.black_scholes import black_scholes_price
from cosfin.models.jump_diffusion import merton_jump_price
from cosfin.numerical.monte_carlo import monte_carlo_european_price


def test_gbm_call_within_confidence_band() -> None:
    _, paths = simulate_gbm_paths(100.0, 0.05, 0.2, 1.0, 400_000, 1,
                                  rng=np.random.default_rng(3))
    result = monte_carlo_european_price(paths[:, -1], 100.0, 0.05, 1.0)
    closed = float(black_scholes_price(100.0, 100.0, 0.05, 0.2, 1.0))
    assert abs(result.price - closed) < 4.0 * result.standard_error
    assert 0.001 < result.standard_error < 0.1
    assert result.n_paths == 400_000


def test_gbm_put_within_confidence_band() -> None:
    _, paths = simulate_gbm_paths(100.0, 0.05, 0.2, 1.0, 400_000, 1,
                                  rng=np.random.default_rng(8))
    result = monte_carlo_european_price(paths[:, -1], 100.0, 0.05, 1.0,
                                        option_type="put")
    closed = float(black_scholes_price(100.0, 100.0, 0.05, 0.2, 1.0,
                                       option_type="put"))
    assert abs(result.price - closed) < 4.0 * result.standard_error


def test_merton_mc_cross_validates_series() -> None:
    _, paths = simulate_merton_paths(100.0, 0.05, 0.2, 1.0, -0.1, 0.15, 1.0,
                                     300_000, 1, rng=np.random.default_rng(5))
    result = monte_carlo_european_price(paths[:, -1], 100.0, 0.05, 1.0)
    series = float(merton_jump_price(100.0, 100.0, 0.05, 0.2, 1.0,
                                     lam=1.0, mu_j=-0.1, sigma_j=0.15))
    assert abs(result.price - series) < 4.0 * result.standard_error


def test_standard_error_scales_inverse_sqrt_n() -> None:
    _, small = simulate_gbm_paths(100.0, 0.05, 0.2, 1.0, 40_000, 1,
                                  rng=np.random.default_rng(21))
    _, large = simulate_gbm_paths(100.0, 0.05, 0.2, 1.0, 160_000, 1,
                                  rng=np.random.default_rng(22))
    se_small = monte_carlo_european_price(small[:, -1], 100.0, 0.05, 1.0).standard_error
    se_large = monte_carlo_european_price(large[:, -1], 100.0, 0.05, 1.0).standard_error
    assert se_small / se_large == pytest.approx(2.0, rel=0.25)


def test_validation() -> None:
    with pytest.raises(ValueError):
        monte_carlo_european_price(np.array([100.0]), 100.0, 0.05, 1.0)
    with pytest.raises(ValueError):
        monte_carlo_european_price(np.array([100.0, 101.0]), -1.0, 0.05, 1.0)
    with pytest.raises(ValueError):
        monte_carlo_european_price(np.array([100.0, np.inf]), 100.0, 0.05, 1.0)
    with pytest.raises(ValueError):
        monte_carlo_european_price(np.array([100.0, 101.0]), 100.0, 0.05, 1.0,
                                   option_type="straddle")  # type: ignore[arg-type]


def test_negative_maturity_rejected() -> None:
    with pytest.raises(ValueError):
        monte_carlo_european_price(np.array([100.0, 101.0]), 100.0, 0.05, -1.0)
