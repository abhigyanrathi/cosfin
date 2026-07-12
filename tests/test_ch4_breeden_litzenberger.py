"""Chapter 4 (Section 4.2) tests: Breeden-Litzenberger density recovery.

The Black-Scholes surface is the controlled experiment: its implied density
is known in closed form (lognormal), so the recovered density can be checked
pointwise, in integrated mass, in its mean (the forward), and by repricing.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from pyfinlib_practice.models.black_scholes import (
    black_scholes_price,
    butterfly_price,
    digital_price,
)
from pyfinlib_practice.models.local_vol import price_from_density, risk_neutral_density

SPOT, RATE, SIGMA, MATURITY = 100.0, 0.05, 0.2, 1.0


@pytest.fixture(scope="module")
def bl_density() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    strikes = np.arange(20.0, 400.0 + 1e-9, 0.05)
    calls = black_scholes_price(SPOT, strikes, RATE, SIGMA, MATURITY)
    return risk_neutral_density(strikes, calls, RATE, MATURITY)


def test_density_nonnegative_and_normalised(
    bl_density: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
) -> None:
    grid, density = bl_density
    assert density.min() > -1e-9  # tiny FP negatives only
    assert np.trapezoid(density, grid) == pytest.approx(1.0, abs=1e-9)


def test_density_mean_is_forward(
    bl_density: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
) -> None:
    grid, density = bl_density
    forward = SPOT * np.exp(RATE * MATURITY)
    assert np.trapezoid(grid * density, grid) == pytest.approx(forward, abs=1e-6)


def test_density_matches_lognormal_pointwise(
    bl_density: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
) -> None:
    grid, density = bl_density
    m = np.log(SPOT) + (RATE - 0.5 * SIGMA**2) * MATURITY
    analytic = np.exp(-0.5 * ((np.log(grid) - m) / SIGMA) ** 2) / (
        grid * SIGMA * np.sqrt(2.0 * np.pi)
    )
    assert np.max(np.abs(density - analytic)) < 1e-7  # O(h^2) FD error, h=0.05


def test_reprice_call_put_and_parity(
    bl_density: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
) -> None:
    grid, density = bl_density
    call = price_from_density(grid, density, RATE, MATURITY, 100.0)
    put = price_from_density(grid, density, RATE, MATURITY, 100.0, option_type="put")
    assert call == pytest.approx(
        float(black_scholes_price(SPOT, 100.0, RATE, SIGMA, MATURITY)), abs=1e-7
    )
    assert put == pytest.approx(
        float(black_scholes_price(SPOT, 100.0, RATE, SIGMA, MATURITY, option_type="put")),
        abs=1e-7,
    )
    forward_leg = SPOT - 100.0 * np.exp(-RATE * MATURITY)
    assert call - put == pytest.approx(forward_leg, abs=1e-7)


def test_digital_by_tail_integration(
    bl_density: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
) -> None:
    # Trapezoid across the payoff jump at K contributes ~ f(K) h / 2 ~ 5e-4.
    grid, density = bl_density
    tail = np.exp(-RATE * MATURITY) * np.trapezoid(density * (grid > 100.0), grid)
    closed = float(digital_price(SPOT, 100.0, RATE, SIGMA, MATURITY))
    assert tail == pytest.approx(closed, abs=2e-3)


def test_butterfly_approximates_density(
    bl_density: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
) -> None:
    # butterfly / h^2 -> e^{-rT} f(K) as h -> 0 (the Ch3 <-> Ch4 bridge).
    grid, density = bl_density
    density_at_100 = density[np.argmin(np.abs(grid - 100.0))]
    fly = float(butterfly_price(SPOT, 100.0, RATE, SIGMA, MATURITY, 0.5))
    assert fly / 0.25 == pytest.approx(
        np.exp(-RATE * MATURITY) * density_at_100, abs=1e-4
    )


def test_validation() -> None:
    with pytest.raises(ValueError):
        risk_neutral_density(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 0.05, 1.0)
    with pytest.raises(ValueError):
        risk_neutral_density(
            np.array([1.0, 2.0, 1.5]), np.array([3.0, 2.0, 1.0]), 0.05, 1.0
        )
    with pytest.raises(ValueError):
        risk_neutral_density(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]), 0.05, 1.0)
    grid = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        price_from_density(grid, np.array([0.1, 0.2, 0.1]), 0.05, 1.0, 2.0,
                           option_type="straddle")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        price_from_density(grid[::-1], np.array([0.1, 0.2, 0.1]), 0.05, 1.0, 2.0)
    with pytest.raises(ValueError):  # shape mismatch
        price_from_density(grid, np.array([0.1, 0.2]), 0.05, 1.0, 2.0)
    with pytest.raises(ValueError):  # fewer than two grid points
        price_from_density(np.array([1.0]), np.array([0.1]), 0.05, 1.0, 2.0)
