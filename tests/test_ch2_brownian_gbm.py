"""Chapter 2 tests: Brownian motion and GBM simulation.

Statistical assertions use seeded generators with tolerances set at ~4x the
relevant CLT standard error, so they are deterministic in CI yet would catch
a genuinely wrong distribution.
"""

from __future__ import annotations

import numpy as np
import pytest

from cosfin.core.brownian import simulate_brownian_paths
from cosfin.core.gbm import gbm_terminal_moments, simulate_gbm_paths


def test_brownian_shapes_and_origin() -> None:
    times, paths = simulate_brownian_paths(7, 11, 2.5, rng=np.random.default_rng(0))
    assert times.shape == (12,)
    assert paths.shape == (7, 12)
    assert times[0] == 0.0 and times[-1] == pytest.approx(2.5)
    assert np.all(paths[:, 0] == 0.0)


def test_brownian_increment_moments() -> None:
    # W_{t+dt} - W_t ~ N(0, dt): mean SE ~ sqrt(dt/N) ~ 0.0016, var SE ~ dt*sqrt(2/N).
    rng = np.random.default_rng(1)
    _, paths = simulate_brownian_paths(200_000, 4, 2.0, rng=rng)
    increments = np.diff(paths, axis=1)
    assert np.abs(increments.mean()) < 0.01
    assert increments.var() == pytest.approx(0.5, rel=0.02)


def test_brownian_quadratic_variation() -> None:
    # sum (dW)^2 -> t (RESULT); sd of the sum is t*sqrt(2/n) ~ 0.01 here.
    _, paths = simulate_brownian_paths(5, 20_000, 1.0, rng=np.random.default_rng(9))
    quadratic_variation = np.sum(np.diff(paths, axis=1) ** 2, axis=1)
    assert np.all(np.abs(quadratic_variation - 1.0) < 0.06)


def test_brownian_reproducible_and_seed_sensitive() -> None:
    _, a = simulate_brownian_paths(3, 5, 1.0, rng=np.random.default_rng(42))
    _, b = simulate_brownian_paths(3, 5, 1.0, rng=np.random.default_rng(42))
    _, c = simulate_brownian_paths(3, 5, 1.0, rng=np.random.default_rng(43))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize(
    ("n_paths", "n_steps", "t_end"),
    [(0, 5, 1.0), (5, 0, 1.0), (5, 5, 0.0), (5, 5, -1.0)],
)
def test_brownian_validation(n_paths: int, n_steps: int, t_end: float) -> None:
    with pytest.raises(ValueError):
        simulate_brownian_paths(n_paths, n_steps, t_end)


def test_gbm_terminal_moments_closed_form() -> None:
    mean, variance = gbm_terminal_moments(100.0, 0.05, 0.2, 1.0)
    assert mean == pytest.approx(100.0 * np.exp(0.05), rel=1e-12)
    expected_var = 100.0**2 * np.exp(0.1) * (np.exp(0.04) - 1.0)
    assert variance == pytest.approx(expected_var, rel=1e-12)


def test_gbm_exact_matches_closed_form_moments() -> None:
    # Exact scheme: one step suffices for the terminal law (RESULT).
    _, paths = simulate_gbm_paths(100.0, 0.05, 0.2, 1.0, 400_000, 1,
                                  rng=np.random.default_rng(3))
    terminal = paths[:, -1]
    mean, variance = gbm_terminal_moments(100.0, 0.05, 0.2, 1.0)
    assert terminal.mean() == pytest.approx(mean, rel=2e-3)
    assert terminal.var() == pytest.approx(variance, rel=0.03)


def test_gbm_discounted_martingale() -> None:
    _, paths = simulate_gbm_paths(100.0, 0.05, 0.2, 1.0, 400_000, 1,
                                  rng=np.random.default_rng(4))
    discounted = np.exp(-0.05) * paths[:, -1]
    standard_error = discounted.std(ddof=1) / np.sqrt(discounted.size)
    assert np.abs(discounted.mean() - 100.0) < 4.0 * standard_error


def test_gbm_euler_weak_error_small() -> None:
    # Euler weak error is O(dt); at 500 steps observed relative mean error ~6e-4.
    _, paths = simulate_gbm_paths(100.0, 0.05, 0.2, 1.0, 200_000, 500,
                                  scheme="euler", rng=np.random.default_rng(2))
    mean, _ = gbm_terminal_moments(100.0, 0.05, 0.2, 1.0)
    assert paths[:, -1].mean() == pytest.approx(mean, rel=5e-3)


def test_gbm_exact_paths_positive_and_start_at_s0() -> None:
    _, paths = simulate_gbm_paths(50.0, 0.0, 0.4, 2.0, 100, 30,
                                  rng=np.random.default_rng(5))
    assert np.all(paths > 0.0)
    assert np.all(paths[:, 0] == 50.0)


def test_gbm_reproducible() -> None:
    _, a = simulate_gbm_paths(100.0, 0.05, 0.2, 1.0, 4, 6, rng=np.random.default_rng(7))
    _, b = simulate_gbm_paths(100.0, 0.05, 0.2, 1.0, 4, 6, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_gbm_validation() -> None:
    with pytest.raises(ValueError):
        simulate_gbm_paths(-1.0, 0.05, 0.2, 1.0, 5, 5)
    with pytest.raises(ValueError):
        simulate_gbm_paths(100.0, 0.05, -0.2, 1.0, 5, 5)
    with pytest.raises(ValueError):
        simulate_gbm_paths(100.0, 0.05, 0.2, 1.0, 0, 5)
    with pytest.raises(ValueError):
        simulate_gbm_paths(100.0, 0.05, 0.2, 1.0, 5, 5, scheme="bogus")  # type: ignore[arg-type]
