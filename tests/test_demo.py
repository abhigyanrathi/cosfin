"""Tests for ``cosfin.demo`` — the compute layer behind the Streamlit app.

Anchors used here:

* Flat-smile-under-GBM: COS prices from the GBM characteristic function,
  inverted through the Ch4 solver, must return the input sigma exactly (to
  solver tolerance). This is a cross-module consistency check between the
  Ch3 pricer, the Ch4 inversion and the Ch6 COS engine.
* Convergence order: the Breeden-Litzenberger density error must shrink at
  the O(h^2) rate of the central second difference, not merely be "small".
* Independent reference: ``lognormal_terminal_pdf`` is checked against
  ``scipy.stats.lognorm``, not against our own formulas.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import lognorm  # type: ignore[import-untyped]

from cosfin.demo import (
    DensityComparison,
    gbm_density_comparison,
    levy_smile,
    lognormal_terminal_pdf,
)
from cosfin.models.characteristic_functions import GBM, Merton

SPOT, R, Q, SIGMA, T = 100.0, 0.05, 0.0, 0.2, 1.0


class TestLognormalTerminalPdf:
    def test_matches_scipy_lognorm(self) -> None:
        s = np.linspace(40.0, 250.0, 301)
        ours = lognormal_terminal_pdf(s, SPOT, R, SIGMA, T)
        scale = SPOT * np.exp((R - 0.5 * SIGMA**2) * T)
        ref = lognorm.pdf(s, s=SIGMA * np.sqrt(T), scale=scale)
        np.testing.assert_allclose(ours, ref, rtol=1e-12)

    def test_integrates_to_one(self) -> None:
        s = np.linspace(1.0, 1000.0, 20_001)
        f = lognormal_terminal_pdf(s, SPOT, R, SIGMA, T)
        assert np.trapezoid(f, s) == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize(
        ("s0", "sigma", "t"),
        [(-1.0, 0.2, 1.0), (100.0, 0.0, 1.0), (100.0, 0.2, 0.0)],
    )
    def test_rejects_bad_scalars(self, s0: float, sigma: float, t: float) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            lognormal_terminal_pdf(100.0, s0, R, sigma, t)

    def test_rejects_nonpositive_s(self) -> None:
        with pytest.raises(ValueError, match="s must be positive"):
            lognormal_terminal_pdf([-1.0, 100.0], SPOT, R, SIGMA, T)


class TestLevySmile:
    def test_gbm_recovers_flat_smile(self) -> None:
        strikes = np.linspace(60.0, 140.0, 17)
        model = GBM(r=R, q=Q, sigma=SIGMA)
        prices, ivs = levy_smile(model, SPOT, strikes, R, T, q=Q, n=512)
        assert prices.shape == ivs.shape == strikes.shape
        assert not np.any(np.isnan(ivs))
        np.testing.assert_allclose(ivs, SIGMA, atol=1e-6)

    def test_merton_produces_a_smile(self) -> None:
        strikes = np.linspace(70.0, 130.0, 13)
        model = Merton(r=R, q=Q, sigma=0.15, lam=0.75, mu_j=-0.10, sigma_j=0.15)
        _, ivs = levy_smile(model, SPOT, strikes, R, T, q=Q, n=512)
        assert not np.any(np.isnan(ivs))
        atm = ivs[len(ivs) // 2]
        assert ivs[0] > atm  # negative-mean jumps steepen the put wing
        assert float(np.max(ivs) - np.min(ivs)) > 0.01  # genuinely non-flat

    def test_out_of_bounds_wing_yields_nan_not_raise(self) -> None:
        # Deep OTM at short maturity with a deliberately coarse expansion:
        # the true price sits below the COS truncation error, the recovered
        # price violates the no-arbitrage band, and the inversion raises
        # internally. levy_smile must swallow that into NaN per strike.
        strikes = np.array([100.0, 400.0])
        model = Merton(r=R, q=Q, sigma=0.15, lam=0.75, mu_j=-0.10, sigma_j=0.15)
        _, ivs = levy_smile(model, SPOT, strikes, R, 0.1, q=Q, n=16)
        assert np.isfinite(ivs[0])
        assert np.isnan(ivs[1])


class TestGbmDensityComparison:
    def make(self, n_strikes: int, n_cos: int = 256) -> DensityComparison:
        return gbm_density_comparison(
            SPOT, R, SIGMA, T, q=Q, k_min=40.0, k_max=250.0, n_strikes=n_strikes, n_cos=n_cos
        )

    def test_cos_density_near_machine_precision(self) -> None:
        cmp_ = self.make(201)
        assert cmp_.max_err_cos < 1e-10

    def test_bl_error_has_second_order_convergence(self) -> None:
        coarse = self.make(101)  # h ~ 2.10
        fine = self.make(201)  # h ~ 1.05
        assert fine.max_err_bl < coarse.max_err_bl
        ratio = coarse.max_err_bl / fine.max_err_bl
        assert 3.0 < ratio < 5.0  # O(h^2): halving h quarters the error

    def test_arrays_align_on_interior_grid(self) -> None:
        cmp_ = self.make(101)
        assert cmp_.strikes.shape == (99,)
        for arr in (cmp_.exact, cmp_.breeden_litzenberger, cmp_.cos):
            assert arr.shape == cmp_.strikes.shape

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_strikes": 2},
            {"k_min": -1.0},
            {"k_min": 250.0, "k_max": 40.0},
        ],
    )
    def test_rejects_bad_grids(self, kwargs: dict[str, float]) -> None:
        base: dict[str, float] = {"k_min": 40.0, "k_max": 250.0, "n_strikes": 101}
        base.update(kwargs)
        with pytest.raises(ValueError):
            gbm_density_comparison(
                SPOT,
                R,
                SIGMA,
                T,
                q=Q,
                k_min=base["k_min"],
                k_max=base["k_max"],
                n_strikes=int(base["n_strikes"]),
            )
