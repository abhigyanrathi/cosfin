"""Pure computation helpers backing the Streamlit demo (``app/streamlit_app.py``).

These live inside the package — not in ``app/`` — so they are covered by the
test suite, checked by ``mypy --strict``, and importable without Streamlit
installed. The app file itself is UI glue only.

Pipelines demonstrated:

* ``levy_smile``: Ch5 characteristic function -> Ch6 COS call prices -> Ch4
  implied-vol inversion. Under GBM the recovered smile must be flat at the
  input sigma (a cross-module consistency anchor); under Merton/VG/CGMY the
  jumps generate a genuine smile.
* ``gbm_density_comparison``: the p.169-style density-recovery exercise —
  Breeden-Litzenberger second differences and the COS density expansion,
  both against the closed-form lognormal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from cosfin.models.black_scholes import black_scholes_price
from cosfin.models.characteristic_functions import GBM, CharacteristicModel
from cosfin.models.local_vol import implied_volatility, risk_neutral_density
from cosfin.numerical.cos import cos_density, truncation_range
from cosfin.numerical.cos import cos_european_price as _cos_price


def lognormal_terminal_pdf(
    s: npt.ArrayLike,
    s0: float,
    mu: float,
    sigma: float,
    t: float,
) -> npt.NDArray[np.float64]:
    """Density of ``S_t`` under GBM ``dS = mu S dt + sigma S dW``.

    RESULT: ``ln S_t ~ N(ln s0 + (mu - sigma^2/2) t, sigma^2 t)``, so

        f(s) = exp( -(ln(s/s0) - (mu - sigma^2/2) t)^2 / (2 sigma^2 t) )
               / (s sigma sqrt(2 pi t)).

    ``mu`` is the *arithmetic* drift (the SDE coefficient); pass ``r - q``
    for the risk-neutral density. Matches the convention of
    ``cosfin.core.gbm.simulate_gbm_paths``.
    """
    if s0 <= 0.0 or sigma <= 0.0 or t <= 0.0:
        raise ValueError("s0, sigma and t must be positive")
    s_arr = np.asarray(s, dtype=np.float64)
    if np.any(s_arr <= 0.0):
        raise ValueError("s must be positive")
    m = np.log(s0) + (mu - 0.5 * sigma**2) * t
    v = sigma * np.sqrt(t)
    z = (np.log(s_arr) - m) / v
    out = np.exp(-0.5 * z**2) / (s_arr * v * np.sqrt(2.0 * np.pi))
    return np.asarray(out, dtype=np.float64)


def levy_smile(
    model: CharacteristicModel,
    spot: float,
    strikes: npt.ArrayLike,
    r: float,
    t: float,
    *,
    q: float = 0.0,
    n: int = 512,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """COS call prices under ``model`` and their Black-Scholes implied vols.

    Returns ``(prices, ivs)`` aligned with ``strikes``. A strike whose COS
    price falls outside the no-arbitrage band — possible in the deep wings,
    where the true price is below the COS truncation/discretisation error —
    yields ``NaN`` in ``ivs`` instead of raising, so one bad wing point
    cannot take down a whole smile. ``q`` must equal the dividend yield
    baked into ``model``; the inversion uses the same carry.
    """
    strikes_arr = np.asarray(strikes, dtype=np.float64)
    prices = _cos_price(model.cf, model.cumulants, spot, strikes_arr, r, t, n, option_type="call")
    ivs = np.full(strikes_arr.shape, np.nan, dtype=np.float64)
    for i, (k, p) in enumerate(zip(strikes_arr, prices, strict=True)):
        try:
            ivs[i] = implied_volatility(float(p), spot, float(k), r, t, option_type="call", q=q)
        except ValueError:
            continue  # price outside no-arbitrage bounds -> leave NaN
    return prices, ivs


@dataclass(frozen=True)
class DensityComparison:
    """Three density estimates on a common interior strike grid, plus errors."""

    strikes: npt.NDArray[np.float64]
    exact: npt.NDArray[np.float64]
    breeden_litzenberger: npt.NDArray[np.float64]
    cos: npt.NDArray[np.float64]
    max_err_bl: float
    max_err_cos: float


def gbm_density_comparison(
    spot: float,
    r: float,
    sigma: float,
    t: float,
    *,
    q: float = 0.0,
    k_min: float,
    k_max: float,
    n_strikes: int = 201,
    n_cos: int = 256,
) -> DensityComparison:
    """Recover the GBM risk-neutral density two ways and compare with the truth.

    (1) Breeden-Litzenberger: analytic Black-Scholes call prices on a uniform
    strike grid, then ``f = e^{rT} d^2C/dK^2`` by central second differences
    (``O(h^2)`` discretisation error — visible, honest, and it shrinks as
    ``n_strikes`` grows). (2) COS: the Fourier-cosine density of
    ``ln(S_T/S_0)`` mapped to the ``S``-density via the Jacobian ``1/s``.
    Both are evaluated on the interior grid that Breeden-Litzenberger
    returns, so the max-abs errors are directly comparable.
    """
    if n_strikes < 3:
        raise ValueError("n_strikes must be at least 3")
    if not 0.0 < k_min < k_max:
        raise ValueError("need 0 < k_min < k_max")
    strikes = np.linspace(k_min, k_max, n_strikes)
    calls = black_scholes_price(spot, strikes, r, sigma, t, option_type="call", q=q)
    k_in, f_bl = risk_neutral_density(strikes, calls, r, t)

    model = GBM(r=r, q=q, sigma=sigma)
    a, b = truncation_range(model.cumulants, t)
    f_log = cos_density(model.cf, np.log(k_in / spot), a, b, n_cos, t)
    f_cos = np.asarray(f_log / k_in, dtype=np.float64)

    f_exact = lognormal_terminal_pdf(k_in, spot, r - q, sigma, t)
    return DensityComparison(
        strikes=k_in,
        exact=f_exact,
        breeden_litzenberger=f_bl,
        cos=f_cos,
        max_err_bl=float(np.max(np.abs(f_bl - f_exact))),
        max_err_cos=float(np.max(np.abs(f_cos - f_exact))),
    )
