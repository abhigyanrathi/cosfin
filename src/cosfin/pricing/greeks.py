"""Black-Scholes Greeks (O&G Chapter 3).

All sensitivities of the Black-Scholes price, broadcast over array-like
inputs. Formulas (RESULT: differentiate the pricing formula; the ``d1``/``d2``
derivative terms cancel through the identity
``S e^{-qT} phi(d1) = K e^{-rT} phi(d2)``):

    delta_call = e^{-qT} N(d1)            delta_put = e^{-qT} (N(d1) - 1)
    gamma      = e^{-qT} phi(d1) / (S sigma sqrt(T))          (call = put)
    vega       = S e^{-qT} phi(d1) sqrt(T)                    (call = put)
    theta_call = -S e^{-qT} phi(d1) sigma / (2 sqrt(T))
                 + q S e^{-qT} N(d1) - r K e^{-rT} N(d2)
    theta_put  = -S e^{-qT} phi(d1) sigma / (2 sqrt(T))
                 - q S e^{-qT} N(-d1) + r K e^{-rT} N(-d2)
    rho_call   = K T e^{-rT} N(d2)        rho_put = -K T e^{-rT} N(-d2)

Sign CONVENTION for theta: the derivative with respect to *calendar time* at
fixed maturity date (``-dV/dT``), quoted per year — the usual "time decay"
number, typically negative for long vanilla positions.

Degenerate limit ``sigma sqrt(T) = 0``: the signed-infinity ``d1``/``d2``
convention of the model layer makes ``N(+/-inf)`` the correct indicators and
``phi(+/-inf) = 0``, so delta, theta and rho collapse to their one-sided
limits with no special-casing here. Gamma alone needs a guard (its true limit
is a Dirac mass at the forward strike); CONVENTION: gamma returns ``0`` on
that set.

``vega`` is defined in ``models.black_scholes`` (the implied-volatility
solvers need it, and models must never import from pricing) and re-exported
here so the Greeks live together at the call site.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from cosfin.models.black_scholes import (
    OptionType,
    d1_d2,
    norm_cdf,
    norm_pdf,
    vega,
)

__all__ = ["delta", "gamma", "rho", "theta", "vega"]


def _require_option_type(option_type: str) -> None:
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def delta(
    spot: npt.ArrayLike,
    strike: npt.ArrayLike,
    r: npt.ArrayLike,
    sigma: npt.ArrayLike,
    t: npt.ArrayLike,
    *,
    option_type: OptionType = "call",
    q: npt.ArrayLike = 0.0,
) -> npt.NDArray[np.float64]:
    """First derivative of the option value in the spot (hedge ratio)."""
    _require_option_type(option_type)
    ta = np.asarray(t, dtype=np.float64)
    qa = np.asarray(q, dtype=np.float64)
    d1, _ = d1_d2(spot, strike, r, sigma, ta, q=qa)
    dividend_discount = np.exp(-qa * ta)
    if option_type == "call":
        return np.asarray(dividend_discount * norm_cdf(d1), dtype=np.float64)
    return np.asarray(dividend_discount * (norm_cdf(d1) - 1.0), dtype=np.float64)


def gamma(
    spot: npt.ArrayLike,
    strike: npt.ArrayLike,
    r: npt.ArrayLike,
    sigma: npt.ArrayLike,
    t: npt.ArrayLike,
    *,
    q: npt.ArrayLike = 0.0,
) -> npt.NDArray[np.float64]:
    """Second derivative in the spot; identical for calls and puts (RESULT)."""
    s = np.asarray(spot, dtype=np.float64)
    sig = np.asarray(sigma, dtype=np.float64)
    ta = np.asarray(t, dtype=np.float64)
    qa = np.asarray(q, dtype=np.float64)
    d1, _ = d1_d2(s, strike, r, sig, ta, q=qa)
    vol_sqrt_t = sig * np.sqrt(ta)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.exp(-qa * ta) * norm_pdf(d1) / (s * vol_sqrt_t)
    return np.asarray(np.where(vol_sqrt_t > 0.0, value, 0.0), dtype=np.float64)


def theta(
    spot: npt.ArrayLike,
    strike: npt.ArrayLike,
    r: npt.ArrayLike,
    sigma: npt.ArrayLike,
    t: npt.ArrayLike,
    *,
    option_type: OptionType = "call",
    q: npt.ArrayLike = 0.0,
) -> npt.NDArray[np.float64]:
    """Calendar-time decay ``-dV/dT`` per year (see sign CONVENTION above)."""
    _require_option_type(option_type)
    s = np.asarray(spot, dtype=np.float64)
    k = np.asarray(strike, dtype=np.float64)
    ra = np.asarray(r, dtype=np.float64)
    sig = np.asarray(sigma, dtype=np.float64)
    ta = np.asarray(t, dtype=np.float64)
    qa = np.asarray(q, dtype=np.float64)

    d1, d2 = d1_d2(s, k, ra, sig, ta, q=qa)
    disc_s = s * np.exp(-qa * ta)
    disc_k = k * np.exp(-ra * ta)

    # phi(+/-inf) = 0 kills the decay term for sigma = 0; only t = 0 needs
    # the safe denominator (0/0 otherwise).
    safe_sqrt_t = np.where(ta > 0.0, np.sqrt(ta), 1.0)
    decay = np.where(ta > 0.0, -disc_s * norm_pdf(d1) * sig / (2.0 * safe_sqrt_t), 0.0)

    if option_type == "call":
        carry = qa * disc_s * norm_cdf(d1) - ra * disc_k * norm_cdf(d2)
    else:
        carry = -qa * disc_s * norm_cdf(-d1) + ra * disc_k * norm_cdf(-d2)
    return np.asarray(decay + carry, dtype=np.float64)


def rho(
    spot: npt.ArrayLike,
    strike: npt.ArrayLike,
    r: npt.ArrayLike,
    sigma: npt.ArrayLike,
    t: npt.ArrayLike,
    *,
    option_type: OptionType = "call",
    q: npt.ArrayLike = 0.0,
) -> npt.NDArray[np.float64]:
    """Sensitivity to the risk-free rate, per unit of rate."""
    _require_option_type(option_type)
    k = np.asarray(strike, dtype=np.float64)
    ra = np.asarray(r, dtype=np.float64)
    ta = np.asarray(t, dtype=np.float64)
    _, d2 = d1_d2(spot, k, ra, sigma, ta, q=q)
    disc_k = k * np.exp(-ra * ta)
    if option_type == "call":
        return np.asarray(ta * disc_k * norm_cdf(d2), dtype=np.float64)
    return np.asarray(-ta * disc_k * norm_cdf(-d2), dtype=np.float64)
