"""The Black-Scholes-Merton closed-form option pricing model (O&G Chapter 3).

Under the BSM assumptions the time-0 value of a European option with spot
``S``, strike ``K``, rate ``r``, continuous dividend yield ``q``, volatility
``sigma`` and maturity ``T`` is (RESULT, via the risk-neutral expectation /
Feynman-Kac):

    call = S e^{-qT} N(d1) - K e^{-rT} N(d2),
    put  = K e^{-rT} N(-d2) - S e^{-qT} N(-d1),

with

    d1 = [ln(S e^{(r-q)T} / K)] / (sigma sqrt(T)) + sigma sqrt(T) / 2,
    d2 = d1 - sigma sqrt(T),

and ``N`` the standard normal CDF. (The ``d1`` form above is algebraically
identical to the textbook ``[ln(S/K) + (r - q + sigma^2/2) T] / (sigma
sqrt(T))`` but groups the forward log-moneyness into a single division.)

Degenerate limit (CONVENTION of this module): where ``sigma sqrt(T) == 0``,
``d1`` and ``d2`` are returned as their one-sided limits ``+/-inf`` (``0`` at
exact forward-ATM). Since ``N(+inf) = 1`` and ``N(-inf) = 0``, every consumer —
price, delta, digital — then collapses to the correct discounted-intrinsic /
indicator limit *by construction*, with no per-function special-casing and no
``nan`` leakage. (A placeholder-denominator approach instead requires each
caller to guard the branch itself; a caller that forgets returns silently
wrong numbers for ``sigma = 0`` or ``T = 0``.)

Dependency rule: the canonical ``vega`` lives here in the model layer because
the implied-volatility solvers (models layer) need it; ``pricing.greeks``
re-exports it. Models never import from ``pricing`` — that direction is what
creates import cycles.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.special import ndtr  # type: ignore[import-untyped]

OptionType = Literal["call", "put"]

_SQRT_2PI = float(np.sqrt(2.0 * np.pi))


def norm_pdf(x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Standard normal PDF ``phi(x) = exp(-x^2/2) / sqrt(2 pi)``.

    Evaluates to exactly ``0.0`` at ``+/-inf`` (``exp(-inf)``), which is what
    makes the degenerate-limit convention of this module self-consistent.
    """
    xa = np.asarray(x, dtype=np.float64)
    return np.asarray(np.exp(-0.5 * xa**2) / _SQRT_2PI, dtype=np.float64)


def norm_cdf(x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Standard normal CDF via ``scipy.special.ndtr`` (fast, vectorised)."""
    xa = np.asarray(x, dtype=np.float64)
    return np.asarray(ndtr(xa), dtype=np.float64)


def _require_option_type(option_type: str) -> None:
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def d1_d2(
    spot: npt.ArrayLike,
    strike: npt.ArrayLike,
    r: npt.ArrayLike,
    sigma: npt.ArrayLike,
    t: npt.ArrayLike,
    *,
    q: npt.ArrayLike = 0.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Black-Scholes ``d1`` and ``d2``, broadcast over array-like inputs.

    Degenerate limit: where ``sigma sqrt(T) == 0`` the *one-sided limits* are
    returned — ``+inf`` if the forward is above the strike, ``-inf`` below,
    ``0`` at exact forward-ATM — so ``N(d1)``, ``N(d2)`` become the correct
    indicator functions and downstream formulas inherit the right limit.
    """
    s = np.asarray(spot, dtype=np.float64)
    k = np.asarray(strike, dtype=np.float64)
    ra = np.asarray(r, dtype=np.float64)
    sig = np.asarray(sigma, dtype=np.float64)
    ta = np.asarray(t, dtype=np.float64)
    qa = np.asarray(q, dtype=np.float64)

    if np.any(s <= 0.0) or np.any(k <= 0.0):
        raise ValueError("spot and strike must be positive")
    if np.any(sig < 0.0):
        raise ValueError("sigma must be non-negative")
    if np.any(ta < 0.0):
        raise ValueError("t must be non-negative")

    vol_sqrt_t = sig * np.sqrt(ta)
    log_forward_moneyness = np.log(s / k) + (ra - qa) * ta

    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = log_forward_moneyness / vol_sqrt_t + 0.5 * vol_sqrt_t
        d2 = d1 - vol_sqrt_t

    degenerate = vol_sqrt_t == 0.0
    if np.any(degenerate):
        limit = np.where(
            log_forward_moneyness > 0.0,
            np.inf,
            np.where(log_forward_moneyness < 0.0, -np.inf, 0.0),
        )
        d1 = np.where(degenerate, limit, d1)
        d2 = np.where(degenerate, limit, d2)
    return np.asarray(d1, dtype=np.float64), np.asarray(d2, dtype=np.float64)


def black_scholes_price(
    spot: npt.ArrayLike,
    strike: npt.ArrayLike,
    r: npt.ArrayLike,
    sigma: npt.ArrayLike,
    t: npt.ArrayLike,
    *,
    option_type: OptionType = "call",
    q: npt.ArrayLike = 0.0,
) -> npt.NDArray[np.float64]:
    """Black-Scholes price of a European call or put (vectorised).

    At ``sigma sqrt(T) == 0`` the signed-infinity ``d1``/``d2`` limits make
    this evaluate to the discounted forward intrinsic value
    ``max(+/-(S e^{-qT} - K e^{-rT}), 0)`` with no explicit branch (RESULT:
    the zero-volatility option is a deterministic forward claim).
    """
    _require_option_type(option_type)
    s = np.asarray(spot, dtype=np.float64)
    k = np.asarray(strike, dtype=np.float64)
    ra = np.asarray(r, dtype=np.float64)
    ta = np.asarray(t, dtype=np.float64)
    qa = np.asarray(q, dtype=np.float64)

    d1, d2 = d1_d2(s, k, ra, sigma, ta, q=qa)
    disc_s = s * np.exp(-qa * ta)
    disc_k = k * np.exp(-ra * ta)

    if option_type == "call":
        price = disc_s * norm_cdf(d1) - disc_k * norm_cdf(d2)
    else:
        price = disc_k * norm_cdf(-d2) - disc_s * norm_cdf(-d1)
    return np.asarray(price, dtype=np.float64)


def vega(
    spot: npt.ArrayLike,
    strike: npt.ArrayLike,
    r: npt.ArrayLike,
    sigma: npt.ArrayLike,
    t: npt.ArrayLike,
    *,
    q: npt.ArrayLike = 0.0,
) -> npt.NDArray[np.float64]:
    """Black-Scholes vega ``dV/dsigma = S e^{-qT} phi(d1) sqrt(T)``.

    Identical for calls and puts (RESULT: differentiate put-call parity in
    ``sigma`` — the forward leg does not depend on it). Canonical definition:
    ``pricing.greeks`` re-exports this function; the implied-volatility
    solvers import it from here so the models layer never depends on pricing.
    In the degenerate limit ``phi(+/-inf) = 0`` gives vega ``0`` off-ATM.
    """
    s = np.asarray(spot, dtype=np.float64)
    ta = np.asarray(t, dtype=np.float64)
    qa = np.asarray(q, dtype=np.float64)
    d1, _ = d1_d2(s, strike, r, sigma, ta, q=qa)
    return np.asarray(s * np.exp(-qa * ta) * norm_pdf(d1) * np.sqrt(ta), dtype=np.float64)


def put_call_parity_residual(
    call: npt.ArrayLike,
    put: npt.ArrayLike,
    spot: npt.ArrayLike,
    strike: npt.ArrayLike,
    r: npt.ArrayLike,
    t: npt.ArrayLike,
    *,
    q: npt.ArrayLike = 0.0,
) -> npt.NDArray[np.float64]:
    """Put-call parity residual ``C - P - (S e^{-qT} - K e^{-rT})``.

    Zero for arbitrage-consistent European prices (RESULT: a static
    replication argument, model-free). Useful as a validation probe for any
    pricer in this library.
    """
    c = np.asarray(call, dtype=np.float64)
    p = np.asarray(put, dtype=np.float64)
    s = np.asarray(spot, dtype=np.float64)
    k = np.asarray(strike, dtype=np.float64)
    ra = np.asarray(r, dtype=np.float64)
    ta = np.asarray(t, dtype=np.float64)
    qa = np.asarray(q, dtype=np.float64)
    return np.asarray(
        c - p - (s * np.exp(-qa * ta) - k * np.exp(-ra * ta)), dtype=np.float64
    )


def digital_price(
    spot: npt.ArrayLike,
    strike: npt.ArrayLike,
    r: npt.ArrayLike,
    sigma: npt.ArrayLike,
    t: npt.ArrayLike,
    *,
    option_type: OptionType = "call",
    q: npt.ArrayLike = 0.0,
) -> npt.NDArray[np.float64]:
    """Cash-or-nothing digital paying one unit of cash if in the money.

    ``call = e^{-rT} N(d2)``, ``put = e^{-rT} N(-d2)`` (RESULT: ``N(d2)`` is
    the risk-neutral exercise probability). CONVENTION: unit cash payout;
    scale by the notional externally. The degenerate limit is exact here too:
    ``N(+/-inf)`` yields the discounted indicator ``e^{-rT} 1{F >< K}``.
    """
    _require_option_type(option_type)
    ra = np.asarray(r, dtype=np.float64)
    ta = np.asarray(t, dtype=np.float64)
    _, d2 = d1_d2(spot, strike, ra, sigma, ta, q=q)
    disc = np.exp(-ra * ta)
    if option_type == "call":
        return np.asarray(disc * norm_cdf(d2), dtype=np.float64)
    return np.asarray(disc * norm_cdf(-d2), dtype=np.float64)


def butterfly_price(
    spot: npt.ArrayLike,
    strike: npt.ArrayLike,
    r: npt.ArrayLike,
    sigma: npt.ArrayLike,
    t: npt.ArrayLike,
    spread: float,
    *,
    q: npt.ArrayLike = 0.0,
) -> npt.NDArray[np.float64]:
    """Long call butterfly ``C(K - h) - 2 C(K) + C(K + h)`` with ``h = spread``.

    A discrete second difference in strike: up to ``O(h^2)`` it is
    proportional to the risk-neutral density at ``K`` (the
    Breeden-Litzenberger link), hence non-negative for arbitrage-free prices
    (RESULT). Guards: ``spread`` must be positive and strictly smaller than
    every strike, otherwise the lower leg ``K - h`` would have a non-positive
    strike and the structure is ill-posed.
    """
    if spread <= 0.0:
        raise ValueError("spread must be positive")
    k = np.asarray(strike, dtype=np.float64)
    if np.any(spread >= k):
        raise ValueError("spread must be strictly smaller than the strike")
    lower = black_scholes_price(spot, k - spread, r, sigma, t, option_type="call", q=q)
    middle = black_scholes_price(spot, k, r, sigma, t, option_type="call", q=q)
    upper = black_scholes_price(spot, k + spread, r, sigma, t, option_type="call", q=q)
    return np.asarray(lower - 2.0 * middle + upper, dtype=np.float64)
