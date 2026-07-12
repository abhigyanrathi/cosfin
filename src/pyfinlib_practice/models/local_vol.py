"""Implied volatility and Breeden-Litzenberger density tools (O&G Chapter 4).

Module name note: kept as ``local_vol`` for API parity with the practice
repository. Contents cover the mentor-confirmed Chapter 4 scope only —
Dupire local volatility, SABR and the arbitrage-free-surface machinery of
Section 4.3 are explicitly out of scope.

* **Section 4.1 — implied volatility.** For a market price strictly inside
  the no-arbitrage bounds, the Black-Scholes price is a strictly increasing,
  continuous function of ``sigma`` on ``(0, inf)`` (vega > 0), so the implied
  volatility exists and is unique (RESULT). Three root-finders are provided:
  Newton-Raphson (quadratic convergence, needs healthy vega), secant
  (derivative-free), and a bracketed Newton/bisection hybrid whose
  convergence is guaranteed by the maintained sign-change bracket.

* **Section 4.2 — Breeden-Litzenberger.** From
  ``C(K) = e^{-rT} int_K^inf (s - K) f(s) ds`` differentiate twice in ``K``:
  ``dC/dK = -e^{-rT} int_K^inf f(s) ds`` and ``d^2C/dK^2 = e^{-rT} f(K)``,
  so the risk-neutral density is ``f(K) = e^{rT} d^2C/dK^2`` (RESULT). Any
  European payoff can then be valued by quadrature against ``f``.

CONVERGENCE CONVENTION (important): all iterative solvers stop on the
*volatility step* ``|sigma_{n+1} - sigma_n| < xtol``, never on the price
residual. For deep in/out-of-the-money options vega is tiny, so a small
price residual is compatible with a ``sigma`` that is still far from the
root; a volatility-step criterion does not have that failure mode.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import numpy.typing as npt

from pyfinlib_practice.models.black_scholes import OptionType, black_scholes_price, vega

Method = Literal["newton", "secant", "bracketed"]

_MIN_VEGA = 1e-12
_SIGMA_FLOOR = 1e-8


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _no_arbitrage_bounds(
    spot: float, strike: float, r: float, t: float, option_type: OptionType, q: float
) -> tuple[float, float]:
    """No-arbitrage price bounds for a European option (RESULT).

    Call: ``max(S e^{-qT} - K e^{-rT}, 0) <= C <= S e^{-qT}``;
    put:  ``max(K e^{-rT} - S e^{-qT}, 0) <= P <= K e^{-rT}``.
    These are the ``sigma -> 0`` and ``sigma -> inf`` limits of the
    Black-Scholes price, which is why a price strictly inside them pins a
    unique implied volatility.
    """
    disc_s = spot * float(np.exp(-q * t))
    disc_k = strike * float(np.exp(-r * t))
    if option_type == "call":
        return max(disc_s - disc_k, 0.0), disc_s
    return max(disc_k - disc_s, 0.0), disc_k


def _validate_solver_inputs(
    price: float, spot: float, strike: float, t: float, option_type: OptionType
) -> None:
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if spot <= 0.0 or strike <= 0.0:
        raise ValueError("spot and strike must be positive")
    if t <= 0.0:
        raise ValueError("t must be positive: implied volatility is undefined at expiry")
    if not np.isfinite(price):
        raise ValueError("price must be finite")


def _validate_target_price(
    price: float,
    spot: float,
    strike: float,
    r: float,
    t: float,
    option_type: OptionType,
    q: float,
) -> None:
    lower, upper = _no_arbitrage_bounds(spot, strike, r, t, option_type, q)
    if not (lower < price < upper):
        raise ValueError(
            f"price {price:.6g} lies outside the open no-arbitrage interval "
            f"({lower:.6g}, {upper:.6g}); no positive implied volatility exists"
        )


def _make_price_error(
    target: float,
    spot: float,
    strike: float,
    r: float,
    t: float,
    option_type: OptionType,
    q: float,
) -> Callable[[float], float]:
    """Return the scalar objective ``f(sigma) = BS(sigma) - target``.

    Strictly increasing in ``sigma`` (vega > 0), so ``f`` has exactly one
    root for an admissible target — the property every solver below relies
    on.
    """

    def price_error(sigma: float) -> float:
        model = black_scholes_price(spot, strike, r, sigma, t, option_type=option_type, q=q)
        return float(model) - target

    return price_error


def _brenner_subrahmanyam_guess(price: float, spot: float, t: float) -> float:
    """ATM implied-volatility approximation used as a starting point.

    From ``C_ATM ~ S sigma sqrt(T / (2 pi))`` invert to
    ``sigma ~ sqrt(2 pi / T) * price / S`` (RESULT, Brenner-Subrahmanyam
    1988). Clipped to ``[1e-3, 5]`` so pathological quotes still yield a
    sane seed.
    """
    guess = float(np.sqrt(2.0 * np.pi / t) * price / spot)
    return float(np.clip(guess, 1e-3, 5.0))


# --------------------------------------------------------------------------- #
# Implied volatility solvers
# --------------------------------------------------------------------------- #
def implied_volatility_newton(
    price: float,
    spot: float,
    strike: float,
    r: float,
    t: float,
    *,
    option_type: OptionType = "call",
    q: float = 0.0,
    initial_guess: float | None = None,
    xtol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """Implied volatility by Newton-Raphson with the analytic vega derivative.

    Iteration: ``sigma <- sigma - (BS(sigma) - price) / vega(sigma)``;
    convergence is quadratic near the root (RESULT: standard Newton theory,
    f is smooth with f' = vega > 0). Fails fast with ``RuntimeError`` when
    vega drops below ``1e-12`` — in that regime the Newton step is
    numerically meaningless and :func:`implied_volatility_bracketed` should
    be used instead. Negative iterates are damped back by halving
    (MODELLING CHOICE: preserves positivity without abandoning the run).
    """
    _validate_solver_inputs(price, spot, strike, t, option_type)
    _validate_target_price(price, spot, strike, r, t, option_type, q)
    f = _make_price_error(price, spot, strike, r, t, option_type, q)

    sigma = initial_guess if initial_guess is not None else _brenner_subrahmanyam_guess(
        price, spot, t
    )
    sigma = max(sigma, _SIGMA_FLOOR)

    for _ in range(max_iter):
        v = float(vega(spot, strike, r, sigma, t, q=q))
        if v < _MIN_VEGA:
            raise RuntimeError(
                "vega below 1e-12: the Newton step is unreliable here; "
                "use implied_volatility_bracketed"
            )
        sigma_next = sigma - f(sigma) / v
        if sigma_next <= 0.0:
            sigma_next = 0.5 * sigma
        if abs(sigma_next - sigma) < xtol:
            return sigma_next
        sigma = sigma_next
    raise RuntimeError(f"Newton-Raphson did not converge within {max_iter} iterations")


def implied_volatility_secant(
    price: float,
    spot: float,
    strike: float,
    r: float,
    t: float,
    *,
    option_type: OptionType = "call",
    q: float = 0.0,
    initial_guess: float | None = None,
    xtol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """Implied volatility by the secant method (no analytic derivative).

    The secant slope ``(f1 - f0) / (s1 - s0)`` replaces vega; convergence
    order is the golden ratio ~1.618 (RESULT). Two seeds are taken as the
    Brenner-Subrahmanyam guess and a 20%-perturbed companion. Kept mainly as
    the Section 4.1 comparison method — production callers should prefer the
    bracketed solver.
    """
    _validate_solver_inputs(price, spot, strike, t, option_type)
    _validate_target_price(price, spot, strike, r, t, option_type, q)
    f = _make_price_error(price, spot, strike, r, t, option_type, q)

    s_prev = initial_guess if initial_guess is not None else _brenner_subrahmanyam_guess(
        price, spot, t
    )
    s_prev = max(s_prev, _SIGMA_FLOOR)
    s_curr = 1.2 * s_prev + 0.01
    f_prev = f(s_prev)

    for _ in range(max_iter):
        f_curr = f(s_curr)
        slope_scale = f_curr - f_prev
        if abs(slope_scale) < 1e-16:
            raise RuntimeError(
                "secant slope vanished (price flat in sigma); "
                "use implied_volatility_bracketed"
            )
        s_next = s_curr - f_curr * (s_curr - s_prev) / slope_scale
        if s_next <= 0.0:
            s_next = 0.5 * s_curr
        if abs(s_next - s_curr) < xtol:
            return s_next
        s_prev, f_prev = s_curr, f_curr
        s_curr = s_next
    raise RuntimeError(f"secant method did not converge within {max_iter} iterations")


def implied_volatility_bracketed(
    price: float,
    spot: float,
    strike: float,
    r: float,
    t: float,
    *,
    option_type: OptionType = "call",
    q: float = 0.0,
    xtol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """Guaranteed-convergence implied volatility: Newton inside a bracket.

    Algorithm: (1) establish ``[lo, hi]`` with ``f(lo) < 0 < f(hi)`` — the
    admissible price guarantees ``f(sigma_floor) < 0``, and ``hi`` is doubled
    from 1.0 until ``f(hi) > 0`` (monotonicity guarantees termination);
    (2) iterate a Newton step, rejecting it for a bisection step whenever it
    would leave the bracket or vega is below ``1e-12``; (3) shrink the
    bracket by the sign of ``f`` each iteration. Bisection fallback bounds
    the bracket width by halving, so convergence to ``xtol`` is guaranteed
    while retaining Newton's quadratic speed near the root (MODELLING
    CHOICE: this is the "combined" solver of Section 4.1 and the default of
    :func:`implied_volatility`).
    """
    _validate_solver_inputs(price, spot, strike, t, option_type)
    _validate_target_price(price, spot, strike, r, t, option_type, q)
    f = _make_price_error(price, spot, strike, r, t, option_type, q)

    lo, hi = _SIGMA_FLOOR, 1.0
    expansions = 0
    while f(hi) < 0.0:
        hi *= 2.0
        expansions += 1
        if expansions > 60:  # pragma: no cover
            # Defensive only: for any price admitted by the bounds check,
            # BS(sigma) -> upper bound > price as sigma -> inf, so f(hi)
            # turns positive long before 2^60. Unreachable by construction.
            raise RuntimeError("failed to bracket the implied volatility")

    sigma = float(np.clip(_brenner_subrahmanyam_guess(price, spot, t), lo, hi))
    if not lo < sigma < hi:
        sigma = 0.5 * (lo + hi)

    for _ in range(max_iter):
        error = f(sigma)
        if error > 0.0:
            hi = sigma
        else:
            lo = sigma
        if hi - lo < xtol:
            return 0.5 * (lo + hi)

        v = float(vega(spot, strike, r, sigma, t, q=q))
        candidate = sigma - error / v if v > _MIN_VEGA else 0.5 * (lo + hi)
        if not lo < candidate < hi:
            candidate = 0.5 * (lo + hi)
        if abs(candidate - sigma) < xtol:
            return candidate
        sigma = candidate
    raise RuntimeError(f"bracketed solver did not converge within {max_iter} iterations")


def implied_volatility(
    price: float,
    spot: float,
    strike: float,
    r: float,
    t: float,
    *,
    option_type: OptionType = "call",
    q: float = 0.0,
    method: Method = "bracketed",
    xtol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """Implied volatility dispatcher.

    CONVENTION: the default method is ``"bracketed"`` — it is the only one
    of the three with guaranteed convergence, and it costs at most a handful
    of extra function evaluations near ATM. ``"newton"`` and ``"secant"``
    remain available for the Section 4.1 method comparison.
    """
    if method == "newton":
        return implied_volatility_newton(
            price, spot, strike, r, t, option_type=option_type, q=q, xtol=xtol, max_iter=max_iter
        )
    if method == "secant":
        return implied_volatility_secant(
            price, spot, strike, r, t, option_type=option_type, q=q, xtol=xtol, max_iter=max_iter
        )
    if method == "bracketed":
        return implied_volatility_bracketed(
            price, spot, strike, r, t, option_type=option_type, q=q, xtol=xtol, max_iter=max_iter
        )
    raise ValueError(f"unknown method {method!r}; expected 'newton', 'secant' or 'bracketed'")


# --------------------------------------------------------------------------- #
# Breeden-Litzenberger
# --------------------------------------------------------------------------- #
def risk_neutral_density(
    strikes: npt.ArrayLike,
    call_prices: npt.ArrayLike,
    r: float,
    t: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Risk-neutral density ``f(K) = e^{rT} d^2C/dK^2`` from call quotes.

    The second strike-derivative is estimated with the non-uniform central
    second difference

        C'' ~ 2 [ (C_{i+1} - C_i)/h_r - (C_i - C_{i-1})/h_l ] / (h_l + h_r),

    which is exact for locally quadratic ``C`` and ``O(h^2)``-accurate on a
    uniform grid (RESULT: Taylor-expand ``C(K +/- h)`` and eliminate the
    first-derivative terms). Returns ``(interior_strikes, density)`` — the
    two boundary strikes have no two-sided neighbour and are dropped rather
    than filled with a lower-order estimate.
    """
    k = np.asarray(strikes, dtype=np.float64)
    c = np.asarray(call_prices, dtype=np.float64)
    if k.ndim != 1 or c.shape != k.shape:
        raise ValueError("strikes and call_prices must be 1-D arrays of equal length")
    if k.size < 3:
        raise ValueError("need at least three strikes for a second difference")
    dk = np.diff(k)
    if np.any(dk <= 0.0):
        raise ValueError("strikes must be strictly increasing")

    h_l, h_r = dk[:-1], dk[1:]
    second = 2.0 * ((c[2:] - c[1:-1]) / h_r - (c[1:-1] - c[:-2]) / h_l) / (h_l + h_r)
    density = np.asarray(np.exp(r * t) * second, dtype=np.float64)
    return k[1:-1], density


def price_from_density(
    strikes: npt.ArrayLike,
    density: npt.ArrayLike,
    r: float,
    t: float,
    strike: float,
    *,
    option_type: OptionType = "call",
) -> float:
    """Value a European option by quadrature against a risk-neutral density.

    ``V = e^{-rT} int payoff(s) f(s) ds`` evaluated with the trapezoid rule
    on the supplied grid (RESULT: risk-neutral valuation). The grid must
    cover the effective support of ``f``; mass beyond the grid edges is the
    (unreported) truncation error, and the kink of the payoff at ``strike``
    contributes an ``O(h^2)`` local quadrature error.
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    k = np.asarray(strikes, dtype=np.float64)
    f = np.asarray(density, dtype=np.float64)
    if k.ndim != 1 or f.shape != k.shape:
        raise ValueError("strikes and density must be 1-D arrays of equal length")
    if k.size < 2:
        raise ValueError("need at least two grid points for quadrature")
    if np.any(np.diff(k) <= 0.0):
        raise ValueError("strikes must be strictly increasing")

    payoff = np.maximum(k - strike, 0.0) if option_type == "call" else np.maximum(strike - k, 0.0)
    return float(np.exp(-r * t) * np.trapezoid(payoff * f, k))
