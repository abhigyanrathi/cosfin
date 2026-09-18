"""The COS method: Fourier-cosine series option pricing (O&G Chapter 6).

Recipe (Fang & Oosterlee 2008; O&G Ch6 presents the same method). Work in
``y = ln(S_T / K)`` with ``x = ln(S_0 / K)``. On a truncation interval
``[a, b]`` the risk-neutral density has the cosine expansion

    f(y) ~ sum'_{k=0}^{N-1} F_k cos(k pi (y - a) / (b - a)),
    F_k  = (2 / (b - a)) Re{ phi(k pi / (b - a)) e^{i k pi (x - a)/(b - a)} },

where ``phi`` is the characteristic function of the *centred* log-return
``ln(S_T / S_0)`` and the ``e^{iux}`` factor shifts it to ``y`` (RESULT:
replace the Fourier-cosine coefficient integral over ``[a, b]`` with the
integral over R — the truncation error — and recognise the characteristic
function). Substituting into the discounted expectation and swapping sum and
integral yields

    V = e^{-rT} sum'_{k} Re{ phi(omega_k) e^{i omega_k (x - a)} } U_k,
    omega_k = k pi / (b - a),

with payoff coefficients ``U_k = (2/(b-a)) int_a^b v(y, T) cos(omega_k
(y - a)) dy`` available in closed form via the ``chi``/``psi`` integrals
below. ``sum'`` halves the ``k = 0`` term (standard cosine-series weight).

Error structure (RESULT, Fang-Oosterlee): for densities with exponentially
decaying cf the series error decays *exponentially* in ``N``; the interval
truncation error is controlled by the cumulant-based range rule of
:func:`truncation_range`. Machine-precision agreement with Black-Scholes at
``N = 256`` is the canonical health check and is enforced in the tests.

Vectorisation: multiple strikes are priced in one call by broadcasting —
per-strike ``[a, b] = [x + a0, x + b0]`` shift with the log-moneyness, so
``omega``, the cf evaluations and the ``chi``/``psi`` terms are 2-D arrays
of shape ``(n_strikes, N)``. This is exact per strike (no shared-interval
approximation) at the cost of evaluating the cf on the 2-D grid.

CONVENTION: digital prices are per unit cash; scale by notional externally.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import numpy.typing as npt

CharacteristicFn = Callable[[npt.ArrayLike, float], npt.NDArray[np.complex128]]
CumulantsFn = Callable[[float], tuple[float, float, float]]
# Local alias, duplicated on purpose: this module stays import-free so it
# can be lifted standalone; Literal aliases are structural, so nothing can
# drift out of sync.
OptionType = Literal["call", "put"]


def truncation_range(
    cumulants: CumulantsFn, t: float, l_trunc: float = 10.0
) -> tuple[float, float]:
    """Cumulant-based truncation interval ``[c1 -/+ L sqrt(c2 + sqrt(c4))]``.

    CONVENTION (Fang-Oosterlee): the width is ``L`` kurtosis-inflated
    standard deviations around the mean; ``L = 10`` suffices for the models
    in this library (heavier-tailed parameterisations may need ``L = 12``,
    exposed as ``l_trunc`` on the pricers). The neglected tail mass — hence
    the truncation error — decays roughly like a Gaussian tail in ``L``.
    """
    if l_trunc <= 0.0:
        raise ValueError("l_trunc must be positive")
    c1, c2, c4 = cumulants(t)
    if c2 <= 0.0:
        raise ValueError("second cumulant must be positive to build a truncation range")
    if c4 < 0.0:
        raise ValueError("fourth cumulant must be non-negative")
    half_width = l_trunc * float(np.sqrt(c2 + np.sqrt(c4)))
    return c1 - half_width, c1 + half_width


def _chi(
    omega: npt.NDArray[np.float64],
    a: npt.NDArray[np.float64],
    c: npt.NDArray[np.float64],
    d: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """``int_c^d e^y cos(omega (y - a)) dy`` in closed form.

    RESULT: two integrations by parts give
    ``[cos(w(d-a)) e^d - cos(w(c-a)) e^c + w sin(w(d-a)) e^d
       - w sin(w(c-a)) e^c] / (1 + w^2)``.
    """
    return np.asarray(
        (
            np.cos(omega * (d - a)) * np.exp(d)
            - np.cos(omega * (c - a)) * np.exp(c)
            + omega * (np.sin(omega * (d - a)) * np.exp(d) - np.sin(omega * (c - a)) * np.exp(c))
        )
        / (1.0 + omega**2),
        dtype=np.float64,
    )


def _psi(
    omega: npt.NDArray[np.float64],
    a: npt.NDArray[np.float64],
    c: npt.NDArray[np.float64],
    d: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """``int_c^d cos(omega (y - a)) dy``; the ``k = 0`` column equals ``d - c``."""
    safe_omega = np.where(omega == 0.0, 1.0, omega)
    values = (np.sin(omega * (d - a)) - np.sin(omega * (c - a))) / safe_omega
    return np.asarray(np.where(omega == 0.0, d - c, values), dtype=np.float64)


def _cos_setup(
    cf: CharacteristicFn,
    cumulants: CumulantsFn,
    spot: float,
    strike: npt.ArrayLike,
    t: float,
    n: int,
    l_trunc: float,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Shared machinery: strikes, per-strike interval, frequencies, ``F_k`` terms."""
    if n < 2:
        raise ValueError("n must be at least 2")
    if spot <= 0.0:
        raise ValueError("spot must be positive")
    if t <= 0.0:
        raise ValueError("t must be positive")
    strikes = np.atleast_1d(np.asarray(strike, dtype=np.float64))
    if strikes.ndim != 1:
        raise ValueError("strike must be a scalar or a 1-D array")
    if np.any(strikes <= 0.0):
        raise ValueError("strikes must be positive")

    a0, b0 = truncation_range(cumulants, t, l_trunc)
    x = np.log(spot / strikes)  # (m,)
    a = (x + a0)[:, None]  # (m, 1)
    b = (x + b0)[:, None]  # (m, 1)

    k = np.arange(n, dtype=np.float64)[None, :]  # (1, n)
    omega = k * np.pi / (b - a)  # (m, n)
    phi = np.asarray(cf(omega, t), dtype=np.complex128)
    if phi.shape != omega.shape:
        raise ValueError("characteristic function must preserve the input shape")
    # Re{ phi(w) e^{i w x} e^{-i w a} } with x - a = -a0 collapsing the shift.
    fk = np.real(phi * np.exp(1j * omega * (x[:, None] - a)))

    weights = np.ones(n, dtype=np.float64)
    weights[0] = 0.5
    return strikes, a, b, omega, np.asarray(fk, dtype=np.float64), weights


def cos_european_price(
    cf: CharacteristicFn,
    cumulants: CumulantsFn,
    spot: float,
    strike: npt.ArrayLike,
    r: float,
    t: float,
    n: int = 256,
    *,
    l_trunc: float = 10.0,
    option_type: OptionType = "call",
) -> npt.NDArray[np.float64]:
    """European call/put by the COS method; vectorised over strikes.

    Payoff coefficients (RESULT, from the ``chi``/``psi`` primitives with
    ``v(y, T) = K (e^y - 1)^+`` resp. ``K (1 - e^y)^+``):

        call: U_k = (2/(b-a)) K [ chi_k(0, b) - psi_k(0, b) ],
        put:  U_k = (2/(b-a)) K [ psi_k(a, 0) - chi_k(a, 0) ].

    Returns a 1-D array of prices matching ``np.atleast_1d(strike)``.
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    strikes, a, b, omega, fk, weights = _cos_setup(cf, cumulants, spot, strike, t, n, l_trunc)
    zero = np.zeros_like(a)
    if option_type == "call":
        payoff_coeff = _chi(omega, a, zero, b) - _psi(omega, a, zero, b)
    else:
        payoff_coeff = _psi(omega, a, a, zero) - _chi(omega, a, a, zero)
    uk = (2.0 / (b - a)) * strikes[:, None] * payoff_coeff
    prices = np.exp(-r * t) * np.sum(weights * fk * uk, axis=1)
    return np.asarray(prices, dtype=np.float64)


def cos_digital_price(
    cf: CharacteristicFn,
    cumulants: CumulantsFn,
    spot: float,
    strike: npt.ArrayLike,
    r: float,
    t: float,
    n: int = 256,
    *,
    l_trunc: float = 10.0,
    option_type: OptionType = "call",
) -> npt.NDArray[np.float64]:
    """Cash-or-nothing digital (unit cash) by the COS method.

    The payoff indicator ``1{y > 0}`` (call) / ``1{y < 0}`` (put) has cosine
    coefficients ``U_k = (2/(b-a)) psi_k(0, b)`` resp. ``(2/(b-a))
    psi_k(a, 0)`` — no ``chi`` term, since no ``e^y`` factor appears. Even
    though the payoff is discontinuous, convergence in ``N`` remains
    exponential: the series error is driven by the decay of the *density*
    coefficients ``F_k``, not the payoff's (RESULT, Fang-Oosterlee).
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    _, a, b, omega, fk, weights = _cos_setup(cf, cumulants, spot, strike, t, n, l_trunc)
    zero = np.zeros_like(a)
    payoff_coeff = (
        _psi(omega, a, zero, b) if option_type == "call" else _psi(omega, a, a, zero)
    )
    uk = (2.0 / (b - a)) * payoff_coeff
    prices = np.exp(-r * t) * np.sum(weights * fk * uk, axis=1)
    return np.asarray(prices, dtype=np.float64)


def cos_density(
    cf: CharacteristicFn,
    y: npt.ArrayLike,
    a: float,
    b: float,
    n: int,
    t: float,
) -> npt.NDArray[np.float64]:
    """Recover the density of the log-return on ``[a, b]`` from its cf.

    ``f(y) ~ (2/(b-a)) sum'_k Re{ phi(omega_k) e^{-i omega_k a} }
    cos(omega_k (y - a))`` (RESULT: the series of the module docstring with
    ``x = 0``). Only valid for ``y`` inside ``[a, b]`` — the cosine series is
    periodic, so evaluations outside the interval alias back into it and are
    meaningless.
    """
    if b <= a:
        raise ValueError("require b > a")
    if n < 2:
        raise ValueError("n must be at least 2")
    if t <= 0.0:
        raise ValueError("t must be positive")
    y_arr = np.atleast_1d(np.asarray(y, dtype=np.float64))
    if np.any(y_arr < a) or np.any(y_arr > b):
        raise ValueError("all evaluation points must lie inside [a, b]")

    k = np.arange(n, dtype=np.float64)
    omega = k * np.pi / (b - a)  # (n,)
    coefficients = (2.0 / (b - a)) * np.real(
        np.asarray(cf(omega, t), dtype=np.complex128) * np.exp(-1j * omega * a)
    )
    weights = np.ones(n, dtype=np.float64)
    weights[0] = 0.5
    density = np.cos((y_arr[:, None] - a) * omega[None, :]) @ (weights * coefficients)
    return np.asarray(density, dtype=np.float64)
