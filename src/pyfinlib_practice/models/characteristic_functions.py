"""Risk-neutral characteristic functions of log-returns (O&G Chapters 5-6).

Each model exposes the characteristic function of the log-return
``Y_t = ln(S_t / S_0)`` under the risk-neutral measure,

    cf(u, t) = E[ exp(i u Y_t) ],

with the drift correction ``omega`` chosen so that the discounted,
dividend-adjusted price is a martingale:

    E[S_t] = S_0 e^{(r - q) t}   <=>   cf(-i, t) = e^{(r - q) t}.

The ``cf(-i, t)`` identity (RESULT: set ``u = -i`` in the definition) is a
model-free correctness probe and is exercised in the tests by evaluating
``cf`` at a complex argument — the implementations below deliberately accept
complex input for exactly that reason.

``cumulants(t)`` returns ``(c1, c2, c4)`` of ``Y_t``, consumed by the COS
truncation-range rule. Cumulant formulas follow Fang & Oosterlee (2008),
Table 2 — the standard reference; they are *not* transcribed from O&G pages
— and every ``c1``/``c2`` is cross-checked in the tests against numerical
derivatives of ``log cf`` at ``u = 0`` (an independent verification path).

Design: frozen dataclasses plus a structural ``Protocol``. The COS engine
takes plain callables, so anything exposing matching ``cf``/``cumulants``
attributes participates without inheritance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt
from scipy.special import gamma as gamma_fn  # type: ignore[import-untyped]


class CharacteristicModel(Protocol):
    """Structural interface consumed by ``pyfinlib_practice.numerical.cos``."""

    def cf(self, u: npt.ArrayLike, t: float) -> npt.NDArray[np.complex128]:
        """Characteristic function of ``ln(S_t / S_0)`` evaluated at ``u``."""
        ...

    def cumulants(self, t: float) -> tuple[float, float, float]:
        """First, second and fourth cumulants ``(c1, c2, c4)`` of the log-return."""
        ...


def _as_complex(u: npt.ArrayLike) -> npt.NDArray[np.complex128]:
    return np.asarray(u, dtype=np.complex128)


def _require_positive_t(t: float) -> None:
    if t <= 0.0:
        raise ValueError("t must be positive")


@dataclass(frozen=True)
class GBM:
    """Geometric Brownian motion: ``Y_t = a t + sigma W_t``, ``a = r - q - sigma^2/2``.

    ``cf(u, t) = exp(i u a t - sigma^2 u^2 t / 2)`` (RESULT: the Gaussian
    characteristic function). Cumulants: ``(a t, sigma^2 t, 0)`` — the
    Gaussian has zero excess kurtosis, so ``c4 = 0``.
    """

    r: float
    q: float
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma <= 0.0:
            raise ValueError("sigma must be positive")

    def _drift(self) -> float:
        return self.r - self.q - 0.5 * self.sigma**2

    def cf(self, u: npt.ArrayLike, t: float) -> npt.NDArray[np.complex128]:
        _require_positive_t(t)
        uc = _as_complex(u)
        exponent = 1j * uc * self._drift() * t - 0.5 * self.sigma**2 * uc**2 * t
        return np.asarray(np.exp(exponent), dtype=np.complex128)

    def cumulants(self, t: float) -> tuple[float, float, float]:
        _require_positive_t(t)
        return self._drift() * t, self.sigma**2 * t, 0.0


@dataclass(frozen=True)
class Merton:
    """Merton jump-diffusion: Brownian part plus compound-Poisson Gaussian jumps.

    With jump intensity ``lam``, log-jumps ``J ~ N(mu_j, sigma_j^2)`` and
    ``kappa = e^{mu_j + sigma_j^2/2} - 1``, the martingale drift is
    ``a = r - q - sigma^2/2 - lam kappa`` and (RESULT: Levy-Khintchine with a
    Gaussian jump measure)

        cf(u, t) = exp( t [ i u a - sigma^2 u^2 / 2
                            + lam (e^{i u mu_j - sigma_j^2 u^2 / 2} - 1) ] ).

    Cumulants (Fang & Oosterlee 2008, Table 2):
    ``c1 = (a + lam mu_j) t``, ``c2 = (sigma^2 + lam (mu_j^2 + sigma_j^2)) t``,
    ``c4 = lam (mu_j^4 + 6 mu_j^2 sigma_j^2 + 3 sigma_j^4) t``.
    """

    r: float
    q: float
    sigma: float
    lam: float
    mu_j: float
    sigma_j: float

    def __post_init__(self) -> None:
        if self.sigma < 0.0 or self.sigma_j < 0.0:
            raise ValueError("sigma and sigma_j must be non-negative")
        if self.lam < 0.0:
            raise ValueError("lam must be non-negative")

    def _kappa(self) -> float:
        return float(np.expm1(self.mu_j + 0.5 * self.sigma_j**2))

    def _drift(self) -> float:
        return self.r - self.q - 0.5 * self.sigma**2 - self.lam * self._kappa()

    def cf(self, u: npt.ArrayLike, t: float) -> npt.NDArray[np.complex128]:
        _require_positive_t(t)
        uc = _as_complex(u)
        jump_cf = np.exp(1j * uc * self.mu_j - 0.5 * self.sigma_j**2 * uc**2)
        exponent = t * (
            1j * uc * self._drift()
            - 0.5 * self.sigma**2 * uc**2
            + self.lam * (jump_cf - 1.0)
        )
        return np.asarray(np.exp(exponent), dtype=np.complex128)

    def cumulants(self, t: float) -> tuple[float, float, float]:
        _require_positive_t(t)
        c1 = (self._drift() + self.lam * self.mu_j) * t
        c2 = (self.sigma**2 + self.lam * (self.mu_j**2 + self.sigma_j**2)) * t
        c4 = self.lam * (self.mu_j**4 + 6.0 * self.mu_j**2 * self.sigma_j**2
                         + 3.0 * self.sigma_j**4) * t
        return c1, c2, c4


@dataclass(frozen=True)
class VarianceGamma:
    """Variance Gamma: Brownian motion with drift, time-changed by a Gamma clock.

    Parameters ``(sigma, theta, beta)`` are the O&G convention: ``beta`` is
    the variance rate of the Gamma subordinator. The martingale correction

        omega = ln(1 - theta beta - sigma^2 beta / 2) / beta

    requires ``theta beta + sigma^2 beta / 2 < 1`` (validated), and (RESULT)

        cf(u, t) = e^{i u (r - q + omega) t}
                   (1 - i u theta beta + sigma^2 beta u^2 / 2)^{-t / beta}.

    Cumulants (Fang & Oosterlee 2008, Table 2):
    ``c1 = (r - q + omega + theta) t``, ``c2 = (sigma^2 + beta theta^2) t``,
    ``c4 = 3 (sigma^4 beta + 2 theta^4 beta^3 + 4 sigma^2 theta^2 beta^2) t``.
    """

    r: float
    q: float
    sigma: float
    theta: float
    beta: float

    def __post_init__(self) -> None:
        if self.sigma < 0.0:
            raise ValueError("sigma must be non-negative")
        if self.beta <= 0.0:
            raise ValueError("beta must be positive")
        if self._log_argument() <= 0.0:
            raise ValueError(
                "VG martingale correction undefined: require "
                "theta*beta + sigma^2*beta/2 < 1"
            )

    def _log_argument(self) -> float:
        return 1.0 - self.theta * self.beta - 0.5 * self.sigma**2 * self.beta

    def _omega(self) -> float:
        return float(np.log(self._log_argument()) / self.beta)

    def cf(self, u: npt.ArrayLike, t: float) -> npt.NDArray[np.complex128]:
        _require_positive_t(t)
        uc = _as_complex(u)
        base = 1.0 - 1j * uc * self.theta * self.beta + 0.5 * self.sigma**2 * self.beta * uc**2
        drift_part = np.exp(1j * uc * (self.r - self.q + self._omega()) * t)
        return np.asarray(drift_part * np.power(base, -t / self.beta), dtype=np.complex128)

    def cumulants(self, t: float) -> tuple[float, float, float]:
        _require_positive_t(t)
        c1 = (self.r - self.q + self._omega() + self.theta) * t
        c2 = (self.sigma**2 + self.beta * self.theta**2) * t
        c4 = 3.0 * (self.sigma**4 * self.beta
                    + 2.0 * self.theta**4 * self.beta**3
                    + 4.0 * self.sigma**2 * self.theta**2 * self.beta**2) * t
        return c1, c2, c4


@dataclass(frozen=True)
class CGMY:
    """CGMY tempered-stable process, optionally with a Brownian component.

    Levy exponent (RESULT, Carr-Geman-Madan-Yor 2002):

        psi(u) = C Gamma(-Y) [ (M - i u)^Y - M^Y + (G + i u)^Y - G^Y ],

    valid for ``Y in (0, 2), Y != 1`` (``Gamma(-Y)`` has poles at 0 and 1;
    O&G exclude ``Y = 1`` as well). ``sigma > 0`` adds an independent
    Brownian component (the extended "CGMY-B" model of O&G Chapter 5);
    ``sigma = 0`` is the pure-jump CGMY. The martingale correction is

        omega = -sigma^2 / 2 - psi(-i),

    which requires ``M > 1`` for ``(M - 1)^Y`` to be a positive real power
    (equivalently: ``E[e^{Y_t}]`` finite), giving

        cf(u, t) = exp( i u (r - q + omega) t - sigma^2 u^2 t / 2 + t psi(u) ).

    Cumulants (Fang & Oosterlee 2008, Table 2; Brownian variance added to c2):
    ``c1 = (r - q + omega) t + C t Gamma(1-Y) (M^{Y-1} - G^{Y-1})``,
    ``c2 = sigma^2 t + C t Gamma(2-Y) (M^{Y-2} + G^{Y-2})``,
    ``c4 = C t Gamma(4-Y) (M^{Y-4} + G^{Y-4})``.
    """

    r: float
    q: float
    C: float
    G: float
    M: float
    Y: float
    sigma: float = 0.0

    def __post_init__(self) -> None:
        if self.C <= 0.0 or self.G <= 0.0:
            raise ValueError("C and G must be positive")
        if self.M <= 1.0:
            raise ValueError("M must exceed 1 for a finite martingale correction")
        if not 0.0 < self.Y < 2.0 or self.Y == 1.0:
            raise ValueError("Y must lie in (0, 2) and differ from 1 (Gamma(-Y) pole)")
        if self.sigma < 0.0:
            raise ValueError("sigma must be non-negative")

    def _exponent(self, uc: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
        g = float(gamma_fn(-self.Y))
        # The M^Y / G^Y reference terms are routed through the *same* complex
        # power routine as the u-dependent terms (via + 0*uc), so psi(0)
        # cancels exactly to 0 and cf(0) == 1 bitwise. Mixing numpy's complex
        # and real pow leaves a ~1e-14 residue in the exponent.
        m_ref = self.M + 0.0 * uc
        g_ref = self.G + 0.0 * uc
        value = self.C * g * (
            (self.M - 1j * uc) ** self.Y
            - m_ref**self.Y
            + (self.G + 1j * uc) ** self.Y
            - g_ref**self.Y
        )
        return np.asarray(value, dtype=np.complex128)

    def _omega(self) -> float:
        g = float(gamma_fn(-self.Y))
        psi_minus_i = float(
            self.C * g * (
                (self.M - 1.0) ** self.Y
                - self.M**self.Y
                + (self.G + 1.0) ** self.Y
                - self.G**self.Y
            )
        )
        return -0.5 * self.sigma**2 - psi_minus_i

    def cf(self, u: npt.ArrayLike, t: float) -> npt.NDArray[np.complex128]:
        _require_positive_t(t)
        uc = _as_complex(u)
        exponent = (
            1j * uc * (self.r - self.q + self._omega()) * t
            - 0.5 * self.sigma**2 * uc**2 * t
            + t * self._exponent(uc)
        )
        return np.asarray(np.exp(exponent), dtype=np.complex128)

    def cumulants(self, t: float) -> tuple[float, float, float]:
        _require_positive_t(t)
        mu = self.r - self.q + self._omega()
        c1 = mu * t + self.C * t * float(gamma_fn(1.0 - self.Y)) * (
            self.M ** (self.Y - 1.0) - self.G ** (self.Y - 1.0)
        )
        c2 = self.sigma**2 * t + self.C * t * float(gamma_fn(2.0 - self.Y)) * (
            self.M ** (self.Y - 2.0) + self.G ** (self.Y - 2.0)
        )
        c4 = self.C * t * float(gamma_fn(4.0 - self.Y)) * (
            self.M ** (self.Y - 4.0) + self.G ** (self.Y - 4.0)
        )
        return c1, c2, c4
