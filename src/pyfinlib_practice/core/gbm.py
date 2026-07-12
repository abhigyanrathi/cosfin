"""Geometric Brownian motion path simulation (O&G Chapter 2).

GBM solves ``dS_t = mu S_t dt + sigma S_t dW_t``. Applying Ito's lemma to
``ln S_t`` (RESULT — the Ito correction is the ``-sigma^2/2`` term):

    d ln S_t = (mu - sigma^2 / 2) dt + sigma dW_t,

so the strong solution is

    S_t = S_0 exp((mu - sigma^2 / 2) t + sigma W_t),

log-normally distributed with ``E[S_t] = S_0 e^{mu t}`` and
``Var[S_t] = S_0^2 e^{2 mu t} (e^{sigma^2 t} - 1)`` (RESULT).

Two schemes are provided:

* ``"exact"`` — steps the closed-form solution; exact in distribution at grid
  points, so it is the correct default.
* ``"euler"`` — Euler-Maruyama on the SDE itself,
  ``S_{i+1} = S_i (1 + mu dt + sigma dW)``; strong order 1/2, weak order 1.
  Kept for studying discretisation error (it can also go negative — a real
  defect of the scheme, deliberately not patched over).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

Scheme = Literal["exact", "euler"]


def simulate_gbm_paths(
    s0: float,
    mu: float,
    sigma: float,
    t_end: float,
    n_paths: int,
    n_steps: int,
    *,
    scheme: Scheme = "exact",
    rng: np.random.Generator | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Simulate GBM paths on a uniform grid.

    Returns ``(times, paths)`` with shapes ``(n_steps + 1,)`` and
    ``(n_paths, n_steps + 1)``; ``paths[:, 0] == s0``.

    MODELLING CHOICE: under the risk-neutral measure pass ``mu = r - q``; this
    function is measure-agnostic and simply simulates the SDE it is given.
    """
    if s0 <= 0.0:
        raise ValueError("s0 must be positive")
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")
    if t_end <= 0.0 or n_paths <= 0 or n_steps <= 0:
        raise ValueError("t_end, n_paths and n_steps must be positive")
    generator = rng if rng is not None else np.random.default_rng()

    dt = t_end / n_steps
    times = np.linspace(0.0, t_end, n_steps + 1)
    shocks = generator.standard_normal((n_paths, n_steps)) * np.sqrt(dt)

    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = s0
    if scheme == "exact":
        log_increments = (mu - 0.5 * sigma**2) * dt + sigma * shocks
        paths[:, 1:] = s0 * np.exp(np.cumsum(log_increments, axis=1))
    elif scheme == "euler":
        growth = 1.0 + mu * dt + sigma * shocks
        paths[:, 1:] = s0 * np.cumprod(growth, axis=1)
    else:  # pragma: no cover - Literal narrows this away for typed callers
        raise ValueError(f"unknown scheme {scheme!r}")
    return times, paths


def gbm_terminal_moments(s0: float, mu: float, sigma: float, t: float) -> tuple[float, float]:
    """Closed-form mean and variance of ``S_t`` under GBM (RESULT).

    ``E[S_t] = S_0 e^{mu t}``,
    ``Var[S_t] = S_0^2 e^{2 mu t} (e^{sigma^2 t} - 1)``.
    """
    mean = s0 * float(np.exp(mu * t))
    variance = s0**2 * float(np.exp(2.0 * mu * t)) * float(np.expm1(sigma**2 * t))
    return mean, variance
