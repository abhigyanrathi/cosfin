"""Merton jump-diffusion path simulation (O&G Chapter 5).

Under the risk-neutral measure the Merton model evolves

    dS_t / S_{t-} = (r - q - lam * kappa) dt + sigma dW_t + (e^J - 1) dN_t,

where ``N`` is a Poisson process with intensity ``lam``, jump sizes
``J ~ N(mu_j, sigma_j^2)`` are i.i.d., and

    kappa = E[e^J] - 1 = exp(mu_j + sigma_j^2 / 2) - 1

is the mean relative jump size. Subtracting the compensator ``lam * kappa``
from the drift makes the discounted, dividend-adjusted price a martingale
(RESULT: E[S_t] = S_0 e^{(r - q) t}).

The log-price increment over a step of length ``dt`` is

    (r - q - sigma^2/2 - lam*kappa) dt + sigma sqrt(dt) Z + sum_{i=1}^{N} J_i,

and conditional on ``N = n`` jumps the jump sum is ``N(n mu_j, n sigma_j^2)``
(RESULT: a sum of i.i.d. Gaussians is Gaussian). Grid marginals are therefore
simulated *exactly*: draw ``N ~ Poisson(lam dt)`` per step, then one Gaussian
for the whole jump sum — no Euler bias, only Monte Carlo noise.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def simulate_merton_paths(
    s0: float,
    r: float,
    sigma: float,
    lam: float,
    mu_j: float,
    sigma_j: float,
    t_end: float,
    n_paths: int,
    n_steps: int,
    *,
    q: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Simulate risk-neutral Merton jump-diffusion paths on a uniform grid.

    Parameters
    ----------
    s0, r, sigma:
        Initial spot, risk-free rate and diffusive volatility.
    lam, mu_j, sigma_j:
        Jump intensity (per year), mean and standard deviation of the
        Gaussian log-jump size.
    t_end, n_paths, n_steps:
        Horizon and grid resolution (``n_steps + 1`` grid points).
    q:
        Continuous dividend yield.
    rng:
        ``numpy.random.Generator``; pass a seeded generator for
        reproducibility (CONVENTION: randomness is injected, never hidden).

    Returns
    -------
    (times, paths):
        Shapes ``(n_steps + 1,)`` and ``(n_paths, n_steps + 1)`` with
        ``paths[:, 0] == s0``. Setting ``lam = 0`` recovers exact GBM.
    """
    if s0 <= 0.0:
        raise ValueError("s0 must be positive")
    if sigma < 0.0 or sigma_j < 0.0:
        raise ValueError("sigma and sigma_j must be non-negative")
    if lam < 0.0:
        raise ValueError("lam must be non-negative")
    if t_end <= 0.0 or n_paths <= 0 or n_steps <= 0:
        raise ValueError("t_end, n_paths and n_steps must be positive")
    generator = rng if rng is not None else np.random.default_rng()

    dt = t_end / n_steps
    times = np.linspace(0.0, t_end, n_steps + 1)
    kappa = float(np.expm1(mu_j + 0.5 * sigma_j**2))
    drift = (r - q - 0.5 * sigma**2 - lam * kappa) * dt

    log_increments = drift + sigma * np.sqrt(dt) * generator.standard_normal((n_paths, n_steps))
    if lam > 0.0:
        counts = generator.poisson(lam * dt, size=(n_paths, n_steps))
        jump_normals = generator.standard_normal((n_paths, n_steps))
        log_increments = log_increments + mu_j * counts + sigma_j * np.sqrt(counts) * jump_normals

    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = s0
    paths[:, 1:] = s0 * np.exp(np.cumsum(log_increments, axis=1))
    return times, paths
