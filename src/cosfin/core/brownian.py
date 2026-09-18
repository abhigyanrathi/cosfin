"""Standard Brownian motion path simulation (O&G Chapter 2).

A standard Brownian motion ``W`` satisfies ``W_0 = 0``, has independent
increments, and ``W_t - W_s ~ N(0, t - s)`` for ``s < t`` (RESULT: the defining
properties; existence via the Kolmogorov extension / Levy construction).

On a uniform grid ``t_i = i * dt`` the exact joint law of ``(W_{t_0}, ...,
W_{t_n})`` is reproduced by cumulatively summing i.i.d. ``N(0, dt)`` draws —
there is no discretisation error in the *distribution* at grid points, only in
what happens between them (RESULT: increments are independent Gaussians).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def simulate_brownian_paths(
    n_paths: int,
    n_steps: int,
    t_end: float,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Simulate standard Brownian motion paths on a uniform time grid.

    Parameters
    ----------
    n_paths, n_steps:
        Number of independent paths and of time steps (grid has ``n_steps + 1``
        points including ``t = 0``).
    t_end:
        Terminal time ``T > 0``.
    rng:
        ``numpy.random.Generator``. Pass a seeded generator for reproducibility
        (CONVENTION: randomness is always injected, never hidden module state).

    Returns
    -------
    (times, paths):
        ``times`` has shape ``(n_steps + 1,)``; ``paths`` has shape
        ``(n_paths, n_steps + 1)`` with ``paths[:, 0] == 0``.
    """
    if n_paths <= 0 or n_steps <= 0:
        raise ValueError("n_paths and n_steps must be positive")
    if t_end <= 0.0:
        raise ValueError("t_end must be positive")
    generator = rng if rng is not None else np.random.default_rng()

    dt = t_end / n_steps
    times = np.linspace(0.0, t_end, n_steps + 1)
    increments = generator.standard_normal((n_paths, n_steps)) * np.sqrt(dt)
    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = 0.0
    np.cumsum(increments, axis=1, out=paths[:, 1:])
    return times, paths
