"""Merton (1976) jump-diffusion European option pricing (O&G Chapter 5).

Conditioning on the number of jumps ``N_T = n``, the terminal log-price is
Gaussian, so each conditional expectation is a Black-Scholes price with
jump-adjusted inputs. Collecting the Poisson weights (and absorbing the
``e^{n(mu_j + sigma_j^2/2)}`` factor into a reweighted intensity) gives the
lognormal-mixture series (RESULT, Merton 1976):

    V = sum_{n >= 0} e^{-lam' T} (lam' T)^n / n!
        * BS(S, K, r_n, sigma_n, T),

with

    kappa   = e^{mu_j + sigma_j^2/2} - 1      (mean relative jump size),
    lam'    = lam (1 + kappa),
    sigma_n^2 = sigma^2 + n sigma_j^2 / T,
    r_n     = r - lam kappa + n (mu_j + sigma_j^2/2) / T,

where the identity ``ln(1 + kappa) = mu_j + sigma_j^2/2`` holds *exactly* and
is used directly (avoids a needless ``log(expm1(...) + 1)`` round trip).

Poisson weights are accumulated with the recursion
``w_{n+1} = w_n * lam' T / (n + 1)`` — no factorials, no overflow. The series
is truncated once the running weight drops below ``weight_tol`` *after* the
index has passed the Poisson mode ``lam' T`` (weights increase up to the mode
before decaying, so truncating earlier would be wrong).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from cosfin.models.black_scholes import OptionType, black_scholes_price


def merton_jump_price(
    spot: npt.ArrayLike,
    strike: npt.ArrayLike,
    r: float,
    sigma: float,
    t: float,
    *,
    lam: float,
    mu_j: float,
    sigma_j: float,
    q: float = 0.0,
    option_type: OptionType = "call",
    max_terms: int = 200,
    weight_tol: float = 1e-16,
) -> npt.NDArray[np.float64]:
    """European call/put under Merton jump-diffusion via the analytic series.

    Vectorised over ``spot``/``strike``. ``lam = 0`` short-circuits to plain
    Black-Scholes (the ``n = 0`` term with unit weight, exactly). Raises
    ``RuntimeError`` if ``max_terms`` is exhausted before the tail weight is
    negligible — for ``lam' T`` beyond ~100 raise ``max_terms`` (the default
    200 covers every parameter set used in Chapters 5-6).

    Each series term reuses :func:`black_scholes_price`, so the degenerate
    ``sigma_n sqrt(T) = 0`` case (``sigma = 0``, ``n = 0``) inherits the
    correct forward-intrinsic limit from the model layer.
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if t <= 0.0:
        raise ValueError("t must be positive")
    if sigma < 0.0 or sigma_j < 0.0:
        raise ValueError("sigma and sigma_j must be non-negative")
    if lam < 0.0:
        raise ValueError("lam must be non-negative")

    if lam == 0.0:
        return black_scholes_price(spot, strike, r, sigma, t, option_type=option_type, q=q)

    log_one_plus_kappa = mu_j + 0.5 * sigma_j**2
    kappa = float(np.expm1(log_one_plus_kappa))
    lam_bar_t = lam * (1.0 + kappa) * t

    weight = float(np.exp(-lam_bar_t))
    total: npt.NDArray[np.float64] | None = None
    for n in range(max_terms):
        sigma_n = float(np.sqrt(sigma**2 + n * sigma_j**2 / t))
        r_n = r - lam * kappa + n * log_one_plus_kappa / t
        term = weight * black_scholes_price(
            spot, strike, r_n, sigma_n, t, option_type=option_type, q=q
        )
        total = term if total is None else total + term
        weight *= lam_bar_t / (n + 1.0)
        if weight < weight_tol and (n + 1.0) > lam_bar_t:
            return np.asarray(total, dtype=np.float64)
    raise RuntimeError(
        f"Merton series not converged after {max_terms} terms; increase max_terms"
    )
