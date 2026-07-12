"""Discrete delta hedging: replication error under GBM and Merton (O&G Ch5).

Experiment: sell one European option at its Black-Scholes value, then run the
self-financing Black-Scholes delta hedge on a discrete grid while the *true*
dynamics are either GBM (``lam = 0``) or Merton jump-diffusion (``lam > 0``).
Report the terminal P&L per path:

    PnL_T = cash_T + Delta_last * S_T - payoff(S_T).

Accounting per step (self-financing, MODELLING CHOICE — hedger prices and
hedges with Black-Scholes at ``sigma_hedge`` regardless of the true
dynamics):

* cash accrues at ``e^{r dt}``;
* the stock position accrues the continuous dividend at ``e^{q dt}``,
  reinvested in shares (so ``Delta`` shares become ``Delta e^{q dt}`` before
  rebalancing — exact treatment, not an approximation);
* rebalance to the new Black-Scholes delta at every interior grid point;
  no rebalance at ``T`` itself (the position is settled instead).

What the experiment shows (RESULT):

* Under GBM the hedge error is pure discretisation noise: mean ~ 0 and
  standard deviation ``O(1/sqrt(n_steps))`` — quadrupling the rebalancing
  frequency roughly halves the P&L dispersion.
* Under Merton the jump risk is *not spanned* by stock and bond: between
  rebalances the stock can gap across the tangent line the delta hedge
  locks in, so the P&L dispersion floors at a strictly positive level no
  matter how fine the grid. This is the market-incompleteness message of
  Chapter 5.

Memory: paths are evolved in place step by step — ``O(n_paths)`` storage,
never the full ``(n_paths, n_steps)`` matrix.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pyfinlib_practice.models.black_scholes import OptionType, black_scholes_price
from pyfinlib_practice.pricing.greeks import delta
from pyfinlib_practice.pricing.payoffs import call_payoff, put_payoff


def delta_hedge_pnl(
    spot: float,
    strike: float,
    r: float,
    sigma: float,
    t: float,
    n_steps: int,
    n_paths: int,
    *,
    option_type: OptionType = "call",
    q: float = 0.0,
    lam: float = 0.0,
    mu_j: float = 0.0,
    sigma_j: float = 0.0,
    sigma_hedge: float | None = None,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.float64]:
    """Terminal P&L of a discretely delta-hedged short option, per path.

    Parameters
    ----------
    spot, strike, r, sigma, t:
        Contract and market inputs; ``sigma`` is the *true* diffusive
        volatility of the simulated dynamics.
    n_steps, n_paths:
        Rebalancing grid resolution and Monte Carlo size.
    option_type, q:
        Contract type and continuous dividend yield.
    lam, mu_j, sigma_j:
        Merton jump intensity and Gaussian log-jump parameters;
        ``lam = 0`` gives pure GBM dynamics.
    sigma_hedge:
        Volatility the hedger plugs into the Black-Scholes price and delta;
        defaults to ``sigma`` (a correctly-calibrated hedger). Setting it
        away from ``sigma`` reproduces the classic hedging-at-the-wrong-vol
        experiment.
    rng:
        ``numpy.random.Generator``; pass a seeded generator for
        reproducibility.

    Returns
    -------
    P&L array of shape ``(n_paths,)``. Under GBM its mean is ~ 0 and its
    standard deviation shrinks like ``1/sqrt(n_steps)``; under Merton the
    dispersion floors at the unhedgeable jump risk.
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if spot <= 0.0 or strike <= 0.0:
        raise ValueError("spot and strike must be positive")
    if t <= 0.0:
        raise ValueError("t must be positive")
    if n_steps <= 0 or n_paths <= 0:
        raise ValueError("n_steps and n_paths must be positive")
    if sigma < 0.0 or sigma_j < 0.0:
        raise ValueError("sigma and sigma_j must be non-negative")
    if lam < 0.0:
        raise ValueError("lam must be non-negative")

    generator = rng if rng is not None else np.random.default_rng()
    hedge_vol = sigma if sigma_hedge is None else sigma_hedge
    if hedge_vol < 0.0:
        raise ValueError("sigma_hedge must be non-negative")

    dt = t / n_steps
    sqrt_dt = float(np.sqrt(dt))
    cash_growth = float(np.exp(r * dt))
    dividend_growth = float(np.exp(q * dt))
    kappa = float(np.expm1(mu_j + 0.5 * sigma_j**2)) if lam > 0.0 else 0.0
    drift = (r - q - 0.5 * sigma**2 - lam * kappa) * dt

    # t = 0: sell the option at the hedger's model value, buy Delta_0 shares.
    price_0 = float(
        black_scholes_price(spot, strike, r, hedge_vol, t, option_type=option_type, q=q)
    )
    delta_prev = np.full(
        n_paths,
        float(delta(spot, strike, r, hedge_vol, t, option_type=option_type, q=q)),
        dtype=np.float64,
    )
    s = np.full(n_paths, float(spot), dtype=np.float64)
    cash = price_0 - delta_prev * s

    for i in range(1, n_steps + 1):
        log_return = drift + sigma * sqrt_dt * generator.standard_normal(n_paths)
        if lam > 0.0:
            counts = generator.poisson(lam * dt, n_paths)
            jump_normals = generator.standard_normal(n_paths)
            log_return = log_return + mu_j * counts + sigma_j * np.sqrt(counts) * jump_normals
        s = s * np.exp(log_return)
        cash = cash * cash_growth
        delta_prev = delta_prev * dividend_growth  # dividends reinvested in shares

        if i < n_steps:
            tau = t - i * dt
            delta_new = np.asarray(
                delta(s, strike, r, hedge_vol, tau, option_type=option_type, q=q),
                dtype=np.float64,
            )
            cash = cash - (delta_new - delta_prev) * s
            delta_prev = delta_new

    payoff = (
        call_payoff(s, strike) if option_type == "call" else put_payoff(s, strike)
    )
    return np.asarray(cash + delta_prev * s - payoff, dtype=np.float64)
