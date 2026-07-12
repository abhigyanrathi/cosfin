"""Monte Carlo European option pricing (O&G Chapters 2 and 5).

Design (MODELLING CHOICE): this module prices from *terminal spots produced
by any simulator* — ``simulate_gbm_paths``, ``simulate_merton_paths`` or a
user-supplied array — rather than baking one dynamic in. Simulation and
estimation are orthogonal concerns; keeping them separate lets one estimator
serve every model and keeps the simulate -> price pipeline visible at the
call site.

Estimator (RESULT: strong law + CLT):

    V_hat = e^{-rT} (1/n) sum payoff(S_T^(i)),
    SE    = e^{-rT} sd(payoff) / sqrt(n),

so the error shrinks like ``O(n^{-1/2})`` independent of dimension, and
``V_hat +/- z * SE`` is an asymptotic confidence interval. ``ddof = 1`` in
the standard deviation gives the unbiased variance estimator.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from pyfinlib_practice.pricing.payoffs import OptionType, call_payoff, put_payoff


class MonteCarloResult(NamedTuple):
    """Point estimate plus its CLT-based one-sigma standard error."""

    price: float
    standard_error: float
    n_paths: int


def monte_carlo_european_price(
    terminal_spots: npt.ArrayLike,
    strike: float,
    r: float,
    t: float,
    *,
    option_type: OptionType = "call",
) -> MonteCarloResult:
    """Discounted-payoff Monte Carlo estimate of a European option price.

    Parameters
    ----------
    terminal_spots:
        Terminal prices ``S_T`` from any simulator (flattened internally).
    strike, r, t:
        Contract strike, risk-free rate and maturity used for discounting.
    option_type:
        ``"call"`` or ``"put"``.
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if strike <= 0.0:
        raise ValueError("strike must be positive")
    if t < 0.0:
        raise ValueError("t must be non-negative")

    s_t = np.asarray(terminal_spots, dtype=np.float64).ravel()
    if s_t.size < 2:
        raise ValueError("need at least two paths to report a standard error")
    if not np.all(np.isfinite(s_t)):
        raise ValueError("terminal_spots must be finite")

    payoff = call_payoff(s_t, strike) if option_type == "call" else put_payoff(s_t, strike)
    discounted = float(np.exp(-r * t)) * payoff
    price = float(np.mean(discounted))
    standard_error = float(np.std(discounted, ddof=1) / np.sqrt(s_t.size))
    return MonteCarloResult(price=price, standard_error=standard_error, n_paths=int(s_t.size))
