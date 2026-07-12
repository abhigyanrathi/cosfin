"""Pricing utilities (O&G Chapters 3 and 5): payoffs, Greeks, delta hedging.

Top layer of the dependency DAG: may import ``core`` and ``models``.
``payoffs`` is a dependency-free leaf so ``numerical`` may import it without
creating a cycle.
"""

from pyfinlib_practice.pricing.greeks import delta, gamma, rho, theta, vega
from pyfinlib_practice.pricing.hedging import delta_hedge_pnl
from pyfinlib_practice.pricing.payoffs import (
    Position,
    breakeven,
    call_payoff,
    digital_call_payoff,
    digital_put_payoff,
    option_pnl,
    put_payoff,
)

__all__ = [
    "Position",
    "breakeven",
    "call_payoff",
    "delta",
    "delta_hedge_pnl",
    "digital_call_payoff",
    "digital_put_payoff",
    "gamma",
    "option_pnl",
    "put_payoff",
    "rho",
    "theta",
    "vega",
]
