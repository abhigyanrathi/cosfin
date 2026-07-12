"""Numerical engines (O&G Chapters 2 and 6): Monte Carlo and the COS method.

Numerical layer of the dependency DAG: ``cos`` is self-contained;
``monte_carlo`` imports only the leaf module ``pricing.payoffs``.
"""

from pyfinlib_practice.numerical.cos import (
    CharacteristicFn,
    CumulantsFn,
    cos_density,
    cos_digital_price,
    cos_european_price,
    truncation_range,
)
from pyfinlib_practice.numerical.monte_carlo import (
    MonteCarloResult,
    monte_carlo_european_price,
)

__all__ = [
    "CharacteristicFn",
    "CumulantsFn",
    "MonteCarloResult",
    "cos_density",
    "cos_digital_price",
    "cos_european_price",
    "monte_carlo_european_price",
    "truncation_range",
]
