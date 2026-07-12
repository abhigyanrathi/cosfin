"""Pricing models (O&G Chapters 3-6): Black-Scholes, implied vol, Levy models.

Model layer of the dependency DAG: may import ``core``, never ``pricing`` or
``numerical``. The canonical ``vega`` lives here (``black_scholes``) because
the implied-volatility solvers need it; ``pricing.greeks`` re-exports it.
"""

from pyfinlib_practice.models.black_scholes import (
    OptionType,
    black_scholes_price,
    butterfly_price,
    d1_d2,
    digital_price,
    norm_cdf,
    norm_pdf,
    put_call_parity_residual,
    vega,
)
from pyfinlib_practice.models.characteristic_functions import (
    CGMY,
    GBM,
    CharacteristicModel,
    Merton,
    VarianceGamma,
)
from pyfinlib_practice.models.jump_diffusion import merton_jump_price
from pyfinlib_practice.models.local_vol import (
    Method,
    implied_volatility,
    implied_volatility_bracketed,
    implied_volatility_newton,
    implied_volatility_secant,
    price_from_density,
    risk_neutral_density,
)

__all__ = [
    "CGMY",
    "GBM",
    "CharacteristicModel",
    "Merton",
    "Method",
    "OptionType",
    "VarianceGamma",
    "black_scholes_price",
    "butterfly_price",
    "d1_d2",
    "digital_price",
    "implied_volatility",
    "implied_volatility_bracketed",
    "implied_volatility_newton",
    "implied_volatility_secant",
    "merton_jump_price",
    "norm_cdf",
    "norm_pdf",
    "price_from_density",
    "put_call_parity_residual",
    "risk_neutral_density",
    "vega",
]
