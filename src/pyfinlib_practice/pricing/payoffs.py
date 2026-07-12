"""Payoff primitives (O&G Chapter 3).

Leaf module by design: imports nothing from the rest of the library, so both
``pricing`` and ``numerical`` may depend on it without creating cycles (the
dependency DAG is documented in the package README). ``OptionType`` is
re-declared locally rather than imported for the same reason — a ``Literal``
alias is structural, so the duplication costs nothing.

CONVENTION: digital payoffs use *strict* inequalities, so at ``S_T == K``
both the digital call and the digital put pay zero. The event has zero
probability under every continuous model in this library, so pricing is
unaffected; the convention only matters for path-wise bookkeeping.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

OptionType = Literal["call", "put"]
Position = Literal["long", "short"]


def call_payoff(spot: npt.ArrayLike, strike: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """European call payoff ``max(S_T - K, 0)``, broadcast over inputs."""
    s = np.asarray(spot, dtype=np.float64)
    k = np.asarray(strike, dtype=np.float64)
    return np.asarray(np.maximum(s - k, 0.0), dtype=np.float64)


def put_payoff(spot: npt.ArrayLike, strike: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """European put payoff ``max(K - S_T, 0)``, broadcast over inputs."""
    s = np.asarray(spot, dtype=np.float64)
    k = np.asarray(strike, dtype=np.float64)
    return np.asarray(np.maximum(k - s, 0.0), dtype=np.float64)


def digital_call_payoff(spot: npt.ArrayLike, strike: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Cash-or-nothing call payoff ``1{S_T > K}`` (unit cash, strict)."""
    s = np.asarray(spot, dtype=np.float64)
    k = np.asarray(strike, dtype=np.float64)
    return np.asarray((s > k).astype(np.float64), dtype=np.float64)


def digital_put_payoff(spot: npt.ArrayLike, strike: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Cash-or-nothing put payoff ``1{S_T < K}`` (unit cash, strict)."""
    s = np.asarray(spot, dtype=np.float64)
    k = np.asarray(strike, dtype=np.float64)
    return np.asarray((s < k).astype(np.float64), dtype=np.float64)


def option_pnl(
    spot: npt.ArrayLike,
    strike: npt.ArrayLike,
    premium: npt.ArrayLike,
    *,
    option_type: OptionType = "call",
    position: Position = "long",
) -> npt.NDArray[np.float64]:
    """Expiry profit and loss of a vanilla position, net of premium.

    Long: ``payoff - premium``; short: ``premium - payoff``. No financing or
    discounting of the premium between trade date and expiry (CONVENTION:
    the usual hockey-stick diagram; add carry externally if needed).
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if position not in ("long", "short"):
        raise ValueError(f"position must be 'long' or 'short', got {position!r}")
    payoff = (
        call_payoff(spot, strike) if option_type == "call" else put_payoff(spot, strike)
    )
    pnl = payoff - np.asarray(premium, dtype=np.float64)
    return np.asarray(pnl if position == "long" else -pnl, dtype=np.float64)


def breakeven(strike: float, premium: float, *, option_type: OptionType = "call") -> float:
    """Expiry spot at which a long position's P&L is zero.

    Call: ``K + premium``; put: ``K - premium``. A negative put breakeven
    means the premium exceeds the strike — the long put can never break
    even (the value is returned unclamped so the caller can detect this).
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if premium < 0.0:
        raise ValueError("premium must be non-negative")
    return float(strike + premium) if option_type == "call" else float(strike - premium)
