"""Chapter 3 tests: payoff primitives, position P&L, breakeven."""

from __future__ import annotations

import numpy as np
import pytest

from cosfin.pricing.payoffs import (
    breakeven,
    call_payoff,
    digital_call_payoff,
    digital_put_payoff,
    option_pnl,
    put_payoff,
)


def test_vanilla_payoffs_and_broadcast() -> None:
    spots = np.array([80.0, 100.0, 120.0])
    np.testing.assert_allclose(call_payoff(spots, 100.0), [0.0, 0.0, 20.0])
    np.testing.assert_allclose(put_payoff(spots, 100.0), [20.0, 0.0, 0.0])
    # Broadcast across strikes as well.
    assert call_payoff(100.0, np.array([90.0, 110.0])).shape == (2,)


def test_digital_payoffs_are_strict_at_the_strike() -> None:
    # CONVENTION: 1{S > K} / 1{S < K} — both pay zero exactly at S == K.
    spots = np.array([99.0, 100.0, 101.0])
    np.testing.assert_allclose(digital_call_payoff(spots, 100.0), [0.0, 0.0, 1.0])
    np.testing.assert_allclose(digital_put_payoff(spots, 100.0), [1.0, 0.0, 0.0])
    assert float(digital_call_payoff(100.0, 100.0) + digital_put_payoff(100.0, 100.0)) == 0.0


def test_option_pnl_long_short_accounting() -> None:
    spots = np.array([80.0, 100.0, 130.0])
    long_call = option_pnl(spots, 100.0, 10.0)
    np.testing.assert_allclose(long_call, [-10.0, -10.0, 20.0])
    short_call = option_pnl(spots, 100.0, 10.0, position="short")
    np.testing.assert_allclose(short_call, -long_call)
    long_put = option_pnl(spots, 100.0, 5.0, option_type="put")
    np.testing.assert_allclose(long_put, [15.0, -5.0, -5.0])


def test_option_pnl_validation() -> None:
    with pytest.raises(ValueError):
        option_pnl(100.0, 100.0, 1.0, option_type="straddle")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        option_pnl(100.0, 100.0, 1.0, position="flat")  # type: ignore[arg-type]


def test_breakeven() -> None:
    assert breakeven(100.0, 4.5) == pytest.approx(104.5)
    assert breakeven(100.0, 4.5, option_type="put") == pytest.approx(95.5)
    # Premium above strike: the long put can never break even — the value is
    # returned unclamped (negative) so callers can detect it.
    assert breakeven(100.0, 120.0, option_type="put") == pytest.approx(-20.0)


def test_breakeven_validation() -> None:
    with pytest.raises(ValueError):
        breakeven(100.0, -1.0)
    with pytest.raises(ValueError):
        breakeven(100.0, 1.0, option_type="straddle")  # type: ignore[arg-type]
