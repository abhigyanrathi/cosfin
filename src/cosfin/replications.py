"""Replication harnesses for the mentor-confirmed Chapter 5-6 items.

Each function returns *structured data* (dataclasses / dicts / arrays); nothing
here prints or plots. ``scripts/make_figures.py`` consumes the same functions
to produce the PNGs, and ``tests/test_replications.py`` anchors them to the
published O&G values.

Covered items:

1. :func:`figure_5_7_data`  -- hedging-frequency P&L histograms under jumps.
2. :func:`figure_5_8_data`  -- CGMYB implied-volatility curves (four panels).
3. :func:`normal_density_recovery` / :func:`lognormal_density_recovery`
   -- COS density recovery for the normal and lognormal (O&G p.169 examples).
4. :func:`table_6_4`, :func:`table_6_5`, :func:`table_6_6`,
   :func:`figure_6_4_data`, :func:`table_6_7`  -- Chapter 6 COS results.
5. :func:`exercise_6_3`  -- Merton: COS vs the analytic series.
6. :func:`table_6_8`  -- Variance Gamma COS convergence.

Every parameter set is tracked in :data:`PARAMETER_PROVENANCE` as CONFIRMED
(printed in the book pages supplied) or ASSUMED (not printed anywhere on the
source pages; an illustrative choice recorded here). Inventing O&G content is
ruled out, so ASSUMED stays ASSUMED until a page settles it.

Payoff CONVENTION resolved by eq. (6.52)/Example 6.3.1: the book's
cash-or-nothing option pays the *strike* K, not unit cash. The library's
:func:`digital_price`/:func:`cos_digital_price` are per unit cash by design;
the Table 6.5 harness multiplies by K. This closes the long-standing
0.273306496 reproduction failure.

Truncation-interval NOTE (tested hypothesis, rejected): the book's Table
6.4/6.5 low-N error columns (e.g. 3.67 at N=16) are orders of magnitude
larger than anything the cumulant-rule interval produces. Two variants were
tested against the printed columns: the core engine's exact per-strike
interval (8.5e-3 at N=16) and ``book_range=True``, a single common interval
in the spirit of the vector formula (6.50) — neither reproduces the printed
digits, so the eq. (6.45) simplified range must be substantially *wider*
than ``c1 -/+ L sqrt(c2 + sqrt(c4))``. The reference *values* anchor
identically in every variant; supply the eq. (6.45) page for digit-level
error-column replication. Either variant here is strictly more accurate
than the book's own runs (our N=64 beats their N=128 at every strike).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import lognorm, norm  # type: ignore[import-untyped]

from cosfin.models.characteristic_functions import (
    CGMY,
    GBM,
    Merton,
    VarianceGamma,
)
from cosfin.models.jump_diffusion import merton_jump_price
from cosfin.models.local_vol import implied_volatility_bracketed
from cosfin.numerical.cos import (
    CharacteristicFn,
    CumulantsFn,
    _chi,
    _psi,
    cos_density,
    cos_digital_price,
    cos_european_price,
    truncation_range,
)
from cosfin.pricing.hedging import delta_hedge_pnl

Real = npt.NDArray[np.float64]

# --------------------------------------------------------------------------- #
# Published reference values (O&G pages supplied by Abhi; see provenance).
# --------------------------------------------------------------------------- #
TABLE_6_4_REF: dict[float, float] = {
    80.0: 20.799226309,
    100.0: 3.659968453,
    120.0: 0.044577814,
}
TABLE_6_5_REF: float = 0.273306496
TABLE_6_7_REF: dict[float, float] = {0.5: 21.679593920, 1.5: 50.279533980}
TABLE_6_8_REF: dict[float, float] = {0.1: 10.993703187, 1.0: 19.099354724}

PARAMETER_PROVENANCE: dict[str, str] = {
    "table_6_4": "CONFIRMED eq (6.51): GBM S0=100, r=0.10, q=0, T=0.1, sigma=0.25; "
    "K in {80,100,120}; references printed in Table 6.4.",
    "table_6_5": "CONFIRMED eq (6.52): S0=100, K=120, r=0.05, q=0, T=0.1, sigma=0.2. "
    "CONVENTION (Example 6.3.1): the cash-or-nothing pays K, so price = "
    "K * unit-cash digital. Closed form reproduces 0.273306496 to 5e-10.",
    "table_6_6": "Density-recovery companion to Figure 6.4 (CGMYB family, eq 6.53). "
    "PENDING: the page with Table 6.6's printed values has not been supplied; "
    "errors below are measured against a high-N self-reference.",
    "table_6_7": "CONFIRMED (pages supplied June 2026): CGMYB C=1, G=5, M=5, "
    "sigma=0.2, S0=K=100, r=0.1, q=0, T=1; refs 21.679593920 (Y=0.5) and "
    "50.279533980 (Y=1.5), reproduced here to <4e-8.",
    "table_6_8": "CONFIRMED (pages supplied June 2026): VG sigma=0.12, theta=-0.14, "
    "beta=0.2, S0=100, K=90, r=0.1, q=0; refs 10.993703187 (T=0.1) and "
    "19.099354724 (T=1), reproduced to <1e-8 at the book's N=4096.",
    "figure_6_4": "CGMYB density recovery, N in {4,8,16,32} (legend printed). "
    "ASSUMED: plotting domain [-2,2] (caption cut off on the supplied photo).",
    "exercise_6_3": "ASSUMED Merton set S0=K in {80..120}, r=0.05, sigma=0.2, lam=1, "
    "mu_j=-0.10, sigma_j=0.15, T=1 (the exercise page does not print one).",
    "figure_5_7": "ASSUMED: Section 5.2.3 states symmetric jumps (mu_j=0) but prints "
    "no parameter set; Figure 5.6's axes imply S0 ~ K ~ 1, T=1. We reuse Figure "
    "5.3's dynamics (r=0.05, sigma=0.2, sigma_j=0.5, xi_p=1) at that scale.",
    "figure_5_8": "CONFIRMED caption p.154: base C=1, G=1, M=5, Y=0.5, "
    "sigma_CGMYB=0.2, r=0.1; panel grids as printed. ASSUMED: S0=100, T=1, q=0 "
    "(not printed).",
}


@dataclass(frozen=True)
class ConvergenceRow:
    """One (N, price) sample of a COS convergence run.

    ``error`` is |price - reference| for the vanilla tables and the *signed*
    ``price - reference`` for Table 6.5 (the book prints signed errors there).
    """

    n: int
    price: float
    error: float
    cpu_seconds: float


# --------------------------------------------------------------------------- #
# Shared-interval COS variant (book's vectorised eq (6.50) setup).
# --------------------------------------------------------------------------- #
def _shared_range_prices(
    cf: CharacteristicFn,
    cumulants: CumulantsFn,
    spot: float,
    strikes: Real,
    r: float,
    t: float,
    n: int,
    *,
    digital: bool,
    l_trunc: float = 10.0,
) -> Real:
    """COS prices with ONE truncation interval ``[a0, b0]`` for every strike.

    The interval is built from the cumulants of ``ln(S_T/S_0)`` only; the
    per-strike log-moneyness ``x`` enters exclusively through the
    ``e^{i omega x}`` phase, in the spirit of the book's single-computation
    vector formula (6.50). Empirical status: this does NOT reproduce the
    printed Table 6.4/6.5 error columns either (see the module NOTE) — the
    book's eq. (6.45) range is evidently much wider. Kept because it is the
    natural common-interval variant and documents the tested hypothesis.
    Call-type payoffs only (all the book tables need). Reuses the
    ``chi``/``psi`` primitives of :mod:`cosfin.numerical.cos` —
    same package, deliberately shared rather than duplicated.
    """
    a, b = truncation_range(cumulants, t, l_trunc)
    x = np.log(spot / strikes)  # (m,)
    k = np.arange(n, dtype=np.float64)[None, :]  # (1, n)
    omega = k * np.pi / (b - a)  # (1, n)
    phi = np.asarray(cf(omega, t), dtype=np.complex128)
    fk = np.real(phi * np.exp(1j * omega * (x[:, None] - a)))

    a_arr, b_arr, zero = np.asarray(a), np.asarray(b), np.asarray(0.0)
    if digital:
        uk = (2.0 / (b - a)) * _psi(omega, a_arr, zero, b_arr)
    else:
        vanilla = _chi(omega, a_arr, zero, b_arr) - _psi(omega, a_arr, zero, b_arr)
        uk = (2.0 / (b - a)) * strikes[:, None] * vanilla
    weights = np.ones(n, dtype=np.float64)
    weights[0] = 0.5
    prices = np.exp(-r * t) * np.sum(weights * fk * uk, axis=1)
    return np.asarray(prices, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Item 4 -- Table 6.4: GBM call convergence (eq 6.51).
# --------------------------------------------------------------------------- #
def table_6_4(
    ns: tuple[int, ...] = (16, 32, 64, 128, 256),
    *,
    book_range: bool = False,
) -> dict[float, list[ConvergenceRow]]:
    """COS European calls under GBM vs the printed Table 6.4 references.

    All three strikes are priced in one call per N (the vector-valued formula
    (6.50) / Remark 6.3.1 point); the recorded CPU time is for that single
    vector computation.
    """
    model = GBM(r=0.10, q=0.0, sigma=0.25)
    strikes = np.array(sorted(TABLE_6_4_REF))
    out: dict[float, list[ConvergenceRow]] = {float(k): [] for k in strikes}
    for n in ns:
        t0 = time.perf_counter()
        if book_range:
            prices = _shared_range_prices(
                model.cf, model.cumulants, 100.0, strikes, 0.10, 0.1, n, digital=False
            )
        else:
            prices = cos_european_price(model.cf, model.cumulants, 100.0, strikes, 0.10, 0.1, n)
        dt = time.perf_counter() - t0
        for strike, price in zip(strikes, prices, strict=True):
            ref = TABLE_6_4_REF[float(strike)]
            out[float(strike)].append(
                ConvergenceRow(n, float(price), abs(float(price) - ref), dt)
            )
    return out


# --------------------------------------------------------------------------- #
# Item 4 -- Table 6.5: cash-or-nothing call convergence (eq 6.52).
# --------------------------------------------------------------------------- #
def table_6_5(
    ns: tuple[int, ...] = (40, 60, 80, 100, 120, 140),
    *,
    book_range: bool = False,
) -> list[ConvergenceRow]:
    """COS cash-or-nothing call vs the printed Table 6.5 reference.

    CONVENTION (Example 6.3.1): the payoff is ``K * 1{S_T > K}``, so the
    library's unit-cash digital is scaled by K = 120. ``error`` here is the
    *signed* ``price - 0.273306496`` because the book prints signed errors.
    """
    model = GBM(r=0.05, q=0.0, sigma=0.2)
    strike = np.array([120.0])
    rows: list[ConvergenceRow] = []
    for n in ns:
        t0 = time.perf_counter()
        if book_range:
            unit = _shared_range_prices(
                model.cf, model.cumulants, 100.0, strike, 0.05, 0.1, n, digital=True
            )
        else:
            unit = cos_digital_price(model.cf, model.cumulants, 100.0, strike, 0.05, 0.1, n)
        dt = time.perf_counter() - t0
        price = 120.0 * float(unit[0])
        rows.append(ConvergenceRow(n, price, price - TABLE_6_5_REF, dt))
    return rows


# --------------------------------------------------------------------------- #
# Item 4 -- Table 6.6 / Figure 6.4: CGMYB density recovery.
# --------------------------------------------------------------------------- #
def _cgmyb_pair() -> dict[float, CGMY]:
    return {
        y: CGMY(r=0.1, q=0.0, C=1.0, G=5.0, M=5.0, Y=y, sigma=0.2) for y in (0.5, 1.5)
    }


def table_6_6(
    ns: tuple[int, ...] = (4, 8, 16, 32, 64, 128),
    domain: tuple[float, float] = (-2.0, 2.0),
    n_grid: int = 400,
    ref_n: int = 512,
) -> dict[float, list[tuple[int, float]]]:
    """Max recovery error of the CGMYB density for increasing N.

    PENDING anchor: measured against a high-N (``ref_n``) self-reference until
    the page with Table 6.6's printed values is supplied; convergence is
    exponential either way (RESULT: Fang-Oosterlee error analysis).
    """
    x = np.linspace(domain[0], domain[1], n_grid)
    out: dict[float, list[tuple[int, float]]] = {}
    for y, model in _cgmyb_pair().items():
        a, b = truncation_range(model.cumulants, 1.0)
        reference = cos_density(model.cf, x, a, b, ref_n, 1.0)
        out[y] = [
            (n, float(np.max(np.abs(cos_density(model.cf, x, a, b, n, 1.0) - reference))))
            for n in ns
        ]
    return out


def figure_6_4_data(
    ns: tuple[int, ...] = (4, 8, 16, 32),
    domain: tuple[float, float] = (-2.0, 2.0),
    n_grid: int = 400,
    ref_n: int = 512,
) -> tuple[Real, dict[float, dict[int, Real]], dict[float, Real]]:
    """Recovered CGMYB densities at the printed N values plus a reference.

    Returns ``(x, {Y: {N: f_N}}, {Y: f_ref})`` for the two Figure 6.4 panels.
    """
    x = np.linspace(domain[0], domain[1], n_grid)
    curves: dict[float, dict[int, Real]] = {}
    refs: dict[float, Real] = {}
    for y, model in _cgmyb_pair().items():
        a, b = truncation_range(model.cumulants, 1.0)
        curves[y] = {n: cos_density(model.cf, x, a, b, n, 1.0) for n in ns}
        refs[y] = cos_density(model.cf, x, a, b, ref_n, 1.0)
    return x, curves, refs


# --------------------------------------------------------------------------- #
# Item 4 -- Table 6.7: CGMYB call convergence.
# --------------------------------------------------------------------------- #
def table_6_7(
    ns: tuple[int, ...] = (32, 64, 128, 256, 384, 512),
) -> dict[float, list[ConvergenceRow]]:
    """COS CGMYB calls at Y in {0.5, 1.5} vs the published Table 6.7 values."""
    out: dict[float, list[ConvergenceRow]] = {}
    for y, model in _cgmyb_pair().items():
        ref = TABLE_6_7_REF[y]
        rows: list[ConvergenceRow] = []
        for n in ns:
            t0 = time.perf_counter()
            price = float(
                cos_european_price(model.cf, model.cumulants, 100.0, 100.0, 0.1, 1.0, n)[0]
            )
            rows.append(ConvergenceRow(n, price, abs(price - ref), time.perf_counter() - t0))
        out[y] = rows
    return out


# --------------------------------------------------------------------------- #
# Item 6 -- Table 6.8: Variance Gamma call convergence.
# --------------------------------------------------------------------------- #
def table_6_8(
    ns: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048, 4096),
) -> dict[float, list[ConvergenceRow]]:
    """COS VG calls at T in {0.1, 1.0} vs the published Table 6.8 values.

    The short maturity is the interesting one: the T=0.1 VG density is so
    peaked that the book (and we) need N = 4096 for full accuracy -- the
    canonical example of cumulant-driven interval width fighting a needle
    density.
    """
    vg = VarianceGamma(r=0.1, q=0.0, sigma=0.12, theta=-0.14, beta=0.2)
    out: dict[float, list[ConvergenceRow]] = {}
    for t, ref in TABLE_6_8_REF.items():
        rows: list[ConvergenceRow] = []
        for n in ns:
            t0 = time.perf_counter()
            price = float(cos_european_price(vg.cf, vg.cumulants, 100.0, 90.0, 0.1, t, n)[0])
            rows.append(ConvergenceRow(n, price, abs(price - ref), time.perf_counter() - t0))
        out[t] = rows
    return out


# --------------------------------------------------------------------------- #
# Item 5 -- Exercise 6.3: Merton via COS vs the analytic series.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Exercise63Row:
    strike: float
    cos_price: float
    series_price: float
    abs_diff: float


def exercise_6_3(
    strikes: tuple[float, ...] = (80.0, 90.0, 100.0, 110.0, 120.0),
) -> list[Exercise63Row]:
    """Two independent Merton pricers, one model (parameters ASSUMED).

    The COS route knows only the characteristic function; the series route
    knows only the lognormal-mixture decomposition. Agreement to ~1e-12 is a
    genuine cross-validation of both.
    """
    model = Merton(r=0.05, q=0.0, sigma=0.20, lam=1.0, mu_j=-0.10, sigma_j=0.15)
    ks = np.array(strikes)
    cos_prices = cos_european_price(
        model.cf, model.cumulants, 100.0, ks, 0.05, 1.0, 1024, l_trunc=12.0
    )
    series = merton_jump_price(100.0, ks, 0.05, 0.20, 1.0, lam=1.0, mu_j=-0.10, sigma_j=0.15)
    return [
        Exercise63Row(float(k), float(c), float(s), abs(float(c) - float(s)))
        for k, c, s in zip(ks, cos_prices, series, strict=True)
    ]


# --------------------------------------------------------------------------- #
# Item 3 -- density recovery for the normal and lognormal (O&G p.169).
# --------------------------------------------------------------------------- #
def normal_density_recovery(
    ns: tuple[int, ...] = (4, 8, 16, 32, 64),
    interval: tuple[float, float] = (-10.0, 10.0),
    n_grid: int = 1001,
) -> list[tuple[int, float]]:
    """Max error recovering the standard normal density from its cf.

    ``cf(u) = e^{-u^2/2}`` (RESULT). Convergence in N is exponential; N = 64
    reaches machine precision on [-10, 10]. PENDING: exact printed p.169
    values (if the book tabulates them) once that page is supplied.
    """

    def cf(u: npt.ArrayLike, t: float) -> npt.NDArray[np.complex128]:
        uc = np.asarray(u, dtype=np.complex128)
        return np.asarray(np.exp(-0.5 * uc**2), dtype=np.complex128)

    a, b = interval
    y = np.linspace(a, b, n_grid)
    analytic = norm.pdf(y)
    return [
        (n, float(np.max(np.abs(cos_density(cf, y, a, b, n, 1.0) - analytic)))) for n in ns
    ]


def lognormal_density_recovery(
    ns: tuple[int, ...] = (8, 16, 32, 64, 128),
    interval: tuple[float, float] = (-10.0, 10.0),
    x_range: tuple[float, float] = (0.05, 5.0),
    n_grid: int = 500,
) -> list[tuple[int, float]]:
    """Max error recovering the lognormal density via a change of variables.

    ``X = e^Z`` with ``Z ~ N(0,1)``: recover the density of Z from its cf on
    the log grid, then ``f_X(x) = f_Z(ln x) / x`` (RESULT: monotone transform
    of densities). This is the standard-normal recovery in disguise, which is
    the pedagogical point of the p.169 pair.
    """

    def cf(u: npt.ArrayLike, t: float) -> npt.NDArray[np.complex128]:
        uc = np.asarray(u, dtype=np.complex128)
        return np.asarray(np.exp(-0.5 * uc**2), dtype=np.complex128)

    a, b = interval
    x = np.linspace(x_range[0], x_range[1], n_grid)
    analytic = lognorm.pdf(x, 1.0)
    out: list[tuple[int, float]] = []
    for n in ns:
        recovered = cos_density(cf, np.log(x), a, b, n, 1.0) / x
        out.append((n, float(np.max(np.abs(recovered - analytic)))))
    return out


# --------------------------------------------------------------------------- #
# Item 1 -- Figure 5.7: hedging-frequency P&L histograms under jumps.
# --------------------------------------------------------------------------- #
FIG_5_7_PARAMS: dict[str, float] = {
    "spot": 1.0,
    "strike": 1.0,
    "r": 0.05,
    "sigma": 0.2,
    "t": 1.0,
    "lam": 1.0,
    "mu_j": 0.0,  # symmetric jumps, as stated in Section 5.2.3
    "sigma_j": 0.5,
}


def figure_5_7_data(
    n_steps_pair: tuple[int, int] = (10, 2000),
    n_paths: int = 1000,
    seed: int = 7,
) -> dict[int, Real]:
    """Terminal hedging P&L per path at the two rebalancing frequencies.

    Parameters are ASSUMED (see provenance). The result that matters is
    qualitative and parameter-robust (RESULT, Section 5.2.3): raising the
    hedging frequency from 10 to 2000 steps does *not* collapse the P&L
    dispersion -- the left tail from jumps survives, because a delta hedge
    only spans the diffusion.
    """
    out: dict[int, Real] = {}
    for n_steps in n_steps_pair:
        out[n_steps] = delta_hedge_pnl(
            FIG_5_7_PARAMS["spot"],
            FIG_5_7_PARAMS["strike"],
            FIG_5_7_PARAMS["r"],
            FIG_5_7_PARAMS["sigma"],
            FIG_5_7_PARAMS["t"],
            n_steps,
            n_paths,
            lam=FIG_5_7_PARAMS["lam"],
            mu_j=FIG_5_7_PARAMS["mu_j"],
            sigma_j=FIG_5_7_PARAMS["sigma_j"],
            rng=np.random.default_rng(seed),
        )
    return out


# --------------------------------------------------------------------------- #
# Item 2 -- Figure 5.8: CGMYB implied-volatility curves.
# --------------------------------------------------------------------------- #
FIG_5_8_BASE: dict[str, float] = {"c": 1.0, "g": 1.0, "m": 5.0, "y": 0.5, "sigma": 0.2}
FIG_5_8_PANELS: dict[str, tuple[float, ...]] = {
    "c": (0.1, 0.2, 0.5, 1.0),
    "g": (0.5, 1.0, 2.0, 3.0),
    "m": (2.0, 3.0, 5.0, 10.0),
    "y": (0.2, 0.4, 0.6, 0.9),
}


def cgmyb_implied_volatility_curve(
    model: CGMY,
    strikes: Real,
    spot: float = 100.0,
    r: float = 0.1,
    t: float = 1.0,
    n_cos: int = 512,
) -> Real:
    """Black-Scholes implied volatilities of CGMYB COS prices, one per strike.

    Wing strikes whose prices sit outside the invertible region come back as
    NaN rather than raising (MODELLING CHOICE: a smile plot wants gaps, not
    crashes).
    """
    prices = cos_european_price(model.cf, model.cumulants, spot, strikes, r, t, n_cos)
    ivs = np.full(strikes.shape, np.nan, dtype=np.float64)
    for i, (strike, price) in enumerate(zip(strikes, prices, strict=True)):
        try:
            ivs[i] = implied_volatility_bracketed(float(price), spot, float(strike), r, t)
        except (ValueError, RuntimeError):
            continue
    return ivs


def figure_5_8_data(
    strikes: Real | None = None,
    spot: float = 100.0,
    r: float = 0.1,
    t: float = 1.0,
    n_cos: int = 512,
) -> tuple[Real, dict[str, dict[float, Real]]]:
    """Implied-volatility curves for the four Figure 5.8 panels.

    Base parameters per the printed caption (C=1, G=1, M=5, Y=0.5,
    sigma=0.2, r=0.1); each panel varies exactly one of C/G/M/Y over the
    printed legend grid while the others stay at base. S0=100 and T=1 are
    ASSUMED (not printed). Returns ``(strikes, {panel: {value: iv_array}})``.
    """
    ks = np.linspace(40.0, 180.0, 29) if strikes is None else strikes
    panels: dict[str, dict[float, Real]] = {}
    for name, values in FIG_5_8_PANELS.items():
        panels[name] = {}
        for value in values:
            params = dict(FIG_5_8_BASE)
            params[name] = value
            model = CGMY(
                r=r,
                q=0.0,
                C=params["c"],
                G=params["g"],
                M=params["m"],
                Y=params["y"],
                sigma=params["sigma"],
            )
            panels[name][value] = cgmyb_implied_volatility_curve(
                model, ks, spot=spot, r=r, t=t, n_cos=n_cos
            )
    return ks, panels
