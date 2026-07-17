"""Streamlit demo for pyfinlib-practice, deployed on Render.

UI glue only: every number shown here is produced by the installed
``pyfinlib_practice`` package (the same code exercised by the test suite).
Computation helpers with any real logic live in ``pyfinlib_practice.demo``
where they are tested and type-checked; this file just wires widgets to them.

Design notes:

* Figures are built with ``matplotlib.figure.Figure`` directly — no pyplot.
  Streamlit re-runs this script in worker threads, and pyplot's global
  figure registry is not thread-safe (it also leaks figures between reruns).
* Heavy calls sit behind ``st.cache_data`` keyed on the full parameter
  tuple, so slider wiggling only recomputes what changed. Model instances
  are frozen dataclasses, hence stable, hashable cache keys.
* Widget bounds are capped (paths, steps, grid sizes) so the app stays
  comfortably inside Render's free-tier 512 MB instance.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import streamlit as st
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pyfinlib_practice import __version__
from pyfinlib_practice.core.gbm import Scheme, simulate_gbm_paths
from pyfinlib_practice.demo import (
    DensityComparison,
    gbm_density_comparison,
    levy_smile,
    lognormal_terminal_pdf,
)
from pyfinlib_practice.models.black_scholes import OptionType, black_scholes_price, vega
from pyfinlib_practice.models.characteristic_functions import (
    CGMY,
    GBM,
    CharacteristicModel,
    Merton,
    VarianceGamma,
)
from pyfinlib_practice.pricing.greeks import delta, gamma, rho, theta

FloatArray = npt.NDArray[np.float64]

st.set_page_config(page_title="pyfinlib-practice demo", page_icon="📈", layout="wide")


# ------------------------------------------------------------------ cached compute
@st.cache_data(show_spinner=False)
def cached_price_curves(
    strike: float, r: float, sigma: float, t: float, q: float, option_type: OptionType
) -> tuple[FloatArray, FloatArray, FloatArray]:
    spots = np.linspace(0.4 * strike, 1.6 * strike, 121)
    prices = black_scholes_price(spots, strike, r, sigma, t, option_type=option_type, q=q)
    deltas = delta(spots, strike, r, sigma, t, option_type=option_type, q=q)
    return spots, prices, deltas


@st.cache_data(show_spinner=False)
def cached_smile(
    model: CharacteristicModel,
    spot: float,
    k_lo: float,
    k_hi: float,
    n_k: int,
    r: float,
    t: float,
    q: float,
    n_cos: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    strikes = np.linspace(k_lo, k_hi, n_k)
    prices, ivs = levy_smile(model, spot, strikes, r, t, q=q, n=n_cos)
    return strikes, prices, ivs


@st.cache_data(show_spinner=False)
def cached_density(
    spot: float, r: float, sigma: float, t: float, q: float, n_strikes: int, n_cos: int
) -> DensityComparison:
    return gbm_density_comparison(
        spot,
        r,
        sigma,
        t,
        q=q,
        k_min=0.35 * spot,
        k_max=2.6 * spot,
        n_strikes=n_strikes,
        n_cos=n_cos,
    )


@st.cache_data(show_spinner=False)
def cached_paths(
    s0: float,
    mu: float,
    sigma: float,
    t: float,
    n_paths: int,
    n_steps: int,
    scheme: Scheme,
    seed: int,
) -> tuple[FloatArray, FloatArray]:
    rng = np.random.default_rng(seed)
    return simulate_gbm_paths(s0, mu, sigma, t, n_paths, n_steps, scheme=scheme, rng=rng)


def new_axes(width: float = 7.0, height: float = 4.0) -> tuple[Figure, Axes]:
    fig = Figure(figsize=(width, height), dpi=110)
    ax = fig.add_subplot()
    ax.grid(alpha=0.3)
    return fig, ax


def as_option_type(label: str) -> OptionType:
    return "call" if label == "call" else "put"


def as_scheme(label: str) -> Scheme:
    return "exact" if label == "exact" else "euler"


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("pyfinlib-practice")
    st.caption(f"v{__version__}")
    st.markdown(
        "Interactive demo of the **AI-written answer-key companion** to the "
        "hand-written `pyfinlib` study project, following Oosterlee & Grzelak, "
        "*Mathematical Modelling and Computation in Finance* (Ch. 2-6).\n\n"
        "Every value on these pages comes from the installed package — the same "
        "code behind the repo's test suite and the reproduced O&G tables.\n\n"
        "[Source on GitHub](https://github.com/abhigyanrathi/pyfinlib-practice)"
    )
    st.divider()
    st.caption(
        "Hosted on Render's free tier: the first request after ~15 idle minutes "
        "waits for a cold start."
    )

st.title("Option pricing, four ways")
tab_bs, tab_smile, tab_density, tab_mc = st.tabs(
    [
        "Black-Scholes & Greeks",
        "Implied-vol smile from jumps",
        "Density recovery",
        "GBM Monte Carlo",
    ]
)

# ------------------------------------------------------------------ tab 1: BS
with tab_bs:
    st.subheader("European option under Black-Scholes (O&G Ch. 3)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        bs_spot = st.number_input("Spot S₀", 1.0, 500.0, 100.0, 1.0)
        bs_strike = st.number_input("Strike K", 1.0, 500.0, 100.0, 1.0)
    with c2:
        bs_sigma = st.number_input("Volatility σ", 0.01, 1.5, 0.20, 0.01)
        bs_t = st.number_input("Maturity T (yrs)", 0.05, 5.0, 1.0, 0.05)
    with c3:
        bs_r = st.number_input("Rate r", -0.05, 0.25, 0.05, 0.005, format="%.3f")
        bs_q = st.number_input("Dividend yield q", 0.0, 0.15, 0.0, 0.005, format="%.3f")
    with c4:
        bs_type = as_option_type(str(st.selectbox("Type", ["call", "put"])))

    args = (bs_spot, bs_strike, bs_r, bs_sigma, bs_t)
    greeks = {
        "Price": float(black_scholes_price(*args, option_type=bs_type, q=bs_q)),
        "Delta": float(delta(*args, option_type=bs_type, q=bs_q)),
        "Gamma": float(gamma(*args, q=bs_q)),
        "Vega": float(vega(*args, q=bs_q)),
        "Theta": float(theta(*args, option_type=bs_type, q=bs_q)),
        "Rho": float(rho(*args, option_type=bs_type, q=bs_q)),
    }
    for col, (name, val) in zip(st.columns(len(greeks)), greeks.items(), strict=True):
        col.metric(name, f"{val:,.4f}")

    spots, prices, deltas = cached_price_curves(bs_strike, bs_r, bs_sigma, bs_t, bs_q, bs_type)
    fig, ax = new_axes()
    ax.plot(spots, prices, label=f"{bs_type} price")
    ax.plot(spots, deltas * float(np.max(prices)), "--", label="delta (rescaled)")
    ax.axvline(bs_spot, color="grey", lw=0.8)
    ax.set_xlabel("spot")
    ax.set_ylabel("value")
    ax.legend()
    st.pyplot(fig, width=700)
    st.caption(
        "Δ, Γ, vega, Θ, ρ are the closed-form Ch. 3 Greeks; theta is per year, "
        "vega and rho per unit (not per 1%)."
    )

# ------------------------------------------------------------------ tab 2: smile
with tab_smile:
    st.subheader("Lévy prices → implied-vol smile (Ch. 5 → Ch. 6 → Ch. 4)")
    st.markdown(
        "Calls are priced by the **COS method** from the model's characteristic "
        "function, then each price is inverted through the bracketed **implied-vol "
        "solver**. Black-Scholes cannot see jumps, so it reports them as a smile. "
        "With the GBM characteristic function the smile is flat — that round trip "
        "is one of the library's cross-module consistency tests."
    )
    left, right = st.columns([1, 2])
    with left:
        model_name = st.selectbox("Model", ["Merton", "Variance Gamma", "CGMY", "GBM (control)"])
        sm_r = st.number_input("Rate r", -0.05, 0.25, 0.05, 0.005, key="sm_r", format="%.3f")
        sm_t = st.number_input("Maturity T", 0.1, 5.0, 1.0, 0.1, key="sm_t")
        sm_n = int(st.select_slider("COS terms N", [64, 128, 256, 512], value=256))
        mny_lo, mny_hi = st.slider("Strike range (moneyness K/S₀)", 0.5, 1.6, (0.7, 1.3), 0.05)

        model: CharacteristicModel
        try:
            if model_name == "Merton":
                m_sig = st.slider("diffusion σ", 0.01, 0.5, 0.15, 0.01)
                m_lam = st.slider("jump intensity λ", 0.0, 3.0, 0.75, 0.05)
                m_mu = st.slider("mean log-jump μⱼ", -0.5, 0.5, -0.10, 0.01)
                m_sj = st.slider("jump vol σⱼ", 0.0, 0.5, 0.15, 0.01)
                model = Merton(r=sm_r, q=0.0, sigma=m_sig, lam=m_lam, mu_j=m_mu, sigma_j=m_sj)
            elif model_name == "Variance Gamma":
                v_sig = st.slider("σ", 0.01, 0.5, 0.12, 0.01)
                v_th = st.slider("θ (skew)", -0.5, 0.5, -0.14, 0.01)
                v_be = st.slider("β (variance rate)", 0.05, 1.0, 0.20, 0.05)
                model = VarianceGamma(r=sm_r, q=0.0, sigma=v_sig, theta=v_th, beta=v_be)
            elif model_name == "CGMY":
                c_c = st.slider("C", 0.1, 3.0, 1.0, 0.1)
                c_g = st.slider("G", 1.0, 15.0, 5.0, 0.5)
                c_m = st.slider("M (needs M > 1)", 1.5, 15.0, 5.0, 0.5)
                c_y = st.slider("Y (Y = 1 excluded: Γ(−Y) pole)", 0.1, 1.9, 0.5, 0.05)
                model = CGMY(r=sm_r, q=0.0, C=c_c, G=c_g, M=c_m, Y=c_y)
            else:
                g_sig = st.slider("σ", 0.01, 0.5, 0.20, 0.01)
                model = GBM(r=sm_r, q=0.0, sigma=g_sig)
        except ValueError as exc:
            st.error(f"Invalid parameters: {exc}")
            st.stop()

    strikes, cos_prices, ivs = cached_smile(
        model, 100.0, 100.0 * mny_lo, 100.0 * mny_hi, 41, sm_r, sm_t, 0.0, sm_n
    )
    n_bad = int(np.sum(np.isnan(ivs)))
    with right:
        fig, ax = new_axes()
        ax.plot(strikes / 100.0, ivs, "o-", ms=3, label="implied vol")
        atm_iv = float(ivs[int(np.argmin(np.abs(strikes - 100.0)))])
        if np.isfinite(atm_iv):
            ax.axhline(atm_iv, color="grey", ls="--", lw=0.8, label=f"ATM = {atm_iv:.4f}")
        ax.set_xlabel("moneyness K / S₀")
        ax.set_ylabel("Black-Scholes implied σ")
        ax.legend()
        st.pyplot(fig, width=700)
        if n_bad:
            st.warning(
                f"{n_bad} wing strike(s) fell below the COS error floor and were "
                "dropped (shown as gaps) — raise N or narrow the strike range."
            )
        with st.expander("COS prices and inverted vols"):
            st.dataframe(
                {
                    "strike": [float(k) for k in strikes],
                    "COS call price": [float(p) for p in cos_prices],
                    "implied vol": [float(v) for v in ivs],
                },
                height=280,
            )

# ------------------------------------------------------------------ tab 3: density
with tab_density:
    st.subheader("Recovering the risk-neutral density (Ch. 4 & Ch. 6, p. 169 exercise)")
    st.markdown(
        "Under GBM the terminal density is known in closed form, which makes it a "
        "clean benchmark: **Breeden-Litzenberger** differentiates call prices twice "
        "in strike (an O(h²) finite difference), while **COS** reconstructs the "
        "density from the characteristic function (spectral accuracy)."
    )
    d1, d2, d3, d4 = st.columns(4)
    de_sigma = d1.number_input("σ", 0.05, 0.8, 0.2, 0.01, key="de_s")
    de_t = d2.number_input("T", 0.1, 5.0, 1.0, 0.1, key="de_t")
    de_nk = int(d3.select_slider("strike grid points", [51, 101, 201, 401], value=201))
    de_ncos = int(d4.select_slider("COS terms N", [64, 128, 256], value=256, key="de_n"))

    cmp_ = cached_density(100.0, 0.05, de_sigma, de_t, 0.0, de_nk, de_ncos)
    fig, ax = new_axes()
    ax.plot(cmp_.strikes, cmp_.exact, "k-", lw=1.2, label="exact lognormal")
    ax.plot(cmp_.strikes, cmp_.breeden_litzenberger, ".", ms=3, label="Breeden-Litzenberger")
    ax.plot(cmp_.strikes, cmp_.cos, "--", label="COS")
    ax.set_xlabel("terminal price s")
    ax.set_ylabel("density f(s)")
    ax.legend()
    st.pyplot(fig, width=700)
    m1, m2 = st.columns(2)
    m1.metric("max |BL − exact|", f"{cmp_.max_err_bl:.2e}")
    m2.metric("max |COS − exact|", f"{cmp_.max_err_cos:.2e}")
    st.caption(
        "Halve the strike spacing and the BL error drops ~4x (second-order "
        "central difference) — the test suite asserts exactly that ratio. The "
        "COS error is already at numerical noise for N ≥ 128."
    )

# ------------------------------------------------------------------ tab 4: MC
with tab_mc:
    st.subheader("GBM path simulation (Ch. 2)")
    p1, p2, p3, p4 = st.columns(4)
    mc_mu = p1.number_input("drift μ", -0.2, 0.4, 0.08, 0.01)
    mc_sigma = p1.number_input("σ", 0.05, 0.8, 0.2, 0.01, key="mc_s")
    mc_t = p2.number_input("T", 0.1, 5.0, 1.0, 0.1, key="mc_t")
    mc_scheme = as_scheme(str(p2.selectbox("scheme", ["exact", "euler"])))
    mc_npaths = int(p3.slider("paths", 10, 2000, 400, 10))
    mc_nsteps = int(p3.slider("steps", 10, 500, 250, 10))
    mc_seed = int(p4.number_input("seed", 0, 10_000, 42, 1))

    times, paths = cached_paths(
        100.0, mc_mu, mc_sigma, mc_t, mc_npaths, mc_nsteps, mc_scheme, mc_seed
    )
    show = min(mc_npaths, 100)
    fig, ax = new_axes()
    ax.plot(times, paths[:show].T, lw=0.5, alpha=0.35)
    ax.set_xlabel("t")
    ax.set_ylabel("S(t)")
    ax.set_title(f"first {show} of {mc_npaths} paths — seed {mc_seed} (deterministic)")
    st.pyplot(fig, width=700)

    terminal = paths[:, -1]
    grid = np.linspace(float(np.min(terminal)) * 0.9, float(np.max(terminal)) * 1.05, 300)
    fig2, ax2 = new_axes()
    ax2.hist(terminal, bins=40, density=True, alpha=0.45, label="simulated S(T)")
    ax2.plot(
        grid, lognormal_terminal_pdf(grid, 100.0, mc_mu, mc_sigma, mc_t), "k-", label="exact pdf"
    )
    ax2.set_xlabel("S(T)")
    ax2.set_ylabel("density")
    ax2.legend()
    st.pyplot(fig2, width=700)

    exact_mean = 100.0 * float(np.exp(mc_mu * mc_t))
    s1, s2 = st.columns(2)
    s1.metric("sample mean S(T)", f"{float(np.mean(terminal)):,.3f}", border=True)
    s2.metric("exact E[S(T)] = S₀·exp(μT)", f"{exact_mean:,.3f}", border=True)
    st.caption(
        "The exact scheme samples the lognormal transition law directly (no "
        "discretisation error at any step count); Euler discretises the SDE and "
        "carries O(Δt) weak error — compare the two at low step counts."
    )
