# pyfinlib-practice

[![CI](https://github.com/abhigyanrathi/pyfinlib-practice/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/abhigyanrathi/pyfinlib-practice/actions/workflows/ci.yml)

Reference ("answer-key") implementations of models and numerical methods
from Oosterlee & Grzelak, *Mathematical Modelling and Computation in
Finance*, Chapters 2–6, plus the mentor-confirmed Chapter 5–6 replication
items. The production library is written independently in a separate
repository; this repo is the AI-written learning companion, used to
rehearse the maths, packaging, tooling, and Git workflow.

Provenance: the v0.2.0 tree was authored by Claude (July 2026), replacing
the v0.1.0 Ch2–4 tree and the never-pushed June Ch5–6 work; it fixes the
v0.1.0 circular import and the degenerate-digital bug, and anchors the COS
pricers to the published O&G table values.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest      # 249 tests, ~15-30 s
.venv/bin/python -m pytest --cov=pyfinlib_practice --cov-report=term-missing
.venv/bin/ruff check src tests
.venv/bin/mypy                  # strict, src + tests, 30 files
```

Requires Python ≥ 3.12, numpy ≥ 2.0, scipy ≥ 1.13.

## Scope

- **Ch2** — Brownian and GBM path simulation (exact and Euler schemes),
  closed-form terminal moments.
- **Ch3** — Black-Scholes calls/puts, cash-or-nothing digitals, butterfly
  spreads, `d1`/`d2`, put-call parity residual, canonical vega.
- **Ch4** — Implied volatility (Newton, secant, bracketed; the mentor list
  of §4.1) and Breeden-Litzenberger density recovery (§4.2). Dupire/SABR
  (§4.3) is out of scope per the confirmed list.
- **Ch5** — Merton jump-diffusion: analytic series pricer, path simulation,
  discrete delta-hedging experiment; characteristic functions for GBM,
  Merton, Variance Gamma and CGMY.
- **Ch6** — COS method: European and digital pricers (per-strike truncation,
  fully vectorised), log-return density recovery, cumulant truncation rule.
- **Replications** (`pyfinlib_practice.replications`) — the mentor-confirmed
  Chapter 5-6 items: Tables 6.4/6.5/6.6/6.7/6.8, Figures 5.7/5.8/6.4,
  Exercise 6.3, and the p.169 normal/lognormal density-recovery examples.
  Data-returning harnesses; `scripts/make_figures.py` renders the PNGs.

## Dependency DAG

```
core  <-  models  <-  pricing
numerical --> pricing.payoffs (leaf) only
```

`models` never imports `pricing` or `numerical`. The canonical `vega` lives
in `models.black_scholes` because the implied-volatility solvers need it;
`pricing.greeks` re-exports the *same object* (asserted by identity in the
test suite). Cold-import isolation is enforced by running every module in a
fresh interpreter, in both `pricing`↔`models` orders — an in-process import
test cannot catch a masked circular import.

## Conventions

- Docstrings tag claims as **RESULT** (provable), **CONVENTION** (a choice
  fixed for consistency) or **MODELLING CHOICE** (a design decision with
  alternatives).
- `ArrayLike` in, `NDArray[float64]` out; scalar inputs broadcast.
- All randomness flows through an injected `numpy.random.Generator`.
- Iterative solvers converge on the **volatility step** (`xtol`), never on
  the price residual — a residual criterion silently accepts wrong sigmas
  where vega is small.
- Degenerate `sigma*sqrt(t) = 0` is handled via signed-infinity `d1`/`d2`
  (one-sided limits), so prices, digitals, delta, theta and rho collapse to
  the correct forward-intrinsic limits with no special-casing; gamma alone
  is guarded to 0 (its true limit is a Dirac mass).
- Digital payoffs are strict at `S_T = K` and priced per unit cash.
- Theta is calendar decay `-dV/dT` per year.
- Docstring style: prose (math-first) for pricing/model functions;
  numpy-style Parameters sections for the many-argument simulators.

## Validation

All three gates green: **249 pytest**, **ruff clean**, **mypy --strict
clean** across 30 files; line coverage **100%** (one documented
`pragma: no cover` on a provably unreachable defensive guard). Anchor policy: external published values where
available, cross-method agreement everywhere, and no invented table values.
Full reproduction transcript with convergence tables: `docs/RESULTS.md`
(regenerate with `python scripts/reproduce_tables.py`).

- Black-Scholes vs the published ATM value 10.4506 and Hull's S=42 example
  (4.76 / 0.81), plus full-precision internal regressions.
- COS(GBM) vs the closed form: max error 5.5e-14 at N=256; digitals 2.5e-16.
- Merton COS vs the independent Chapter-5 series: 4.8e-13 across 5 strikes —
  two unrelated algorithms, one model.
- Pure-jump CGMY (C,G,M,Y) = (1,5,5,0.5), S=K=100, r=0.1, T=1 vs the
  Fang–Oosterlee (2008) reference **19.812948843118**: reproduced to
  **5.7e-9 untuned** — a genuinely external anchor.
- Martingale probes `cf(-i, t) = e^{(r-q)t}` at complex argument: ≤ 2e-16
  for all four models; cumulants vs finite differences of `log cf`: ≤ 5e-9.
- Breeden-Litzenberger on a Black-Scholes surface: density 1.1e-8 pointwise
  vs the lognormal, total mass 1 − 6e-12, mean = forward to 2.5e-9,
  repricing error ≤ 2e-9.
- Hedging: GBM P&L-dispersion ratio 3.82 for 13 vs 208 steps (theory 4) and
  1.93 for 13 vs 52 (theory 2); under Merton, dispersion is 8.8× GBM at 208
  steps and mean P&L ≈ −2.46 — the seller who charges the diffusion value
  10.45 for an option truly worth 12.76 loses the difference on average.
  That is the Chapter-5 incompleteness message rendered as a test.
- **O&G table anchors** (printed values from supplied pages): Table 6.4
  (eq 6.51) Black-Scholes references reproduced to 4e-10 and COS to 1.8e-14
  by N=64; Table 6.5 reference 0.273306496 reproduced to 5e-10 once the
  Example-6.3.1 pays-K cash convention is applied — this closes the
  long-standing reproduction failure; Table 6.7 (CGMYB, C=1,G=5,M=5,σ=0.2)
  21.679593920 / 50.279533980 reproduced to 1.8e-9 / ≤5.9e-8 — the Y=1.5
  heavy-tail sum is platform-sensitive at ~2e-8 (3.5e-8 Linux, 5.9e-8
  Windows; see `docs/RESULTS.md`); Table 6.8 (VG)
  10.993703187 / 19.099354724 reproduced to 8.5e-9 at the book's N=4096.

**Remaining anchor gaps** (everything else above is anchored to printed
values): Table 6.6's printed numbers and the p.169 printed values (if
tabulated) await those pages; the eq (6.45) simplified-range definition is
needed only to reproduce the book's *error columns* digit-for-digit (a
shared-interval variant was tested and also does not match — the book's
range is evidently much wider than the cumulant rule); Figure 5.7's parameter set and
Figure 5.8's S0/T are not printed in the book and stay ASSUMED by
construction.

## v0.2.0 design changes vs v0.1.0 (the Ch2–4 tree)

1. Canonical `vega` in `models.black_scholes`, re-exported by
   `pricing.greeks` (DAG-correct; kills the v0.1.0 models↔pricing circular
   import).
2. Signed-infinity `d1`/`d2` for `sigma*sqrt(t) = 0` — eliminates the
   digital-at-`sigma=0` bug class by construction.
3. Butterfly guard also rejects `spread >= strike` (main only guards
   `spread <= 0`).
4. Implied vol: `xtol` on the volatility step, Brenner-Subrahmanyam seed,
   safeguarded Newton-in-bracket as the default method; the open-interval
   bounds check rejects deep-ITM quotes that are FP-indistinguishable from
   intrinsic (implied vol is information-theoretically unrecoverable there —
   documented and tested rather than silently mispriced).
5. Breeden-Litzenberger uses the non-uniform central second difference and
   returns the interior grid explicitly.
6. Characteristic functions are frozen dataclasses behind a structural
   `Protocol`; `cf` deliberately accepts complex `u` so the martingale
   identity is testable; CGMY reference powers are routed through the same
   complex-pow path as the `u`-dependent terms so `cf(0) == 1` exactly.
7. Merton series uses the exact identity `ln(1+kappa) = mu_j + sigma_j^2/2`,
   recursive Poisson weights (no factorials), and mode-aware truncation.
8. COS is fully vectorised over strikes with an exact per-strike truncation
   interval (2-D broadcast), not a shared-interval approximation.
9. Monte Carlo returns a `MonteCarloResult` NamedTuple
   (price, standard_error, n_paths) and is composable over any simulator.
10. Hedging streams in O(n_paths) memory with exact dividend accrual
    `e^{q dt}` reinvested in shares, and exposes `sigma_hedge` for the
    wrong-vol experiment.
11. `gbm_terminal_moments` helper; `option_type` is validated at every
    entry point (the v0.1.0 else-branch accepted arbitrary strings as puts).

## Known issues and open anchors

- **Resolved:** Table 6.5 (0.273306496) — eq (6.52) supplied the parameters
  (S0=100, K=120, r=0.05, T=0.1, σ=0.2) and Example 6.3.1 fixed the payoff
  convention (the cash-or-nothing pays K). Anchored in
  `tests/test_replications.py`.
- Table 6.6 printed values and any p.169 printed table: pages pending;
  harnesses measure against high-N self-references meanwhile.
- eq (6.45) simplified integration range: page pending. Tested and
  rejected: neither the per-strike interval nor a shared cumulant-rule
  interval reproduces the printed low-N error columns (book: 3.67 at N=16;
  ours: <9e-3), so the book's range must be materially wider. Reference
  values anchor regardless; the page is needed only for digit-exact error
  columns.
- Figure 5.7 parameters and Figure 5.8 S0/T are not printed in the book;
  the ASSUMED choices are recorded in `PARAMETER_PROVENANCE`.
