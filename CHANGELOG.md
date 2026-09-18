# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Streamlit demo app (`app/streamlit_app.py`) with its compute layer
  `pyfinlib_practice.demo` and test coverage, deployed via a Render
  Blueprint (`render.yaml`); `.python-version` pin added (`998cf1f`).
- Demo notebook `notebooks/pyfinlib_practice_demo.ipynb` (`15c8e89`).
- `docs/RESULTS.md`, a full reproduction transcript for the Chapter 5-6
  convergence tables, plus a CI badge in the README (`260afaa`).
- Project instructions (`CLAUDE.md`), task specifications under
  `docs/tasks/`, a decisions log (`docs/decisions.md`), and permission
  guardrails in `.claude/settings.json` (`8af59c4`).

### Changed
- Table 6.7 (CGMYB, Y=1.5) documented as platform-sensitive rather than
  claiming a single cross-platform figure (`260afaa`).
- README revised twice for clarity on project purpose and scope
  (`a4b45ef`, `0c56264`); Streamlit sidebar wording adjusted (`553d6a1`).
- Package renamed `pyfinlib_practice` → `cosfin`; GitHub repo renamed
  `pyfinlib-practice` → `cosfin` to follow. See `docs/tasks/task_1b.md`.

### Fixed
- Ruff violations in the demo notebook that were blocking CI (`6f92dd5`).
- Two README precision claims in the Validation section corrected to the
  values actually measured, and reworded to note platform sensitivity: the
  COS(GBM) max error (5.5e-14 to 5.9e-14) and the martingale probe bound
  (2e-16, which sits below float64 machine epsilon, to 2.3e-16) (`abf9df0`).

## [0.2.0] - 2026-07-12

Full Chapter 2-6 rebuild on top of the 0.1.0 Ch2-4 tree (`ce56441`).

### Added
- Fourier-cosine (COS) pricing engine: European and digital pricers with
  per-strike truncation intervals, fully vectorised, plus log-return
  density recovery and the cumulant truncation rule.
- Levy characteristic functions as frozen dataclasses behind a structural
  `Protocol`: GBM, Merton, Variance Gamma and CGMY.
- Merton jump-diffusion analytic series pricer and path simulation; a
  discrete delta-hedging experiment (`pricing/hedging.py`).
- `pyfinlib_practice.replications` harness for Tables 6.4-6.8, Figures
  5.7/5.8/6.4 and Exercise 6.3, with per-table parameter provenance
  tracked as CONFIRMED or ASSUMED.
- `scripts/reproduce_tables.py` and `scripts/make_figures.py`.

### Fixed
- `models` to `pricing` circular import: the canonical `vega` now lives in
  `models.black_scholes`, where the implied-volatility solvers need it, and
  `pricing.greeks` re-exports the same object. The models layer no longer
  imports from pricing in any direction.
- `digital_price` returned silently wrong values where `sigma * sqrt(t)`
  was zero. `d1_d2` now returns the one-sided signed-infinity limits, so
  every consumer collapses to the correct discounted-indicator limit by
  construction rather than requiring each caller to guard the branch.
- `butterfly_price` silently priced with an invalid, non-positive lower-leg
  strike when `spread >= strike`: a finite but structurally wrong price
  exactly at the boundary, `nan` strictly beyond it. Now raises
  `ValueError`.

### Changed
- `option_type` is validated at every entry point; the previous
  else-branch accepted arbitrary strings as puts.
- **Breaking:** the `MonteCarloResult` field `std_error` was renamed
  `standard_error`. The NamedTuple itself predates this release.
- Implied volatility gained a unified `implied_volatility()` entry point
  dispatching over the three existing solvers, defaulting to the bracketed
  method. The solvers themselves, their `xtol` convergence rule and the
  Brenner-Subrahmanyam seed all predate this release.
- Removed the empty `utils/` package; `norm_pdf` and `norm_cdf` live in
  `models/black_scholes.py`.

## [0.1.0] - 2026-06-26

Initial Chapter 2-4 build (`4447bd1`): Brownian motion and GBM path
simulation, Monte Carlo pricing, Black-Scholes prices and Greeks, implied
volatility solvers, and Breeden-Litzenberger density recovery. Ships the
CI workflow, LICENSE and packaging metadata.

Never formally tagged in git. Reconstructed from the literal
`version = "0.1.0"` string in `pyproject.toml` at `4447bd1` and `a4b45ef`,
both superseded by the 0.2.0 rebuild.
