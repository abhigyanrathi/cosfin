# pyfinlib (practice)

A quantitative finance library implementing models and numerical methods from
Oosterlee & Grzelak, *Mathematical Modelling and Computation in Finance*
(Chapters 1–8).

> **About this repository.** This is a **practice** repo. It
> is used to rehearse packaging, tooling, and Git workflow.

## Layout

```
src/pyfinlib/
    core/        stochastic processes — Brownian motion, GBM
    models/      asset price models — Black–Scholes, local volatility
    pricing/     payoffs and Greeks
    numerical/   Monte Carlo, PDE/FDM, Fourier methods
    utils/       shared helpers
tests/           pytest suite (mirrors src/ structure)
docs/            derivations and implementation notes
notebooks/       demonstration notebooks
```

## Install (development)

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Checks

```powershell
ruff check .
mypy
pytest
```

