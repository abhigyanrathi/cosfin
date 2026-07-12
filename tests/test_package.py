"""Package-level tests: cold-import isolation, export integrity, versioning.

The cold-import matrix runs every module in its own interpreter, in both
historical import orders. Rationale (a hard-won lesson from this repo's own
history): a shared test process caches partially-initialised modules and can
mask circular imports entirely — the models <-> pricing cycle that shipped
in v0.1.0 was invisible to any in-process import test.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import pyfinlib_practice

MODULES = [
    "pyfinlib_practice",
    "pyfinlib_practice.core",
    "pyfinlib_practice.core.brownian",
    "pyfinlib_practice.core.gbm",
    "pyfinlib_practice.core.jump_diffusion",
    "pyfinlib_practice.models",
    "pyfinlib_practice.models.black_scholes",
    "pyfinlib_practice.models.characteristic_functions",
    "pyfinlib_practice.models.jump_diffusion",
    "pyfinlib_practice.models.local_vol",
    "pyfinlib_practice.numerical",
    "pyfinlib_practice.numerical.cos",
    "pyfinlib_practice.numerical.monte_carlo",
    "pyfinlib_practice.pricing",
    "pyfinlib_practice.replications",
    "pyfinlib_practice.pricing.greeks",
    "pyfinlib_practice.pricing.hedging",
    "pyfinlib_practice.pricing.payoffs",
]


def _cold_import(statement: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", statement],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("module", MODULES)
def test_cold_import_each_module(module: str) -> None:
    _cold_import(f"import {module}")


def test_cold_import_pricing_then_models() -> None:
    # The exact order that tripped the latent v0.1.0 cycle.
    _cold_import("import pyfinlib_practice.pricing; import pyfinlib_practice.models")


def test_cold_import_models_then_pricing() -> None:
    _cold_import("import pyfinlib_practice.models; import pyfinlib_practice.pricing")


def test_cold_import_local_vol_directly() -> None:
    _cold_import("from pyfinlib_practice.models.local_vol import implied_volatility")


def test_version() -> None:
    assert pyfinlib_practice.__version__ == "0.2.0"


@pytest.mark.parametrize(
    "subpackage", ["core", "models", "numerical", "pricing"]
)
def test_all_exports_resolve(subpackage: str) -> None:
    import importlib

    module = importlib.import_module(f"pyfinlib_practice.{subpackage}")
    for name in module.__all__:
        assert getattr(module, name) is not None


def test_vega_reexport_is_canonical_object() -> None:
    from pyfinlib_practice.models import black_scholes
    from pyfinlib_practice.pricing import greeks

    assert greeks.vega is black_scholes.vega
