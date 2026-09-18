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

import cosfin

MODULES = [
    "cosfin",
    "cosfin.core",
    "cosfin.core.brownian",
    "cosfin.core.gbm",
    "cosfin.core.jump_diffusion",
    "cosfin.models",
    "cosfin.models.black_scholes",
    "cosfin.models.characteristic_functions",
    "cosfin.models.jump_diffusion",
    "cosfin.models.local_vol",
    "cosfin.numerical",
    "cosfin.numerical.cos",
    "cosfin.numerical.monte_carlo",
    "cosfin.pricing",
    "cosfin.replications",
    "cosfin.pricing.greeks",
    "cosfin.pricing.hedging",
    "cosfin.pricing.payoffs",
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
    _cold_import("import cosfin.pricing; import cosfin.models")


def test_cold_import_models_then_pricing() -> None:
    _cold_import("import cosfin.models; import cosfin.pricing")


def test_cold_import_local_vol_directly() -> None:
    _cold_import("from cosfin.models.local_vol import implied_volatility")


def test_version() -> None:
    assert cosfin.__version__ == "0.3.0"


@pytest.mark.parametrize(
    "subpackage", ["core", "models", "numerical", "pricing"]
)
def test_all_exports_resolve(subpackage: str) -> None:
    import importlib

    module = importlib.import_module(f"cosfin.{subpackage}")
    for name in module.__all__:
        assert getattr(module, name) is not None


def test_vega_reexport_is_canonical_object() -> None:
    from cosfin.models import black_scholes
    from cosfin.pricing import greeks

    assert greeks.vega is black_scholes.vega
