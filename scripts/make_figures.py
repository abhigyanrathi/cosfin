"""Generate the Chapter 5-6 replication figures as PNGs.

Run from the repo root with the package (and the ``plots`` extra) installed:
``python scripts/make_figures.py [output_dir]``. Default output: ``figures/``
(git-ignored; the PNGs are fully reproducible from the seeded harnesses in
``pyfinlib_practice.replications``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm, norm

from pyfinlib_practice.numerical.cos import cos_density
from pyfinlib_practice.replications import (
    FIG_5_7_PARAMS,
    figure_5_7_data,
    figure_5_8_data,
    figure_6_4_data,
)


def fig_5_7(out: Path) -> None:
    data = figure_5_7_data()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, n_steps in zip(axes, (10, 2000), strict=True):
        ax.hist(data[n_steps], bins=60, range=(-0.6, 0.1), color="0.35", edgecolor="black")
        ax.set_title(f"P&L at time T, {n_steps} time steps")
        ax.set_xlabel("P&L(T)")
    fig.suptitle(
        "Figure 5.7 replication: hedging frequency vs P&L under Merton jumps "
        "(parameters ASSUMED; see provenance)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out / "fig_5_7.png", dpi=150)
    plt.close(fig)


def fig_5_8(out: Path) -> None:
    ks, panels = figure_5_8_data()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    titles = {"c": "Effect of C", "g": "Effect of G", "m": "Effect of M", "y": "Effect of Y"}
    for ax, name in zip(axes.ravel(), ("c", "g", "m", "y"), strict=True):
        for value, ivs in panels[name].items():
            ax.plot(ks, 100.0 * ivs, marker="o", ms=3, label=f"{name.upper()}={value:g}")
        ax.set_title(f"{titles[name]} on implied volatility")
        ax.set_xlabel("K")
        ax.set_ylabel("implied volatility [%]")
        ax.legend(fontsize=8)
    fig.suptitle(
        "Figure 5.8 replication: CGMYB implied vols "
        "(base C=1, G=1, M=5, Y=0.5, sigma=0.2, r=0.1; S0=100, T=1 assumed)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out / "fig_5_8.png", dpi=150)
    plt.close(fig)


def fig_6_4(out: Path) -> None:
    x, curves, refs = figure_6_4_data()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, y in zip(axes, (0.5, 1.5), strict=True):
        for n, f in curves[y].items():
            ax.plot(x, f, lw=1.0, label=f"N={n}")
        ax.plot(x, refs[y], "k--", lw=1.0, label="N=512 ref")
        ax.set_title(f"CGMYB density recovery, Y={y}")
        ax.set_xlabel("x")
        ax.legend(fontsize=8)
    fig.suptitle("Figure 6.4 replication: COS density recovery (domain [-2,2] assumed)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_6_4.png", dpi=150)
    plt.close(fig)


def density_recovery(out: Path) -> None:
    def cf(u: object, t: float) -> np.ndarray:
        uc = np.asarray(u, dtype=np.complex128)
        return np.exp(-0.5 * uc**2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    y = np.linspace(-5.0, 5.0, 801)
    for n in (4, 8, 16):
        axes[0].plot(y, cos_density(cf, y, -10.0, 10.0, n, 1.0), lw=1.0, label=f"N={n}")
    axes[0].plot(y, norm.pdf(y), "k--", lw=1.0, label="exact")
    axes[0].set_title("Standard normal recovery")
    axes[0].legend(fontsize=8)
    x = np.linspace(0.05, 5.0, 500)
    for n in (8, 16, 32):
        axes[1].plot(x, cos_density(cf, np.log(x), -10.0, 10.0, n, 1.0) / x, lw=1.0, label=f"N={n}")
    axes[1].plot(x, lognorm.pdf(x, 1.0), "k--", lw=1.0, label="exact")
    axes[1].set_title("Lognormal recovery (change of variables)")
    axes[1].legend(fontsize=8)
    fig.suptitle("p.169 examples replication: COS density recovery for increasing N", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "density_recovery.png", dpi=150)
    plt.close(fig)


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("figures")
    out.mkdir(parents=True, exist_ok=True)
    fig_5_7(out)
    fig_5_8(out)
    fig_6_4(out)
    density_recovery(out)
    print(f"wrote 4 figures to {out}/  (fig 5.7 params: {FIG_5_7_PARAMS})")


if __name__ == "__main__":
    main()
