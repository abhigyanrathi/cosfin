"""Print every mentor table/exercise item against the published references.

Run from the repo root with the package installed:
``python scripts/reproduce_tables.py``. Companion to ``make_figures.py``
(which renders Figures 5.7/5.8/6.4); everything here is text output.
"""

from __future__ import annotations

from pyfinlib_practice.replications import (
    TABLE_6_4_REF,
    TABLE_6_5_REF,
    TABLE_6_7_REF,
    TABLE_6_8_REF,
    exercise_6_3,
    lognormal_density_recovery,
    normal_density_recovery,
    table_6_4,
    table_6_5,
    table_6_6,
    table_6_7,
    table_6_8,
)

BOOK_6_4 = {
    80.0: {16: 3.67, 32: 7.44e-1, 64: 1.92e-2, 128: 1.31e-7, 256: 5.68e-14},
    100.0: {16: 3.87, 32: 5.28e-1, 64: 1.52e-2, 128: 3.87e-7, 256: 1.44e-13},
    120.0: {16: 3.17, 32: 8.13e-1, 64: 2.14e-2, 128: 3.50e-7, 256: 1.26e-13},
}
BOOK_6_5 = {40: -2.26e-1, 60: 2.60e-3, 80: 3.59e-5, 100: -4.85e-7, 120: 1.29e-9, 140: -9.82e-13}


def report_table_6_4() -> None:
    print("Table 6.4  GBM call (eq 6.51: S0=100, r=0.10, sigma=0.25, T=0.1)")
    print("  NOTE: printed error columns imply a much wider eq-(6.45) range than the")
    print("  cumulant rule; both variants here are more accurate (see README).")
    rows = table_6_4()
    for strike, series in rows.items():
        print(f"  K={strike:g}  ref {TABLE_6_4_REF[strike]}")
        print(f"    {'N':>4}  {'price':>16}  {'|err| ours':>12}  {'|err| book':>12}")
        for row in series:
            print(
                f"    {row.n:>4}  {row.price:>16.10f}  {row.error:>12.2e}"
                f"  {BOOK_6_4[strike][row.n]:>12.2e}"
            )


def report_table_6_5() -> None:
    print("\nTable 6.5  cash-or-nothing call (eq 6.52; pays K -- Example 6.3.1)")
    print(f"  ref {TABLE_6_5_REF}   (signed errors, as printed)")
    print(f"    {'N':>4}  {'price':>16}  {'err ours':>12}  {'err book':>12}")
    for row in table_6_5():
        print(
            f"    {row.n:>4}  {row.price:>16.12f}"
            f"  {row.error:>+12.2e}  {BOOK_6_5[row.n]:>+12.2e}"
        )


def report_table_6_6() -> None:
    print("\nTable 6.6  CGMYB density recovery, max |f_N - f_ref| on [-2,2]")
    print("  (printed values PENDING that page; measured vs the N=512 reference)")
    for y, series in table_6_6().items():
        line = "  ".join(f"N={n}: {err:.2e}" for n, err in series)
        print(f"  Y={y}:  {line}")


def report_table_6_7() -> None:
    print("\nTable 6.7  CGMYB call (C=1, G=5, M=5, sigma=0.2, S0=K=100, r=0.1, T=1)")
    for y, series in table_6_7().items():
        print(f"  Y={y}  ref {TABLE_6_7_REF[y]}")
        for row in series:
            print(f"    N={row.n:>4}  {row.price:>16.10f}  |err| {row.error:.2e}")


def report_table_6_8() -> None:
    print("\nTable 6.8  VG call (sigma=0.12, theta=-0.14, beta=0.2, S0=100, K=90, r=0.1)")
    for t, series in table_6_8().items():
        print(f"  T={t}  ref {TABLE_6_8_REF[t]}")
        for row in series:
            print(f"    N={row.n:>4}  {row.price:>16.10f}  |err| {row.error:.2e}")


def report_exercise_6_3() -> None:
    print("\nExercise 6.3  Merton: COS (characteristic function) vs analytic series")
    print(f"  {'K':>6}  {'COS':>16}  {'series':>16}  {'abs diff':>10}")
    for row in exercise_6_3():
        print(
            f"  {row.strike:>6g}  {row.cos_price:>16.10f}"
            f"  {row.series_price:>16.10f}  {row.abs_diff:>10.2e}"
        )


def report_density_recovery() -> None:
    print("\nItem 3  density recovery on [-10,10] (p.169 examples), max errors")
    normal = "  ".join(f"N={n}: {e:.2e}" for n, e in normal_density_recovery())
    logn = "  ".join(f"N={n}: {e:.2e}" for n, e in lognormal_density_recovery())
    print(f"  normal:    {normal}")
    print(f"  lognormal: {logn}")


def main() -> None:
    report_table_6_4()
    report_table_6_5()
    report_table_6_6()
    report_table_6_7()
    report_table_6_8()
    report_exercise_6_3()
    report_density_recovery()


if __name__ == "__main__":
    main()
