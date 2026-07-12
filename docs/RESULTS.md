# Reproduced results — Oosterlee & Grzelak Chapter 5–6 tables

This document is the formatted transcript of

```bash
python scripts/reproduce_tables.py
```

run against the printed reference values from Oosterlee & Grzelak,
*Mathematical Modelling and Computation in Finance*. Every number below is
regenerable from a clean clone; nothing is hand-entered except the printed
book references themselves.

**Environments.** The committed numbers are from Linux, CPython 3.12.3,
numpy 2.4.4, scipy 1.17.1 (the CI platform). An independent run on
Windows 11 / CPython 3.14 reproduced every table digit-for-digit at the
printed precision with one exception — the Table 6.7 Y=1.5 heavy-tail case —
documented in note 3 below.

**How to read the plateaus.** The references are printed to 8–9 decimal
places, so the best achievable |error| against them is the rounding of the
last printed digit (half-ulp ≈ 5e-10 at 9 dp). Where our error column
plateaus at that level, the book has run out of digits — not the pricer out
of accuracy.

**Book error columns.** Tables 6.4/6.5 print the book's own error columns,
generated with the simplified integration range of eq. (6.45), a page we do
not yet have. Both our per-strike interval and a shared cumulant-rule
interval are materially *more* accurate at low N than the printed columns,
which implies the eq. (6.45) range is much wider than the cumulant rule.
The printed columns are reproduced here verbatim for comparison only.

---

## Table 6.4 — GBM European call (eq. 6.51)

Parameters: S0=100, r=0.10, sigma=0.25, T=0.1.

| K | Ref (printed) | N | Price (COS) | \|err\| ours | \|err\| book |
|---|---|---|---|---|---|
| 80 | 20.799226309 | 16 | 20.7918221679 | 7.40e-03 | 3.67e+00 |
| | | 32 | 20.7992260658 | 2.43e-07 | 7.44e-01 |
| | | 64 | 20.7992263087 | 3.27e-10 | 1.92e-02 |
| | | 128 | 20.7992263087 | 3.27e-10 | 1.31e-07 |
| | | 256 | 20.7992263087 | 3.27e-10 | 5.68e-14 |
| 100 | 3.659968453 | 16 | 3.6590011549 | 9.67e-04 | 3.87e+00 |
| | | 32 | 3.6599683735 | 7.95e-08 | 5.28e-01 |
| | | 64 | 3.6599684533 | 3.25e-10 | 1.52e-02 |
| | | 128 | 3.6599684533 | 3.25e-10 | 3.87e-07 |
| | | 256 | 3.6599684533 | 3.25e-10 | 1.44e-13 |
| 120 | 0.044577814 | 16 | 0.0435108693 | 1.07e-03 | 3.17e+00 |
| | | 32 | 0.0445776644 | 1.50e-07 | 8.13e-01 |
| | | 64 | 0.0445778141 | 7.33e-11 | 2.14e-02 |
| | | 128 | 0.0445778141 | 7.33e-11 | 3.50e-07 |
| | | 256 | 0.0445778141 | 7.33e-11 | 1.26e-13 |

All three strikes converge by N=64 and plateau at the printed-reference
rounding (references are 9 dp).

## Table 6.5 — cash-or-nothing call (eq. 6.52, pays-K convention)

Parameters: S0=100, K=120, r=0.05, T=0.1, sigma=0.2. The printed reference
0.273306496 is reproduced only under the Example 6.3.1 convention that the
digital pays K (not one unit of cash); this closed a long-standing
reproduction failure. Errors are signed, as the book prints them.

| N | Price (COS) | err ours | err book |
|---|---|---|---|
| 40 | 0.273306492092 | -3.91e-09 | -2.26e-01 |
| 60 | 0.273306496497 | +4.97e-10 | +2.60e-03 |
| 80 | 0.273306496497 | +4.97e-10 | +3.59e-05 |
| 100 | 0.273306496497 | +4.97e-10 | -4.85e-07 |
| 120 | 0.273306496497 | +4.97e-10 | +1.29e-09 |
| 140 | 0.273306496497 | +4.97e-10 | -9.82e-13 |

## Table 6.6 — CGMYB density recovery

Max |f_N − f_ref| on [−2, 2]. **The book's printed values for this table
are pending** (page not yet supplied); until then the error is measured
against the model's own N=512 reference, so this is a self-convergence
check, not an external anchor.

| Y | N=4 | N=8 | N=16 | N=32 | N=64 | N=128 |
|---|---|---|---|---|---|---|
| 0.5 | 7.09e-01 | 4.26e-01 | 1.11e-01 | 3.25e-03 | 3.85e-07 | 6.94e-18 |
| 1.5 | 2.06e-01 | 9.46e-02 | 8.55e-03 | 1.98e-06 | 0.00e+00 | 0.00e+00 |

Y=1.5 converges faster than Y=0.5 because the CGMY characteristic function
decays like exp(−c·|u|^Y): larger Y, faster decay, faster cosine-series
convergence. The exact zeros mean N=64 already agrees with the N=512
reference to the last representable bit on the evaluation grid.

## Table 6.7 — CGMYB European call

Parameters: C=1, G=5, M=5, sigma=0.2, S0=K=100, r=0.1, T=1.

| Y | Ref (printed) | N | Price (COS) | \|err\| |
|---|---|---|---|---|
| 0.5 | 21.67959392 | 32 | 21.2946785258 | 3.85e-01 |
| | | 64 | 21.6795795357 | 1.44e-05 |
| | | 128–512 | 21.6795939182 | 1.76e-09 |
| 1.5 | 50.27953398 | 32 | 47.4317737721 | 2.85e+00 |
| | | 64–512 | 50.2795339453 | 3.47e-08 |

The Y=1.5 row is platform-sensitive: Windows/CPython 3.14 converges to
50.2795339215 (|err| 5.85e-08), a cross-platform spread of ~2.4e-08
absolute, ~5e-10 relative. See note 3.

## Table 6.8 — Variance Gamma European call

Parameters: sigma=0.12, theta=−0.14, beta=0.2, S0=100, K=90, r=0.1.

| T | Ref (printed) | N | Price (COS) | \|err\| |
|---|---|---|---|---|
| 0.1 | 10.993703187 | 64 | 10.9896339187 | 4.07e-03 |
| | | 128 | 10.9944001152 | 6.97e-04 |
| | | 256 | 10.9937073720 | 4.19e-06 |
| | | 512 | 10.9936963867 | 6.80e-06 |
| | | 1024 | 10.9937026175 | 5.69e-07 |
| | | 2048 | 10.9937032668 | 7.98e-08 |
| | | 4096 | 10.9937031785 | 8.51e-09 |
| 1.0 | 19.099354724 | 64 | 19.0993434612 | 1.13e-05 |
| | | 128 | 19.0993547171 | 6.86e-09 |
| | | 256–4096 | 19.0993547242 | ~2.0e-10 |

The non-monotone T=0.1 column (error rises from N=256 to N=512 before
falling again) is expected, not a defect: at short maturity the VG density
is not smooth, so the cosine expansion loses spectral convergence and
decays algebraically with oscillation. Demonstrating exactly this is the
point of the book's table; at T=1.0 the density is smooth again and the
plateau is the printed-reference rounding.

## Exercise 6.3 — Merton: COS vs analytic series

Two independent pricers for the same model — the Fourier-cosine engine on
the Merton characteristic function versus the Chapter-5 Poisson-weighted
Black-Scholes series. Agreement to ~1e-13 is cross-method validation, not
a round-trip.

| K | COS | Series | \|diff\| |
|---|---|---|---|
| 80 | 25.9555349170 | 25.9555349170 | 3.13e-13 |
| 90 | 18.6983161868 | 18.6983161868 | 1.14e-13 |
| 100 | 12.7612885936 | 12.7612885936 | 4.01e-13 |
| 110 | 8.2583486989 | 8.2583486989 | 4.78e-13 |
| 120 | 5.0905502904 | 5.0905502904 | 1.99e-13 |

(The |diff| column varies in its last digits across platforms at the
1e-13 level; the prices themselves are digit-identical.)

## p. 169 examples — density recovery on [−10, 10]

Max pointwise error of the COS-recovered density against the analytic
normal and lognormal densities.

| Density | N=4 | N=8 | N=16 | N=32 | N=64 | N=128 |
|---|---|---|---|---|---|---|
| normal | 2.54e-01 | 1.08e-01 | 7.18e-03 | 4.04e-07 | 2.45e-16 | — |
| lognormal | — | 6.75e-01 | 6.36e-02 | 6.47e-06 | 9.99e-16 | 9.99e-16 |

Both reach machine precision — the clean spectral-convergence picture for
smooth densities, and the contrast case for Table 6.8's T=0.1 column.

---

## Notes

1. **Plateaus are the book's rounding, not ours.** Wherever an error column
   flattens near 5e-10 (9-dp references) or 5e-9 (8-dp references), the
   converged price agrees with the printed value to every published digit.
2. **eq. (6.45) is the only missing piece for the error columns.** The
   reference *values* anchor regardless; the page is needed solely to
   reproduce the book's printed low-N error columns digit-for-digit.
3. **Heavy-tail platform sensitivity (Table 6.7, Y=1.5).** With Y=1.5 the
   characteristic function decays slowly, the truncation interval is wide,
   and the cosine sum runs long oscillatory terms through libm's complex
   exponential — so the converged value differs across platforms at ~2e-8
   absolute. The same regime sets the floor for the put-call-parity
   consistency test in `tests/test_ch6_cos.py`, whose tolerance (1e-6) is
   chosen with two orders of headroom over the worst observed
   cross-platform residual. Everything outside this regime reproduces
   digit-identically across Linux and Windows.
4. **Anchor policy.** External printed values wherever supplied,
   cross-method agreement everywhere, self-convergence only where the book
   page is pending (Table 6.6) — and gaps stay gaps until pages arrive.
