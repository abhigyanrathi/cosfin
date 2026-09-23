# Decisions log

Notable decisions made during development, and why.

---

## Package naming

Renamed the package and repo from `pyfinlib` to `cosfin`. `pyfinlib` was
already taken on PyPI by an active, unrelated Rust-backed quant library
(`sarsoo/finlib`).

## Test-count drift in the README

The README claimed 249 tests across 30 files. The actual numbers, verified
by running the suite directly, are 264 tests across 33 files. The README
was corrected to match. This drift -- a documentation claim silently going
stale while the code moved on -- became one of the two anchor bugs for
`numeric-claim`, the companion validation tool: a check that catches when a
published number and the underlying reality have quietly diverged.

## CGMY error floor as a validation anchor

The pure-jump CGMY case (Table 6.7, Y=1.5) reproduces the Fang-Oosterlee
(2008) published reference to roughly 2.4e-8, against a machine-precision
prediction near 1e-13. That gap is real and platform-sensitive rather than
a bug -- it's the second anchor case for `numeric-claim`.

## Two precision claims in the README were tighter than actually measured

Two lines in the Validation section overstated precision: the COS(GBM) max
error was written as 5.5e-14 but measures 5.8e-14 to 5.9e-14 depending on
platform, and the martingale-probe bound was written as under 2e-16, which
sits below float64 machine epsilon (2.22e-16) and can't actually hold as
stated. Both were rewritten to state the real, platform-qualified numbers,
matching how Table 6.7's platform sensitivity is already documented
elsewhere in the same section.

## digital_price at zero volatility

`digital_price` (and everything built on `d1_d2`) returned silently wrong
values whenever `sigma * sqrt(t)` was exactly zero, because the formula
divides by that term. Fixed by having `d1_d2` return the correct one-sided
signed-infinity limit in that case, so every downstream consumer -- price,
delta, digital, vega -- collapses to the right degenerate value
automatically instead of needing a separate zero-volatility branch in each
caller.

## butterfly_price at the spread boundary

`butterfly_price` was worse than a simple edge case: for `spread > strike`
it returned `nan`, but exactly at `spread == strike` it returned a finite,
plausible-looking, wrong price, because the division landed on exactly
zero rather than a small number. Now raises `ValueError` for
`spread >= strike`.

## MonteCarloResult field rename

Renamed the `std_error` field to `standard_error` for clarity. This is a
breaking change to the public interface.

## Provenance check against Grzelak's reference code

Since Oosterlee & Grzelak's co-author publishes reference code under
BSD-3-Clause, I checked whether any of this implementation derives from
his code rather than independently from the book. Summary and method are
in `docs/provenance-audit.md`. Conclusion: no derivation found, no
attribution obligation triggered.