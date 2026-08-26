# Provenance audit: `src/pyfinlib_practice/` vs the Grzelak reference repositories

**Status:** report only. No attribution added, no LICENSE modified, no code
rewritten. Findings are for Abhi to act on or not.

**Date:** 2026-08-26
**Auditor:** Claude Code session (AI-assisted, per this repo's standing disclosure)

## Question being answered

Lech Grzelak, co-author of Oosterlee & Grzelak, publishes reference code for
the textbook under **BSD-3-Clause**, which carries an attribution
requirement. If any implementation here derives from *his code* rather than
from the *book text*, an unmet LICENSE/NOTICE obligation exists and must be
resolved before publication or a PyPI release.

Sources examined (cloned at audit time, `--depth 1`):

| Repo | Branch | License | Copyright line |
|---|---|---|---|
| `LechGrzelak/QuantFinanceBook` | `master` | BSD 3-Clause | `Copyright (c) 2024, leszek` |
| `LechGrzelak/Computational-Finance-Course` | `main` | BSD 3-Clause | `Copyright (c) 2024, leszek` |

## The distinction this audit turns on

Implementing a **published textbook formula** is not a BSD-3 concern. Two
correct implementations of the COS method will necessarily share the same
`chi`/`psi` closed forms and the same cosine-series structure, because the
*mathematics* dictates them. The book's equations are not Grzelak's
repository's intellectual property.

The BSD-3 concern is narrower: copying **Grzelak's particular code** — his
non-book-mandated variable names, his function decomposition, his comment
phrasing, his file organisation. Every finding below states explicitly which
of the two any observed similarity falls into.

## Method

1. Both repos cloned in full and read verbatim from disk (not via
   summarising fetch), so naming and comment comparisons are exact.
2. Repo-wide mechanical comparisons run across all files (results below).
3. Line-by-line structural comparison for the highest-risk modules, where
   filename or concept correspondence was closest.

### Repo-wide mechanical results

These four measurements cover **all** files on both sides:

| Measurement | Result |
|---|---|
| Shared function names (any) | **0** |
| Identical comment lines ≥35 chars | **0** |
| O&G table reference values present in Grzelak's code | **0 of 8** |
| Global `np.random` seeding | Grzelak: **84 files**; here: **0 files** (injected `Generator` in 4) |

The zero function-name overlap is not a near-miss. Grzelak writes
`GeneratePathsGBMEuler`, `CallPutOptionPriceCOSMthd`, `Chi_Psi`,
`BS_Call_Option_Price`, `ImpliedVolatility`. This repo writes
`simulate_gbm_paths`, `cos_european_price`, `_chi`/`_psi`,
`black_scholes_price`, `implied_volatility`. The concepts correspond because
the book dictates them; not one identifier does.

The reference-value result matters independently: all eight Table 6.4–6.8 and
Fang–Oosterlee anchors are absent from Grzelak's code, consistent with this
repo's own `PARAMETER_PROVENANCE` claim that they were transcribed from
**printed book pages**, not lifted from his scripts.

### Whole-repo structural observation

Grzelak's repos are script-per-figure: module-level globals, a
`mainCalculation()` invoked at import, `matplotlib` calls inline, no package
structure, no tests, no type hints. This repo is an installable
`mypy --strict` package with an enforced dependency DAG and 264 tests. Wholesale
derivation would have had to survive a total reorganisation. Note also that
Grzelak's repos contain extensive Heston, Hull-White, SABR, CIR and Bates
material; this repo implements Ch2–6 only and has no Heston at all. Lifted
code would plausibly have brought some of it along.

## Confidence scale

| Level | Meaning |
|---|---|
| **High — derived** | ≥2 signals beyond what the math forces: non-book-mandated naming match, matching decomposition, or near-verbatim comments. |
| **High — independent** | This repo's conventions hold throughout; naming and decomposition differ from the candidate; verified by direct line-by-line reading. |
| **Medium — independent** | Same conclusion, but resting on repo-wide signals and signature comparison rather than full line-by-line reading of both sides. |
| **Low** | No credible candidate exists, or the file is too trivial for the question to be meaningful. Which one is stated. |

---

## Per-file findings

### `numerical/cos.py`
**Book concept:** Ch6, the COS method.
**Candidates checked:** `cfc@main:Lecture 08.../CallPut_COS_Method.py`,
`CashOrNothing_COS_Method.py`, `COS_Normal_Density_Recovery.py`;
`qfb@master:PythonCodes/Chapter 06/Tab06_04.py` (near-identical to the former).
**Judgment:** Independently written.
**Evidence:**
- *Truncation range* — the decisive difference. Grzelak hardcodes
  `a,b = ∓L·√tau`. This repo computes the **cumulant-based** interval
  `c1 ∓ L√(c2+√c4)` in a dedicated `truncation_range()`, and takes
  `cumulants` as a required argument. Different algorithm step, not a
  restyling.
- *Per-strike vs shared interval* — this repo builds a separate `[a,b]` per
  strike as a 2-D `(m,n)` broadcast; Grzelak uses one shared interval and an
  `np.outer` product.
- *Decomposition* — Grzelak returns a dict `{"chi":…, "psi":…}` from one
  `Chi_Psi`; this repo has two array-returning functions `_chi`, `_psi`.
- *Naming* — `H_k`/`u`/`CP` vs `uk`/`omega`/`option_type`.
- **Divergent algebra.** In Grzelak's `Chi_Psi`, the first term of `expr2`
  is not multiplied by `exp(d)`, while this repo's `_chi` does multiply it.
  A transcription would have inherited the quirk. This is the single
  strongest independence signal in the audit.
**Book vs Grzelak:** shared `chi`/`psi` closed forms and the halved `k=0`
term are Fang–Oosterlee (2008), cited as such in this file's docstring.
**Confidence:** High — independent.

### `pricing/hedging.py`
**Book concept:** Ch5, discrete delta hedging.
**Candidates checked:** `cfc@main:Lecture 11.../BS_Hedging.py`,
`HedgingWithJumps.py`.
**Judgment:** Independently written.
**Evidence:**
- *Memory model* — Grzelak allocates full `(NoOfPaths, NoOfSteps+1)` `PnL`,
  `CallM` and `DeltaM` matrices. This repo streams in `O(n_paths)`, evolving
  `s` in place, and documents that choice.
- *Structure* — Grzelak pre-generates all paths via `GeneratePathsGBM`
  returning a dict, then loops. This repo fuses simulation and rebalancing
  into a single loop.
- *Accounting* — a `PnL` recursion vs an explicit `cash` account with
  `cash_growth`/`dividend_growth` factors.
- *Feature set* — this repo has continuous dividends reinvested in shares,
  Merton jumps, and `sigma_hedge` for the wrong-vol experiment. None exist
  in Grzelak's file.
**Book vs Grzelak:** the self-financing discrete-hedge experiment is the
Chapter 5 set piece; both implement it.
**Confidence:** High — independent.

### `models/local_vol.py`
**Book concept:** Ch4 implied volatility and Breeden-Litzenberger. (The
filename is misleading — no Dupire/SABR here; its own docstring says it
exists "for API parity".)
**Candidates checked:** `cfc@main:Lecture 04.../ImpliedVolatility.py`;
`qfb@master:PythonCodes/Chapter 04/*`.
**Judgment:** Independently written.
**Evidence:**
- *Opposite convergence criterion* — Grzelak's loop uses `error=abs(g)`, a
  **price-residual** rule, with the volatility-step rule present but
  commented out. This repo converges on the **volatility step** (`xtol`) and
  its docstring argues explicitly why a residual rule is unreliable where
  vega is small. Deliberately the opposite choice.
- *Scope* — Grzelak has one bare Newton loop driven by module-level globals,
  printing each iteration. This repo has three solvers (Newton, secant,
  bracketed), a dispatcher, no-arbitrage bounds checks, and a
  Brenner-Subrahmanyam seed.
**Book vs Grzelak:** Newton-on-vega for implied vol is standard textbook
material.
**Confidence:** High — independent.

### `models/characteristic_functions.py`
**Book concept:** Ch5, Lévy characteristic functions (GBM, Merton, VG, CGMY).
**Candidates checked:** `qfb@master:PythonCodes/Chapter 05/*` (`ChFCGMY`,
`ChFVG`, `ChFForMertonModel`).
**Judgment:** Independently written.
**Evidence:**
- *Martingale correction* — Grzelak computes `omega = -1/tau·log(varPhi(-i))`,
  a numerical log of an exponential. This repo computes `omega = -sigma²/2 −
  psi(-i)` directly from the closed-form Lévy exponent.
- *Structure* — Grzelak returns bare `lambda`s from factory functions; this
  repo uses frozen dataclasses behind a structural `Protocol`, with
  `__post_init__` parameter validation (`M > 1`, `Y ∈ (0,2)\{1}`) that has no
  counterpart in his code.
- *Bespoke numerics* — the `+ 0.0*uc` trick routing reference powers through
  the complex path so `cf(0) == 1` bitwise is specific to this repo.
**Book vs Grzelak:** the exponent
`C·Γ(-Y)[(M-iu)^Y − M^Y + (G+iu)^Y − G^Y]` is the published CGMY (2002)
formula, cited as such here.
**Confidence:** High — independent.

### `core/gbm.py`
**Book concept:** Ch2, GBM path simulation.
**Candidates checked:** `cfc@main:Lecture 09.../EulerConvergence_GBM.py`,
`Lecture 03.../GBM_ABM_paths.py`.
**Judgment:** Independently written.
**Evidence:** Grzelak computes the Euler path `S1` and exact path `S2` as two
hardcoded parallel computations inside one function; this repo has a single
`scheme: Literal["exact","euler"]` dispatch. Injected `Generator` vs
`np.random.normal` with moment-matching. Full validation and
RESULT/MODELLING-CHOICE docstrings vs none.
**Book vs Grzelak:** the Euler-Maruyama discretisation of GBM is forced by
the math.
**Confidence:** High — independent.

### `models/jump_diffusion.py`
**Book concept:** Ch5, Merton (1976) jump-diffusion analytic series pricer.
**Candidates checked:** `MertonCallPrice`, all four occurrences —
`qfb@master:PythonCodes/Chapter 05/Fig05_05.py`,
`PythonCodes/Chapter 06/Exe06_03.py`,
`Solutions to Exercises/Chapter 5/.../Exercise_5_11_a.py`,
`Solutions to Exercises/Chapter 6/.../Exercise_6_3.py`. Confirmed all four
are the same function pasted across his own files (whitespace/comment
differences only) via direct diff — one comparison covers all four.
**Judgment:** Independently written. Verified by full line-by-line reading
plus a numerical cross-check (below), not signature comparison alone —
this is the file the earlier pass flagged as needing that closer look.
**Evidence:**
- **Different discounting architecture**, the decisive difference. This
  repo folds the jump compensation into a modified discount rate
  `r_n = r - lam*kappa + n*ln(1+kappa)/T` and modified intensity
  `lam' = lam*(1+kappa)`, then calls `black_scholes_price(r_n, sigma_n, ...)`
  once per term as a black box — matching the standard literature form this
  file's own docstring cites (Merton 1976). Grzelak keeps the *unadjusted*
  rate and intensity, instead shifting the *mean of the conditional
  lognormal* (`muX(n)`) by the compensator, computing an undiscounted
  `value_n` inline with its own re-derived `d1`/`d2`, and applying one
  `exp(-r*tau)*exp(-xiP*tau)` factor to the whole sum at the end. Both are
  valid decompositions of the same formula; one composes through an
  existing pricer via parameter substitution, the other duplicates BS math
  inline with a different algebraic grouping. Not a restyling — a different
  derivation route to the same result.
- **Verified equivalent, not just argued equivalent:** transcribed
  Grzelak's formula into an isolated script and ran both side by side
  across five parameter sets (lam in {0.5, 1, 3, 8}, tau in {0.1, 0.5, 1, 2}).
  Absolute difference ranged from **9.4e-16 to 5.7e-14** across the five
  cases (case-by-case: 3.73e-14, 5.68e-14, 8.88e-15, 9.44e-16, 3.20e-14) —
  confirms they're the same math via two structurally different routes, not
  a coincidence of the test cases chosen.
- **Truncation strategy is opposite in character.** Grzelak hardcodes
  `range(1, 20)`, recomputing `factorial(k)` from scratch every iteration.
  This repo uses an incremental recursive weight (`weight *= lam_bar_t /
  (n+1)`) with an adaptive tolerance stop gated on having passed the Poisson
  mode first — no factorial anywhere. A transcription does not acquire a
  different, more numerically robust algorithm on its own.
- **Zero naming overlap, including the idiosyncratic terms.** Grzelak names
  the jump intensity `xiP` (not lambda, not "lam") and the compensator
  `helpExp` — both distinctive enough to be real fingerprints if copied.
  Grepped this repo's entire `src/`: **zero hits for either.**
- `kappa`: Grzelak computes `exp(x) - 1`; this repo uses `np.expm1(x)`,
  numerically better for small `x` — a choice made writing fresh code with
  modern numpy idioms, not one preserved by transcription.
- Grzelak's file uses `np.complex` and `np.math.factorial`, both removed
  from numpy years ago (>=1.24) — unmaintained ~2018 script code, doesn't
  run on this repo's `numpy>=2.0` floor as-is.
- No input validation, no edge cases, no `lam=0` fast path in Grzelak's
  version; all present here, consistent with every other file in this audit.
**Book vs Grzelak:** the Poisson-weighted lognormal-mixture series is
Merton (1976); both parties implement the same published formula by
necessarily-similar high-level structure (a sum over conditional
Black-Scholes terms), which is why the decomposition-level difference above
is the evidence that actually matters here, not the shared top-level shape.
**Confidence:** High — independent.

**Methodology note: a range-figure error, caught and corrected.** This
numerical cross-check first reported "3.7e-14 – 5.7e-14" as the five-case
agreement range — actually just the first two printed rows, read as the
set's min/max rather than computed as such. The real range is **9.4e-16 –
5.7e-14**; three of five cases fell below the original stated floor. Cause:
reading tool output directly instead of computing summary statistics from
it. Corrected in the entry above. Left here because it demonstrates this
audit's own verify-against-raw-output discipline lapsing inside the
artifact meant to enforce that discipline, not only in the code under
audit.

### `core/brownian.py`, `core/jump_diffusion.py`, `models/black_scholes.py`, `numerical/monte_carlo.py`, `pricing/greeks.py`
**Book concepts:** Ch2 Brownian motion; Ch5 Merton/Poisson paths; Ch3
Black-Scholes; Ch9-equivalent Monte Carlo; Ch3 Greeks.
**Candidates checked:** `PythonCodes/Chapter 02,03/*`;
`Lecture 03,05,09,11/*` — specifically `GeneratePathsBM`, `GeneratePathsMerton`,
`GeneratePathsPoisson`, `BS_Call_Put_Option_Price`, `BS_Delta`, `BS_Gamma`,
`BS_Vega`, `EUOptionPriceFromMCPaths`.
**Judgment:** Independently written.
**Evidence:** no shared identifier, comment, or reference value with any
candidate (repo-wide measurements above). This repo's conventions hold
uniformly: snake_case, injected `Generator`, `ArrayLike`→`NDArray` typing,
`ValueError` guards, RESULT/CONVENTION docstring tags — none of which appear
anywhere in Grzelak's code. Grzelak's counterparts are all script-style with
global RNG and dict returns.
**Book vs Grzelak:** all shared content is standard published formulas
(BSM price and Greeks, Merton's Poisson-weighted paths, Euler schemes).
**Confidence:** Medium — independent. Judgment rests on repo-wide signals
and signature comparison rather than full line-by-line reading of both
sides for each file, unlike `models/jump_diffusion.py` above, which was
promoted to High after exactly that closer read.

### `pricing/payoffs.py`
**Book concept:** Ch3 payoff functions.
**Candidates checked:** `AssetOfNothingPayoff`, `DigitalPayoffValuation`,
`PayoffValuation` in both repos.
**Judgment:** Independently written, though the question is close to vacuous
here — `np.maximum(s - k, 0.0)` admits essentially one spelling.
**Confidence:** Low — the file is too mathematically forced for derivation
to be a meaningful question.

### `replications.py`
**Book concept:** harnesses reproducing O&G Tables 6.4–6.8, Figures
5.7/5.8/6.4, Exercise 6.3.
**Candidates checked:** `qfb@master:PythonCodes/Chapter 06/Tab06_04.py`
through `Tab06_08.py`, `Exe06_03.py`, `Fig06_04.py` — the closest
*filename* correspondence anywhere in this audit (`table_6_4` ↔ `Tab06_04.py`,
`exercise_6_3` ↔ `Exe06_03.py`).
**Judgment:** Independently written.
**Evidence:** despite the naming correspondence, **none of the eight
reference values this module hardcodes appear anywhere in Grzelak's code** —
they came from printed book pages, as `PARAMETER_PROVENANCE` claims with its
15 CONFIRMED/ASSUMED tags. Grzelak's `Tab06_*.py` files are near-copies of
his own `CallPut_COS_Method.py` with different `K` and `N`; this module is a
data-returning harness with no plotting and no `mainCalculation()`. The
correspondence is to the **book's table numbering**, which both parties
follow, not to his code.
**Book vs Grzelak:** table numbering is the book's.
**Confidence:** High — independent.

### `demo.py`
**Judgment:** Unrelated. Streamlit UI compute layer; Grzelak's repos predate
and contain no Streamlit code.
**Confidence:** Low — no credible candidate exists.

### `__init__.py` (root, `core/`, `models/`, `numerical/`, `pricing/`)
**Judgment:** Unrelated. Re-export shims and `__all__` lists. Grzelak's repos
have no package structure at all, so no counterpart exists.
**Confidence:** Low — too trivial for the question to be meaningful.

---

## Summary

**No file in `src/pyfinlib_practice/` shows evidence of derivation from
either Grzelak repository.** Every similarity found traces to a published
formula from the book or the wider literature (Fang–Oosterlee 2008,
CGMY 2002, standard BSM), not to Grzelak's particular code.

On this evidence no BSD-3-Clause attribution obligation to those two
repositories is triggered. Four points of caution:

1. Five files carry **Medium** rather than High confidence — the conclusion
   there rests on repo-wide signals rather than exhaustive line-by-line
   reading. (`models/jump_diffusion.py` was in this group originally; a
   targeted line-by-line read plus a numerical cross-check against
   Grzelak's `MertonCallPrice` promoted it to High — see that entry above
   for the method, which is the template for closing out the rest of this
   group if that's ever wanted.)
2. This audit covers only the two named repositories. It says nothing about
   any other source.
3. The audit examined the **current** state of both repos. Code present here
   could in principle derive from a historical version.
4. Auditing an AI-assisted codebase for derivation has an inherent limit: a
   model may reproduce patterns from training data without any file ever
   being copied. The evidence here — divergent algebra in `_chi`, the
   opposite convergence criterion, the different truncation rule — argues
   against that, since a reproduced-from-memory implementation would be more
   likely to match, not contradict, the reference.

**No action taken. No action recommended in either direction — that is Abhi's
call.**
