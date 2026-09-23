# Provenance check: independence from Grzelak's reference code

Lech Grzelak, co-author of Oosterlee & Grzelak, publishes reference code
for the textbook under BSD-3-Clause, which carries an attribution
requirement if code here derives from his code rather than from the book
text. This checks whether that's the case.

## Sources checked

- `LechGrzelak/QuantFinanceBook` (branch `master`)
- `LechGrzelak/Computational-Finance-Course` (branch `main`)

Both cloned in full and read directly, so naming and comment comparisons
are exact rather than based on a summary.

## The distinction that matters

Implementing a published textbook formula isn't a BSD-3 concern -- two
correct COS-method implementations will share the same closed forms
because the math forces them to. The concern is narrower: copying
Grzelak's particular code -- his variable names, his function
decomposition, his comment phrasing -- none of which the book requires.

## Repo-wide results

| Measurement | Result |
|---|---|
| Shared function names | 0 |
| Identical comment lines >= 35 chars | 0 |
| Grzelak's reference values matching this repo's table anchors | 0 of 8 |
| Global RNG seeding (Grzelak: pervasive; here: none -- injected Generator instead) | 84 files vs 0 |

Grzelak names things like `GeneratePathsGBMEuler`, `Chi_Psi`,
`BS_Call_Option_Price`. This repo uses `simulate_gbm_paths`, `_chi`/`_psi`,
`black_scholes_price`. Every concept matches because the book dictates it;
no identifier does.

## Per-file notes

**numerical/cos.py** -- Grzelak hardcodes a fixed truncation range
(`+/-L*sqrt(tau)`) and computes it once, shared across all strikes. This
repo derives the range from the cumulants per strike, as a vectorised 2-D
computation. Different algorithm step, not a restyling. Grzelak's `Chi_Psi`
also has an asymmetry in one term (`expr2`'s first term isn't multiplied
by `exp(d)`, where the corresponding term here is) -- a transcription
would have inherited that.

**pricing/hedging.py** -- Grzelak pre-generates full path matrices and
loops over them afterward. This repo streams in O(n_paths) memory, fusing
simulation and rebalancing into one pass, and handles continuous
dividends and jumps, which his version doesn't.

**models/local_vol.py** (implied volatility) -- Grzelak's Newton loop
converges on the price residual; this repo deliberately converges on the
volatility step instead, because a residual criterion silently accepts
wrong volatilities when vega is small. Opposite choice, not a variant.

**models/characteristic_functions.py** -- Grzelak computes the martingale
correction as a numerical log of an exponential; this repo computes it
directly from the closed-form Levy exponent. Structured as frozen
dataclasses behind a Protocol with parameter validation, which has no
counterpart in Grzelak's bare-lambda factories.

**core/gbm.py** -- Grzelak hardcodes the Euler and exact schemes as two
separate parallel computations. This repo dispatches on a single scheme
parameter.

**models/jump_diffusion.py** (Merton) -- the closest naming
correspondence in the audit by concept, so checked most carefully,
including a numerical cross-check: transcribed Grzelak's formula
standalone and ran it against this repo's implementation across five
parameter sets. They agree to 9.4e-16 to 5.7e-14, confirming they compute
the same thing through different derivations. Grzelak folds the jump
compensation into the conditional lognormal's mean and re-derives
Black-Scholes math inline; this repo instead calls the existing
`black_scholes_price` function with a modified discount rate and
intensity per term. Grzelak also hardcodes a fixed 20-term sum
recomputing factorials each time; this repo uses an adaptive,
recursive-weight stopping rule with no factorials. Zero overlap on
Grzelak's distinctive naming (`xiP`, `helpExp`) anywhere in this repo.

**core/brownian.py, core/jump_diffusion.py, models/black_scholes.py,
numerical/monte_carlo.py, pricing/greeks.py** -- no shared identifiers,
comments, or reference values with Grzelak's equivalents. Consistent with
this repo's conventions throughout (injected RNG, typed arrays, explicit
validation), none of which appear in his script-style code.

**pricing/payoffs.py** -- too mathematically forced to be a meaningful
comparison (`max(S-K, 0)` has one natural spelling).

**replications.py** -- despite matching Grzelak's file naming by table
number (his convention, both follow it), none of the eight reference
values this module hardcodes appear in his code -- they're transcribed
from the book's printed pages.

**demo.py, package __init__.py files** -- no counterpart exists in
Grzelak's repos at all (no Streamlit code, no package structure).

## Conclusion

No file shows evidence of deriving from either Grzelak repository. Every
similarity traces to a formula the book or the wider published literature
(Fang-Oosterlee 2008, CGMY 2002) requires, not to his particular code. No
BSD-3-Clause attribution obligation is triggered.

This covers only these two repositories, and only their current state --
it says nothing about any other source, and it can't rule out derivation
from a historical version of either repo that no longer exists.