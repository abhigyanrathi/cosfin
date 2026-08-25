# Decisions log

Append one entry per Claude Code session. Newest at the bottom.

Format: date, what was decided or done, why, and what was explicitly *not* done.

---

## 2026-08-21 — orchestration session (no code)

**Locked:**

- Package and import name: `cosfin`. `pyfinlib` is taken on PyPI by an active
  Rust-backed quant library (`sarsoo/finlib`, v0.0.10, June 2025).
- GitHub repo renames `pyfinlib-practice` → `cosfin`.
- Render: new service `cosfin-demo`; old `pyfinlib-practice-demo` deleted after
  cutover. Render identifies services by the `name` field in `render.yaml`, so
  a name change provisions a new service rather than renaming the old one.
- Release version `0.3.0`. Not 0.2.0 — that tag points at pre-rename code.
- The empty `pyfinlib` production repo is deleted. The hand-written-deliverable
  plan is dropped; this repo is the portfolio piece, AI-assisted by design.
- Commit `322c01c` is immutable — regression fixture for `numeric-claim`.
- `numeric-claim` (import `numeric_claim`) is the name for the tool previously
  called "attest". `attest` on PyPI belongs to Dag Odenhall.

**Verified independently by fresh clone and full gate run:**

- 264 tests pass; ruff clean; mypy clean over **33** source files; 942
  statements at 100% coverage.
- `README.md:93` claims 249 tests and 30 files. Both wrong. Live and public
  since 12 July.
- No CRR, binomial, Cox-Ross-Rubinstein, Abramowitz-Stegun, or QuantLib code
  exists anywhere in this repo or its full git history. `git log -S` across all
  refs returned nothing.
- Every `.py` file in the `pyfinlib` production repo is 0 bytes; last commit
  2026-06-03.

**Superseded, kept visible rather than deleted:**

- The two-repo integrity model (hand-written `pyfinlib` + AI-written practice
  repo). Dropped by decision. The Claude Code guard-hook config built to
  enforce it is not deployed.
- `numeric-claim`'s original motivating bugs (a stale CRR convergence figure
  and an Abramowitz-Stegun precision ceiling at 8e-6) could not be located in
  any tracked artifact. Replaced with two real, measured, reproducible catches:
  the 249→264 README drift at `322c01c`, and the CGMY Y=1.5 error floor at
  ~2.4e-8 against a machine-precision prediction of ~1e-13.

**Open:**

- `numeric-claim` fingerprint boundary (function body / call graph / package
  edge) — undecided.
- `numeric-claim` multiple-comparisons correction for stochastic claims —
  Bonferroni is a placeholder, not a decision.
- Whether any code here derives from Grzelak's BSD-3 repos — pending the
  provenance audit in Task 1a.
