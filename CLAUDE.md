# pyfinlib-practice → cosfin — project instructions

Quantitative finance library implementing Oosterlee & Grzelak Ch2–6:
Black-Scholes, implied volatility, Lévy/jump models, and a Fourier-cosine (COS)
pricing engine.

**This repo is the deliverable.** Implementation here is AI-assisted, by
design and with the mentor informed. That disclosure is on the record — never
remove or soften it in the README or the Streamlit sidebar.

## Quality gates — run in CI's order, always

```
ruff check .     # expect: All checks passed!
mypy             # expect: no issues found in 33 source files
pytest -q        # expect: 264 passed
```

**Install all three extras before typechecking:** `pip install -e ".[dev,app,plots]"`.
With only `[dev]`, mypy emits 7 spurious errors in `app/streamlit_app.py` from
missing streamlit/matplotlib stubs. That is an environment artifact, not a
regression. Do not "fix" it in code.

Verified state, independently re-checked 21 Aug 2026 by fresh clone and full
gate run: v0.2.0, 264 tests, 100% line coverage over 942 statements in 19
source files, ruff and mypy --strict clean across 33 files, CI green on the
3.12 / 3.13 matrix.

Note that **19** (coverage source files) and **33** (mypy-checked files: 19 src
+ 13 tests + 1 app) are both correct and measure different things. State which
one you mean. The README claimed 30 for months and it was simply wrong.

## Immutable commit — do not touch

**`ce56441` must never be amended, rebased, or rewritten.** It carries two
independent regression fixtures for the `numeric-claim` project. Both must
survive in history exactly as committed:

1. **README count drift.** `README.md` at this commit states 249 tests and
   30 source files; the suite at the same commit actually runs 264 tests
   across 33 files. This is a documentation-vs-reality claim, not a code bug. `998cf1f`
   later corrected the pytest/mypy comment lines (32, 35) but missed the
   `## Validation` prose paragraph (lines 94-95 on `main` today), which is
   left deliberately stale as the live, currently-reproducible fixture.
2. **CGMY/Table 6.7 error floor.** Documented separately in "Known gaps"
   below — the Y=1.5 error floor sits ~5 orders above the machine-precision
   prediction. This is a numerical bug, not a doc bug.

`numeric-claim`'s validation gate must catch both. A tool that catches only
one is not validated. All corrections land in new commits on top of
`ce56441`, never inside it.

## Conventions

- Python floor 3.12, `numpy>=2.0`, `scipy>=1.13`. src-layout.
- `mypy --strict` over `src`, `tests`, `app`. No exceptions.
- ruff select `E,F,I,N,UP,B,SIM,RUF` with no blanket ignores.
- snake_case arguments — `s0`, `strike`, `t`, `mu_j`, `sigma_j`. Never `S0`,
  `K`, `T`. The `N` (pep8-naming) rule is on deliberately.
- scipy imports carry inline `# type: ignore[import-untyped]`, not a pyproject
  override.
- Docstrings label RESULT / CONVENTION / MODELLING CHOICE.
- Anchor tests against **published** values. Round-trips through the same
  pricer cannot catch a pricer bug.
- Circular-import testing requires **separate process invocations** per module.
  A shared process caches partially-initialised modules and masks the failure.
  This repo has a confirmed history of exactly that bug.

## Known gaps — do not paper over these

- **Table 6.6** is validated against its own N=512 self-reference because the
  book page was never supplied. It is a convergence check disguised as a
  benchmark. Say so whenever it comes up.
- **eq. (6.45)** page still missing; needed only to reproduce the book's
  printed low-N error columns in Tables 6.4/6.5.
- **Figures 5.7 / 5.8** use assumed illustrative parameters not printed in the
  source pages. Flagged in the repo, not hidden.
- **Table 6.7, Y=1.5** is platform-sensitive at ~2.4e-8 absolute between Linux
  and Windows. Documented in `docs/RESULTS.md` note 3. This is also
  `numeric-claim`'s error-budget test fixture — do not "fix" or hide it.
- **Ch7 and Ch8 are unbuilt.** Heston does not exist here.
- `norm_pdf` / `norm_cdf` sit in `models/black_scholes.py` rather than a
  `utils/` layer. Known DAG deviation, accepted.

## Current task

See `docs/tasks/task_1a.md`, then `docs/tasks/task_1b.md`. Do not start 1b
until 1a has been verified externally.

Ch8 Heston is **not** currently in scope. Do not start it.

## Working agreements

- Never `git add -A`. Stage explicitly by path. It caused a real CI failure
  here (commit `15c8e89` swept in a notebook with 17 ruff violations under the
  message "Update description", which killed lint before mypy or pytest ran).
- `--no-pager` on all `git log` / `git diff` / `git show`.
- Descriptive commit messages. Never "Update description."
- Present options with tradeoffs before touching files. If you hit a decision
  these instructions do not cover: **stop and ask.** Do not default.
- Report raw command output, not summaries. "264 tests pass" is a claim; the
  pytest output is the fact.
- Append to `docs/decisions.md` at the end of every session.
- Deployment is Render Blueprint with `autoDeploy: true` on commit, **not**
  gated on GitHub Actions. A commit that breaks CI still deploys.
- Be terse. Push back rather than validate.
