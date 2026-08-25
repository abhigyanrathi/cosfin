# Claude Code Task 1b — cosfin: package rename, repo rename, Render migration, PyPI release

**Prerequisite:** Task 1a work items 1–3 complete and verified. Do not start until the README numeric corrections have landed and the provenance audit exists.
**Session hygiene:** separate session from 1a. `/clear` first. Plan mode — this touches 33+ files and three external services.

---

## Hard constraints

1. **Do not modify any file under the package's `src/` tree except import statements and the package directory name.** No implementation changes.
2. **Do not amend, rebase, or rewrite commit `ce56441`.** It is the origin commit for the `numeric-claim` fixture and must keep containing the stale "249 pytest" README claim in its own historical snapshot.
3. Use `git mv` for the package directory rename, not delete-and-recreate. History must survive `git log --follow`.
4. **Never `git add -A`.** Stage explicitly.
5. **Always `--no-pager`** on git log/diff/show.
6. **Gate:** if Task 1a's provenance audit found derivation from `LechGrzelak/QuantFinanceBook` or `Computational-Finance-Course` (BSD-3-Clause, attribution required), **stop before Phase 5** and escalate. Do not publish to PyPI with an unsatisfied attribution obligation.
7. If you propose a decision this spec does not cover: stop and ask.

---

## Decisions already made — do not revisit

| Item | Value |
|---|---|
| PyPI distribution name | `cosfin` |
| Python import name | `cosfin` |
| GitHub repo name | `cosfin` (renamed from `pyfinlib-practice`) |
| Render service name | `cosfin-demo` (new service; old `pyfinlib-practice-demo` deleted after cutover) |
| Release version | `0.3.0` (new tag). Not 0.2.0 — that tag points at pre-rename code. |
| Empty `pyfinlib` repo | **Delete** (Phase 6) |

---

## Phase 1 — package rename (branch, nothing pushed)

Work on a branch. Nothing in this phase touches GitHub, Render, or PyPI.

- `git mv src/pyfinlib_practice src/cosfin`
- Update every reference to `pyfinlib_practice` / `pyfinlib-practice`. Known distribution across 33 files, heaviest first: `tests/test_package.py` (26), `app/streamlit_app.py` (12), `notebooks/pyfinlib_practice_demo.ipynb` (10), `src/.../replications.py` (6), `src/.../demo.py` (6), `README.md` (6), plus 27 more files with 1–5 each, plus `pyproject.toml` (2), `render.yaml` (1), `scripts/*`.
- Rename `notebooks/pyfinlib_practice_demo.ipynb` → `notebooks/cosfin_demo.ipynb`.
- `pyproject.toml`: `name = "cosfin"`, bump `version = "0.3.0"`, and update the `[tool.setuptools.package-data]` key so `py.typed` still ships. **Verify `py.typed` is present in a built wheel — do not assume.**
- `render.yaml`: `name: cosfin-demo`.
- README: restructure so the COS/Fourier engine is the organizing spine, with Black-Scholes and Monte Carlo as supporting modules. The package name now foregrounds Ch6; the document should match. Keep all corrected numbers from Task 1a.

## Phase 2 — verify before anything leaves the machine

Fresh clone of the branch into a clean directory, new venv, `pip install -e ".[dev,app,plots]"`.

- `ruff check .` → clean
- `mypy` → clean, 33 source files
- `pytest -q` → 264 passed
- `python -c "import cosfin; print(cosfin.__file__)"` → succeeds
- `grep -ri "pyfinlib" . --exclude-dir=.git --exclude-dir=.venv` → **zero hits**, other than any deliberate historical mention in CHANGELOG
- Cold-import stress test: in **five separate processes**, import each of `cosfin.pricing`, `cosfin.models`, `cosfin.numerical`, `cosfin.core`, `cosfin.replications` first. All five must succeed. (A shared process caches partially-initialised modules and masks circular-import failures — this repo has a confirmed history of exactly that bug.)
- `python -c "from cosfin.pricing import vega; from cosfin.models import vega as v2; print(vega is v2)"` → `True`

Do not proceed until every one of these passes with pasted output.

## Phase 3 — merge and push

Merge branch to main, push. Confirm CI goes green on GitHub Actions before continuing.

## Phase 4 — GitHub repo rename

Abhi performs this in the GitHub UI: rename `pyfinlib-practice` → `cosfin`. GitHub creates automatic redirects from the old URL.

Then, locally:
- `git remote set-url origin https://github.com/abhigyanrathi/cosfin.git`
- `git --no-pager remote -v` to confirm
- `git fetch` to confirm the new remote works

Update the repo's GitHub description and topics to match the new framing.

## Phase 5 — Render migration

**Gate: only after the Phase 4 rename is live.**

Render identifies services by the `name` field in `render.yaml`, so `cosfin-demo` will be provisioned as a new service rather than renaming the existing one. This deliberately avoids depending on whether the old Blueprint binding survived the repo rename.

1. In the Render dashboard: New → Blueprint → select the `cosfin` repo.
2. Confirm the build succeeds and `cosfin-demo.onrender.com` serves the Streamlit app.
3. Exercise the app — confirm it actually renders plots, not just returns 200 on the health check.
4. **Only then** delete the old `pyfinlib-practice-demo` service.
5. Update the README demo link and any badge URLs.

Note: `autoDeploy: true` on the On Commit trigger means deploys are **not** gated on GitHub Actions CI. A commit that breaks CI still deploys. Consider switching to After CI Checks Pass — flag this to Abhi as a recommendation, do not change it unilaterally.

## Phase 6 — delete the empty `pyfinlib` repo

Abhi performs this. Irreversible.

Before deleting, confirm the repo contains nothing not already superseded: 7 commits, all config and docs, every `.py` file 0 bytes. Verified 21 Aug 2026. Re-verify at time of deletion rather than trusting this line.

## Phase 7 — PyPI release

**Gate: provenance audit clean (constraint 6).**

PyPI upload is irreversible. **A version number, once uploaded, can never be reused — even after deletion.** A botched upload burns `0.3.0` permanently.

1. Build: `python -m build` → sdist + wheel in `dist/`.
2. Inspect the wheel contents. Confirm `cosfin/py.typed` is present and no test or notebook files leaked in.
3. `twine check dist/*`
4. **Upload to TestPyPI first.** In a clean venv, `pip install -i https://test.pypi.org/simple/ cosfin`, then `import cosfin` and run one real pricing call.
5. Only after TestPyPI verifies: upload to PyPI.
6. In a fresh venv on a clean machine path: `pip install cosfin`, `import cosfin`, run one real pricing call.
7. Tag `v0.3.0` and push the tag. Add the CHANGELOG entry.

**Authentication:** use a scoped API token, never an account password. Better: set up **PyPI Trusted Publishing** (OIDC from GitHub Actions) so no long-lived token exists anywhere. Recommend this to Abhi; it is the current best practice and removes a credential from his machine entirely.

---

## Acceptance criteria

Not complete until all are demonstrated with pasted raw output:

1. Fresh clone of `cosfin` from GitHub, clean venv, `[dev,app,plots]` install: ruff clean, mypy clean on 33 files, 264 tests pass.
2. `grep -ri "pyfinlib" .` on that clone returns zero hits outside `.git` and deliberate CHANGELOG history.
3. Five separate-process cold-import tests all pass; `pricing.vega is models.vega` → `True`.
4. `cosfin-demo.onrender.com` serves the working app; old service deleted.
5. `pip install cosfin` in a clean venv succeeds, `import cosfin` works, one pricing call returns a correct value checked against a known anchor (ATM European call `S=K=100, r=0.05, σ=0.20, T=1` → `10.450583572185565`).
6. Every numeric claim in the published README matches live output from that clean install exactly.
7. `git --no-pager log --oneline` shows `ce56441` present and unmodified.
8. Empty `pyfinlib` repo deleted.

## Report back

Append to `docs/decisions.md`. Report to the orchestrating chat:
- exact commands and raw output, not summaries
- anything that differed from this spec's expectations
- anything you stopped and did not do, and why

A self-report of "released successfully" is a claim, not a fact. Paste the `pip install` output from a clean venv.
