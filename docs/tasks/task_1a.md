# Claude Code Task 1a — pyfinlib-practice: correctness pass + provenance audit

**Repo:** `pyfinlib-practice` (local: adjust path as needed)
**Session hygiene:** one phase, `/clear` before starting. Plan mode first — this touches more than three files.
**Blocked work NOT in this task:** PyPI release, GitHub repo rename, Render reconnection. Those are Task 1b, pending a name decision.

---

## Hard constraints

1. **Do not modify any file under `src/pyfinlib_practice/`.** No implementation changes. This task is docs, metadata, and audit only.
2. **Do not amend, rebase, or rewrite commit `ce56441`.** It is the origin commit for the `numeric-claim` fixture — its `README.md` states 249 tests and 30 source files against a suite that, at that commit, already ran 264 tests across 33 files. `998cf1f` later corrected the pytest/mypy comment lines but missed the `## Validation` prose paragraph (lines 94-95 on `main` today), which still says "249 pytest" / "across 30 files." See Work Item 1 for the carve-out. All other fixes go in new commits on top of `ce56441`.
3. **Never `git add -A`.** Stage explicitly by path. (Root cause of a prior CI failure: commit `15c8e89` swept in a notebook with 17 ruff violations.)
4. **Always `--no-pager`** on `git log` / `git diff` / `git show`.
5. Descriptive commit messages. Not "Update description."
6. If you propose a decision this spec does not cover: **stop and ask.** Do not default.

---

## Verified ground truth (independently confirmed 21 Aug 2026)

Confirmed by fresh anonymous clone, venv install with `[dev,app,plots]`, and full gate run:

| Item | Verified value |
|---|---|
| `pytest --collect-only -q` | **264 tests collected** |
| `pytest -q` | **264 passed** |
| `ruff check .` | All checks passed |
| `mypy` (config: `files = ["src","tests","app"]`, `strict = true`) | **Success: no issues found in 33 source files** |
| Coverage | 942 statements, **100%** |
| HEAD | `6f92dd5` "style: fix ruff violations in demo notebook (unblocks CI)", 2026-07-25 17:26:34 +0530 |
| Tag | `v0.2.0` at `ce56441` |
| src LOC / tests LOC | 2854 / 1723 |

**mypy gotcha:** running `mypy` with only `[dev]` installed produces 7 spurious errors in `app/streamlit_app.py` (missing `streamlit` / `matplotlib` stubs). Install `[dev,app,plots]` before claiming a clean typecheck. Document this in the README validation section.

---

## Work item 1 — README numeric corrections

**File:** `README.md`, line 93 (the `## Validation` section).

Lines 32/35 already correctly say 264/33. Lines 94-95 (the `## Validation` section) still claim **"249 pytest"** and **"across 30 files."** **Do not fix lines 94-95** - that paragraph is `numeric-claim`'s live validation fixture and must keep contradicting lines 32/35 on `main`. If you believe it should be corrected, stop and ask; do not default to fixing it here.

Before editing, re-run all three gates yourself and confirm the numbers against live output. Do not take the table above on faith — re-derive it. If any number disagrees with your run, **stop and report the discrepancy** rather than writing either number.

Then scan the whole README for any other numeric claim (LOC counts, coverage, chapter counts, Python versions, dependency floors) and verify each one the same way. Report every claim found and whether it verified.

## Work item 2 — CHANGELOG

No `CHANGELOG.md` exists. Create one, Keep a Changelog format, backfilled from the 11 commits in `git log`.

Real entries only, derived from actual commit contents. Notable content that belongs in it:
- v0.2.0 (`ce56441`): full Ch2–6 rebuild; fixed the `models/ ↔ pricing/` circular import; fixed the `digital_price` σ=0 silent numerical error (returned ≈0.5306 instead of ≈0.9512); hardened `butterfly_price` to raise `ValueError` instead of returning `nan` when `spread >= strike`.
- `998cf1f`: Streamlit demo app + Render blueprint deployment.
- `260afaa`: `docs/RESULTS.md` reproduction transcript, CI badge, platform-honest Table 6.7 claim.

Do not invent version numbers or dates. Only `v0.2.0` is tagged; earlier work goes under an `Unreleased` or `0.1.0` heading only if the git history supports it.

## Work item 3 — provenance audit (report only, no changes)

Lech Grzelak publishes the textbook's official code publicly under **BSD-3-Clause**:
- `LechGrzelak/QuantFinanceBook`
- `LechGrzelak/Computational-Finance-Course`

BSD-3 carries an attribution requirement. If any implementation in this repo derives from those repos rather than from the book text, a LICENSE/NOTICE obligation exists and must be satisfied before any publication or PyPI release.

**Deliverable:** `docs/provenance-audit.md` containing, per source file in `src/pyfinlib_practice/`:
- whether it appears derived from, independently written against, or unrelated to the Grzelak repos
- the specific evidence for that judgment (structural similarity, variable naming, comment phrasing, algorithm-step ordering)
- an explicit confidence level

**Report findings. Do not act on them.** Do not add attribution, do not modify LICENSE, do not rewrite code. Abhi decides what follows.

If you cannot access the Grzelak repos, say so plainly and mark the audit as not performed. Do not produce an audit that guesses.

## Work item 4 — stale docs in the *production* `pyfinlib` repo

Separate repo. Currently public, and its `README.md` opens by describing "a production-grade Python library implementing quantitative finance models... Chapters 1–8" over a repo where **every `.py` file is 0 bytes**.

Also stale:
- `README.md:41` — "Work in progress. Deadline: end of June 2026."
- `mentor_roadmap.md:12-13` — "End of June 2026... 28 days remaining as of June 2, 2026." (79 days past.)
- `mentor_roadmap.md:43` — June-1 log entry attributes "Ch4 and Ch6 practice items" to the mentor. This is a **misattribution**: those were the **Chapter 3** items (BS call, put-call parity, Delta, Digital, Butterfly). The genuine Ch4 list came 23 June. Ch6 has never had a confirmed function list.
- `topics_covered.md` — marks every Ch1–8 section ✗; untouched since 2 June.
- `progress_pyfinlib.md` — describes the practice repo as "Details TBD."

**Deliverable:** a proposed rewrite of `pyfinlib/README.md` that accurately describes an empty scaffold, plus corrections to the three tracking files. **Propose as a diff; do not commit** until Abhi approves the wording — this repo's public description is a positioning decision, not a mechanical fix.

---

## Acceptance criteria

Task is not complete until all of the following are demonstrated with pasted raw command output:

1. On a **fresh clone** into a clean directory, with `pip install -e ".[dev,app,plots]"`, the commands `ruff check .`, `mypy`, and `pytest -q` all pass, and **every numeric claim in `README.md` matches that output exactly, except lines 94-95, which must still read "249 pytest" / "across 30 files," left deliberately stale.** Not approximately — exactly, for every other claim.
2. `CHANGELOG.md` exists, and every entry in it maps to a real commit SHA that you can cite.
3. `docs/provenance-audit.md` exists with a per-file judgment and stated confidence, or an explicit statement that the audit could not be performed and why.
4. `git --no-pager log --oneline` shows `ce56441` still present and unmodified in history.
5. A proposed `pyfinlib/README.md` rewrite exists as an uncommitted diff.

## Report back

At the end of the session, append to `docs/decisions.md`, and report to the orchestrating chat:
- exact commands run and their raw output (not summaries)
- every numeric claim found and its verified value
- any discrepancy between this spec's ground-truth table and your own run
- anything you stopped and did not do, and why

A self-report of "tests pass" is a claim, not a fact. Paste the output.
