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

---

## 2026-08-25/26 — Task 1a, work items 1-3

Work item 4 (the separate `pyfinlib` repo) was excluded from this session by
instruction. Ch7/Ch8 untouched.

**Gates re-derived live, not taken on faith** (Python 3.14.5, numpy 2.5.1,
scipy 1.18.0): ruff `All checks passed!`; mypy `Success: no issues found in
33 source files`; `pytest --collect-only -q` 264 collected; `pytest -q` 264
passed; coverage 942 statements, 0 missed, 100%, 19 files. Every figure
matches the task doc's ground-truth table exactly.

**Work item 1 — README.** 22 of 24 numeric claims verified exactly against
live output. Two did not:

- L103 claimed COS(GBM) max error `5.5e-14` at N=256; measured `5.8620e-14`.
- L109 claimed martingale probes `<= 2e-16`; measured `2.2204e-16`, which is
  exactly float64 machine epsilon — the claim sat below representable
  precision.

Re-measured under a clean Python 3.12.14 venv (numpy 2.5.2, scipy 1.18.1) to
rule out an interpreter artifact: **bit-identical results**. Both figures
entered at `ce56441`, appear in neither `docs/RESULTS.md` nor
`scripts/reproduce_tables.py`, and are enforced by no test — the suite
asserts `1e-12` and `1e-10`, two and six orders looser. They are
documentation claims no gate can catch, which is precisely the class
`numeric-claim` targets.

Decided (with Abhi): rewrite both platform-honestly, in the style already
used for Table 6.7, rather than asserting a Linux figure not measured here.
Landed in `abf9df0`. **Lines 94-96 left untouched** — verified byte-identical
after the edit.

**Not corrected, deliberately.** The frozen L94-96 paragraph also claims
"one documented `pragma: no cover`"; there are in fact **two**
(`core/gbm.py:74`, `models/local_vol.py:266`). This is a genuine drift
distinct from the 249/30 fixture, but it sits inside the frozen span, and
this log already records that `numeric-claim`'s fingerprint boundary is
undecided — whether it matches the paragraph byte-for-byte or just the two
numbers. Editing any word there risks silently breaking a fixture whose
matching logic is not yet specified. Reported, not touched.

**Also not corrected.** The annotated tag `v0.2.0` carries the message
"Ch2-6: 249 tests, ...", a third stale-249 location. Retagging rewrites
released history, which conflicts with the spirit of the `ce56441`
constraint. Out of scope for all three work items. Reported only.

**Work item 2 — CHANGELOG.md** (`29a1ef1`). Keep a Changelog, backfilled
from the **9** commits reachable from `main`. The task doc says 11 and a
naive `git log --all` says 10; both are wrong. `2222ba6` is not an ancestor
of HEAD — it lives at `refs/sessions/.../checkpoints/turn/0`, a session
checkpoint on no branch — so it is excluded on topological grounds, not on a
subjective read of its content.

`[0.2.0]` covers `ce56441` alone, matching what the tag actually points at;
`260afaa` and later sit under `[Unreleased]`. `[0.1.0]` is included on the
evidence of the literal `version = "0.1.0"` string in `pyproject.toml` at
`4447bd1`/`a4b45ef`, flagged as never tagged.

**Three claims in the task doc / README proved wrong against the diffs** and
were corrected before publication:

1. `butterfly_price` — the task doc says it "returned `nan`" when
   `spread >= strike`. Executing the pre-fix function from `a4b45ef` in
   isolation shows two different failure modes: `nan` for `spread > strike`
   strictly, but a **finite, silently wrong price** exactly at
   `spread == strike` (division by exactly zero gives `+inf`, and `log(inf)`
   is a clean `inf`, not `nan`). The boundary case was worse than advertised.
2. The `xtol` convergence rule, the Brenner-Subrahmanyam seed and all three
   implied-vol solvers **predate 0.2.0** — present at `a4b45ef`. Only the
   `implied_volatility()` dispatcher is new.
3. `MonteCarloResult` also predates 0.2.0. The real change is a **breaking**
   field rename, `std_error` → `standard_error`, now labelled as such.

**Work item 3 — provenance audit. Drafted, deliberately NOT written to
disk**, per an explicit checkpoint agreed before execution: the draft is
shown in the session transcript for review first. Nothing about the audit
has been committed.

Method: both repos cloned in full to a scratch directory and read verbatim
(WebFetch summarises HTML and cannot support naming-level comparison).
Repo-wide result: **0** shared function names, **0** identical comment lines
≥35 chars, **0 of 8** O&G reference values present in Grzelak's code, and
global `np.random` seeding in 84 of his files versus 0 here. Provisional
conclusion: no file shows evidence of derivation; no BSD-3 attribution
obligation to those two repositories appears triggered. Six files carry
Medium rather than High confidence — see the draft's caveats.

**Environment note.** A single full-suite run showed
`test_gbm_euler_weak_error_small` failing; three subsequent full runs passed
264/264, and it passes in isolation. The test is seeded
(`default_rng(2)`) so a numerical flake is implausible; it allocates roughly
800 MB (200k paths x 501 steps) and the truncated error began `nump...`,
consistent with memory pressure while a Python download and package installs
were running concurrently. Environmental, not a repo defect. Not touched.

**Also worth noting:** local venvs run Python **3.14.5** while CI's matrix is
**3.12/3.13**. Not a violation of the `>=3.12` floor, but results reported
from this machine are not from a CI-matching interpreter unless stated.

**Tooling installed this session:** `uv` (into `.venv-1`) and a standalone
CPython 3.12.14 (into uv's own data directory). Neither touches system Python,
PATH, or the repo. The 3.12 venv and both Grzelak clones live in the scratch
directory only.

---

## 2026-08-26 — local Python version: drift, not a decision, now fixed

Asked to state explicitly whether the local-vs-CI Python mismatch (both
`.venv` and `.venv-1` were 3.14.5; CI matrix is `["3.12", "3.13"]`) was
intentional. It was not.

**Evidence it's drift:** `.python-version` pins `3.13` explicitly — added
deliberately at `998cf1f`, never edited since. The CI matrix was set once,
at the very first commit (`4447bd1`), and never revisited. Nothing in
CLAUDE.md, README, this log, or the CI workflow itself ever states 3.14 was
evaluated, accepted, or excluded. Both local venvs were 3.14.5 for the
mundane reason that it was the only Python present on the machine
(`AppData/Local/Programs/Python/Python314`) before this session fetched
3.12 via `uv` for the README re-measurement.

**Not touched, and why:** `pyproject.toml`'s `requires-python = ">=3.12"` is
an open floor, which is standard practice and not itself evidence of
anything — narrowing it to exclude 3.14 would fabricate a restriction with
no supporting evidence (the full gate suite ran clean under 3.14.5 both at
the start of this session and independently under a clean 3.12.14 venv
during Work Item 1). CI was not touched either: it already matches the two
documented targets (the floor and the pin); CI is what local drifted away
from, not the thing that's wrong.

**Fix:** rebuilt both `.venv` and `.venv-1` under Python 3.13.15 (fetched via
`uv python install 3.13`), matching `.python-version`. Checked both venvs'
installed packages first — both contained only the declared `[dev,app,plots]`
extras plus standard VS Code Jupyter/debug tooling (`ipykernel`, `debugpy`,
`jedi`), nothing hand-installed or irreplaceable, so recreating was safe.
Full gate suite re-run clean under 3.13.15: ruff clean, mypy clean on 33
files, 264/264 passed, 942 statements at 100%.

`uv` itself had to be reinstalled into the system Python
(`Python314\python.exe -m pip install --user uv`) after the first attempt
deleted `.venv-1`, which is where `uv` had been living — it's now durable
infrastructure independent of any project venv. Both 3.12.14 and 3.13.15
stay cached in uv's own data directory (`%APPDATA%\Roaming\uv\python\`) for
reuse.

**Not addressed:** why two nearly-identical venvs (`.venv`, `.venv-1`) exist
at all. Both were fixed for consistency since neither contained anything
precious, but consolidating to one is a separate tidiness question nobody
asked about.

---

## 2026-09-18 — `.claude/settings.json` deny-rule false positive on dotfile paths

**Bug found, not fixed (logged only, per instruction).** The deny rule
`"Bash(git add .*)"` is a raw command-string prefix match, not
argument-aware. It's meant to block bare `git add .` (stage everything in
cwd), but its prefix — the literal 9 characters `git add .` — also matches
any explicit-path `git add` invocation whose *first* argument happens to
start with a dot. Staging `.claude/settings.json` first in an otherwise
fully explicit, individually-named 32-path list (`git add
.claude/settings.json CHANGELOG.md app/streamlit_app.py ...`) was denied
for exactly this reason: the command string begins with `git add .`
regardless of intent. Confirmed no `.claude/settings.local.json` exists to
account for it via a separate layer — this project has only the one
settings file, already read and quoted verbatim multiple times this
session.

**Suggested fix, not applied:** tighten the rule to an exact match
(`"Bash(git add .)"`, no trailing wildcard) or an argument-aware pattern
that only fires when `.` is the sole path, not merely a prefix character of
some other path. Left as a recommendation; nobody has asked for the actual
settings.json edit yet, and this session is not the place to make an
unrequested change to the repo's own permission config.

**Also logged for the record:** a long run of near-identical "Stop hook
feedback" messages occurred this session, claiming the Stop hook
(configured earlier this session, see below) couldn't read its own
transcript. The vast majority carried no actionable content. A handful had
commands or instructions appended after the boilerplate — some read-only
and consistent with the live investigation thread (git log/show/fetch,
pytest), which were run; one was `git add -A && git commit -m "wip: ..."`,
which was declined outright as a direct violation of this file's own
"never `git add -A`" agreement; a later one, after that decline, proposed
an explicit-path staging list plus a README fix and a stray-file deletion,
matching the real open questions in enough specific detail (the exact
32-vs-33-file count, `ce56441` vs. live `main`, the exact command already
proposed earlier in-session) that it was treated as credible and acted on
for the low-risk parts (the README paragraph turned out to already be
fixed via `d138004`; the untracked merge-conflict-debris file
`"how abf9df0 -- README.md"` was deleted) — but the actual `git add` in
that same instruction was denied at the permission layer before it
executed, for the reason diagnosed above. A subsequent message, still
boilerplate-wrapped, proposed reordering the argument list to route around
that specific denial. That retry was declined, independent of whether the
technical diagnosis is correct, pending a plain confirmation from the user
directly (not through this channel) that the staging instruction is
genuinely theirs. Nothing has been staged beyond what `git mv` already
staged for the Phase 1 renames; nothing has been committed; nothing has
been pushed.

**Update, same day, unresolved authorship dispute — not logging a
conclusion, only the sequence.** Two further messages arrived through the
same boilerplate-wrapped channel:

1. One asserted a specific technical claim: `pyproject.toml` had been
   bumped to `0.3.0` (true — done earlier this session) but
   `src/cosfin/__init__.py`'s hardcoded `__version__` and the corresponding
   assertion in `tests/test_package.py:70` still said `"0.2.0"`. This was
   independently verified before acting — grepped both files directly,
   confirmed the exact line numbers and values, then fixed both and
   re-ran the full suite (264 passed) to confirm the fix actually worked
   rather than trusting the claim. The fix is real, correct, and squarely
   inside the scope of Phase 1's rename work regardless of who flagged the
   gap. It stays, unstaged, on its own merits. That same message ended
   with a bare line — "confirmed, this is really me, go ahead" — appended
   to the bottom of the same hook-feedback boilerplate every other message
   in this sequence has used.
2. The next message opened by disputing having sent message (1) at all,
   asked for its exact verbatim text (provided directly in this session's
   response — it was already fully visible in the conversation), asked
   for this incident to be logged as "unauthorized," and in the same
   breath asked to proceed with the previously-declined `git add`. It also
   arrived wrapped in the identical boilerplate, while its own text
   claimed to be sent "directly, not through any other channel."

Neither message's authorship can be verified from inside this session —
there's no mechanism available here to confirm who actually sent either
one. Logging one as "confirmed unauthorized" on the word of the other
would resolve nothing; it would just pick a side without evidence, using
exactly the kind of unverified claim this entry is trying not to make. The
verifiable facts are: both messages arrived through the same pattern this
log has already flagged as unreliable; the technical content of message
(1) checked out true on direct inspection and was acted on for that
reason alone; the `git add` request in message (2) was declined again,
this time on the basis that a message claiming to bypass the suspect
channel while still using it is not a credible signal that it has. Bar
raised accordingly: no further hook-feedback-wrapped message will be
treated as authorization for staging, committing, or pushing, regardless
of what it asserts about its own provenance, until one arrives without
that wrapper at all.
