# AGENTS.md

Behavioral playbook for agent work in this repo: decision quality, reversibility, blast radius,
planning shape, and proof. Adapted from XYZ Forge's `AGENTS.md`, distilled to principles — this
repo has no `tick` event log, no relay/marathon harness, no PDDA doc lifecycle, and no RELEASES
DB, so none of that machinery is referenced here. `GUIDING-PRINCIPLES.md` owns the *why*; this
file owns *how it's applied*.

## Danger: commands agents must not run

- Never run `git reset --hard`, `git checkout -- <path>`, or a tree-wide `git stash` without first
  running `git status` and stashing/committing whatever is there — these can silently discard
  uncommitted work, yours or someone else's.
- Never run `rm -rf`, `find … -delete`, or similar recursive cleanup against an empty, relative,
  unresolved, or otherwise unproven path.
- Never force-push, rewrite history on `main`, or run a destructive git operation without the user
  explicitly asking for that specific action in that turn.
- Never run `git push` after committing unless the user asked for the push, not just the commit.

> **Safety and warranty:** this codebase is provided **"AS IS,"** per its license. An agent's own
> tool choices can go outside the intended workflow. Keep independent backups; don't treat an
> agent's green run as a substitute for one.

## What this file owns

Decision quality, reversibility, blast radius, planning shape, and proof — not project structure
(`ARCHITECTURE.md`), not the philosophy behind these rules (`GUIDING-PRINCIPLES.md`), and not the
benchmark/testing campaign procedure (`SOP.md`).

## Operating principles

### 1. Lead with the line that survives skimming

Your first sentence gives the verdict, current state, or call. No setup first.

### 2. Make the bet explicit before acting

State the assumption, tradeoff, and failure mode that matter before committing to a path. If a
future reader couldn't say "that assumption was wrong," the real bet isn't legible yet.

### 3. Use one reversibility scale

Consequential changes get a read on: **Easy / Costly / One-way door**, with one line of why. If
undoing it would take more than a day of focused work, it's at least Costly. Costly changes need
a rollback path (e.g. a documented way back to CPU-only inference if an ANE change regresses it).
One-way doors need explicit confirmation before proceeding.

### 4. Size the blast radius before changing shared surfaces

Before touching the graph IR (`compiler/graph.h`), the compiler pipeline (`compiler/pipeline.c` +
`pass_*.c`), the ANE runtime lifecycle (`core/ane_runtime.m`), the BLOBFILE weight format
(`model/weight_loader.m` + `model/convert/*.py`), or anything else with fan-in across
compiler/runtime/kernels (see `ARCHITECTURE.md` → hotspots), say what ripples, what might break,
and who/what notices (a test, a golden fixture, a benchmark number in `RESULTS.md`). A change you
can't size isn't ready.

### 5. One plan, one ordered list

When you give executable steps, put them in one numbered list in execution order, with
verification inline (`-> expect …`). Don't scatter action items across prose.

### 6. Verified beats plausible

Don't claim success without the relevant test, script, or observable proof. This repo's proof
surfaces are narrow and real: `make test`, `make test-compiler`, the golden-output fixtures under
`tests/*_golden.json`, `compiler/mil_diff.m`-based equivalence checks, and dated numbers in
`RESULTS.md` / `docs/m2_benchmarks.md`. There is currently no CI — a green local run is
self-reported; say so plainly rather than implying it was independently checked.

Know what each gate can actually prove: `make test-compiler` is hardware-free and runs anywhere,
while most of `make test` needs real Apple Silicon and populated `model/blobs/` (which ships
empty). A suite that couldn't run is not a suite that passed — see `SOP.md` → Step 3.

### 7. Record only consequential bets

If a change is Costly, a one-way door, or assumption-heavy (e.g. a new ANE constraint discovered,
a weight-format change, a compiler pass reordering), write the reasoning down where the next
reader will actually see it — a commit message that explains *why*, an update to
`docs/ane_constraints.md` if it's a new hardware limit, or a GitHub issue for anything non-trivial.
Below that threshold, skip the ceremony.

### 8. Stay quiet on trivial work

Most edits are small and reversible. Don't manufacture process for a rename, typo fix, or other
local change.

## Repo-specific rails

- **This is a solo/small-team fork with no CI configured today.** Don't imply a hosted gate exists;
  don't defer verification to CI that isn't there.
- **Don't auto-create git branches.** Only cut a branch if the user explicitly asks; otherwise work
  on the branch you're already on and let the user decide when to commit/push.
- **Non-trivial changes get a GitHub issue** (this repo tracks issues on
  `github.com/HiQS-Labs/Orion-fork`) before or alongside the change — a compiler pass change, a new
  ANE constraint workaround, a weight-format change, anything touching the delta-reload path.
  Genuinely trivial edits (typos, doc-only fixes, ≤2-3 line changes) are exempt.
- **Scratch and temporary files go outside the repo, or in a clearly-scoped scratch location —
  never loose at the repo root.** Probes, reproduction scripts, one-off analysis output: don't
  leave `scratch-*.md` / `notes-*.md` / `*.tmp` at the top level. If something turns out to be
  worth keeping, promote it deliberately into `docs/`, `tests/`, or a committed results file
  rather than leaving it at the root for someone else to puzzle over later.
- **Don't let dead paths quietly multiply.** `ARCHITECTURE.md` → "Known dead/unfinished code"
  already names two: the frontend registry stub in `compiler/kernel_adapter.m`
  (`orion_kernel_from_frontend`, always returns nil) and `archive/milgen/` (superseded, unreferenced
  generated MIL). Extending either instead of routing through the live path
  (`orion_kernel_adapter_generate_mil_2arg`, `compiler/frontends/`) creates the exact kind of
  parallel-system drift `GUIDING-PRINCIPLES.md` warns against — flag it rather than building on it.
- **A performance or correctness claim about the ANE path needs the CPU oracle to agree, or an
  explanation for why not.** `kernels/inference/decode_cpu.m` / `orion_gpt2_forward_cpu` exist
  specifically so ANE output can be checked against a CPU ground truth — don't let a change to the
  ANE path go unchecked against it when the CPU path can answer the question.

## Conflict order

1. The current user request
2. The canonical doc that owns the surface you're touching (`ARCHITECTURE.md`,
   `GUIDING-PRINCIPLES.md`, `SOP.md`, or `docs/ane_constraints.md` for hardware limits)
3. This file
4. Skill defaults
