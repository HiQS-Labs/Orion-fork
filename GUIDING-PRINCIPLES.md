# Guiding Principles

North star for **Orion** — a from-scratch compiler and runtime that runs small LLMs on Apple's
Neural Engine (ANE), local-only, no CoreML/Metal/GPU/cloud. When a choice is unclear, the option
that keeps the system correct, honest about hardware limits, and free of duplicated logic wins.
`AGENTS.md` is the behavioral playbook; `ARCHITECTURE.md` is the structural map; this is the *why*.

Adapted from XYZ Forge's `GUIDING-PRINCIPLES.md`, distilled to the parts that apply to a
single-repo ML compiler/runtime project — the multi-agent coordination material (event logs,
relay containment, marathon load rails) isn't reproduced here because nothing in this repo does
that job.

## The North Star

There is no perfect architecture and no finished codebase. The bar is not perfection — it is that
every change leaves this project **more durable, more reversible, and less duplicated** than it
found it, and that the three stay in balance:

- **Durable** — it removes the root cause and the next planned change builds on it, rather than
  being torn out when the next model or ANE quirk shows up.
- **Reversible** — the cost of being wrong is known and bounded before the change lands. A change
  nobody can undo is a bet, not a fix, and gets treated as one.
- **DRY** — nothing canonical lives in two places where it can drift. One graph IR, one codegen
  path, one weight format — not a second one growing quietly beside it.

**Do not build a new layer, pass, frontend, or weight format when an existing one can be extended
easily, logically, and safely.** Extending `compiler/builder.c`, an existing frontend, or an
existing pass beats standing up a parallel path, even when the parallel path is faster to write
today — the parallel path is what later has to be kept in sync, or quietly rots (see
`archive/milgen/` and the dead frontend-registry stub in `compiler/kernel_adapter.m`, both named
in `ARCHITECTURE.md`). If an existing abstraction genuinely can't carry the new case, say so in
one line, with the reason, before forking.

These three pull against each other, and that tension is the decision, not a problem to average
away: the most durable fix is often the least reversible (baking a new ANE workaround deep into
the compiler pipeline), and collapsing two near-duplicate frontends is a DRY win that can widen
the blast radius across every model that uses them. Name the trade and pick; don't split the
difference by building both.

This section is canonical. `AGENTS.md` owns how it's applied to a given change — the
reversibility scale and blast-radius sizing — and the principles below are the build-time and
done-time expression of it. Where another doc restates this, that doc is the copy and this is the
original.

## Purpose

Orion turns a model definition into MIL text (`compiler/`), compiles it against the private
`AppleNeuralEngine.framework`, and runs it with CPU picking up whatever the ANE structurally can't
do (final logits over a huge vocab, loss/optimizer state, anything needing an op the ANE MIL
compiler rejects). The goal: local inference and training on Apple Silicon that is fast because it
avoids the ANE's recompile cost (delta reload), and correct because every ANE code path has a CPU
oracle it can be checked against.

## The quality bar

Any change that touches the compiler, runtime, or kernels is high-quality only when it's all four:

- **Correct** — checked against a golden fixture, a CPU-oracle comparison, or a compiler
  equivalence diff (`compiler/mil_diff.m`) — not eyeballed.
- **Honest about hardware limits** — a documented ANE constraint (`docs/ane_constraints.md`) is
  load-bearing, not incidental. A change that silently works around one without updating that doc
  is setting a trap for the next person who hits it.
- **Measured, not asserted** — a performance claim cites a number in `RESULTS.md` or
  `docs/m2_benchmarks.md`, with enough context (date, hardware, mode) to reproduce it, not a vague
  "faster now."
- **Resolved to one path** — a fix lands in the live path (the frontend/pass/kernel actually
  wired into `orion_kernel_adapter_generate_mil_2arg` and the CLI), not a second copy that only
  some callers use.

Fail one, and the change isn't done.

## How it's built

1. **Correctness before speed.** A compiler or kernel change that isn't checked against a golden
   fixture or the CPU oracle is unverified, however fast it makes things.
2. **ANE constraints are load-bearing, not incidental.** `docs/ane_constraints.md` is the record
   of empirically-discovered hardware/compiler limits (no `concat`, weight-budget ceilings, the
   ~119-compile-per-process ceiling that motivates delta reload). A change that runs into a new
   one adds it there — the next person who hits the same wall shouldn't have to rediscover it.
3. **The CPU path is the oracle, not a legacy fallback.** `kernels/inference/decode_cpu.m` and
   `orion_gpt2_forward_cpu` exist to keep the ANE path honest. Letting it drift out of sync with
   the ANE path removes the one ground truth this system has.
4. **Build durable, not band-aid.** Durable means it removes the root cause and the next planned
   change builds on it — not a patch torn out when the next model or ANE quirk lands. A band-aid
   is wasted work unless something genuinely needs one now, and it gets flagged as temporary if so.
5. **Least code that clears the bar.** Prefer extending an existing frontend/pass/builder op over
   writing a new one; the smallest change that stays correct and durable wins. Net-new code is a
   cost to justify. Deleting dead code (see `archive/milgen/`, the registry stub) counts as
   progress.
6. **Honest reporting.** Surface what failed and why — never present an untested change as
   verified, or a partial benchmark run as a finished one. There's no CI here to catch an
   overclaim; the honesty has to come from the report itself.
7. **Docs are meant to reflect current reality.** `README.md`, `ARCHITECTURE.md`, `RESULTS.md`,
   and `docs/*.md` should describe what the code actually does. If they disagree with the code,
   the docs are the bug — fix the doc, or flag the drift, don't let it sit.
8. **Done means verified.** "Done" is `make test` / `make test-compiler` passing and, for
   anything performance-related, a number recorded somewhere durable — not work that merely looks
   finished. Say plainly when verification was partial or skipped.
9. **Non-trivial changes get a signal stream.** A compiler pass change, a new ANE workaround, a
   weight-format change — anything beyond a 2-3 line fix — gets a GitHub issue so there's a
   record of why it happened, not just what changed.

## Applying this

Adding a feature or weighing a tradeoff, ask: *does this keep the system correct, honest about
what the ANE can and can't do, and free of a second copy of logic that already exists somewhere
else? And is "done" provable by running `make test`?* If any answer is no, reconsider.

---

## Appendix: doc review checklist

When reviewing a doc in this repo (architecture notes, a benchmark writeup, a PR description),
check:

1. **Correctness backed?** Any perf or correctness claim without a fixture, oracle comparison, or
   dated number → flag it.
2. **Reuse honored?** A new frontend/pass/format where an existing one could extend → ask why not.
3. **Done verifiable?** Names a runnable check (`make test`, `make test-compiler`, a specific
   golden fixture). None named → low-quality signal.
4. **Drift reduced, not created?** No duplicated docs, no code path left half-migrated.
5. **Next action singular?** One explicit next step, not buried in prose.
6. **Destructive or irreversible steps flagged?** Surfaced before executing, not after.
