# Standard Operating Procedure (SOP): Benchmark & Correctness Runs

> **Scope & relationship to the other two docs:**
> - **`GUIDING-PRINCIPLES.md`** is the canonical source for the durable/reversible/DRY bar and
>   "extend what exists, don't fork a parallel system." Nothing here restates it.
> - **`AGENTS.md`** owns repo-wide behavioral governance: reversibility scale, blast-radius
>   sizing, verified-beats-plausible, issue-first for non-trivial changes.
> - **`SOP.md` (this file)** is the tactical procedure for running and recording a benchmark or
>   correctness campaign against Orion's compiler/runtime — the kind of run that produces a number
>   that ends up in `RESULTS.md` or `docs/m2_benchmarks.md`, or a finding that needs a fix.

Adapted from XYZ Forge's `SOP.md`, distilled to what applies to a solo repo with no CI, no
telemetry harness, and no results-ingestion tooling — the ATE variation-matrix machinery, LM
Studio triage classifier, and `TESTS-RESULTS/` registry from the source doc aren't reproduced
here because none of it exists in this repo. What's kept is the *shape*: verify locally first,
isolate anything long or destructive, record the result somewhere durable, file real defects
immediately.

## 1. Governance & principles

- **Verified beats plausible.** Any performance or correctness claim about this repo backed by a
  run must be reproducible from what's committed — a dated entry in `RESULTS.md` /
  `docs/m2_benchmarks.md`, or a passing golden fixture under `tests/`. A number that only exists
  in chat or a local terminal isn't evidence.
- **Isolate anything long-running or destructive.** Don't run a multi-hour benchmark sweep, a
  fuzz loop, or anything that mutates model weights/checkpoints on disk in the working tree you're
  actively editing. Use a separate clone, or at minimum a scratch directory outside the repo for
  generated artifacts (checkpoints, temp weight files).
- **File confirmed defects immediately.** When a benchmark or test run turns up a real,
  reproducible defect — an ANE constraint violation, a golden-output mismatch, a perf regression
  against a recorded baseline — open the GitHub issue right away rather than sitting on it; filing
  is cheap and reversible. Only genuinely ambiguous findings get *offered* ("want this filed?")
  instead of filed outright.

## 2. Standard workflow

```
[1. Scope the run]
        |
        v
[2. Isolate if long/destructive]  --> (separate clone or scratch dir)
        |
        v
[3. Local gate preflight]         --> (make test && make test-compiler)
        |
        v
[4. Execute the run]              --> (./orion bench ..., or a targeted test)
        |
        v
[5. Compare against baseline]     --> (RESULTS.md / docs/m2_benchmarks.md / golden fixture)
        |
        v
[6. Triage any finding]           --> (file GH issue if it's a real, reproducible defect)
        |
        v
[7. Record & close out]           --> (update RESULTS.md / docs / open PR)
```

## 3. Step by step

### Step 1: Scope the run
State what's being measured or verified and why (a specific change, a regression check, a new
hardware configuration). If it's non-trivial, this doubles as the GitHub issue per `AGENTS.md`.

### Step 2: Isolate if the run is long or destructive
For anything that writes checkpoints, patches weight files in place, or runs long enough to be
disruptive to interactively work in the same tree:

```bash
git clone . ../Orion-fork-<topic>
cd ../Orion-fork-<topic>
```

Short, read-only benchmark runs (a single `./orion bench` invocation) don't need this.

### Step 3: Local gate preflight
Before trusting any number or fixing anything based on a run:

```bash
make test
make test-compiler
```

There's no CI here — this is the entire gate. If either fails, the run downstream of it isn't
trustworthy evidence of anything.

### Step 4: Execute the run
Use the CLI directly:

```bash
./orion bench --ane            # ANE kernel/e2e latency
./orion infer  --ane --prompt "..."   # spot-check a specific behavior
./orion train  --steps N              # a training-step timing/correctness pass
```

Note the exact command, model/bucket, and mode (CPU / hybrid / full-ANE) — that context is what
makes the resulting number reproducible later.

### Step 5: Compare against baseline
Check the new result against what's already recorded in `RESULTS.md` or
`docs/m2_benchmarks.md`, or against the relevant golden fixture
(`tests/forward_golden.json`, `tests/test_infer_golden.json`, `tests/tokenizer_golden.json`) via
the matching test binary. A result with nothing to compare against is a first data point, not yet
a claim of "faster" or "regressed."

### Step 6: Triage any finding
- **Real, reproducible defect** (ANE constraint hit, golden mismatch, confirmed regression): file
  the GitHub issue now, with the exact command and the numbers/diff as evidence.
- **Ambiguous** (flaky, hardware-dependent, unclear root cause): offer to file rather than filing
  silently, and say what's unclear.

### Step 7: Record & close out
1. Update `RESULTS.md` and/or `docs/m2_benchmarks.md` with the dated result.
2. If a fix landed, reference the run/finding in the commit message or PR description.
3. Open the PR against `main` (this repo has no `development` branch — everything targets `main`).

## 4. Maintainer defaults (optional downstream)

These exist to keep the maintainer's own flow predictable. Downstream forks are free to ignore
them.

- **Don't auto-create branches.** A branch gets cut only when the user explicitly asks for one,
  per `AGENTS.md`. In the meantime, work happens on whatever branch is already checked out.
- **Direct commits to `main` happen only when explicitly requested.** Otherwise, non-trivial work
  gets its own branch + PR so there's a review point before it lands.
- **Never push without being asked.** Committing and pushing are separate asks — a request to
  commit doesn't imply a request to push.
