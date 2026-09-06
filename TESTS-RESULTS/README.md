# TESTS-RESULTS

Committed receipts for benchmark and correctness campaigns run against this repo.

`SOP.md` §1 is the rule this directory exists to satisfy: **a number that only exists in a
terminal is not evidence.** `benchmarks/baseline.json` is gitignored and never leaves the machine
that produced it, so anything anyone else needs to see lands here instead.

## Layout

```text
TESTS-RESULTS/
├── README.md                              # this file — the schema contract
└── YYYY-MM-DD-<campaign-name>/
    ├── raw-metrics-<machine-slug>.json    # one file per machine
    ├── logs/                              # verbatim console output, per machine
    └── SUMMARY.md                         # human-readable cross-device comparison
```

**One file per machine, not one per campaign.** Several machines contribute to a campaign, often
days apart; a single shared `raw-metrics.json` would collide. The comparison across machines
lives in `SUMMARY.md`, which is written once the machines have reported.

The campaign directory is named for the date the campaign *opened*, not the date each machine
ran. A machine joining late still writes into the original directory.

`<machine-slug>` is lowercase-kebab of the chip: `m1-pro`, `m1-max`, `m4-pro`.

## Record schema

`raw-metrics-<machine-slug>.json` is a JSON **array** of records, one per benchmark invocation.

| Field | Meaning |
|---|---|
| `timestamp` | UTC, `date -u +%Y-%m-%dT%H:%M:%SZ` |
| `machine` | `scutil --get ComputerName` (fallback `hostname -s`) |
| `chip` | `sysctl -n machdep.cpu.brand_string` |
| `hw_model` | `sysctl -n hw.model` |
| `cores` | `{total, performance, efficiency, ane}` |
| `memory_gb` | `sysctl -n hw.memsize` / 1024³ |
| `os_version` | `sw_vers -productVersion` |
| `git_commit` / `branch` | `git rev-parse --short HEAD`, `git rev-parse --abbrev-ref HEAD` |
| `target` | Source area exercised, e.g. `kernels/training/stories_train` |
| `benchmark` | Stable identifier for the run, e.g. `train_50_steps_delta` |
| `command` | The exact command, so the run is reproducible |
| `wall_clock_seconds` | Total elapsed |
| `timing_breakdown` | Per-phase ms, benchmark-specific keys |
| `throughput` | `{tokens_per_sec, tflops, ...}` as applicable |
| `status` | `PASS` / `FAIL` / `PARTIAL` |
| `notes` | One line of prose when the number needs a caveat |
| `log` | **Required.** Relative path to the verbatim console receipt |

`log` is mandatory. Most `bench` subcommands print human tables to stderr and emit no
machine-readable output (only `bench kernels` writes JSONL to stdout), so most numbers here are
transcribed by hand. The log is what makes a transcribed number checkable.

### Orion-specific fields

Records from an ANE run should also carry, where meaningful:

| Field | Meaning |
|---|---|
| `compile_count_start` / `compile_count_end` | `orion_compile_count()`; flat means delta compile held |
| `programs_compiled_at_startup` | 72 for Stories110M |
| `weight_kernels_patched` | 60 for Stories110M — 5 of 6 per layer; `sdpa_bwd2` has no weights |
| `delta_path_taken` | false if any step logged the full-recompile fallback |
| `ane_generation` | M1 / M2 / M3 / M4 — the axis campaigns exist to vary |
| `rss_peak_mb` | Memory ceiling, which matters on 16GB machines |

## Rules

1. **Commit the logs.** A record without its log is an assertion.
2. **Record failures as findings, not as absent rows.** A `COMPILE FAILED` on one chip and not
   another is the most valuable thing a multi-machine campaign produces. Give it `status: FAIL`
   and a `notes` line; do not drop it.
3. **Do not edit `RESULTS.md` from a campaign.** That file is the upstream author's record on an
   M4 Max nobody in this fork owns. It is a reference guide, not our baseline.
4. **A new hardware constraint goes in `docs/ane_constraints.md`,** per
   `GUIDING-PRINCIPLES.md` #2 — a constraint rediscovered later is a constraint that cost twice.
