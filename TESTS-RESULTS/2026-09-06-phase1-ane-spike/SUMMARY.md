# Phase 1 ANE Spike — Cross-Device Summary

Campaign for [HiQS-Labs/Orion-fork#1](https://github.com/HiQS-Labs/Orion-fork/issues/1), Phase 1
of the Cactus Needle Oracle arc ([XYZ-forge#467](https://github.com/HiQS-Labs/XYZ-forge/issues/467)).

**Question under test:** does delta compile hold on real Neural Engine hardware — 72 ANE programs
compiled once, then every training step patching weights into the already-compiled programs
instead of recompiling, keeping `orion_compile_count()` flat and staying clear of the ANE's
~119-compiles-per-process ceiling?

| Machine | Status | Reported |
|---|---|---|
| M1 Pro / 16GB | Complete | 2026-09-06 |
| M1 Max | Not yet run | — |
| M4 Pro | Not yet run | — |

---

## Headline: training works on M1, inference does not

**The delta-compile claim holds on M1 Pro.** 72 programs compiled once in 11.1s; all 50 training
steps took the delta path with zero fallbacks and zero process restarts; loss fell 12.23 → 10.36
on real TinyStories tokens, which is what distinguishes a real training loop from one that is
merely cycling the compiler.

**The GPT-2 inference path is dead on M1-generation ANE.** All five inference kernels fail to
compile. Isolated cause: the kernels declare **fp32 program input/output**, which an
M1-generation ANE rejects outright. Recorded as
[`docs/ane_constraints.md` #18](../../docs/ane_constraints.md). The training path is unaffected
because it is fp16 `[1,C,1,S]` end to end.

That asymmetry — **training fine, inference dead, on the same chip** — is the most important
thing this campaign has produced so far, and it was invisible until someone ran the code on
something other than an M4.

---

## Results table

Absolute latencies are **recorded, not graded**. `RESULTS.md` is the upstream author's Mac Studio
M4 Max 64GB and is a reference guide only; the comparison that matters is between the three
machines in this campaign, and it cannot be drawn until the other two report.

| Metric | M1 Pro / 16GB | M1 Max | M4 Pro |
|---|---|---|---|
| `make test-compiler` | 4/4 PASS | — | — |
| Startup compile (72 programs) | 11.1 s | — | — |
| Delta path taken | 50/50 steps | — | — |
| Delta patch, avg | 837.7 ms | — | — |
| — of which save / patch | 476 ms / 362 ms | — | — |
| Avg train time per step | 3099.7 ms | — | — |
| Avg total step time | 3956.5 ms | — | — |
| Throughput | 0.449 TFLOPS | — | — |
| Loss, step 1 → 50 | 12.2324 → 10.3588 | — | — |
| Compiles during training | 0 | — | — |
| Peak RSS (training) | 2.25 GB | — | — |
| `bench kernels` (GPT-2) | **FAIL** — all 5 kernels | — | — |
| `bench inference --ane` | **FAIL** — silent CPU fallback | — | — |
| `bench swap` | **FAIL** — compile at iter 0 | — | — |
| CPU inference baseline | 65 tok/s | — | — |

### Reading the numbers

- **`grad_accum` defaulted to 10**, not the 4 in `RESULTS.md`. The canonical command in the issue
  passes no `--grad_accum`, so `STORIES_ACCUM_STEPS` = 10 applies. Every machine in this campaign
  will do the same, so the cross-device comparison is sound — but `avg_train_ms` is over ten
  microbatches and must not be held against upstream's four. `avg_recompile_ms` is per-step and
  is comparable.
- **The `--ane` inference number is a CPU number.** 63 tok/s against 65 tok/s for the explicit CPU
  run. The benchmark prints `mode: ANE full` and reports a throughput regardless of whether any
  layer reached the ANE. Do not read an `--ane` figure on M1 without checking the log for
  `falling back to CPU`.
- **Delta patch is stable, not just fast**: 50 samples spanning 812.3–937.4 ms, a 15% spread with
  no drift across the run. Whatever the absolute number is on other silicon, the mechanism is not
  degrading over steps.

---

## Findings

### 1. fp32 program I/O is rejected by M1-generation ANE — blocking

Severity: blocks the entire GPT-2 inference path and two of four benchmarks on M1.

Isolated by compiling one MIL program twice with only the I/O dtype varied, everything else
byte-identical:

```
io_dtype=fp32  -> FAILED
io_dtype=fp16  -> COMPILED
```

The generated MIL shows the fp32 is in the function signature, not the body, which is already
fp16 throughout:

```
func main<ios18>(tensor<fp32, [1,768,1,64]> x) {
    tensor<fp16, [1,768,1,1]> lnf_g = const()[...];
```

`compiler/frontends/gpt2_final.h` states the fp32 contract in its own header comment, so this is
deliberate upstream design that happens to be M4-only. Whether the M4 genuinely accepts fp32 I/O
is **unverified by this campaign** — the M4 Pro leg will settle it.

### 2. `scripts/download_data.sh` pointed at a dataset that no longer exists — fixed

`huggingface.co/datasets/karpathy/llama2c` returns 401 and is absent from HuggingFace's listing
of that author's datasets. Worse than the dead link: the script ran `curl -L` without `--fail`,
so it wrote the 29-byte body `Invalid username or password.` to
`data/tinystories_data00.bin` and **exited 0**, after which its own `if [ -f "$DEST" ]` guard
would treat that corpse as a valid cached dataset on every subsequent run.

Fixed in this campaign: the script now fetches the raw `roneneldan/TinyStories` text and
tokenizes it locally via `scripts/tokenize_tinystories.py` using the Llama2 sentencepiece model
already committed at `tokenizer/data/llama2_tokenizer.model`, with `--fail`, a size floor, and
cleanup of partial output. Produces 5,070,729 tokens (9.7 MB) — roughly 99x what a 50-step run
consumes.

### 3. The llama converter needs a checkpoint the plan never fetched — fixed in the issue

`hf_to_blobs_llama.py` has `--checkpoint` as a **required** argument and does not download
anything itself, unlike the GPT-2 converter. `model/weights/stories110M.bin` has to be curl'd
from `karpathy/tinyllamas` first. The issue's precondition omitted both facts.

### 4. `bench` telemetry is mostly not machine-readable

Only `bench kernels` writes JSONL to stdout. `bench training`, `bench inference`, and `bench swap`
print human tables to stderr only, so every number in `raw-metrics-m1-pro.json` except the kernel
rows was transcribed by hand from a log. The logs are committed so the transcription is
checkable, but adding `--json` to the other three subcommands is worth doing before this campaign
is repeated.

---

## Handoff to Phase 3

**Not written yet — this needs all three machines.** What is already settled for Needle:

Orion's delta-compile mechanism is real and works on M1-generation silicon, so the core premise
of running Needle training on the ANE via Orion survives contact with hardware. But **anything
Needle inherits from the GPT-2 inference path will not compile on M1**, and the fix — moving
program I/O to fp16 — touches every GPT-2 frontend plus the host-side IOSurface element sizing
(constraint #14 makes those two changes inseparable). Phase 3 should assume fp16-only program
boundaries from the start rather than discovering this again.

Open until the other machines report: whether fp32 I/O is M1-only or broader, and whether the
~4x gap between this machine's step time and the upstream M4 Max figure is generation, memory
(16GB vs 64GB), or the `grad_accum` difference.

---

## Reproducing

```bash
make && make test-compiler
curl -L --fail -o model/weights/stories110M.bin \
  https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin
pip install sentencepiece numpy torch transformers
python model/convert/hf_to_blobs_llama.py --checkpoint model/weights/stories110M.bin --output model/blobs/stories110m/
python model/convert/hf_to_blobs_gpt2.py --output model/blobs/gpt2_124m/
bash scripts/download_data.sh
cp -R model/blobs/stories110m model/blobs/stories110m_train   # training mutates weights in place
./orion train --weights model/blobs/stories110m_train --dataset data/tinystories_data00.bin --steps 50 --checkpoint_every 25
```

Then the four `bench` subcommands per the issue. Full commands are in the `command` field of every
record in `raw-metrics-<machine-slug>.json`.
