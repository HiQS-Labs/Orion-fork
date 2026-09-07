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
| M1 Max / 64GB | Complete | 2026-09-07 |
| M4 Pro / 24GB | Complete | 2026-09-06 |

**All three machines have now reported.** Phase 1 is complete.

---

## Headline: delta compile holds on both generations; the inference blocker is M1-only

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

**The M4 Pro leg settles it: constraint #18 is M1-generation-specific.** All five GPT-2 kernels
compile on M4 Pro with the fp32 program I/O completely unchanged, and `bench swap` — which dies at
iteration 0 on M1 — runs 100/100 compile-evict cycles. Same binary, same commit, same flags. Per
the decision table in [#1](https://github.com/HiQS-Labs/Orion-fork/issues/1), that puts
[#2](https://github.com/HiQS-Labs/Orion-fork/issues/2) in the **bounded compatibility-fix** branch
rather than the "nobody in this fleet can run ANE inference" branch: move program I/O to fp16 so
one codebase runs on both generations.

**Delta compile is not merely present on M4, it is comfortably faster.** 542.5 ms average patch
against 837.7 ms on M1 Pro (1.54x), and startup compile of 72 programs in 3.9 s against 11.1 s
(2.8x). The mechanism holds on both generations tested, which is the single result Phase 1 existed
to establish.

**M1 Max closes the campaign and reproduces M1 Pro exactly where it counts.** The loss curve is
*bit-identical* to M1 Pro — 12.2324 → 10.3588 at both endpoints, to four decimals — which is the
expected result for two machines sharing the same M1-generation ANE and the same fp16 accumulation
order, and is stronger evidence than any latency number that the two machines are running the same
computation. Delta path 50/50, zero fallbacks, compile count flat at 72, all five GPT-2 kernels
COMPILE FAILED, `bench swap` dead at iteration 0. Constraint #18 is therefore a property of the M1
generation, not of one laptop.

**M1 Max was measured twice, and the second run settles it.** Run 1 (2026-09-06) was taken on a
host carrying a persistent foreign load — a Python process at ~92% CPU that had been running for
2 days 21 hours — and came out ~23% slower than M1 Pro on a chip whose ANE and CPU topology are
identical to M1 Pro's. Run 2 (2026-09-07) repeated it after a reboot with Time Machine stopped and
nothing above 10% CPU. **M1 Max is now 2.0% faster than M1 Pro on average total step, and 5-7%
faster on every `bench training` phase** — the expected direction for the same ANE and CPU
topology with twice the memory bandwidth. The table below reports run 2; run 1 is retained in
`raw-metrics-m1-max.json` as the evidence behind finding 7.

**The surprise nobody asked for: on M4 Pro, ANE inference is slower than CPU.** 160 tok/s on the
ANE against 255 tok/s on the CPU, on the same machine, with the ANE path genuinely executing — 49
programs compiled and cached, zero fallback lines in the log. So the M4's reward for fixing #2 is
an inference path that works and still loses to the CPU at this shape. See finding 5.

---

## Results table

Absolute latencies are **recorded, not graded**. `RESULTS.md` is the upstream author's Mac Studio
M4 Max 64GB and is a reference guide only; the comparison that matters is between the machines in
this campaign. Two of three have reported, so the M1-vs-M4 comparison below is real; the M1 Max
column stays blank until it runs.

| Metric | M1 Pro / 16GB | M1 Max / 64GB | M4 Pro / 24GB |
|---|---|---|---|
| `make test-compiler` | 4/4 PASS | 4/4 PASS | 4/4 PASS |
| Startup compile (72 programs) | 11.1 s | 9.4 s | **3.9 s** |
| Delta path taken | 50/50 steps | 50/50 steps | 50/50 steps |
| Delta patch, avg | 837.7 ms | 815.5 ms | **542.5 ms** |
| — of which save / patch | 476 ms / 362 ms | 441.0 ms / 374.5 ms | 349.4 ms / 193.1 ms |
| Delta patch spread | 15% | 16% | 20% |
| Avg train time per step | 3099.7 ms | 3050.7 ms | 2206.6 ms |
| Avg total step time | 3956.5 ms | 3878.2 ms | 2762.4 ms |
| Throughput | 0.449 TFLOPS | 0.456 TFLOPS | 0.631 TFLOPS |
| Loss, step 1 → 50 | 12.2324 → 10.3588 | 12.2324 → 10.3588 | 12.2326 → 10.4402 |
| Compiles during training | 0 | 0 | 0 |
| Peak RSS (training) | 2.25 GB | 2.35 GB | 2.92 GB |
| `bench kernels` (GPT-2) | **FAIL** — all 5 kernels | **FAIL** — all 5 kernels | **PASS** — all 5 compile |
| `bench inference --ane` | **FAIL** — silent CPU fallback | **FAIL** — silent CPU fallback | **PASS** — 160 tok/s, real ANE |
| `bench swap` | **FAIL** — compile at iter 0 | **FAIL** — compile at iter 0 | **PASS** — 100/100, RSS 1.52x |
| CPU inference baseline | 65 tok/s | 62 tok/s | 255 tok/s |

All three machines ran the same commands against a **binary-identical Orion**. The commits differ
(M1 Pro `1a629ac`, M4 Pro `f9e9f93`, M1 Max `238816c`) because each machine committed its own
results before the next ran, but `git diff 1a629ac..238816c -- compiler core kernels apps model
tokenizer Makefile` is empty: every change in that range is docs, `scripts/`, or `TESTS-RESULTS/`.
The three columns compare the same binary.

The M1 Max column is **run 2, the clean-host re-run**. Its run-1 figures (delta patch 1031.2 ms,
avg total step 4869.0 ms, 0.366 TFLOPS) are retained in `raw-metrics-m1-max.json` marked
`superseded_by`, and are the subject of finding 7.

**The two M1 machines land on top of each other, as they should.** Same 16-core ANE, same 8P+2E
CPU topology; M1 Max's edge (2.0% on total step, 5-7% on `bench training` phases, 15% on startup
compile) tracks its higher memory bandwidth. Three independent runs across two M1 machines
produced the *same loss to four decimals* — 12.2324 → 10.3588 — which is the strongest evidence in
this campaign that the delta-compile path is deterministic and genuinely training.

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
- **Delta patch is stable, not just fast**: on M1 Pro, 50 samples spanning 812.3–937.4 ms, a 15%
  spread with no drift across the run; on M4 Pro, 507.2–615.5 ms, a 20% spread, likewise flat.
  Whatever the absolute number is on other silicon, the mechanism is not degrading over steps on
  either generation.
- **The M4 Pro training run is deterministic.** It was executed twice from freshly re-converted
  pristine weights and produced a final loss of 10.4402 both times, to four decimals. The dataset
  step is deterministic too — 5,070,729 tokens on both machines, from a locally tokenized corpus.
  A repeated number is worth more than a single one, and this one cost 2.5 minutes.
- **The M1 Pro / M4 Pro loss curves are not identical and should not be** — 10.3588 against
  10.4402 after 50 steps from the same starting loss of ~12.232. Same seed and same data, different
  fp16 accumulation order on different silicon. The trajectories agree; the last digits do not.
- **ANE is not automatically the fast path.** On M4 Pro the CPU decode beats the ANE decode by
  1.6x. Phase 3 should not assume that reaching the ANE is the same as winning.

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
deliberate upstream design that happens to be M4-only.

**Settled 2026-09-06 by the M4 Pro leg: the M4 does genuinely accept fp32 program I/O.** All five
kernels compile, and the `bench swap` synthetic program that dies at iteration 0 on M1 completes
100/100 cycles. The constraint is M1-generation-specific, so #2 is a **bounded compatibility fix**
— move program I/O to fp16 so one codebase serves both generations — and not evidence that the
fp32 contract is wrong everywhere. `docs/ane_constraints.md` #18 has been updated from "not yet
checked on M4" to the measured result.

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

### 5. On M4 Pro, the ANE inference path loses to the CPU

With #2's blocker absent, M4 Pro is the first machine in this campaign where an `--ane` inference
number means what it says: 49 programs compiled and cached, zero CPU-fallback lines. The number it
means is **160 tok/s, against 255 tok/s for the CPU path on the same machine** — the ANE is 0.63x
the CPU.

This is not a defect and it is not a regression; it is what single-token decode looks like on a
processor built for wide batched work. Prefill is 6.9 ms on ANE against 4.1 ms on CPU, and the
per-kernel evals from `bench kernels` are all under 0.2 ms, so the time is not going into the
kernels — it is going into per-invocation dispatch across 49 programs to produce one token.

It matters for Phase 3 because it separates two questions that are easy to conflate: *can Needle
reach the ANE* (yes, on M4) and *should it* (not for this access pattern). The training path,
which batches real work per step, is where the ANE earns its place — and that is the path that
already works on both generations.

### 6. `bench training` silently overwrites the pristine weights

The Step 4 command in [#1](https://github.com/HiQS-Labs/Orion-fork/issues/1) is
`./orion bench training --steps 20`, with no `--weights`. That defaults to
`model/blobs/stories110m` (`apps/cli/commands/bench.m:317`) — the **pristine reference set** — and
the trainer writes weights back in place on every delta step, which is exactly what Precondition 4
warns about for `./orion train`. Following the issue verbatim on this machine trained 20 steps into
the reference blobs; they were re-converted from the checkpoint and the benchmark re-run against
`stories110m_train` for the record above. The M1 Pro record used `--weights model/blobs/stories110m_train`,
so machine 1 evidently hit or anticipated this, but the issue text was never corrected.

Fix is one flag in the issue's Step 4. Worth doing before the M1 Max run.

**Resolved.** Step 4 was corrected on 2026-09-06 — it now passes
`--weights model/blobs/stories110m_train` and carries a ⚠️ callout explaining why the flag is not
optional. The M1 Max run followed the corrected text, and its pristine blobs were verified
untouched by mtime afterwards, so the fix is confirmed working by the machine it was written for.

*(An earlier revision of this paragraph, and the M1 Max issue comment, both claimed Step 4 was
"still uncorrected." That was wrong — written from this finding's original wording without
re-reading the live issue. Corrected here rather than silently deleted, because the campaign's own
rule is that a wrong number gets a correction, not an edit.)*

### 7. The M1 Max host was contended, and the campaign has no way to say by how much

The M1 Max leg ran on a machine carrying a persistent foreign load: a Python 3.14 process at ~92%
CPU with an elapsed time of **2 days 21 hours**, plus `mds_stores` (Spotlight) at ~37% and
`WindowServer` at ~47%. The 1-minute load average taken immediately after the training run was
**69.66** on a 10-core machine.

The effect is visible in the numbers rather than merely suspected. M1 Max and M1 Pro have the same
16-core ANE and the same 8P+2E CPU topology, so on this workload they should land within noise of
each other. M1 Max instead came out **23% slower** on average total step time (4869.0 ms against
3956.5 ms), and its per-step delta spread was **807–1548 ms (91%)** against M1 Pro's 15% and M4
Pro's 20%. A wide, drifting spread on otherwise identical silicon is what contention looks like.

**Resolved 2026-09-07 by re-running on a rebooted host.** The operator rebooted and stopped Time
Machine; the re-run waited for the post-boot storm to decay (load 174 → 3.8 over 29 minutes) and
fired only once nothing exceeded 10% CPU. Run 2 confirms the hypothesis exactly:

| | Run 1 (contended) | Run 2 (clean) | M1 Pro |
|---|---|---|---|
| Delta patch, avg | 1031.2 ms | **815.5 ms** | 837.7 ms |
| Delta spread | **92%** | **16%** | 15% |
| Avg total step | 4869.0 ms | **3878.2 ms** | 3956.5 ms |
| Throughput | 0.366 TFLOPS | **0.456 TFLOPS** | 0.449 TFLOPS |
| Loss step 1 → 50 | 12.2324 → 10.3588 | 12.2324 → 10.3588 | 12.2324 → 10.3588 |

The spread is the tell. It collapsed from 92% to 16%, landing beside M1 Pro's 15% and M4 Pro's
20% — a wide, drifting spread on identical silicon was contention, exactly as suspected, and it is
a better contention detector than the mean. The "M1 Max is 23% slower" artifact is gone: it is now
2.0% *faster* than M1 Pro, in line with its higher memory bandwidth.

Three things worth keeping from this:

1. **The correctness results never moved.** Compile success/failure is a property of the ANE
   compiler, not of host load, and the loss curve was bit-identical across all three runs. Those
   are the results Phase 1 existed to produce, and they were trustworthy even from the bad run.
2. **A load check before a measured run is cheap and would have saved this one.** `uptime` plus
   the top few CPU consumers takes seconds. `SOP.md` should say so, and does not yet.
3. **Post-reboot is not the same as quiet.** The re-run's first sample was load **174** at five
   minutes uptime — worse than the contention it was replacing. Spotlight, Apple asset downloads,
   Backblaze and a 786 GB Time Machine backup all start at boot. Waiting for decay took 29 minutes;
   running immediately after a reboot would have produced a third bad number.

---

## Handoff to Phase 3

**All three machines have reported. Phase 1 is complete.** M1 Max ran last and behaved exactly as
M1 Pro did — same compile failures, same delta-compile success, and a loss curve identical to four
decimals — so the expectation recorded here before it ran was correct, and nothing in the plan below
changed as a result.

**Which ANE generations are viable for Needle: both tested generations are viable for training,
only M4 is viable for inference.** Orion's delta-compile mechanism — 72 programs compiled once,
then weight patching in place — held on M1 Pro and M4 Pro alike: 50/50 steps on the delta path,
zero fallbacks, zero process restarts, compile count flat at 72, and a falling loss curve on both
proving patched weights genuinely reach the ANE rather than the loop cycling the compiler. M4 Pro
is 1.54x faster per patch and 2.8x faster at startup, so newer silicon helps but is not required.
The constraint Needle must design around is **fp32 program I/O, which M1-generation ANE rejects
outright and M4 accepts** (`docs/ane_constraints.md` #18, verified in both directions here): every
GPT-2 inference frontend declares fp32 at the program boundary, so on M1 the entire inference path
fails to compile while the fp16 training path is untouched. Phase 3 should assume **fp16-only
program boundaries from the start** — that is the one change that makes a single codebase run on
both generations, and constraint #14 couples it to host-side IOSurface element sizing, so the
frontend dtype change and the host buffer change must land together or the failure mode is silent
wrong data rather than a compile error. Two further limits are worth carrying forward: `bench
kernels` reports a **GPT-2 working set of ~67.1 MB against a 32 MB SRAM budget**, so Needle's
layers will need to fit a budget this model already exceeds; and on M4 Pro the ANE inference path,
once it compiles, is **slower than the CPU** (160 vs 255 tok/s), so reaching the ANE and benefiting
from it are separate questions — the batched training step is where the ANE pays, not single-token
decode.

No longer open: whether fp32 I/O is M1-only — it is, confirmed in both directions across two M1
machines and one M4. Whether an M1 Max behaves as an M1 Pro — it does, on every load-independent
axis measured.

**M1-generation latency budget for Phase 3 planning, measured on a quiet host:** ~3.9 s per
training step at `grad_accum` 10, of which ~0.82 s is the delta patch (0.44 s writing weights to
disk, 0.37 s reloading into the ANE), plus a one-off ~9-11 s to compile 72 programs at startup.
M4 Pro does the same work in ~2.8 s per step with a 3.9 s startup. Both M1 machines agree within
2%, so this budget can be treated as the M1 generation's, not one machine's.

Still open, and out of scope for this phase: the gap against upstream's M4 Max `RESULTS.md` figures
remains unattributed between generation, memory, and the `grad_accum` 10-vs-4 difference.

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

Then the four `bench` subcommands. **Pass `--weights model/blobs/stories110m_train` to `bench
training`** — the issue's Step 4 omits it and the default overwrites the pristine blobs (finding 6).
Prefix each run with `/usr/bin/time -l` to capture peak RSS, which the `bench` summaries do not
report for the `train` subcommand. Full commands are in the `command` field of every
record in `raw-metrics-<machine-slug>.json`.
