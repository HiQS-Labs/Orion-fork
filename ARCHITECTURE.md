# Orion Architecture

Orion is a local AI runtime for training and running small LLMs (GPT-2 124M, Llama2-style
Stories110M) directly on Apple Silicon's Neural Engine (ANE) — no CoreML, no Metal, no GPU,
no cloud. It ships its own graph IR and MIL (Model Intermediate Language) compiler, talks to
the private `AppleNeuralEngine.framework` directly, and works around the ANE's ~119-compile
per-process limit with a "delta reload" scheme that patches weights into already-compiled
programs instead of recompiling them.

Language mix: Objective-C/Objective-C++ (`.m`/`.h`) for almost everything, a couple of Python
scripts for offline weight conversion, a stub Swift macOS app, built with a plain `Makefile`
(`xcrun clang`, frameworks: Foundation, IOSurface, Accelerate) — no Xcode project required for
the CLI/runtime.

## The three layers

```
apps/cli (infer | train | bench)          <- entry points, arg parsing, orchestration
        |
compiler/  (frontends -> graph IR -> passes -> codegen)   <- model def -> MIL text
        |
core/ + kernels/  (ane_runtime, program_cache, kv_cache, decode/prefill/train kernels)
        |
AppleNeuralEngine.framework (private, dlopen'd)  +  CPU (Accelerate/BLAS)
```

Every layer above is reachable from `apps/cli/commands/bench.m`, which is the only CLI command
that drives the compiler pipeline directly (frontends → validate → pipeline → codegen); `infer.m`
and `train.m` only go through the pre-wired kernel adapters in `kernels/inference/` and
`kernels/training/`.

## 1. Compiler — model definition to MIL text

- **IR** (`compiler/graph.h/.c`): `OrionGraph` is a flat, pointer-free struct — a fixed
  `nodes[4096]` array plus named `inputs[16]`/`outputs[16]` tables. `OrionNode` holds an
  `OrionOp` enum (1:1 with MIL ops), shape in ANE layout `[batch,channels,1,seq]`, and up to 8
  input indices into the same array. No heap pointers inside a node, so `orion_graph_free` is a
  plain `free()`.
- **Builder** (`compiler/builder.c/.h`): fluent `orion_gb_*` API — primitive ops (`conv1x1`,
  `add/sub/mul`, `matmul`, `reshape`, `cast`, activations, reduces) plus composites built purely
  from primitives (`orion_gb_layernorm`, `orion_gb_gelu`, `orion_gb_silu`, `orion_gb_rmsnorm`,
  `orion_gb_linear`). This is the shared vocabulary every frontend and `patterns.c` build on.
- **Frontends** (`compiler/frontends/*.c`): one per model piece — `gpt2_decode.c` (traced in
  detail: per-layer decode split into a `proj` graph and an `ffn` graph, each its own MIL
  program), plus `gpt2_prefill.c`, `gpt2_final.c`, `lora.c`, `stories_train.c`,
  `classifier_softmax.c` (not traced in depth — see Unknowns).
- **Patterns** (`compiler/patterns.c`): multi-node composites above the builder —
  `orion_pattern_attention`, `orion_pattern_ffn`, `orion_pattern_residual`,
  `orion_pattern_cast_to_fp16/fp32` — used directly by the frontends.
- **Passes** (`compiler/pass_*.c`), run via `orion_pipeline_optimize` (`identity → cast → dce`,
  fixpoint loop, max 20 iterations) and `orion_pipeline_ane_passes` (`sram` budget estimate →
  `uniform_outputs`):
  - `pass_identity.c` — eliminate identity nodes, rewrite downstream refs.
  - `pass_cast.c` — hoist/eliminate redundant casts.
  - `pass_conv_bias.c` — conv+bias fusion, **present but disabled**: ANE MIL doesn't support
    `bias=` on conv.
  - `pass_dce.c` — mark-and-sweep liveness from graph outputs.
  - `pass_sram.c` — per-node tensor byte size for SRAM budget (analysis, not a rewrite).
  - `pass_uniform_outputs.c` — enforce ANE's uniform-output-channel-count constraint.
  - `pass_ane_validate.c` — collects ANE-specific constraint violations, separate from the
    generic `validate.c` (which rejects empty graphs, the banned `CONCAT` op — ANE has no
    concat — dead/dangling references, and cycles via `topo.c`'s Kahn's-algorithm sort).
- **Codegen** (`compiler/codegen.m`, `orion_codegen_mil`): topo-sorts the graph and walks it
  emitting textual MIL (`program(1.3)`, `func main<ios18>`) — the sole place the graph IR
  becomes MIL text.
- **Kernel adapter** (`compiler/kernel_adapter.m`): `orion_kernel_adapter_generate_mil_2arg` is
  the single choke point — frontend → validate → optimize → codegen → free — that every
  layer-specific frontend funnels through (hence its high fan-in in the call graph). A separate
  frontend *registry* (`orion_kernel_from_frontend`) exists but its `generate_mil` stub always
  returns nil; real kernels bypass it and call the adapter directly — **this registry path looks
  like dead/unfinished code**.
- `mil_diff.m` — normalizes and diffs two MIL text blobs for compiler-equivalence tests, not part
  of the compile path itself.

## 2. Runtime & kernels — compiled MIL to running tensors

- **ANE runtime** (`core/ane_runtime.m/.h`): `orion_ane_init()` dlopens
  `AppleNeuralEngine.framework` and resolves private classes by name (must run first).
  `orion_compile_mil()` builds an `_ANEInMemoryModelDescriptor` from MIL text + a weight dict,
  writes MIL text and every weight blob to a per-model temp directory (the ANE compiler reads
  weights from disk, not memory), compiles, loads, and wraps the result in an opaque
  `OrionProgram*`. `orion_eval()` wraps IOSurface-backed inputs/outputs in `_ANEIOSurfaceObject`
  and issues an `_ANERequest`. `orion_release_program()` unloads and cleans up the temp dir.
- **Delta reload** — the key perf trick, built around a hard ~119-compile-per-process ANE limit:
  - `orion_program_patch_weights()` reuses a donor program's compiled `net.plist`, writes only
    new weight blobs, and reloads without recompiling.
  - `orion_program_reload_weights()` goes further: unload → overwrite weight files in place →
    reload the *same* model object, no new descriptor at all.
  - Per `README.md`/`RESULTS.md`: 72 ANE programs compiled once (~4.5s) at training start, then
    every step reloads weights instead of recompiling — 4,200ms→494ms per step (8.5x), 0 compiles
    during steady-state training.
- **Program cache** (`core/ane_program_cache.m/.h`): keyed by
  `"<kernel_name>:<layer_idx>:<weights_id>:<bucket>"`; ObjC wrapper `dealloc` calls
  `orion_release_program`, so eviction is safe. `orion_cache_evict(weights_id)` drops all entries
  for a training checkpoint step.
- **Weights**: `model/weight_loader.m` reads the custom **BLOBFILE** format (128-byte header,
  magic `0xDEADBEEF`, fp16 payload) into CPU fp32 arrays; `model/convert/hf_to_blobs_gpt2.py` /
  `hf_to_blobs_llama.py` (offline, Python + HuggingFace `torch`/`transformers`) are the only
  producers of this format — Python never runs at inference/training time, only for one-time
  conversion. `model/model_registry.m` is a static table of model configs + valid ANE bucket
  sizes (`gpt2_124m`: {32,64,128,256,512,1024}; `stories110m`: none). `core/bucket.h` picks the
  smallest bucket ≥ actual sequence length, mapping variable-length prompts onto fixed
  pre-compiled MIL program shapes.
- **Inference kernels** (`kernels/inference/`):
  - `prefill_ane.m` — embed (CPU) → transpose into an IOSurface → 12× (ANE attention → ANE FFN)
    → ANE final LN → CPU logits, populating the KV cache as it goes.
  - `decode_ane.m` — per layer: ANE `decode_proj` (LN1+QKV) → CPU cross-attention against the KV
    cache + output-proj + residual → ANE `decode_ffn` (LN2+FFN+residual). Final logits are
    computed on CPU because the vocab embedding (`wte`) is too large for the ANE.
  - `decode_cpu.m` — a full parallel CPU implementation, used both as a correctness oracle and
    as an actual fallback inference path. Three inference modes exist end to end: CPU-only,
    ANE-prefill+CPU-decode (hybrid), full-ANE.
  - `kv_cache.m` — flat CPU fp32 buffer `[n_layer, n_head, max_seq, head_dim]`, no ANE
    involvement.
- **Training kernels** (`kernels/training/stories_train.m`): per layer, ANE forward
  (`fwd_attn`→CPU residual→`fwd_ffn`→CPU residual) then ANE backward
  (`ffn_bwd`→CPU→`sdpa_bwd1`→`sdpa_bwd2`→`qkv_bwd`→CPU). Loss, Adam optimizer state, and
  weight-gradient accumulation are **permanently CPU-side by design** — ANE weights are baked
  constants at compile time, `dW` needs `cblas_sgemm`, NLL loss needs `gather` (not in MIL), and
  the classifier's 32000-channel backward conv is rejected by ANE. `data_loader.m` mmaps a
  Karpathy-style pretokenized `uint16` file (no tokenizer at train time). Each training step
  writes updated weights to disk and calls `orion_program_reload_weights()` on **5 of the 6**
  per-layer ANE kernel handles (`patch_layer`, `stories_train.m:993-1023`) — `sdpa_bwd2` has no
  weights and its program is left untouched. If any layer's patch fails it falls back to a full
  `orion_trainer_recompile` (`stories_train.m:1042`).
- **Compile budget guard** (`orion_trainer_needs_restart`, `stories_train.m:1055-1064`): the
  trainer tracks `orion_compile_count()` against `STORIES_MAX_COMPILES` (100, deliberately
  conservative against the ~119 hard limit) and reports when a process restart is needed before
  the next full recompile would fit — `n_layers * 6` compiles.
- Supporting: `core/lora_adapter` (LoRA A/B matrices into IOSurface tensors, hot-swappable
  without recompiling), `core/checkpoint` (binary checkpoint format, magic `BLZT`, compatible
  with a prior "ANEgpt" project), `core/iosurface_tensor` (the CPU↔ANE handoff buffer type,
  fp16 `[1,C,1,S]`), `core/profiler` (latency/throughput capture feeding `bench`).

## 3. Entry points & tooling

- **CLI** (`apps/cli/main.m` — thin dispatcher to 3 subcommands):
  - `infer` — tokenize (GPT-2 BPE) → load BLOBFILE weights → ANE or CPU prefill → decode loop
    with graceful ANE→CPU fallback at each step.
  - `train` — Stories110M only; loads pretokenized binary data directly, no tokenizer;
    checkpoint save/resume; drives the training kernels above.
  - `bench` — the only command that touches the compiler pipeline directly; per-kernel ANE
    latency, e2e throughput, training step-breakdown, with baseline regression tracking.
- **macapp** (`apps/macapp/*.swift`) — **stub only** (`// TODO(M5)`), no bridging header, not
  wired to the C core, not built by the Makefile. Planned GUI shell, not yet real.
- **Tokenizers**: `tokenizer/gpt2_bpe` is live (used by `infer`/`bench`/tests).
  `tokenizer/sentencepiece_wrap` (Llama-style, 32k vocab) is implemented and tested
  (`test_sp_tokenizer.m`) but **no CLI command currently calls it** — `train` consumes
  pre-tokenized binaries directly.
- **`archive/milgen/`** — hand-written/generated legacy MIL code, confirmed dead: nothing outside
  `archive/` references it, and it's absent from the Makefile's source lists. Fully superseded by
  the live `compiler/frontends` + `pipeline`/`codegen` path.
- **`docs/`** — `ane_api_reference.md` (private framework calling convention, credited to
  upstream `maderix/ANEgpt`), `ane_constraints.md` (catalog of empirically-found ANE MIL
  limits — no concat, weight-budget ceilings), `m2_benchmarks.md` (dated hardware benchmark
  table).
- **Build**: plain `Makefile`, `xcrun clang`, `-fobjc-arc`; links Foundation, IOSurface,
  Accelerate; targets `all`/`test`/`test-compiler`/`bench`/`clean`. Quick start is `make` then
  `./orion infer` / `./orion train`.
- **`tests/`** (30 files) span every layer: graph IR/passes/compiler-equivalence, ANE runtime +
  program cache/swap, inference (CPU/hybrid/full-ANE, golden-output comparisons), training
  kernels, tokenizers — plus Python golden-data generators.

## Known dead/unfinished code

- `compiler/kernel_adapter.m`'s frontend *registry* path (`orion_kernel_from_frontend`) — stub,
  always returns nil; real kernels bypass it.
- `compiler/pass_conv_bias.c` — implemented but disabled in the pipeline (ANE limitation).
- `archive/milgen/` — legacy, unreferenced.
- `apps/macapp/` — Swift GUI stub, not integrated.
- `tokenizer/sentencepiece_wrap` — implemented and tested, but not called from any CLI command
  today.

## Open questions (not verified by this pass)

- Exact bodies of `compiler/frontends/gpt2_prefill.c`, `gpt2_final.c`, `lora.c`,
  `stories_train.c`, `classifier_softmax.c` (only `gpt2_decode.c` was traced in depth).
  `lora.c` and `stories_train.c` are the largest frontends and likely worth a follow-up pass.
  `compiler/patterns.c` was only read up to its attention-pattern head.
  What `compiler/pass_ane_validate.c` specifically checks (only its scaffolding was read).
- `orion_program_patch_weights` still exists in `core/ane_runtime.m`, but the training delta path
  does not use it (`patch_layer`, `stories_train.m:993-1023`, calls `orion_program_reload_weights`).
  Who its live callers are was not traced — possibly another dead path.
- Whether the full-recompile fallback (`orion_trainer_recompile`, wired at
  `stories_train.m:1042`) is ever actually taken in practice, or only on a patch failure that
  never occurs.
- `model/convert/hf_to_blobs_llama.py` was assumed structurally parallel to the GPT-2 converter
  by naming/README, not read directly.

---
*Generated via a 3-lane parallel recon (compiler pipeline / runtime & kernels / entry points &
tooling) over the codebase-memory graph + direct source reads, 2026-09-06.*
