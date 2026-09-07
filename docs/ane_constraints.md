# ANE Constraints Reference

> Practical reference for anyone working with Apple Neural Engine private APIs.
>
> Some constraints below were first documented by [maderix](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615) through hardware-level benchmarking (marked with **[maderix]**). The weight-budget constraints (#15-#17) were characterized by [@tyrauber](https://github.com/tyrauber) in [#3](https://github.com/mechramc/Orion/issues/3) (marked with **[tyrauber]**) and independently reproduced here. The remaining MIL IR and memory constraints were discovered empirically during Orion development on M4 Max.

---

## 1. `concat` MIL Op Rejected by ANE Compiler

**What happens:** `ANECCompile() FAILED: err=()` for any program using `concat(axis=1, values=(...))`.

**Workaround:** Use multi-output MIL programs (`orion_mil_program_multi`) that return separate IOSurfaces per output instead of concatenating tensors into one.

**Discovered:** T064-T071 (ANE Training Kernels). All 7 training kernels using concat failed; `qkvBwd` (single summed output, no concat) was the only one that compiled.

---

## 2. Multi-Output Requires Uniform Output Buffer Sizes

**What happens:** `ANEProgramProcessRequestDirect() Failed with status=0x1d : Program Inference error` at eval time when output IOSurfaces have different allocation sizes.

**Workaround:** Pad all multi-output IOSurfaces to the max channel size across all outputs for that kernel. For example, if a kernel outputs `[d,s]` and `[h,s]` where `h > d`, allocate all output surfaces at `[h,s]`.

**Discovered:** T072 (Training Step). fwdFFN had mixed-size outputs `[d,s]` and `[h,s]`. Same fix applied to ffnBwd and sdpaBwd1 outputs.

---

## 3. Multi-Output Surfaces Ordered Alphabetically by MIL Variable Name

**What happens:** Output surfaces arrive in alphabetical order by their MIL variable name, NOT by their position in the MIL return tuple. Reading outputs in tuple order produces silently wrong data.

**Example:** MIL returns `(q32, k32, v32)` but actual output order is `k32, q32, v32`. You must provide output surfaces as `{ioK, ioQ, ioV}`.

**Workaround:** Always provide output IOSurfaces in alphabetical order of their MIL variable names. Or name your variables so alphabetical order matches your intended order.

**Discovered:** T101 (ANE Decode Step). Prefill's multi-output `(attn_res32, k32, v32)` had worked by coincidence -- it was already alphabetical.

---

## 4. Minimum IOSurface Allocation ~49KB for Eval

**What happens:** `ANEProgramProcessRequestDirect() Failed with status=0x1d` at eval time. Programs compile fine but fail at eval when IOSurface allocations are too small. seq=1 tensors with 768 channels = 3072 bytes -- too small.

**Minimum working size:** `[768, 16]` = 49,152 bytes (48KB). ANE internally uses a stride of 16 for the seq dimension in padded surfaces.

**Workaround:** Use `ORION_DECODE_SEQ = 16` as the minimum decode bucket. Place token data at seq position 0 and zero-pad positions 1-15.

**Discovered:** T100 (ANE Decode MIL). Invalidated T099 spike results which had reported seq=1 working (environment may have changed between sessions).

---

## 5. ~119 Compile Limit Per Process **[maderix]**

**What happens:** After approximately 119 calls to `ANECCompile()`, the compiler silently fails or the process becomes unstable. The ANE compiler leaks internal state that cannot be reclaimed.

**Workaround:** Track compile count with `orion_compile_count()`. When approaching the budget, checkpoint state to disk and call `exec()` to restart the process. exec() overhead is negligible (~50ms).

**Discovered:** First documented by [maderix](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615). Confirmed in T004 (Upstream Validation). ANEgpt uses the same exec() restart strategy. 83% of wall time is compile, so a program cache (T084) is essential.

---

## 6. SDPA Causal Masks Ignored by ANE **[maderix/ANEgpt]**

**What happens:** If you pass a causal mask to ANE's SDPA operation, it compiles and runs but the mask has no effect. Attention scores are computed as if no mask exists, producing incorrect outputs for autoregressive generation.

**Workaround:** Decompose attention manually:
```
Q @ K^T  -->  apply causal mask (additive -inf)  -->  softmax  -->  @ V
```
Implemented via `orion_mil_causal_attention` using explicit matmul + mask + softmax + matmul.

**Discovered:** T022 (MIL Causal Attention). Also confirmed in upstream maderix/ANE and ANEgpt codebases.

---

## 7. Weights Baked at Compile Time **[maderix]**

**What happens:** Overwriting BLOBFILE weight files on disk and reloading does NOT change model outputs. The weights are embedded into the compiled ANE program binary at compile time. There is no way to update weights without recompiling.

**Workaround:** Recompilation is mandatory for weight updates. For training: compile with new weights, cache the program, evict the old one. The program cache (T084) manages this lifecycle.

**Discovered:** First documented by [maderix](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine). Confirmed in T001-T007 (Upstream Validation) on M4 Max hardware.

---

## 8. BLOBFILE Offset is `uint64(64)`, Not `uint64(128)`

**What happens:** MIL weight references with the wrong offset cause silent garbage reads. The compiled program reads weight data from the wrong position in the blob, producing incorrect outputs with no error.

**Details:** The BLOBFILE format has a 128-byte header, but the MIL `const()` weight reference offset points to the chunk header at byte 64, not the start of the file (byte 0) or end of the full header (byte 128).

```
MIL: const(name="w", val=blob(file="weight.blob", offset=uint64(64)))
```

The weight blob offset in the weight dict is 0 (start of blob data), not 64.

**Discovered:** T019-T023 (MIL Builder Helpers). Also confirmed in T099 (Single-Token Spike).

---

## 9. `milText` Must Be `NSData*`, Not `NSString*`

**What happens:** Passing an `NSString*` to `_ANEInMemoryModelDescriptor.milText` causes a crash or silent failure. The API expects raw UTF-8 bytes.

**Workaround:**
```objc
NSString *milString = [self generateMIL];
NSData *milData = [milString dataUsingEncoding:NSUTF8StringEncoding];
descriptor.milText = milData;
```

Always convert your MIL text string to `NSData*` with UTF-8 encoding before passing to the descriptor.

**Discovered:** T008 (Hello MIL Proof-of-Concept) and documented in T009 (ANE API Reference).

---

## 10. `gelu` Is Not a Valid MIL Op

**What happens:** Using `gelu(x)` in MIL text causes `ANECCompile() FAILED`. The `gelu` activation is not in the ANE compiler's supported op set despite appearing in MIL documentation.

**Workaround:** Decompose to the tanh approximation manually:

```
gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

Implemented as `orion_mil_gelu` using `tanh`, `mul`, `add`, and `pow` MIL ops which are all supported.

**Discovered:** T021 (MIL GELU + SiLU). `silu` also requires decomposition but was implemented alongside.

---

## 11. Weight Dict Must Be `@{}`, Not `nil`, for Weight-Free Programs

**What happens:** Passing `nil` as the weight dictionary to `ANEProgramProcessRequestDirect()` causes a crash, even for programs that have no weights (e.g., softmax, activation-only kernels).

**Workaround:**
```objc
NSDictionary *weights = @{};  // empty dict, NOT nil
ANEProgramProcessRequestDirect(program, request, weights);
```

Always pass an empty `NSDictionary` for weight-free programs.

**Discovered:** T008 (Hello MIL Proof-of-Concept). The `z = add(x, y)` proof-of-concept has no weights but still requires `@{}`.

---

## 12. Multi-Input Requires Uniform Input Buffer Sizes

**What happens:** `ANEProgramProcessRequestDirect() Failed with status=0x1d` at eval time when input IOSurfaces have different allocation sizes — even though the MIL declares them with different shapes.

**Example:** LoRA program with `x [1,768,1,32]` and `lora_A [1,768,1,16]`: allocating surfaces as `orion_tensor_create(768, 32)` and `orion_tensor_create(768, 16)` fails at eval. Both surfaces must be allocated with `orion_tensor_create(768, 32)`.

**Workaround:** Allocate all input IOSurfaces with the same `allocSize` (use the maximum). Write packed data into the beginning of oversized surfaces — ANE reads the flat buffer as packed `[1,C,1,S]` where S is the MIL-declared shape, regardless of actual surface size.

**Discovered:** T163 (LoRA Tests). Same constraint as outputs (#2 above).

---

## 13. Multi-Input Surfaces Ordered Alphabetically by MIL Parameter Name

**What happens:** Input surfaces provided in the wrong order produce silently wrong data. The output is non-zero but mathematically incorrect — inputs are mapped to the wrong MIL parameters.

**Example:** MIL declares `func main(x, lora_A, lora_B)` — declaration order. Inputs must be provided as `{lora_A, lora_B, x}` — alphabetical order by parameter name.

**Workaround:** Sort input surfaces alphabetically by MIL parameter name. Same rule as output ordering (#3 above).

**Discovered:** T163 (LoRA Tests). Tested all 6 permutations of 3 inputs; only alphabetical order produces correct output.

---

## 14. ANE Reads Flat Buffer as Packed Shape Data

**What happens:** When a MIL parameter declares shape `[1,C,1,S]` but the IOSurface is larger (e.g., allocated for `[1,C,1,S_max]`), ANE reads the first `C*S` contiguous fp16 values from the flat buffer. It does NOT use stride or padding — the data must be packed.

**Example:** `lora_A` declared as `[1,768,1,16]` in MIL, surface allocated as `[1,768,1,32]`. ANE reads bytes 0..24575 (768*16*2 bytes) as packed channel-major data. If you write data with stride-32 layout (interleaving data and padding per channel), ANE reads the wrong values.

**Workaround:** Write adapter data packed at the start of the surface: `orion_tensor_write_f32(surf, packed_data, C*S)`. Do NOT pad to the surface's full width.

**Discovered:** T163 (LoRA Tests). Stride-padded data produced ~260x smaller LoRA contribution than expected.

---

## 15. Maximum 16 BLOBFILE Weight Tensors Per Program **[tyrauber]**

**What happens:** A program with 17 or more BLOBFILE-backed weight tensors fails to compile with `InvalidMILProgram`. 16 compiles fine.

**The budget counts tensors, not bytes, and not just conv weights.** Three properties, all measured:

**a) It is a count budget, not a byte budget.** The ceiling is 16 across a 2304x range in per-weight size:

```
C     bytes/weight   ceiling   total at ceiling
16    512            16        0.01 MB
64    8192           16        0.12 MB
256   131072         16        2.00 MB
768   1179648        16        18.00 MB
```

**b) Shape variety is irrelevant.** Mixing shapes within one program does not shift the ceiling, even at realistic FFN dimensions or extreme aspect ratios:

```
uniform 64            -> 16    (0.12 MB)
alternating 64/256    -> 16    (0.50 MB)
cycling 64/128/256    -> 16    (0.56 MB)
FFN-like 768/3072     -> 16    (72.00 MB)
lopsided 32/1024      -> 16    (1.00 MB)
```

**c) Every BLOBFILE tensor costs a full slot, regardless of size.** A 128-byte bias vector consumes the same budget as a 1.18 MB weight matrix:

```
conv only         -> 16 conv  (16 blobs)
conv + bias each  ->  8 conv  (16 blobs)
```

**Practical impact:** this is the binding limit on mega-kernel fusion, and it is stricter than "16 layers of weights". A linear layer *with a bias* costs **two** slots, not one. Budget in blobs, not in conceptual weights: count every `const()` that references a BLOBFILE, including biases and norm weight vectors.

Inline constants — conv `strides`/`pad`/`dilations` attributes and similar — do not count against this budget. See #16 for the one case where an inline scalar does cost a slot.

**Verification:** `experiments/ane_weight_limit_probe.m`, modes `size`, `shape` and `kind`. Measured on M4, macOS 26.5.2:

```
14 conv -> SUCCESS    17 conv -> FAILED
15 conv -> SUCCESS    18 conv -> FAILED
16 conv -> SUCCESS
```

**Discovered:** [@tyrauber](https://github.com/tyrauber) in [#3](https://github.com/mechramc/Orion/issues/3) on M4 Max / macOS 15. Independently reproduced on M4 / macOS 26.5.2; the size, shape and bias-slot properties characterized here.

---

## 16. Scalar-Operand Ops Cost One Weight Slot **[tyrauber]**

**What happens:** A program containing `pow()`, `add()`, or `mul()` against an *inline scalar constant* drops the blob ceiling from 16 to 15. At 16 blobs plus any such op, compilation fails.

`mul(scalar)` carrying the same penalty as `add(scalar)` is worth noting: it means the cost attaches to feeding an inline scalar into an elementwise op, not to a specific opcode.

**Details:** The penalty does **not** stack — a program using several of these still compiles at 15 blobs. Unary elementwise ops with no constant operand (`sqrt`, `tanh`, `sigmoid`, `exp`) carry no penalty and compile fine at 16:

```
16 conv + sigmoid      -> SUCCESS     16 conv + mul(scalar) -> FAILED
16 conv + tanh         -> SUCCESS     16 conv + add(scalar) -> FAILED
16 conv + sqrt         -> SUCCESS     16 conv + pow(const)  -> FAILED
16 conv + exp          -> SUCCESS
```

**Consequence for activation lowering:** how an activation is expressed decides whether it costs budget. `SiLU(x) = x * sigmoid(x)` uses no scalar constant and stays at 16. Rewriting it via the identity `sigmoid(x) = 0.5*(tanh(0.5x)+1)` introduces scalar `mul` and `add`, dropping the ceiling to 15:

```
SiLU via sigmoid  -> 16
SiLU via tanh     -> 15
```

Both are mathematically exact; only the second spends a slot. In a program that already contains a norm the point is moot, since `pow(x, -0.5)` has already reduced the budget to 15 and the penalty does not stack.

**Why RMSNorm appears to break things:** RMSNorm is built on `pow(x, -0.5)`, so any program containing one silently inherits the -1 penalty. This originally looked like a rule about "mixing norm and linear weight types"; the real cause is the `pow()` op. There is no weight-type mixing rule.

**Verification:** measured on M4, macOS 26.5.2:

```
16 conv + sqrt         -> SUCCESS    16 conv + add(scalar) -> FAILED
16 conv + tanh         -> SUCCESS    15 conv + add(scalar) -> SUCCESS
16 conv + sigmoid      -> SUCCESS    16 conv + pow(const)  -> FAILED
16 conv + exp          -> SUCCESS    15 conv + pow(const)  -> SUCCESS
16 conv + add + pow    -> FAILED     15 conv + add + pow   -> SUCCESS
```

**Interaction with #15:** the penalty applies to the total blob budget, not to a conv-only count. A program with `pow()` and paired conv+bias blobs caps at 7 convs (14 blobs), consistent with a 15-blob ceiling:

```
conv + bias        -> 8 conv (16 blobs)
conv + bias + pow  -> 7 conv (14 blobs)
```

**Workaround:** budget 15 blobs for any program containing a norm, or restructure to avoid `pow()` where an unpenalized op will do.

**Discovered:** [@tyrauber](https://github.com/tyrauber) in [#3](https://github.com/mechramc/Orion/issues/3), correcting the initial "mixed weight type" framing. Independently reproduced.

---

## 17. `rsqrt` Is Not a Valid MIL Op **[tyrauber]**

**What happens:** Any program containing `rsqrt` fails to compile, regardless of weight count — including a program with zero weights.

**Workaround:** use `pow(x, -0.5)`, which is what `orion_mil_rmsnorm` does. Note this incurs the #16 penalty. See also #10 (`gelu` is likewise unavailable and must be expanded).

**Verification:** measured on M4, macOS 26.5.2:

```
15 conv + rsqrt -> FAILED
 0 conv + rsqrt -> FAILED
```

**Discovered:** [@tyrauber](https://github.com/tyrauber) in [#3](https://github.com/mechramc/Orion/issues/3). Independently reproduced.

---

## 18. fp32 Program I/O Rejected on M1-Generation ANE

**What happens:** A program whose `func main` inputs or outputs are declared `fp32` fails to compile on an M1-generation Neural Engine with `ANECCompile() FAILED` / `CompilationFailure`. The same program with `fp16` I/O compiles. Internal `cast` to and from fp16 does not help — the rejection is at the program boundary, not in the body.

This is an **ANE-generation difference**, not a MIL-validity problem — the same text compiles on the M4 the upstream code was written against. **Confirmed by direct measurement on 2026-09-06**, not merely inferred: on an M4 Pro (Mac16,8, macOS 15.6) all five GPT-2 kernels compile with the fp32 I/O unchanged, and `bench swap` runs 100/100 compile-evict cycles — the exact configuration that fails on all five kernels and at swap iteration 0 on M1 Pro.

**Blast radius on M1 is the entire GPT-2 inference path, not just one benchmark.** Every GPT-2 frontend declares fp32 program input (`compiler/frontends/gpt2_final.h` documents it in as many words: `Input: fp32 [1, d_model, 1, bucket]`), so on an M1 Pro all five inference kernels — `prefill_attn`, `prefill_ffn`, `final_ln`, `decode_proj`, `decode_ffn` — fail to compile, `./orion bench kernels` produces no rows at all, and `./orion infer --ane` silently falls back to CPU for every layer. The synthetic program in `bench_swap` fails for the same reason.

The **Stories110M training path is unaffected** and works normally on M1, because its kernels are fp16 `[1,C,1,S]` end to end via `core/iosurface_tensor`. That asymmetry — training fine, inference dead — is the single most important thing to know about running this repo on M1-generation hardware.

**Symptom:** on an M1 Pro, `./orion bench swap` fails at iteration 0 on every bucket (32/64/128/256), and `./orion bench kernels` reports `COMPILE FAILED` for all five kernels:

```
bench swap: compile failed at iter 0
  prefill_attn_L0       COMPILE FAILED
  prefill_ffn_L0        COMPILE FAILED
  final_ln              COMPILE FAILED
  decode_proj_L0        COMPILE FAILED
  decode_ffn_L0         COMPILE FAILED
ANE compile error: ... _ANECompiler : ANECCompile() FAILED ... err=(CompilationFailure)
```

`./orion bench inference --ane` does *not* report failure in its summary — it prints `mode: ANE full` and a throughput number produced entirely by the CPU fallback (63 tok/s, against 65 tok/s for the explicit CPU run). Do not read an `--ane` inference number on M1 as an ANE number without checking the log for `falling back to CPU`.

**Workaround:** declare program I/O `fp16` and cast on the host side. Note this interacts with #14 — the ANE reads the flat IOSurface as packed shape data, so the host buffer element size must change with the declared dtype.

**Verification:** isolated on M1 Pro (MacBookPro18,1), macOS 15.7.5, by compiling one MIL program twice with only the I/O dtype varied and everything else byte-identical:

```
io_dtype=fp32  -> FAILED
io_dtype=fp16  -> COMPILED
```

The generated MIL for a failing kernel confirms where the fp32 enters — it is the function signature, not the body, which is already fp16:

```
func main<ios18>(tensor<fp32, [1,768,1,64]> x) {
    tensor<fp16, [1,768,1,1]> lnf_g = const()[...];
```

**Cross-generation confirmation (M4 Pro, Mac16,8, macOS 15.6, 2026-09-06).** The same binary, the
same fp32 program I/O, the same commit — all five kernels compile:

```
prefill_attn_L0    compile  88.53 ms    eval avg 0.1490 ms    SRAM ~23.3 MB
prefill_ffn_L0     compile  87.33 ms    eval avg 0.1942 ms    SRAM ~18.4 MB
final_ln           compile  44.89 ms    eval avg 0.0939 ms    SRAM ~0.4 MB
decode_proj_L0     compile  60.33 ms    eval avg 0.1184 ms    SRAM ~7.0 MB
decode_ffn_L0      compile  87.70 ms    eval avg 0.1808 ms    SRAM ~18.1 MB
```

So the constraint is **M1-generation-specific**. The fix remains the fp16 workaround above, because
one codebase has to run on both — but it is a bounded compatibility fix, not a correction to
something universally wrong.

**Second M1-generation machine confirms it is generational, not per-machine (M1 Max, Mac13,1,
macOS 15.6.1, 2026-09-07).** The prediction recorded here before that machine ran — "expected to
fail with M1 Pro" — is now a measurement. All five kernels `COMPILE FAILED`, `bench swap` died at
iteration 0, and `./orion bench inference --ane` logged `ANE decode failed at step 0, falling back
to CPU` while still printing `mode: ANE full` and 38 tok/s against 51 tok/s for the explicit CPU
run — the same trap, on a second chip. Two M1-generation machines with different form factors,
memory sizes and OS point releases fail identically; one M4 passes. The dtype at the program
boundary is the variable.

**Discovered:** Orion-fork Phase 1 ANE spike, 2026-09-06 ([#1](https://github.com/HiQS-Labs/Orion-fork/issues/1)). Confirmed failing on **M1 Pro and M1 Max**, confirmed **passing** on M4 Pro. Receipts: `TESTS-RESULTS/2026-09-06-phase1-ane-spike/`.

---

## Quick Reference Table

| # | Constraint | Severity | Symptom | Source |
|---|-----------|----------|---------|--------|
| 1 | No `concat` op | Compile fail | `ANECCompile() FAILED` | Orion |
| 2 | Uniform output buffer sizes | Eval fail | `status=0x1d` | Orion |
| 3 | Alphabetical output ordering | Silent wrong data | Outputs swapped | Orion |
| 4 | Minimum ~49KB IOSurface | Eval fail | `status=0x1d` | Orion |
| 5 | ~119 compile limit | Process instability | Silent fail / crash | maderix |
| 6 | SDPA masks ignored | Silent wrong data | Unmasked attention | maderix/ANEgpt |
| 7 | Weights baked at compile | Silent stale data | Old weights used | maderix |
| 8 | BLOBFILE offset is 64 | Silent wrong data | Garbage weights | Orion |
| 9 | milText must be NSData* | Crash | Immediate crash | Orion |
| 10 | No `gelu` MIL op | Compile fail | `ANECCompile() FAILED` | Orion |
| 11 | Weight dict must be `@{}` | Crash | Immediate crash | Orion |
| 12 | Uniform input buffer sizes | Eval fail | `status=0x1d` | Orion |
| 13 | Alphabetical input ordering | Silent wrong data | Inputs misassigned | Orion |
| 14 | Flat buffer = packed shape data | Silent wrong data | ~260x smaller values | Orion |
| 15 | Max 16 BLOBFILE weights (bias included) | Compile fail | `InvalidMILProgram` | tyrauber |
| 16 | Scalar-operand ops cost a slot | Compile fail | `InvalidMILProgram` at 16 | tyrauber |
| 17 | No `rsqrt` op | Compile fail | `InvalidMILProgram` | tyrauber |
| 18 | fp32 program I/O rejected on M1 ANE (M4 accepts it) | Compile fail (all GPT-2 inference) | `ANECCompile() FAILED` | Orion |
