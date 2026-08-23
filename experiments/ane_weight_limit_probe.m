// ane_weight_limit_probe.m — Independent verification of the ANE weight-count ceiling (#15-#17)
//
// Builds synthetic MIL programs with N chained 1x1 conv weights, optionally followed
// by one "penalty" op, and reports whether ANECCompile() accepts them. No model blobs
// required — all weights are generated in memory.
//
// Compile only — programs are never evaluated, so IOSurface sizing (#4) is irrelevant.
// Each mode stays well under the ~119 compiles-per-process limit (#5); run modes in
// separate processes rather than combining them.
//
// Modes:
//   base   (default) verify the #15-#17 constraint model
//   size   does the ceiling depend on weight BYTES or on weight COUNT?
//   shape  does mixing weight shapes within one program shift the ceiling?
//   kind   do bias-style [1,C,1,1] blobs consume slots like [C,C,1,1] conv weights?
//   silu   does PR #2's sigmoid -> tanh-identity SiLU rewrite cost budget?
//
// Build:
//   xcrun clang -O2 -fobjc-arc -DACCELERATE_NEW_LAPACK -I . -I core -I compiler \
//     -framework Foundation -framework IOSurface -framework Accelerate -ldl \
//     experiments/ane_weight_limit_probe.m core/ane_runtime.m core/iosurface_tensor.m \
//     -o build/ane_weight_limit_probe
// Run:
//   ./build/ane_weight_limit_probe [base|size|shape|kind|silu]
//
// Set DUMP_MIL=1 to print a sample generated program instead of compiling.

#import <Foundation/Foundation.h>
#import <stdio.h>
#import <stdlib.h>
#import <string.h>
#import "ane_runtime.h"

#define SEQ 16     // minimum decode bucket (#4)

typedef enum {
    PEN_NONE = 0,
    PEN_SQRT, PEN_TANH, PEN_SIGMOID, PEN_EXP,
    PEN_ADD_SCALAR, PEN_POW, PEN_RSQRT, PEN_ADD_AND_POW,
    PEN_MUL_SCALAR, PEN_SILU_SIGMOID, PEN_SILU_TANH,
} PenaltyOp;

// A program shape: `dims` cycles across the conv chain, so consecutive weights
// can differ in shape and size. A single-entry dims list gives uniform weights.
typedef struct {
    const int *dims;
    int        n_dims;
    bool       bias;      // give every conv its own [1,out,1,1] BLOBFILE bias
    PenaltyOp  pen;
} Config;

static const char *penalty_name(PenaltyOp p) {
    switch (p) {
        case PEN_NONE:        return "none";
        case PEN_SQRT:        return "sqrt";
        case PEN_TANH:        return "tanh";
        case PEN_SIGMOID:     return "sigmoid";
        case PEN_EXP:         return "exp";
        case PEN_ADD_SCALAR:  return "add(scalar)";
        case PEN_POW:         return "pow(const)";
        case PEN_RSQRT:       return "rsqrt";
        case PEN_ADD_AND_POW: return "add+pow";
        case PEN_MUL_SCALAR:  return "mul(scalar)";
        case PEN_SILU_SIGMOID: return "silu via sigmoid";
        case PEN_SILU_TANH:   return "silu via tanh";
    }
    return "?";
}

// Create a BLOBFILE (128-byte header + fp16 data). Header format from
// experiments/spike_e5_binary.m; the MIL const() offset of 64 points at the
// chunk header inside this layout (#8).
static NSData *make_blobfile(int count) {
    size_t fp16_size = (size_t)count * sizeof(uint16_t);
    size_t total = 128 + fp16_size;
    uint8_t *b = (uint8_t *)calloc(total, 1);
    b[0] = 1; b[4] = 2;
    b[64] = 0xEF; b[65] = 0xBE; b[66] = 0xAD; b[67] = 0xDE; b[68] = 1;
    *(uint32_t *)(b + 72) = (uint32_t)fp16_size;
    *(uint32_t *)(b + 80) = 128;
    _Float16 *fp16 = (_Float16 *)(b + 128);
    for (int i = 0; i < count; i++) fp16[i] = (_Float16)0.0625f;
    return [NSData dataWithBytesNoCopy:b length:total freeWhenDone:YES];
}

// Emit the MIL ops for one penalty, consuming `in` and producing `out_name`.
static void append_penalty(NSMutableString *m, PenaltyOp pen, int ch,
                           NSString *in, NSString *out_name) {
    switch (pen) {
        case PEN_NONE:
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@ = identity(x=%@)[name=string(\"%@\")];\n",
                ch, SEQ, out_name, in, out_name];
            break;
        case PEN_SQRT: case PEN_TANH: case PEN_SIGMOID: case PEN_EXP: case PEN_RSQRT: {
            const char *op = (pen == PEN_SQRT)    ? "sqrt"
                           : (pen == PEN_TANH)    ? "tanh"
                           : (pen == PEN_SIGMOID) ? "sigmoid"
                           : (pen == PEN_EXP)     ? "exp" : "rsqrt";
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@ = %s(x=%@)[name=string(\"%@\")];\n",
                ch, SEQ, out_name, op, in, out_name];
            break;
        }
        case PEN_ADD_SCALAR:
            [m appendFormat:@"        fp16 %@_k = const()[name=string(\"%@_k\"), val=fp16(0.5)];\n",
                out_name, out_name];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@ = add(x=%@, y=%@_k)[name=string(\"%@\")];\n",
                ch, SEQ, out_name, in, out_name, out_name];
            break;
        case PEN_POW:
            [m appendFormat:@"        fp16 %@_e = const()[name=string(\"%@_e\"), val=fp16(2.0)];\n",
                out_name, out_name];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@ = pow(x=%@, y=%@_e)[name=string(\"%@\")];\n",
                ch, SEQ, out_name, in, out_name, out_name];
            break;
        case PEN_MUL_SCALAR:
            [m appendFormat:@"        fp16 %@_k = const()[name=string(\"%@_k\"), val=fp16(0.5)];\n",
                out_name, out_name];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@ = mul(x=%@, y=%@_k)[name=string(\"%@\")];\n",
                ch, SEQ, out_name, in, out_name, out_name];
            break;

        // Pre-PR lowering: SiLU(x) = x * sigmoid(x)
        case PEN_SILU_SIGMOID:
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@_sig = sigmoid(x=%@)[name=string(\"%@_sig\")];\n",
                ch, SEQ, out_name, in, out_name];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@ = mul(x=%@, y=%@_sig)[name=string(\"%@\")];\n",
                ch, SEQ, out_name, in, out_name, out_name];
            break;

        // PR #2 lowering: sigmoid(x) = 0.5 * (tanh(0.5x) + 1)
        case PEN_SILU_TANH:
            [m appendFormat:@"        fp16 %@_half = const()[name=string(\"%@_half\"), val=fp16(0.5)];\n",
                out_name, out_name];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@_hx = mul(x=%@, y=%@_half)[name=string(\"%@_hx\")];\n",
                ch, SEQ, out_name, in, out_name, out_name];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@_th = tanh(x=%@_hx)[name=string(\"%@_th\")];\n",
                ch, SEQ, out_name, out_name, out_name];
            [m appendFormat:@"        fp16 %@_one = const()[name=string(\"%@_one\"), val=fp16(1.0)];\n",
                out_name, out_name];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@_onep = add(x=%@_th, y=%@_one)[name=string(\"%@_onep\")];\n",
                ch, SEQ, out_name, out_name, out_name, out_name];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@_sg = mul(x=%@_onep, y=%@_half)[name=string(\"%@_sg\")];\n",
                ch, SEQ, out_name, out_name, out_name, out_name];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@ = mul(x=%@, y=%@_sg)[name=string(\"%@\")];\n",
                ch, SEQ, out_name, in, out_name, out_name];
            break;

        case PEN_ADD_AND_POW:
            [m appendFormat:@"        fp16 %@_k = const()[name=string(\"%@_k\"), val=fp16(0.5)];\n",
                out_name, out_name];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@_a = add(x=%@, y=%@_k)[name=string(\"%@_a\")];\n",
                ch, SEQ, out_name, in, out_name, out_name];
            [m appendFormat:@"        fp16 %@_e = const()[name=string(\"%@_e\"), val=fp16(2.0)];\n",
                out_name, out_name];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@ = pow(x=%@_a, y=%@_e)[name=string(\"%@\")];\n",
                ch, SEQ, out_name, out_name, out_name, out_name];
            break;
    }
}

// Build: x -> conv_0 [+bias] -> ... -> conv_{n-1} [+bias] -> [penalty] -> out
// Returns the MIL text; fills wdict and reports total weight bytes.
static NSString *build_mil(Config cfg, int n_conv, NSMutableDictionary *wdict,
                           size_t *out_bytes) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({"
        @"{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        @"{\"coremlc-version\", \"3505.4.1\"}, "
        @"{\"coremltools-component-milinternal\", \"\"}, "
        @"{\"coremltools-version\", \"9.0\"}"
        @"})]\n{\n"];

    int in_ch0 = cfg.dims[0];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", in_ch0, SEQ];

    [m appendString:@"        string c_pt = const()[name=string(\"c_pt\"), val=string(\"valid\")];\n"];
    [m appendString:@"        tensor<int32, [2]> c_st = const()[name=string(\"c_st\"), val=tensor<int32, [2]>([1,1])];\n"];
    [m appendString:@"        tensor<int32, [4]> c_pd = const()[name=string(\"c_pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendString:@"        tensor<int32, [2]> c_dl = const()[name=string(\"c_dl\"), val=tensor<int32, [2]>([1,1])];\n"];
    [m appendString:@"        int32 c_gr = const()[name=string(\"c_gr\"), val=int32(1)];\n"];

    size_t bytes = 0;
    NSString *cur = @"x";
    int cur_ch = in_ch0;

    for (int i = 0; i < n_conv; i++) {
        int out_ch = cfg.dims[(i + 1) % cfg.n_dims];

        NSString *wpath = [NSString stringWithFormat:@"@model_path/weights/w%d.bin", i];
        wdict[wpath] = @{@"offset": @0, @"data": make_blobfile(out_ch * cur_ch)};
        bytes += (size_t)out_ch * cur_ch * 2;

        [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> w%d = const()[name=string(\"w%d\"), "
                        @"val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"%@\"), offset=uint64(64)))];\n",
            out_ch, cur_ch, i, i, out_ch, cur_ch, wpath];
        [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> c%d = conv("
                        @"dilations=c_dl, groups=c_gr, pad=c_pd, pad_type=c_pt, strides=c_st, "
                        @"weight=w%d, x=%@)[name=string(\"c%d\")];\n",
            out_ch, SEQ, i, i, cur, i];
        cur = [NSString stringWithFormat:@"c%d", i];

        if (cfg.bias) {
            NSString *bpath = [NSString stringWithFormat:@"@model_path/weights/b%d.bin", i];
            wdict[bpath] = @{@"offset": @0, @"data": make_blobfile(out_ch)};
            bytes += (size_t)out_ch * 2;
            [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> b%d = const()[name=string(\"b%d\"), "
                            @"val=tensor<fp16, [1,%d,1,1]>(BLOBFILE(path=string(\"%@\"), offset=uint64(64)))];\n",
                out_ch, i, i, out_ch, bpath];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> a%d = add(x=%@, y=b%d)[name=string(\"a%d\")];\n",
                out_ch, SEQ, i, cur, i, i];
            cur = [NSString stringWithFormat:@"a%d", i];
        }
        cur_ch = out_ch;
    }

    append_penalty(m, cfg.pen, cur_ch, cur, @"out");
    [m appendString:@"    } -> (out);\n}\n"];
    if (out_bytes) *out_bytes = bytes;
    return m;
}

static bool try_compile(Config cfg, int n_conv, size_t *out_bytes) {
    @autoreleasepool {
        NSMutableDictionary *wdict = [NSMutableDictionary dictionary];
        NSString *mil = build_mil(cfg, n_conv, wdict, out_bytes);
        char tag[64];
        snprintf(tag, sizeof(tag), "probe_%dconv", n_conv);
        OrionProgram *p = orion_compile_mil(mil.UTF8String, wdict, tag);
        if (p) { orion_release_program(p); return true; }
        return false;
    }
}

// Largest n in [0,hi] that compiles. Assumes monotonicity: if n compiles, n-1 does.
static int find_ceiling(Config cfg, int hi, size_t *bytes_at_ceiling) {
    int lo = 0, best = -1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        size_t b = 0;
        if (try_compile(cfg, mid, &b)) {
            best = mid;
            if (bytes_at_ceiling) *bytes_at_ceiling = b;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return best;
}

static Config uniform_cfg(const int *dim, bool bias, PenaltyOp pen) {
    Config c = { dim, 1, bias, pen };
    return c;
}

// ---------------------------------------------------------------------------

static void mode_base(void) {
    printf("=== ANE weight-limit probe: constraint model (#15-#17) ===\n");
    printf("weights are [64,64,1,1] fp16, seq=%d\n\n", SEQ);
    static const int d64 = 64;

    printf("--- Baseline: conv weights only ---\n");
    for (int n = 14; n <= 18; n++)
        printf("  %2d conv           -> %s\n", n,
               try_compile(uniform_cfg(&d64, false, PEN_NONE), n, NULL) ? "SUCCESS" : "FAILED");

    printf("\n--- Penalty ops at 16 vs 15 conv ---\n");
    PenaltyOp ops[] = { PEN_SQRT, PEN_TANH, PEN_SIGMOID, PEN_EXP, PEN_ADD_SCALAR, PEN_POW };
    for (unsigned i = 0; i < sizeof(ops)/sizeof(ops[0]); i++) {
        bool at16 = try_compile(uniform_cfg(&d64, false, ops[i]), 16, NULL);
        bool at15 = try_compile(uniform_cfg(&d64, false, ops[i]), 15, NULL);
        printf("  16 conv + %-12s -> %-7s | 15 conv + %-12s -> %s\n",
               penalty_name(ops[i]), at16 ? "SUCCESS" : "FAILED",
               penalty_name(ops[i]), at15 ? "SUCCESS" : "FAILED");
    }

    printf("\n--- Stacking ---\n");
    printf("  16 conv + add+pow  -> %s\n", try_compile(uniform_cfg(&d64, false, PEN_ADD_AND_POW), 16, NULL) ? "SUCCESS" : "FAILED");
    printf("  15 conv + add+pow  -> %s\n", try_compile(uniform_cfg(&d64, false, PEN_ADD_AND_POW), 15, NULL) ? "SUCCESS" : "FAILED");

    printf("\n--- rsqrt ---\n");
    printf("  15 conv + rsqrt    -> %s\n", try_compile(uniform_cfg(&d64, false, PEN_RSQRT), 15, NULL) ? "SUCCESS" : "FAILED");
    printf("   0 conv + rsqrt    -> %s\n", try_compile(uniform_cfg(&d64, false, PEN_RSQRT), 0,  NULL) ? "SUCCESS" : "FAILED");
}

// Is the ceiling a COUNT budget or a BYTE budget? If bytes, small weights
// should permit far more than 16 and large weights far fewer.
static void mode_size(void) {
    printf("=== Does weight SIZE shift the ceiling? ===\n");
    printf("uniform [C,C,1,1] weights; searching n in [0,40]\n\n");
    printf("  %-8s %-14s %-10s %s\n", "C", "bytes/weight", "ceiling", "total bytes at ceiling");
    static const int sizes[] = { 16, 32, 64, 128, 256, 512, 768 };
    for (unsigned i = 0; i < sizeof(sizes)/sizeof(sizes[0]); i++) {
        size_t bytes = 0;
        int ceil = find_ceiling(uniform_cfg(&sizes[i], false, PEN_NONE), 40, &bytes);
        printf("  %-8d %-14zu %-10d %.2f MB\n",
               sizes[i], (size_t)sizes[i]*sizes[i]*2, ceil, bytes / (1024.0*1024.0));
    }
}

// Does mixing weight SHAPES within one program change the ceiling?
static void mode_shape(void) {
    printf("=== Does weight SHAPE VARIETY shift the ceiling? ===\n");
    printf("dims cycle across the conv chain; searching n in [0,40]\n\n");

    static const int uniform64[]  = { 64 };
    static const int two[]        = { 64, 256 };              // square-ish alternation
    static const int three[]      = { 64, 128, 256 };         // three distinct shapes
    static const int ffn[]        = { 768, 3072 };            // real FFN up/down proj
    static const int lopsided[]   = { 32, 1024 };             // extreme aspect ratio

    struct { const char *name; const int *dims; int n; } cases[] = {
        { "uniform 64",              uniform64, 1 },
        { "alternating 64/256",      two,       2 },
        { "cycling 64/128/256",      three,     3 },
        { "FFN-like 768/3072",       ffn,       2 },
        { "lopsided 32/1024",        lopsided,  2 },
    };

    printf("  %-24s %-10s %s\n", "shape pattern", "ceiling", "total bytes at ceiling");
    for (unsigned i = 0; i < sizeof(cases)/sizeof(cases[0]); i++) {
        Config c = { cases[i].dims, cases[i].n, false, PEN_NONE };
        size_t bytes = 0;
        int ceil = find_ceiling(c, 40, &bytes);
        printf("  %-24s %-10d %.2f MB\n", cases[i].name, ceil, bytes / (1024.0*1024.0));
    }
}

// Do small bias-style [1,C,1,1] blobs consume budget like [C,C,1,1] conv weights?
// This is the corrected form of the original "mixed weight type" question.
static void mode_kind(void) {
    printf("=== Do bias blobs consume weight slots? ===\n");
    printf("each conv optionally paired with its own [1,C,1,1] BLOBFILE bias\n\n");
    static const int d64 = 64;

    size_t b1 = 0, b2 = 0;
    int no_bias = find_ceiling(uniform_cfg(&d64, false, PEN_NONE), 40, &b1);
    int with_bias = find_ceiling(uniform_cfg(&d64, true,  PEN_NONE), 40, &b2);

    printf("  conv only          -> ceiling %d conv  (%d blobs total)\n", no_bias, no_bias);
    printf("  conv + bias each   -> ceiling %d conv  (%d blobs total)\n", with_bias, with_bias * 2);
    printf("\n");
    if (with_bias * 2 == no_bias)
        printf("  => budget counts TOTAL blobs (bias costs a full slot)\n");
    else if (with_bias == no_bias)
        printf("  => budget counts CONV weights only (bias blobs are free)\n");
    else
        printf("  => neither: bias blobs cost a partial/different amount\n");

    // Cross-check the unified model: if the budget is N total blobs and pow()
    // costs one slot, then conv+bias with pow() should cap at floor(15/2) = 7.
    int with_bias_pow = find_ceiling(uniform_cfg(&d64, true, PEN_POW), 40, NULL);
    printf("\n  conv + bias + pow  -> ceiling %d conv  (%d blobs total)\n",
           with_bias_pow, with_bias_pow * 2);
    printf("  => consistent with a %d-blob budget under pow()\n", with_bias_pow * 2 + 1);
}


// Does PR #2's SiLU rewrite (sigmoid -> 0.5*(tanh(0.5x)+1)) cost budget?
static void mode_silu(void) {
    printf("=== Does the SiLU lowering affect the blob budget? ===\n");
    printf("comparing sigmoid-based SiLU against the tanh-identity rewrite\n\n");
    static const int d64 = 64;

    struct { const char *label; PenaltyOp pen; } cases[] = {
        { "no activation",     PEN_NONE },
        { "sigmoid (alone)",   PEN_SIGMOID },
        { "mul(scalar)",       PEN_MUL_SCALAR },
        { "add(scalar)",       PEN_ADD_SCALAR },
        { "SiLU via sigmoid",  PEN_SILU_SIGMOID },
        { "SiLU via tanh",     PEN_SILU_TANH },
    };

    printf("  %-20s %s\n", "program contains", "conv ceiling");
    for (unsigned i = 0; i < sizeof(cases)/sizeof(cases[0]); i++) {
        int c = find_ceiling(uniform_cfg(&d64, false, cases[i].pen), 24, NULL);
        printf("  %-20s %d\n", cases[i].label, c);
    }

    // With a norm already present the budget is 15 regardless (#16 non-stacking),
    // so the rewrite should cost nothing extra there.
    printf("\n  With pow() already in the program (norm present):\n");
    int pow_only = find_ceiling(uniform_cfg(&d64, false, PEN_POW), 24, NULL);
    int pow_plus = find_ceiling(uniform_cfg(&d64, false, PEN_ADD_AND_POW), 24, NULL);
    printf("    pow alone          %d\n", pow_only);
    printf("    pow + add(scalar)  %d\n", pow_plus);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        const char *mode = (argc > 1) ? argv[1] : "base";

        if (getenv("DUMP_MIL")) {
            static const int d = 64;
            NSMutableDictionary *dd = [NSMutableDictionary dictionary];
            printf("%s\n", build_mil(uniform_cfg(&d, true, PEN_NONE), 2, dd, NULL).UTF8String);
            return 0;
        }

        if (!orion_ane_init()) { fprintf(stderr, "ANE init failed\n"); return 1; }

        if      (!strcmp(mode, "base"))  mode_base();
        else if (!strcmp(mode, "size"))  mode_size();
        else if (!strcmp(mode, "shape")) mode_shape();
        else if (!strcmp(mode, "kind"))  mode_kind();
        else if (!strcmp(mode, "silu"))  mode_silu();
        else { fprintf(stderr, "unknown mode: %s (base|size|shape|kind|silu)\n", mode); return 2; }

        printf("\ncompiles used: %d\n", orion_compile_count());
    }
    return 0;
}
