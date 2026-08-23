// ane_weight_limit_probe.m — Independent verification of the ANE weight-count ceiling (#3)
//
// Builds synthetic MIL programs with N chained 1x1 conv weights, optionally followed
// by one "penalty" op, and reports whether ANECCompile() accepts them. No model blobs
// required — all weights are generated in memory.
//
// Verifies the corrected constraint model from issue #3 (tyrauber):
//   - baseline ceiling of 16 conv weights
//   - pow() / add(scalar) each cost one weight slot (15 max)
//   - penalties do not stack
//   - rsqrt is rejected outright
//
// Compile only — programs are never evaluated, so IOSurface sizing (#4) is irrelevant.
// Stays well under the ~119 compiles-per-process limit (#5).
//
// Build:
//   xcrun clang -O2 -fobjc-arc -DACCELERATE_NEW_LAPACK -I . -I core -I compiler \
//     -framework Foundation -framework IOSurface -framework Accelerate -ldl \
//     experiments/ane_weight_limit_probe.m core/ane_runtime.m core/iosurface_tensor.m \
//     -o build/ane_weight_limit_probe
// Run:
//   ./build/ane_weight_limit_probe
//
// Set DUMP_MIL=1 to print a sample generated program instead of compiling.

#import <Foundation/Foundation.h>
#import <stdio.h>
#import "ane_runtime.h"

#define CH  64     // channels; weights are [CH,CH,1,1] fp16 = 8KB each
#define SEQ 16     // minimum decode bucket (#4)

typedef enum {
    PEN_NONE = 0,
    PEN_SQRT,
    PEN_TANH,
    PEN_SIGMOID,
    PEN_EXP,
    PEN_ADD_SCALAR,
    PEN_POW,
    PEN_RSQRT,
    PEN_ADD_AND_POW,
} PenaltyOp;

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
    }
    return "?";
}

// Emit the MIL ops for one penalty, consuming `in` and producing `out_name`.
static void append_penalty(NSMutableString *m, PenaltyOp pen,
                           NSString *in, NSString *out_name) {
    switch (pen) {
        case PEN_NONE:
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@ = identity(x=%@)[name=string(\"%@\")];\n",
                CH, SEQ, out_name, in, out_name];
            break;
        case PEN_SQRT: case PEN_TANH: case PEN_SIGMOID: case PEN_EXP: case PEN_RSQRT: {
            const char *op = (pen == PEN_SQRT)    ? "sqrt"
                           : (pen == PEN_TANH)    ? "tanh"
                           : (pen == PEN_SIGMOID) ? "sigmoid"
                           : (pen == PEN_EXP)     ? "exp" : "rsqrt";
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@ = %s(x=%@)[name=string(\"%@\")];\n",
                CH, SEQ, out_name, op, in, out_name];
            break;
        }
        case PEN_ADD_SCALAR:
            [m appendFormat:@"        fp16 %@_k = const()[name=string(\"%@_k\"), val=fp16(0.5)];\n",
                out_name, out_name];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@ = add(x=%@, y=%@_k)[name=string(\"%@\")];\n",
                CH, SEQ, out_name, in, out_name, out_name];
            break;
        case PEN_POW:
            [m appendFormat:@"        fp16 %@_e = const()[name=string(\"%@_e\"), val=fp16(2.0)];\n",
                out_name, out_name];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@ = pow(x=%@, y=%@_e)[name=string(\"%@\")];\n",
                CH, SEQ, out_name, in, out_name, out_name];
            break;
        case PEN_ADD_AND_POW:
            [m appendFormat:@"        fp16 %@_k = const()[name=string(\"%@_k\"), val=fp16(0.5)];\n",
                out_name, out_name];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@_a = add(x=%@, y=%@_k)[name=string(\"%@_a\")];\n",
                CH, SEQ, out_name, in, out_name, out_name];
            [m appendFormat:@"        fp16 %@_e = const()[name=string(\"%@_e\"), val=fp16(2.0)];\n",
                out_name, out_name];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@ = pow(x=%@_a, y=%@_e)[name=string(\"%@\")];\n",
                CH, SEQ, out_name, out_name, out_name, out_name];
            break;
    }
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

// Build a program: x -> conv_0 -> ... -> conv_{n-1} -> [penalty] -> out
static NSString *build_mil(int n_conv, PenaltyOp pen, NSMutableDictionary *wdict) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:
        @"program(1.3)\n"
        @"[buildInfo = dict<string, string>({"
        @"{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        @"{\"coremlc-version\", \"3505.4.1\"}, "
        @"{\"coremltools-component-milinternal\", \"\"}, "
        @"{\"coremltools-version\", \"9.0\"}"
        @"})]\n{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", CH, SEQ];

    // Shared conv parameters
    [m appendString:@"        string c_pt = const()[name=string(\"c_pt\"), val=string(\"valid\")];\n"];
    [m appendString:@"        tensor<int32, [2]> c_st = const()[name=string(\"c_st\"), val=tensor<int32, [2]>([1,1])];\n"];
    [m appendString:@"        tensor<int32, [4]> c_pd = const()[name=string(\"c_pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendString:@"        tensor<int32, [2]> c_dl = const()[name=string(\"c_dl\"), val=tensor<int32, [2]>([1,1])];\n"];
    [m appendString:@"        int32 c_gr = const()[name=string(\"c_gr\"), val=int32(1)];\n"];

    // Identity-ish weights: [CH,CH,1,1], small non-zero values.
    NSString *cur = @"x";
    for (int i = 0; i < n_conv; i++) {
        NSString *path = [NSString stringWithFormat:@"@model_path/weights/w%d.bin", i];
        wdict[path] = @{@"offset": @0, @"data": make_blobfile(CH * CH)};

        [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> w%d = const()[name=string(\"w%d\"), "
                        @"val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"%@\"), offset=uint64(64)))];\n",
            CH, CH, i, i, CH, CH, path];
        [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> c%d = conv("
                        @"dilations=c_dl, groups=c_gr, pad=c_pd, pad_type=c_pt, strides=c_st, "
                        @"weight=w%d, x=%@)[name=string(\"c%d\")];\n",
            CH, SEQ, i, i, cur, i];
        cur = [NSString stringWithFormat:@"c%d", i];
    }

    append_penalty(m, pen, cur, @"out");
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

static bool try_compile(int n_conv, PenaltyOp pen) {
    @autoreleasepool {
        NSMutableDictionary *wdict = [NSMutableDictionary dictionary];
        NSString *mil = build_mil(n_conv, pen, wdict);
        char tag[64];
        snprintf(tag, sizeof(tag), "probe_%dconv_%s", n_conv, penalty_name(pen));
        OrionProgram *p = orion_compile_mil(mil.UTF8String, wdict, tag);
        if (p) { orion_release_program(p); return true; }
        return false;
    }
}

int main(void) {
    @autoreleasepool {
        if (!orion_ane_init()) { fprintf(stderr, "ANE init failed\n"); return 1; }

        if (getenv("DUMP_MIL")) {
            NSMutableDictionary *d = [NSMutableDictionary dictionary];
            printf("%s\n", build_mil(2, PEN_NONE, d).UTF8String);
            return 0;
        }
        printf("=== ANE weight-limit probe (independent verification of issue #3) ===\n");
        printf("channels=%d seq=%d, weights are [%d,%d,1,1] fp16\n\n", CH, SEQ, CH, CH);

        // 1. Baseline ceiling: sweep conv count with no penalty op.
        printf("--- Baseline: conv weights only ---\n");
        for (int n = 14; n <= 18; n++)
            printf("  %2d conv           -> %s\n", n, try_compile(n, PEN_NONE) ? "SUCCESS" : "FAILED");

        // 2. Which ops cost a slot, at 16 and 15 conv.
        printf("\n--- Penalty ops at 16 vs 15 conv ---\n");
        PenaltyOp ops[] = { PEN_SQRT, PEN_TANH, PEN_SIGMOID, PEN_EXP, PEN_ADD_SCALAR, PEN_POW };
        for (unsigned i = 0; i < sizeof(ops)/sizeof(ops[0]); i++) {
            bool at16 = try_compile(16, ops[i]);
            bool at15 = try_compile(15, ops[i]);
            printf("  16 conv + %-12s -> %-7s | 15 conv + %-12s -> %s\n",
                   penalty_name(ops[i]), at16 ? "SUCCESS" : "FAILED",
                   penalty_name(ops[i]), at15 ? "SUCCESS" : "FAILED");
        }

        // 3. Do penalties stack?
        printf("\n--- Stacking ---\n");
        printf("  16 conv + add+pow  -> %s\n", try_compile(16, PEN_ADD_AND_POW) ? "SUCCESS" : "FAILED");
        printf("  15 conv + add+pow  -> %s\n", try_compile(15, PEN_ADD_AND_POW) ? "SUCCESS" : "FAILED");

        // 4. rsqrt, including with no weights at all.
        printf("\n--- rsqrt ---\n");
        printf("  15 conv + rsqrt    -> %s\n", try_compile(15, PEN_RSQRT) ? "SUCCESS" : "FAILED");
        printf("   0 conv + rsqrt    -> %s\n", try_compile(0,  PEN_RSQRT) ? "SUCCESS" : "FAILED");

        printf("\ncompiles used: %d\n", orion_compile_count());
    }
    return 0;
}
