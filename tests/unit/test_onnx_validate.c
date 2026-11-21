/*
 * SNEPPX - ONNX model validator test.
 *
 * Exercises SneppX_onnx_validate on:
 *   (a) a valid 2-node Gemm->Relu graph                      -> expect 0
 *   (b) a node whose input is undeclared                     -> expect -1 ("not declared")
 *   (c) a graph output not produced by any node (0 nodes)    -> expect -1 ("not produced")
 *   (d) a malformed/truncated model (ir_version only)        -> expect -1
 */

#include "onnx_format.h"
#include "test_gtest.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#if defined(_MSC_VER)
#  include <process.h>
#  define SNEPPX_PID() ((unsigned)_getpid())
#else
#  include <unistd.h>
#  define SNEPPX_PID() ((unsigned)getpid())
#endif

#define IN_FEATURES  2
#define OUT_FEATURES 3

static float kWeights[OUT_FEATURES * IN_FEATURES] = {
    1.0f, 2.0f,
    3.0f, 4.0f,
    5.0f, 6.0f
};
static float kBias[OUT_FEATURES] = { 10.0f, 20.0f, 30.0f };

static void unique_path(char* out, size_t n, const char* tag) {
    const char* t = getenv("TMP");
    if (!t) t = getenv("TEMP");
    if (!t) t = getenv("TMPDIR");
    if (!t) t = ".";
    snprintf(out, n, "%s/sneppx_onnxv_%s_%u.onnx", t, tag, SNEPPX_PID());
}

TEST(test_onnx_validate, suite) {
    int failures = 0;
    char path[512];
    char err[256];

    /* ---- (a) valid graph: X[batch,2] -> Gemm(transB=1) -> hidden -> Relu -> Y ---- */
    {
        SneppXOnnxDim xdims[2] = { {0, "batch"}, {IN_FEATURES, NULL} };
        SneppXOnnxDim ydims[2] = { {0, "batch"}, {OUT_FEATURES, NULL} };
        SneppXOnnxValueInfo inputs[1]  = { { "X", xdims, 2, 0 } };
        SneppXOnnxValueInfo outputs[1] = { { "Y", ydims, 2, 0 } };
        size_t wdims[2] = { OUT_FEATURES, IN_FEATURES };
        size_t bdims[1] = { OUT_FEATURES };
        SneppXOnnxInitializer inits[2] = {
            { "W", wdims, 2, 0, kWeights, OUT_FEATURES * IN_FEATURES },
            { "B", bdims, 1, 0, kBias, OUT_FEATURES }
        };
        SneppXOnnxAttr tattr = { "transB", 2, 1, 0.0f, NULL, 0, NULL, 0 }; /* INT, i=1 */
        const char* gin[3] = { "X", "W", "B" };
        const char* gout[1] = { "hidden" };
        const char* relu_in[1] = { "hidden" };
        const char* relu_out[1] = { "Y" };
        SneppXOnnxNode nodes[2] = {
            { "Gemm", "n_gemm", gin, 3, gout, 1, &tattr, 1 },
            { "Relu", "n_relu", relu_in, 1, relu_out, 1, NULL, 0 }
        };

        unique_path(path, sizeof(path), "valid");
        if (SneppX_onnx_save_graph(path, "sneppx_valid", "SNEPPX-Alg", "1.3.0",
                                   10, 1, 14,
                                   inputs, 1, outputs, 1,
                                   inits, 2, nodes, 2) != 0) {
            fprintf(stderr, "FAIL(a): export failed\n"); failures++; goto after_a;
        }
        err[0] = 0;
        if (SneppX_onnx_validate(path, err, sizeof(err)) != 0) {
            fprintf(stderr, "FAIL(a): valid graph reported invalid: %s\n", err); failures++;
        } else {
            fprintf(stderr, "info(a): valid graph accepted (err='%s')\n", err);
        }
        remove(path);
    }
after_a:

    /* ---- (b) undeclared node input ---- */
    {
        SneppXOnnxDim ydims[2] = { {0, "batch"}, {OUT_FEATURES, NULL} };
        SneppXOnnxDim bxdims[2] = { {0, "batch"}, {IN_FEATURES, NULL} };
        SneppXOnnxValueInfo inputs[1]  = { { "X", bxdims, 2, 0 } };
        SneppXOnnxValueInfo outputs[1] = { { "Y", ydims, 2, 0 } };
        const char* relu_in[1] = { "ghost" };   /* never declared */
        const char* relu_out[1] = { "Y" };
        SneppXOnnxNode nodes[1] = {
            { "Relu", "n_relu", relu_in, 1, relu_out, 1, NULL, 0 }
        };
        unique_path(path, sizeof(path), "undecl");
        SneppX_onnx_save_graph(path, "sneppx_bad", "SNEPPX-Alg", "1.3.0",
                               10, 1, 14, inputs, 1, outputs, 1, NULL, 0, nodes, 1);
        err[0] = 0;
        int rc = SneppX_onnx_validate(path, err, sizeof(err));
        if (rc == 0) { fprintf(stderr, "FAIL(b): undeclared input accepted\n"); failures++; }
        else if (!strstr(err, "not declared") && !strstr(err, "ghost")) {
            fprintf(stderr, "FAIL(b): wrong error: %s\n", err); failures++;
        } else {
            fprintf(stderr, "info(b): rejected undeclared input: %s\n", err);
        }
        remove(path);
    }

    /* ---- (c) graph output not produced (0 nodes) ---- */
    {
        SneppXOnnxDim ydims[2] = { {0, "batch"}, {OUT_FEATURES, NULL} };
        SneppXOnnxDim xdims[2] = { {0, "batch"}, {IN_FEATURES, NULL} };
        SneppXOnnxValueInfo inputs[1]  = { { "X", xdims, 2, 0 } };
        SneppXOnnxValueInfo outputs[1] = { { "Y", ydims, 2, 0 } };
        SneppXOnnxNode nodes_empty[1] = { { "Identity", "noop", NULL, 0, NULL, 0, NULL, 0 } };
        unique_path(path, sizeof(path), "unprod");
        /* note: node count = 0 -> output Y never produced */
        SneppX_onnx_save_graph(path, "sneppx_bad2", "SNEPPX-Alg", "1.3.0",
                               10, 1, 14, inputs, 1, outputs, 1, NULL, 0, nodes_empty, 0);
        err[0] = 0;
        int rc = SneppX_onnx_validate(path, err, sizeof(err));
        if (rc == 0) { fprintf(stderr, "FAIL(c): unproduced output accepted\n"); failures++; }
        else if (!strstr(err, "not produced") && !strstr(err, "Y")) {
            fprintf(stderr, "FAIL(c): wrong error: %s\n", err); failures++;
        } else {
            fprintf(stderr, "info(c): rejected: %s\n", err);
        }
        remove(path);
    }

    /* ---- (d) malformed/truncated model (ir_version only, no graph) ---- */
    {
        unique_path(path, sizeof(path), "trunc");
        FILE* f = fopen(path, "wb");
        if (f) {
            /* field 1 (ir_version), wire 0, value 10 -> bytes 0x08 0x0a */
            unsigned char hdr[2] = { 0x08, 0x0a };
            fwrite(hdr, 1, 2, f);
            fclose(f);
        }
        err[0] = 0;
        int rc = SneppX_onnx_validate(path, err, sizeof(err));
        if (rc == 0) { fprintf(stderr, "FAIL(d): malformed model accepted\n"); failures++; }
        else {
            fprintf(stderr, "info(d): rejected malformed model: %s\n", err);
        }
        remove(path);
    }

    if (failures) { fprintf(stderr, "RESULT: %d failure(s)\n", failures); FAIL() << "early exit (legacy return 1)"; }
    fprintf(stderr, "PASS: ONNX validator (valid + 3 invalid cases)\n");
    return;
}
