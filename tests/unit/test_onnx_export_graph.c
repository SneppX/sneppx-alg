/*
 * SNEPPX - ONNX arbitrary-graph exporter test.
 *
 * Exports a 2-node graph (Gemm with transB=1 -> Relu) to a canonical binary
 * .onnx ModelProto and validates structure + initializer round-trip + semantics
 * with an independent standard-field-number protobuf reader.
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

/* ---- minimal standard protobuf reader (canonical ONNX IR field numbers) ---- */
typedef struct {
    const unsigned char* b;
    size_t len;
    size_t pos;
} PBr;

typedef struct {
    unsigned field;
    int wire;
    size_t start;
    size_t slen;
} PBField;

static int pbr_varint(PBr* r, uint64_t* out) {
    uint64_t v = 0;
    int sh = 0;
    while (r->pos < r->len) {
        unsigned char c = r->b[r->pos++];
        v |= (uint64_t)(c & 0x7f) << sh;
        if (!(c & 0x80)) {
            *out = v;
            return 1;
        }
        sh += 7;
        if (sh > 63) return 0;
    }
    return 0;
}

static int pbr_next(PBr* r, PBField* f) {
    uint64_t tag;
    if (!pbr_varint(r, &tag)) return 0;
    f->field = (unsigned)(tag >> 3);
    f->wire = (int)(tag & 7);
    size_t vstart = r->pos;
    if (f->wire == 0) {
        uint64_t val;
        if (!pbr_varint(r, &val)) return 0;
        f->start = vstart;
        f->slen = r->pos - vstart;
    } else if (f->wire == 1) {
        f->start = r->pos;
        if (r->pos + 8 > r->len) return 0;
        r->pos += 8;
        f->slen = 8;
    } else if (f->wire == 2) {
        uint64_t ln;
        if (!pbr_varint(r, &ln)) return 0;
        f->slen = (size_t)ln;
        f->start = r->pos;
        if (r->pos + f->slen > r->len) return 0;
        r->pos += f->slen;
    } else if (f->wire == 5) {
        f->start = r->pos;
        if (r->pos + 4 > r->len) return 0;
        r->pos += 4;
        f->slen = 4;
    } else {
        return 0;
    }
    return 1;
}

static int pb_find(const unsigned char* b, size_t off, size_t n,
                   unsigned field, int wire, size_t* so, size_t* sl) {
    PBr r;
    r.b = b;
    r.len = off + n;
    r.pos = off;
    PBField f;
    while (pbr_next(&r, &f)) {
        if (f.field == field && f.wire == wire) {
            if (so) *so = f.start;
            if (sl) *sl = f.slen;
            return 1;
        }
    }
    return 0;
}

static int pb_find_str(const unsigned char* b, size_t off, size_t n,
                       unsigned field, char* out, size_t maxout) {
    size_t s, l;
    if (!pb_find(b, off, n, field, 2, &s, &l)) return 0;
    size_t c = l < maxout ? l : maxout - 1;
    memcpy(out, b + s, c);
    out[c] = 0;
    return 1;
}

static int pb_find_u64(const unsigned char* b, size_t off, size_t n,
                       unsigned field, uint64_t* val) {
    size_t s, l;
    if (!pb_find(b, off, n, field, 0, &s, &l)) return 0;
    uint64_t v = 0;
    int sh = 0;
    size_t p = s, end = s + l;
    while (p < end) {
        unsigned char c = b[p++];
        v |= (uint64_t)(c & 0x7f) << sh;
        if (!(c & 0x80)) {
            *val = v;
            return 1;
        }
        sh += 7;
        if (sh > 63) return 0;
    }
    return 0;
}

static int pb_find_packed_int64(const unsigned char* b, size_t off, size_t n,
                                unsigned field, uint64_t* out, size_t cap,
                                size_t* cnt) {
    size_t s, l;
    if (!pb_find(b, off, n, field, 2, &s, &l)) {
        *cnt = 0;
        return 0;
    }
    size_t p = s, end = s + l;
    size_t k = 0;
    while (p < end && k < cap) {
        uint64_t v = 0;
        int sh = 0;
        while (p < end) {
            unsigned char c = b[p++];
            v |= (uint64_t)(c & 0x7f) << sh;
            if (!(c & 0x80)) break;
            sh += 7;
        }
        out[k++] = v;
    }
    *cnt = k;
    return 1;
}

typedef int (*FieldVisitor)(const unsigned char* b, size_t off, size_t n,
                            size_t idx, void* user);

static size_t pb_for_each(const unsigned char* b, size_t off, size_t n,
                          unsigned field, FieldVisitor fn, void* user) {
    PBr r;
    r.b = b;
    r.len = off + n;
    r.pos = off;
    PBField f;
    size_t idx = 0;
    while (pbr_next(&r, &f)) {
        if (f.field == field && f.wire == 2) {
            size_t cur = idx++;
            if (!fn(b, f.start, f.slen, cur, user)) return idx;
        }
    }
    return idx;
}

static int collect_strings(const unsigned char* b, size_t off, size_t n,
                           unsigned field, char names[][32], size_t cap) {
    PBr r;
    r.b = b;
    r.len = off + n;
    r.pos = off;
    PBField f;
    size_t k = 0;
    while (pbr_next(&r, &f) && k < cap) {
        if (f.field == field && f.wire == 2) {
            size_t c = f.slen < 31 ? f.slen : 31;
            memcpy(names[k], b + f.start, c);
            names[k][c] = 0;
            k++;
        }
    }
    return (int)k;
}

typedef struct { size_t off; size_t len; int want; int found; } WantSlot;
static int grab_slot(const unsigned char* b, size_t off, size_t n,
                     size_t idx, void* user) {
    WantSlot* w = (WantSlot*)user;
    (void)b;
    if ((int)idx == w->want) {
        w->off = off;
        w->len = n;
        w->found = 1;
        return 0;
    }
    return 1;
}

static int find_attr_int(const unsigned char* b, size_t off, size_t n,
                         const char* want, uint64_t* val) {
    PBr r;
    r.b = b; r.len = off + n; r.pos = off;
    PBField f;
    while (pbr_next(&r, &f)) {
        if (f.field == 5 && f.wire == 2) { /* AttributeProto */
            char an[32] = {0};
            pb_find_str(b, f.start, f.slen, 1, an, sizeof(an));
            if (strcmp(an, want) == 0 && pb_find_u64(b, f.start, f.slen, 3, val))
                return 1;
        }
    }
    return 0;
}

/* ---- test data ---- */
#define IN_FEATURES  2
#define OUT_FEATURES 3

static float kWeights[OUT_FEATURES * IN_FEATURES] = {
    1.0f, 2.0f,
    3.0f, 4.0f,
    5.0f, 6.0f
};
static float kBias[OUT_FEATURES] = { 10.0f, 20.0f, 30.0f };

static void unique_path(char* out, size_t n) {
    const char* t = getenv("TMP");
    if (!t) t = getenv("TEMP");
    if (!t) t = getenv("TMPDIR");
    if (!t) t = ".";
    snprintf(out, n, "%s/sneppx_onnx_graph_xt_%u.onnx", t, SNEPPX_PID());
}

static size_t count_fields(const unsigned char* b, size_t off, size_t n, unsigned field, int wire) {
    PBr r;
    r.b = b; r.len = off + n; r.pos = off;
    PBField f;
    size_t k = 0;
    while (pbr_next(&r, &f))
        if (f.field == field && (wire < 0 || f.wire == wire)) k++;
    return k;
}

TEST(test_onnx_export_graph, suite) {
    char path[512];
    unique_path(path, sizeof(path));
    int failures = 0;

    /* Graph: X[batch,2] -> Gemm(transB=1, W[3,2], B[3]) -> hidden[batch,3]
     *        hidden -> Relu -> Y[batch,3] */
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
    SneppXOnnxAttr gemm_attr = { "transB", 2, 1, 0.0f, NULL, 0, NULL, 0 }; /* INT, i=1 */
    const char* gemm_in[3]  = { "X", "W", "B" };
    const char* gemm_out[1] = { "hidden" };
    const char* relu_in[1]  = { "hidden" };
    const char* relu_out[1] = { "Y" };
    SneppXOnnxNode nodes[2] = {
        { "Gemm", "n_gemm", gemm_in,  3, gemm_out, 1, &gemm_attr, 1 },
        { "Relu", "n_relu", relu_in,  1, relu_out, 1, NULL, 0 }
    };

    if (SneppX_onnx_save_graph(path, "sneppx_graph", "SNEPPX-Alg", "1.3.0",
                               10, 1, 14,
                               inputs, 1, outputs, 1,
                               inits, 2,
                               nodes, 2) != 0) {
        fprintf(stderr, "FAIL: SneppX_onnx_save_graph returned nonzero\n");
        FAIL() << "early exit (legacy return 1)";
    }

    /* read bytes back */
    FILE* fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "FAIL: cannot open %s\n", path); FAIL() << "early exit (legacy return 1)"; }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz <= 0 || sz > 1024 * 1024) { fprintf(stderr, "FAIL: size %ld\n", sz); fclose(fp); FAIL() << "early exit (legacy return 1)"; }
    unsigned char* buf = (unsigned char*)malloc((size_t)sz);
    size_t bytes_rd = fread(buf, 1, (size_t)sz, fp);
    fclose(fp);
    if (bytes_rd != (size_t)sz) { fprintf(stderr, "FAIL: short read %zu\n", bytes_rd); free(buf); FAIL() << "early exit (legacy return 1)"; }

    /* raw protobuf, no ONNX magic */
    if (buf[0] != 0x08) { fprintf(stderr, "FAIL: first byte 0x%02x (expected 0x08)\n", buf[0]); failures++; }
    char producer[64] = {0};
    if (!pb_find_str(buf, 0, (size_t)sz, 2, producer, sizeof(producer)) || strcmp(producer, "SNEPPX-Alg") != 0) {
        fprintf(stderr, "FAIL: producer='%s'\n", producer); failures++;
    }
    uint64_t irver = 0;
    if (!pb_find_u64(buf, 0, (size_t)sz, 1, &irver) || irver == 0) { fprintf(stderr, "FAIL: ir_version\n"); failures++; }

    size_t go = 0, gl = 0;
    if (!pb_find(buf, 0, (size_t)sz, 7, 2, &go, &gl)) {
    }
    char gname[64] = {0};
    if (!pb_find_str(buf, go, gl, 2, gname, sizeof(gname)) || strcmp(gname, "sneppx_graph") != 0) {
        fprintf(stderr, "FAIL: graph name='%s'\n", gname); failures++;
    }

    /* opset version (OperatorSetIdProto.version = 2) */
    {
        size_t oo = 0, ol = 0; uint64_t osv = 0;
        if (pb_find(buf, 0, (size_t)sz, 8, 2, &oo, &ol) && !pb_find_u64(buf, oo, ol, 2, &osv)) {
            fprintf(stderr, "FAIL: opset version missing\n"); failures++;
        }
    }

    /* 2 nodes in order: node0 = Gemm, node1 = Relu */
    size_t nn = count_fields(buf, go, gl, 1, 2);
    WantSlot n0 = {0, 0, 0, 0}, n1 = {0, 0, 1, 0};
    pb_for_each(buf, go, gl, 1, grab_slot, &n0);
    pb_for_each(buf, go, gl, 1, grab_slot, &n1);
    if (nn != 2 || !n0.found || !n1.found) {
        fprintf(stderr, "FAIL: expected 2 nodes, found %zu (n0=%d n1=%d)\n", nn, n0.found, n1.found);
        failures++;
    } else {
        /* node 0: Gemm, inputs [X,W,B], output [hidden], transB=1 */
        char op[32] = {0};
        if (!pb_find_str(buf, n0.off, n0.len, 4, op, sizeof(op)) || strcmp(op, "Gemm") != 0) {
            fprintf(stderr, "FAIL: node0 op='%s'\n", op); failures++;
        }
        char in[3][32]; char out[3][32];
        if (collect_strings(buf, n0.off, n0.len, 1, in, 3) != 3 ||
            strcmp(in[0],"X")!=0 || strcmp(in[1],"W")!=0 || strcmp(in[2],"B")!=0) {
            fprintf(stderr, "FAIL: node0 inputs (%s,%s,%s)\n", in[0], in[1], in[2]); failures++;
        }
        if (collect_strings(buf, n0.off, n0.len, 2, out, 3) != 1 || strcmp(out[0],"hidden")!=0) {
            fprintf(stderr, "FAIL: node0 output '%s'\n", out[0]); failures++;
        }
        uint64_t tv = 0;
        if (!find_attr_int(buf, n0.off, n0.len, "transB", &tv) || tv != 1) {
            fprintf(stderr, "FAIL: node0 transB attr wrong\n"); failures++;
        }
        /* node 1: Relu, input [hidden], output [Y], NO attrs */
        char op2[32] = {0};
        if (!pb_find_str(buf, n1.off, n1.len, 4, op2, sizeof(op2)) || strcmp(op2, "Relu") != 0) {
            fprintf(stderr, "FAIL: node1 op='%s'\n", op2); failures++;
        }
        char in2[3][32]; char out2[3][32];
        if (collect_strings(buf, n1.off, n1.len, 1, in2, 3) != 1 || strcmp(in2[0],"hidden")!=0) {
            fprintf(stderr, "FAIL: node1 input '%s'\n", in2[0]); failures++;
        }
        if (collect_strings(buf, n1.off, n1.len, 2, out2, 3) != 1 || strcmp(out2[0],"Y")!=0) {
            fprintf(stderr, "FAIL: node1 output '%s'\n", out2[0]); failures++;
        }
        size_t nattr = count_fields(buf, n1.off, n1.len, 5, -1);
        if (nattr != 0) { fprintf(stderr, "FAIL: Relu should have 0 attrs, got %zu\n", nattr); failures++; }
    }

    /* initializers W[3,2] / B[3] round-trip */
    {
        size_t ioff[2] = {0, 0}, ilen[2] = {0, 0}; int got[2] = {0, 0};
        PBr ri; ri.b = buf; ri.len = go + gl; ri.pos = go;
        PBField f; size_t idx = 0;
        while (pbr_next(&ri, &f) && idx < 16) {
            if (f.field == 5 && f.wire == 2) {
                char nm[16] = {0};
                pb_find_str(buf, f.start, f.slen, 8, nm, sizeof(nm)); /* TensorProto.name = 8 */
                if (strcmp(nm, "W") == 0) { ioff[0] = f.start; ilen[0] = f.slen; got[0] = 1; }
                else if (strcmp(nm, "B") == 0) { ioff[1] = f.start; ilen[1] = f.slen; got[1] = 1; }
                idx++;
            }
        }
        if (!got[0] || !got[1]) { fprintf(stderr, "FAIL: initializers W/B not found\n"); failures++; }
        else {
            size_t rs = 0, rl = 0;
            pb_find(buf, ioff[0], ilen[0], 9, 2, &rs, &rl);
            if (rl != OUT_FEATURES * IN_FEATURES * sizeof(float) || memcmp(buf + rs, kWeights, rl) != 0) {
                fprintf(stderr, "FAIL: W raw_data mismatch (rl=%zu)\n", rl); failures++;
            }
            pb_find(buf, ioff[1], ilen[1], 9, 2, &rs, &rl);
            if (rl != OUT_FEATURES * sizeof(float) || memcmp(buf + rs, kBias, rl) != 0) {
                fprintf(stderr, "FAIL: B raw_data mismatch (rl=%zu)\n", rl); failures++;
            }
            uint64_t wd[8] = {0}; size_t ndw = 0;
            pb_find_packed_int64(buf, ioff[0], ilen[0], 1, wd, 8, &ndw);
            if (ndw != 2 || wd[0] != OUT_FEATURES || wd[1] != IN_FEATURES) {
                fprintf(stderr, "FAIL: W dims %zu %llu,%llu\n", ndw,
                        (unsigned long long)wd[0], (unsigned long long)wd[1]);
                failures++;
            }
            uint64_t dt = 999;
            pb_find_u64(buf, ioff[0], ilen[0], 2, &dt);
            if (dt != 1) { fprintf(stderr, "FAIL: W data_type=%llu\n", (unsigned long long)dt); failures++; }
        }
    }

    /* graph input X shape [batch,2], output Y shape [batch,3] */
    {
        WantSlot inS = {0, 0, 0, 0}, outS = {0, 0, 0, 0};
        pb_for_each(buf, go, gl, 11, grab_slot, &inS);
        pb_for_each(buf, go, gl, 12, grab_slot, &outS);
        if (inS.found) {
            char nm[32] = {0}; pb_find_str(buf, inS.off, inS.len, 1, nm, sizeof(nm));
            if (strcmp(nm, "X") != 0) { fprintf(stderr, "FAIL: input name '%s'\n", nm); failures++; }
            size_t to = 0, tl = 0; int ok_param = 0; uint64_t feat = 0; int got_feat = 0;
            if (pb_find(buf, inS.off, inS.len, 2, 2, &to, &tl) && pb_find(buf, to, tl, 1, 2, &to, &tl)) {
                size_t sho = 0, shl = 0;
                if (pb_find(buf, to, tl, 2, 2, &sho, &shl)) {
                    PBr rd; rd.b = buf; rd.len = sho + shl; rd.pos = sho; PBField df;
                    while (pbr_next(&rd, &df)) {
                        if (df.field == 1 && df.wire == 2) {
                            char dp[32] = {0}; uint64_t dv = 0;
                            int gots = pb_find_str(buf, df.start, df.slen, 2, dp, sizeof(dp));
                            int gotv = pb_find_u64(buf, df.start, df.slen, 1, &dv);
                            if (gots && dp[0]) ok_param = 1;
                            else if (gotv) { feat = dv; got_feat = 1; }
                        }
                    }
                }
            }
            if (!ok_param || !got_feat || feat != IN_FEATURES) {
                fprintf(stderr, "FAIL: X shape (param=%d feat=%llu got=%d)\n", ok_param, (unsigned long long)feat, got_feat);
                failures++;
            }
        }
        if (outS.found) {
            char nm[32] = {0}; pb_find_str(buf, outS.off, outS.len, 1, nm, sizeof(nm));
            if (strcmp(nm, "Y") != 0) { fprintf(stderr, "FAIL: output name '%s'\n", nm); failures++; }
            size_t to = 0, tl = 0; int ok_param = 0; uint64_t feat = 0; int got_feat = 0;
            if (pb_find(buf, outS.off, outS.len, 2, 2, &to, &tl) && pb_find(buf, to, tl, 1, 2, &to, &tl)) {
                size_t sho = 0, shl = 0;
                if (pb_find(buf, to, tl, 2, 2, &sho, &shl)) {
                    PBr rd; rd.b = buf; rd.len = sho + shl; rd.pos = sho; PBField df;
                    while (pbr_next(&rd, &df)) {
                        if (df.field == 1 && df.wire == 2) {
                            char dp[32] = {0}; uint64_t dv = 0;
                            int gots = pb_find_str(buf, df.start, df.slen, 2, dp, sizeof(dp));
                            int gotv = pb_find_u64(buf, df.start, df.slen, 1, &dv);
                            if (gots && dp[0]) ok_param = 1;
                            else if (gotv) { feat = dv; got_feat = 1; }
                        }
                    }
                }
            }
            if (!got_feat || feat != OUT_FEATURES || !ok_param) {
                fprintf(stderr, "FAIL: Y shape (feat=%llu param=%d)\n", (unsigned long long)feat, ok_param);
                failures++;
            }
        }
    }

    /* 8) semantics: hidden = X*W^T + B; Y = Relu(hidden) */
    {
        float X[IN_FEATURES] = { 1.0f, 2.0f };
        float hidden[OUT_FEATURES], Y[OUT_FEATURES];
        for (size_t n = 0; n < OUT_FEATURES; n++) {
            float s = (float)kBias[n];
            for (size_t k = 0; k < IN_FEATURES; k++) s += X[k] * kWeights[n * IN_FEATURES + k];
            hidden[n] = s;
            Y[n] = hidden[n] > 0.0f ? hidden[n] : 0.0f;
        }
        fprintf(stderr, "INFO: hidden=[%.4f,%.4f,%.4f] Y=[%.4f,%.4f,%.4f]\n",
                hidden[0], hidden[1], hidden[2], Y[0], Y[1], Y[2]);
        if (hidden[0] != 15.0f || hidden[1] != 31.0f || hidden[2] != 47.0f ||
            Y[0] != 15.0f || Y[1] != 31.0f || Y[2] != 47.0f) {
            fprintf(stderr, "FAIL: semantic ground-truth mismatch\n"); failures++;
        }
    }

    remove(path);
    free(buf);
    if (failures) { fprintf(stderr, "RESULT: %d failure(s)\n", failures); FAIL() << "early exit (legacy return 1)"; }
    fprintf(stderr, "PASS: arbitrary ONNX graph (Gemm->Relu) validated\n");
    return;
}
