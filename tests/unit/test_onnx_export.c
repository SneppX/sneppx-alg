/*
 * SNEPPX - ONNX binary exporter test.
 *
 * Verifies that SNEPPX_onnx_save_linear emits a CANONICAL ONNX ModelProto
 * (raw protobuf, the format onnxruntime / tf2onnx consume) by writing a linear
 * Gemm model and parsing the bytes back with a small STANDARD-field-number
 * protobuf reader (independent of the non-standard in-tree reader).
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

/* ---- minimal standard protobuf reader ---- */
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

static int pb_find_str(const unsigned char* b, size_t off, size_t n,
                       unsigned field, char* out, size_t maxout) {
    size_t s, l;
    if (!pb_find(b, off, n, field, 2, &s, &l)) return 0;
    size_t c = l < maxout ? l : maxout - 1;
    memcpy(out, b + s, c);
    out[c] = 0;
    return 1;
}

/* decode a packed repeated int64 (field) within [off,off+n) */
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

/* iterate every occurrence of a wire-2 field, calling fn per occurrence */
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

/* collect all wire-2 field "field" strings into names[] (cap); returns count */
static size_t collect_strings(const unsigned char* b, size_t off, size_t n,
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
    return k;
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
    snprintf(out, n, "%s/sneppx_onnx_export_xt_%u.onnx", t, SNEPPX_PID());
}

TEST(test_onnx_export, suite) {
    char path[512];
    unique_path(path, sizeof(path));

    int failures = 0;

    /* export a real linear model: Y = X * W^T + B, transB=1 */
    if (SNEPPX_onnx_save_linear(path, "sneppx_linear", "X", "Y",
                                IN_FEATURES, OUT_FEATURES,
                                kWeights, kBias) != 0) {
        fprintf(stderr, "FAIL: SNEPPX_onnx_save_linear returned nonzero\n");
        FAIL() << "early exit (legacy return 1)";
    }

    /* read bytes back */
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "FAIL: cannot open exported file %s\n", path);
        FAIL() << "early exit (legacy return 1)";
    }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz <= 0 || sz > 1024 * 1024) {
        fprintf(stderr, "FAIL: exported file size %ld unexpected\n", sz);
        fclose(fp);
        FAIL() << "early exit (legacy return 1)";
    }
    unsigned char* buf = (unsigned char*)malloc((size_t)sz);
    size_t bytes_rd = fread(buf, 1, (size_t)sz, fp);
    fclose(fp);
    if (bytes_rd != (size_t)sz) {
        fprintf(stderr, "FAIL: short read %zu\n", bytes_rd);
        free(buf);
        FAIL() << "early exit (legacy return 1)";
    }

    /* 1) raw protobuf, no "ONNX" magic prefix (first byte is ir_version tag 0x08) */
    if (buf[0] != 0x08) {
        fprintf(stderr, "FAIL: first byte 0x%02x, expected 0x08 (ir_version tag; raw protobuf)\n", buf[0]);
        failures++;
    }

    /* 2) ModelProto top-level */
    uint64_t irver = 0;
    if (!pb_find_u64(buf, 0, (size_t)sz, 1, &irver) || irver == 0) {
        fprintf(stderr, "FAIL: ModelProto.ir_version missing/0\n");
        failures++;
    }
    char producer[64] = {0};
    if (!pb_find_str(buf, 0, (size_t)sz, 2, producer, sizeof(producer)) ||
        strcmp(producer, "SNEPPX-Alg") != 0) {
        fprintf(stderr, "FAIL: producer_name='%s' (expected SNEPPX-Alg)\n", producer);
        failures++;
    }
    size_t go = 0, gl = 0;
    if (!pb_find(buf, 0, (size_t)sz, 7, 2, &go, &gl)) {
        fprintf(stderr, "FAIL: ModelProto.graph field 7 not found\n");
        remove(path);
        free(buf);
    }

    /* 3) opset_import (field 8) -> OperatorSetIdProto.version (field 2) */
    {
        size_t oo = 0, ol = 0;
        uint64_t osv = 0;
        if (pb_find(buf, 0, (size_t)sz, 8, 2, &oo, &ol)) {
            if (!pb_find_u64(buf, oo, ol, 2, &osv) || osv == 0) {
                fprintf(stderr, "FAIL: opset version missing/0\n");
                failures++;
            }
        } else {
            fprintf(stderr, "FAIL: opset_import field 8 missing\n");
            failures++;
        }
    }

    /* 4) GraphProto name */
    {
        char gname[128] = {0};
        if (!pb_find_str(buf, go, gl, 2, gname, sizeof(gname)) ||
            strcmp(gname, "sneppx_linear") != 0) {
            fprintf(stderr, "FAIL: graph.name='%s' (expected sneppx_linear)\n", gname);
            failures++;
        }
    }

    /* 5) exactly 1 Gemm node with inputs [X,W,B], output [Y], transB=1 */
    {
        WantSlot nodeSlot = {0, 0, 0, 0};
        size_t nnodes = pb_for_each(buf, go, gl, 1, grab_slot, &nodeSlot);
        if (nnodes != 1 || !nodeSlot.found) {
            fprintf(stderr, "FAIL: expected 1 node, found %zu\n", nnodes);
            failures++;
        } else {
            char op[32] = {0};
            if (!pb_find_str(buf, nodeSlot.off, nodeSlot.len, 4, op, sizeof(op)) ||
                strcmp(op, "Gemm") != 0) {
                fprintf(stderr, "FAIL: node op_type='%s' (expected Gemm)\n", op);
                failures++;
            }
            /* inputs field 1 */
            char ins[3][32];
            size_t nin = collect_strings(buf, nodeSlot.off, nodeSlot.len, 1, ins, 3);
            int inputs_ok = (nin == 3 &&
                             strcmp(ins[0], "X") == 0 &&
                             strcmp(ins[1], "W") == 0 &&
                             strcmp(ins[2], "B") == 0);
            if (!inputs_ok) {
                fprintf(stderr, "FAIL: node inputs wrong (n=%zu: %s,%s,%s)\n",
                        nin, ins[0], ins[1], ins[2]);
                failures++;
            }
            /* output field 2 */
            char outs[3][32];
            size_t nout = collect_strings(buf, nodeSlot.off, nodeSlot.len, 2, outs, 3);
            if (nout != 1 || strcmp(outs[0], "Y") != 0) {
                fprintf(stderr, "FAIL: node output wrong (n=%zu: %s)\n", nout, outs[0]);
                failures++;
            }
            /* transB attribute: type=INT(2), i=1 */
            int trans_ok = 0;
            PBr ra;
            ra.b = buf; ra.len = nodeSlot.off + nodeSlot.len; ra.pos = nodeSlot.off;
            PBField f;
            while (pbr_next(&ra, &f)) {
                if (f.field == 5 && f.wire == 2) {
                    char aname[32] = {0};
                    uint64_t atype = 0, aint = 0;
                    pb_find_str(buf, f.start, f.slen, 1, aname, sizeof(aname));
                    pb_find_u64(buf, f.start, f.slen, 20, &atype);
                    pb_find_u64(buf, f.start, f.slen, 3, &aint);
                    if (strcmp(aname, "transB") == 0 && atype == 2 && aint == 1) trans_ok = 1;
                }
            }
            if (!trans_ok) {
                fprintf(stderr, "FAIL: Gemm transB attribute missing/wrong\n");
                failures++;
            }
        }
    }

    /* 6) initializers: 2 (W [N,K], B [N]); round-trip raw_data == kWeights/kBias */
    {
        size_t ioff[2] = {0, 0};
        size_t ilen[2] = {0, 0};
        int got[2] = {0, 0};
        PBr ri;
        ri.b = buf; ri.len = go + gl; ri.pos = go;
        PBField f;
        size_t idx = 0;
        while (pbr_next(&ri, &f) && idx < 64) {
            if (f.field == 5 && f.wire == 2) {
                char nm[16] = {0};
                pb_find_str(buf, f.start, f.slen, 8, nm, sizeof(nm)); /* TensorProto.name = 8 */
                if (strcmp(nm, "W") == 0) { ioff[0] = f.start; ilen[0] = f.slen; got[0] = 1; }
                else if (strcmp(nm, "B") == 0) { ioff[1] = f.start; ilen[1] = f.slen; got[1] = 1; }
                idx++;
            }
        }
        if (!got[0] || !got[1]) {
            fprintf(stderr, "FAIL: expected initializers W and B (w=%d b=%d)\n", got[0], got[1]);
            failures++;
        } else {
            /* W dims [N=3,K=2], data_type FLOAT(1), raw_data == kWeights */
            {
                size_t off = ioff[0], n = ilen[0];
                uint64_t dims[8] = {0};
                size_t nd = 0;
                if (pb_find_packed_int64(buf, off, n, 1, dims, 8, &nd)) {
                    if (nd != 2 || dims[0] != OUT_FEATURES || dims[1] != IN_FEATURES) {
                        fprintf(stderr, "FAIL: W dims wrong (nd=%zu %llu,%llu)\n", nd,
                                (unsigned long long)dims[0], (unsigned long long)dims[1]);
                        failures++;
                    }
                } else {
                    fprintf(stderr, "FAIL: W dims not found\n");
                    failures++;
                }
                size_t rs = 0, rl = 0;
                if (pb_find(buf, off, n, 9, 2, &rs, &rl)) {
                    if (rl != OUT_FEATURES * IN_FEATURES * sizeof(float) ||
                        memcmp(buf + rs, kWeights, rl) != 0) {
                        fprintf(stderr, "FAIL: W raw_data mismatch (rl=%zu)\n", rl);
                        failures++;
                    }
                } else {
                    fprintf(stderr, "FAIL: W raw_data field 9 missing\n");
                    failures++;
                }
                uint64_t dt = 999;
                pb_find_u64(buf, off, n, 2, &dt);
                if (dt != 1) {
                    fprintf(stderr, "FAIL: W data_type=%llu (expected 1 FLOAT)\n", (unsigned long long)dt);
                    failures++;
                }
            }
            /* B dims [N=3], raw_data == kBias */
            {
                size_t off = ioff[1], n = ilen[1];
                uint64_t dims[8] = {0};
                size_t nd = 0;
                if (pb_find_packed_int64(buf, off, n, 1, dims, 8, &nd)) {
                    if (nd != 1 || dims[0] != OUT_FEATURES) {
                        fprintf(stderr, "FAIL: B dims wrong (nd=%zu %llu)\n", nd, (unsigned long long)dims[0]);
                        failures++;
                    }
                }
                size_t rs = 0, rl = 0;
                if (pb_find(buf, off, n, 9, 2, &rs, &rl)) {
                    if (rl != OUT_FEATURES * sizeof(float) ||
                        memcmp(buf + rs, kBias, rl) != 0) {
                        fprintf(stderr, "FAIL: B raw_data mismatch (rl=%zu)\n", rl);
                        failures++;
                    }
                }
            }
        }
    }

    /* 7) graph input X (field 11) shape [batch, in] and output Y (field 12) shape [batch, out] */
    {
        WantSlot inSlot = {0, 0, 0, 0}, outSlot = {0, 0, 0, 0};
        pb_for_each(buf, go, gl, 11, grab_slot, &inSlot);
        pb_for_each(buf, go, gl, 12, grab_slot, &outSlot);
        if (inSlot.found) {
            char nm[32] = {0};
            if (!pb_find_str(buf, inSlot.off, inSlot.len, 1, nm, sizeof(nm)) || strcmp(nm, "X") != 0) {
                fprintf(stderr, "FAIL: graph input name='%s' (expected X)\n", nm);
                failures++;
            }
            size_t to = 0, tl = 0;
            if (pb_find(buf, inSlot.off, inSlot.len, 2, 2, &to, &tl) &&     /* TypeProto */
                pb_find(buf, to, tl, 1, 2, &to, &tl)) {                      /* tensor_type */
                size_t sho = 0, shl = 0;
                if (pb_find(buf, to, tl, 2, 2, &sho, &shl)) {                /* shape */
                    PBr rd; rd.b = buf; rd.len = sho + shl; rd.pos = sho;
                    PBField df;
                    int first_param = 0;
                    char pparam[32] = {0};
                    uint64_t feat = 0; int got_feat = 0;
                    while (pbr_next(&rd, &df)) {
                        if (df.field == 1 && df.wire == 2) {                 /* dim */
                            char dp[32] = {0}; uint64_t dv = 0;
                            int gots = pb_find_str(buf, df.start, df.slen, 2, dp, sizeof(dp));
                            int gotv = pb_find_u64(buf, df.start, df.slen, 1, &dv);
                            if (gots && dp[0]) { if (!first_param) { strncpy(pparam, dp, 31); pparam[31]=0; first_param = 1; } }
                            else if (gotv) { feat = dv; got_feat = 1; }
                        }
                    }
                    if (!first_param || strcmp(pparam, "batch") != 0 || !got_feat || feat != IN_FEATURES) {
                        fprintf(stderr, "FAIL: X shape wrong (batch='%s' feat=%llu got=%d)\n", pparam, (unsigned long long)feat, got_feat);
                        failures++;
                    }
                }
            }
        }
        if (outSlot.found) {
            char nm[32] = {0};
            if (!pb_find_str(buf, outSlot.off, outSlot.len, 1, nm, sizeof(nm)) || strcmp(nm, "Y") != 0) {
                fprintf(stderr, "FAIL: graph output name='%s' (expected Y)\n", nm);
                failures++;
            }
            size_t to = 0, tl = 0;
            if (pb_find(buf, outSlot.off, outSlot.len, 2, 2, &to, &tl) &&
                pb_find(buf, to, tl, 1, 2, &to, &tl)) {                     /* tensor_type */
                size_t sho = 0, shl = 0;
                if (pb_find(buf, to, tl, 2, 2, &sho, &shl)) {               /* shape */
                    PBr rd; rd.b = buf; rd.len = sho + shl; rd.pos = sho;
                    PBField df;
                    uint64_t feat = 0; int got_feat = 0; int first_param = 0;
                    while (pbr_next(&rd, &df)) {
                        if (df.field == 1 && df.wire == 2) {
                            char dp[32] = {0}; uint64_t dv = 0;
                            int gots = pb_find_str(buf, df.start, df.slen, 2, dp, sizeof(dp));
                            int gotv = pb_find_u64(buf, df.start, df.slen, 1, &dv);
                            if (gots && dp[0]) first_param = 1;
                            else if (gotv) { feat = dv; got_feat = 1; }
                        }
                    }
                    if (!got_feat || feat != OUT_FEATURES || !first_param) {
                        fprintf(stderr, "FAIL: Y shape wrong (feat=%llu got=%d param=%d)\n", (unsigned long long)feat, got_feat, first_param);
                        failures++;
                    }
                }
            }
        }
    }

    /* 8) semantic truth: Y = X*W^T + B with W stored row-major [n][k] */
    {
        float X[IN_FEATURES] = { 1.0f, 2.0f };
        float Y[OUT_FEATURES];
        for (size_t n = 0; n < OUT_FEATURES; n++) {
            float s = (float)kBias[n];
            for (size_t k = 0; k < IN_FEATURES; k++) s += X[k] * kWeights[n * IN_FEATURES + k];
            Y[n] = s;
        }
        /* Y[0] = 1*1 + 2*2 + 10 = 15; Y[1] = 1*3 + 2*4 + 20 = 31; Y[2] = 1*5 + 2*6 + 30 = 47 */
        fprintf(stderr, "INFO: Y = X*W^T + B = [%.4f, %.4f, %.4f]\n", Y[0], Y[1], Y[2]);
        if (Y[0] != 15.0f || Y[1] != 31.0f || Y[2] != 47.0f) {
            fprintf(stderr, "FAIL: expected Y=[15,31,47]\n");
            failures++;
        }
    }

    remove(path);
    free(buf);

    if (failures) {
        fprintf(stderr, "RESULT: %d assertion failure(s)\n", failures);
        FAIL() << "early exit (legacy return 1)";
    }
    fprintf(stderr, "PASS: ONNX linear export validated against standard ONNX IR field numbers\n");
    return;
}
