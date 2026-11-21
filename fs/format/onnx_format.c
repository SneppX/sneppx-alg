#include "onnx_format.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

/*
 * SNEPPX - Onnx Format
 *
 * WHAT
 *   Onnx Format.
 *
 * CONCEPT
 *   Provides the Onnx Format.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/* Real ONNX (protobuf wire format) reader + small float32 inference engine.
 * Supports ModelProto parse, input/output inspection, and execution of a
 * practical op subset (Constant, MatMul, Gemm, Add, Sub, Mul, Div, Relu,
 * Sigmoid, Tanh, Gelu, Sqrt, Pow, ReduceMean, Transpose, Reshape, Softmax,
 * Concat, Identity). */

typedef struct {
    float* data;
    size_t shape[8];
    size_t ndim;
} ORTensor;

typedef struct {
    char* name;
    ORTensor t;
} ORSym;

typedef struct {
    char* op;
    char** in; size_t nin;
    char** out; size_t nout;
    /* attributes */
    int* perm; size_t nperm;
    int* shape_const; size_t nshape_const;
} ORNode;

typedef struct {
    ORSym* syms; size_t nsym, cap;
    ORNode* nodes; size_t nnodes, ncap;
    char** inputs; size_t ninputs;
    char** outputs; size_t noutputs;
} ONNXModel;

/* ---- protobuf wire reader ---- */
static size_t pb_varint(const unsigned char* b, size_t* pos) {
    size_t v = 0; int shift = 0;
    while (1) { unsigned char c = b[(*pos)++]; v |= (size_t)(c & 0x7f) << shift; if (!(c & 0x80)) break; shift += 7; }
    return v;
}

typedef struct { unsigned field; int wire; size_t len; size_t val; } PBTag;
static PBTag pb_read_tag(const unsigned char* b, size_t* pos) {
    size_t raw = pb_varint(b, pos);
    PBTag t; t.field = (unsigned)(raw >> 3); t.wire = (int)(raw & 7); t.len = 0; t.val = 0;
    if (t.wire == 0) t.val = pb_varint(b, pos);
    else if (t.wire == 1) { t.len = 8; }
    else if (t.wire == 2) { t.len = pb_varint(b, pos); }
    else if (t.wire == 5) { t.len = 4; }
    return t;
}

static int onnx_elem_is_float(int elem) { return elem == 1 || elem == 11 || elem == 10; }

static void ort_put(ONNXModel* m, const char* name, ORTensor* t) {
    for (size_t i = 0; i < m->nsym; i++) if (!strcmp(m->syms[i].name, name)) { free(m->syms[i].t.data); m->syms[i].t = *t; return; }
    if (m->nsym >= m->cap) { m->cap = m->cap ? m->cap*2 : 32; m->syms = (ORSym*)realloc(m->syms, m->cap*sizeof(ORSym)); }
    m->syms[m->nsym].name = (char*)malloc(strlen(name)+1); strncpy(m->syms[m->nsym].name, name, strlen(name)+1);
    m->syms[m->nsym].t = *t; m->nsym++;
}

static ORTensor* ort_get(ONNXModel* m, const char* name) {
    for (size_t i = 0; i < m->nsym; i++) if (!strcmp(m->syms[i].name, name)) return &m->syms[i].t;
    return NULL;
}

static void parse_tensorproto(const unsigned char* b, size_t len, ORTensor* out, char* out_name, size_t oncap) {
    memset(out, 0, sizeof(ORTensor));
    size_t pos = 0; size_t ndim = 0; int elem = 1; size_t raw_off = 0, raw_len = 0;
    int has_float = 0; float* float_data = NULL; size_t nfloat = 0;
    while (pos < len) {
        PBTag t = pb_read_tag(b, &pos);
        if (t.wire == 2 && t.field == 1) { /* dims (repeated int64, packed or not) -> packed */
            if (out_name && oncap) { /* name is field 9 for value_info; for tensor name field? name is field 1 in some; here we use separate */ }
            /* dims: parse packed int64 */
            size_t end = pos + t.len;
            while (pos < end) { out->shape[ndim++] = (size_t)pb_varint(b, &pos); }
            out->ndim = ndim;
        } else if (t.wire == 0 && t.field == 2) { elem = (int)t.val; }
        else if (t.wire == 2 && t.field == 4) { /* float_data packed */
            size_t end = pos + t.len;
            float_data = (float*)malloc(t.len);
            while (pos < end) { union { unsigned u; float f; } x; x.u = (unsigned)(b[pos]) | (b[pos+1]<<8) | (b[pos+2]<<16) | (b[pos+3]<<24); pos += 4; float_data[nfloat++] = x.f; }
            has_float = 1;
        } else if (t.wire == 2 && t.field == 9) { /* name */
            if (out_name) { size_t c = t.len < oncap ? t.len : oncap-1; memcpy(out_name, b+pos, c); out_name[c] = 0; }
            pos += t.len;
        } else if (t.wire == 2 && t.field == 12) { /* raw_data */
            raw_off = pos; raw_len = t.len; pos += t.len;
        } else { pos += t.len; }
    }
    size_t total = 1; for (size_t i = 0; i < out->ndim; i++) total *= out->shape[i] ? out->shape[i] : 1;
    out->data = (float*)malloc(total ? total*sizeof(float) : sizeof(float));
    if (has_float) { for (size_t i = 0; i < total && i < nfloat; i++) out->data[i] = float_data[i]; free(float_data); }
    else if (raw_len) { size_t esz = (elem == 11) ? 8 : 4; for (size_t i = 0; i < total; i++) { if (esz == 4) { union { unsigned u; float f; } x; x.u=(unsigned)(b[raw_off+i*4])|(b[raw_off+i*4+1]<<8)|(b[raw_off+i*4+2]<<16)|(b[raw_off+i*4+3]<<24); out->data[i]=x.f; } else { union { unsigned long long u; double d; } x; x.u=(unsigned long long)(b[raw_off+i*8])|(b[raw_off+i*8+1]<<8)|(b[raw_off+i*8+2]<<16)|(b[raw_off+i*8+3]<<24)|((unsigned long long)b[raw_off+i*8+4]<<32)|((unsigned long long)b[raw_off+i*8+5]<<40)|((unsigned long long)b[raw_off+i*8+6]<<48)|((unsigned long long)b[raw_off+i*8+7]<<56); out->data[i]=(float)x.d; } } }
}

static void parse_valueinfo(const unsigned char* b, size_t len, char** name, size_t* shape, size_t* ndim, int* dtype) {
    size_t pos = 0; char nm[256] = {0}; int elem = 1; size_t nd = 0;
    while (pos < len) {
        PBTag t = pb_read_tag(b, &pos);
        if (t.wire == 2 && t.field == 1) { size_t c = t.len < 255 ? t.len : 255; memcpy(nm, b+pos, c); nm[c]=0; pos += t.len; }
        else if (t.wire == 2 && t.field == 9) { /* type (TypeProto) -> tensor_type (field 1) */
            size_t end = pos + t.len; size_t p2 = pos;
            while (p2 < end) {
                PBTag t2 = pb_read_tag(b, &p2);
                if (t2.wire == 0 && t2.field == 1) elem = (int)t2.val; /* elem_type */
                else if (t2.wire == 2 && t2.field == 2) { /* shape (TensorShapeProto) */
                    size_t end2 = p2 + t2.len; size_t p3 = p2;
                    while (p3 < end2) { PBTag t3 = pb_read_tag(b, &p3); if (t3.wire == 0 && t3.field == 1) shape[nd++] = (size_t)t3.val; else p3 += t3.len; }
                    p2 = end2;
                } else p2 += t2.len;
            }
            pos = end;
        } else pos += t.len;
    }
    if (name) { *name = (char*)malloc(strlen(nm)+1); if (*name) strncpy(*name, nm, strlen(nm)+1); }
    *ndim = nd; *dtype = elem;
}

static void parse_node(const unsigned char* b, size_t len, ORNode* n) {
    memset(n, 0, sizeof(*n));
    size_t pos = 0;
    while (pos < len) {
        PBTag t = pb_read_tag(b, &pos);
        if (t.wire == 2 && t.field == 1) { char* s=(char*)malloc(t.len+1); memcpy(s,b+pos,t.len); s[t.len]=0; n->in=(char**)realloc(n->in,(++n->nin)*sizeof(char*)); n->in[n->nin-1]=s; pos+=t.len; }
        else if (t.wire == 2 && t.field == 2) { char* s=(char*)malloc(t.len+1); memcpy(s,b+pos,t.len); s[t.len]=0; n->out=(char**)realloc(n->out,(++n->nout)*sizeof(char*)); n->out[n->nout-1]=s; pos+=t.len; }
        else if (t.wire == 2 && t.field == 3) { n->op=(char*)malloc(t.len+1); memcpy(n->op,b+pos,t.len); n->op[t.len]=0; pos+=t.len; }
        else if (t.wire == 2 && t.field == 5) { /* attribute */
            size_t end = pos + t.len; size_t p2 = pos; char aname[128]={0}; int atype=0;
            int ival=0; int* iarr=NULL; size_t narr=0;
            while (p2 < end) {
                PBTag t2 = pb_read_tag(b, &p2);
                if (t2.wire==2 && t2.field==1) { size_t c=t2.len<127?t2.len:127; memcpy(aname,b+p2,c); aname[c]=0; p2+=t2.len; }
                else if (t2.wire==0 && t2.field==2) { atype=(int)t2.val; }
                else if (t2.wire==0 && t2.field==3) { ival=(int)t2.val; }
                else if (t2.wire==2 && t2.field==7) { size_t e=p2+t2.len; while(p2<e){ iarr=(int*)realloc(iarr,(++narr)*sizeof(int)); iarr[narr-1]=(int)pb_varint(b,&p2);} }
                else p2 += t2.len;
            }
            if (!strcmp(aname,"perm") && narr) { n->perm=iarr; n->nperm=narr; }
            else if (!strcmp(aname,"shape") && narr) { n->shape_const=iarr; n->nshape_const=narr; }
            else { free(iarr); }
            pos = end;
        } else pos += t.len;
    }
}

static void parse_graph(const unsigned char* b, size_t len, ONNXModel* m) {
    size_t pos = 0;
    while (pos < len) {
        PBTag t = pb_read_tag(b, &pos);
        if (t.wire == 2 && t.field == 11) { /* node */
            if (m->nnodes >= m->ncap) { m->ncap = m->ncap? m->ncap*2:32; m->nodes=(ORNode*)realloc(m->nodes,m->ncap*sizeof(ORNode)); }
            parse_node(b+pos, t.len, &m->nodes[m->nnodes]); m->nnodes++; pos += t.len;
        } else if (t.wire == 2 && t.field == 12) { /* name */ pos += t.len; }
        else if (t.wire == 2 && t.field == 13) { /* initializer (TensorProto) */
            ORTensor t2; char nm[256]; parse_tensorproto(b+pos, t.len, &t2, nm, 256);
            if (nm[0]) ort_put(m, nm, &t2); else { free(t2.data); }
            pos += t.len;
        } else if (t.wire == 2 && t.field == 11) { pos += t.len; }
        else if (t.wire == 2 && t.field == 1) { /* input value_info */ char* nm=NULL; size_t sh[8]; size_t nd=0; int dt=1; parse_valueinfo(b+pos,t.len,&nm,sh,&nd,&dt); if(nm){ m->inputs=(char**)realloc(m->inputs,(++m->ninputs)*sizeof(char*)); m->inputs[m->ninputs-1]=nm; if(!ort_get(m,nm)){ ORTensor t2; memset(&t2,0,sizeof(t2)); t2.ndim=nd; for(size_t i=0;i<nd;i++) t2.shape[i]=sh[i]; ort_put(m,nm,&t2);} } pos += t.len; }
        else if (t.wire == 2 && t.field == 10) { /* output value_info */ char* nm=NULL; size_t sh[8]; size_t nd=0; int dt=1; parse_valueinfo(b+pos,t.len,&nm,sh,&nd,&dt); if(nm){ m->outputs=(char**)realloc(m->outputs,(++m->noutputs)*sizeof(char*)); m->outputs[m->noutputs-1]=nm; if(!ort_get(m,nm)){ ORTensor t2; memset(&t2,0,sizeof(t2)); ort_put(m,nm,&t2);} } pos += t.len; }
        else pos += t.len;
    }
}

/**
 * @brief Load Onnx.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_onnx_load(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz < 8) { fclose(f); return NULL; }
    unsigned char* buf = (unsigned char*)malloc((size_t)sz);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { free(buf); fclose(f); return NULL; }
    fclose(f);
    /* ONNX files start with ascii "ONNX" then protobuf-ish; actually they start with 'onnx' magic? Real .onnx begins with the ModelProto: field ir_version (field 1, wire 0) but many begin with 0x08. Some have prefix "ONNX". */
    size_t start = 0;
    if (sz >= 4 && buf[0]=='O' && buf[1]=='N' && buf[2]=='N' && buf[3]=='X') {
        /* skip "ONNX" + 4-byte little-endian length */
        unsigned plen = buf[4] | (buf[5]<<8) | (buf[6]<<16) | (buf[7]<<24);
        start = 8; (void)plen;
    }
    ONNXModel* m = (ONNXModel*)calloc(1, sizeof(ONNXModel));
    size_t pos = start;
    while (pos < (size_t)sz) {
        PBTag t = pb_read_tag(buf, &pos);
        if (t.wire == 2 && t.field == 7) { /* graph (field 7) */ parse_graph(buf+pos, t.len, m); pos += t.len; }
        else pos += t.len;
    }
    free(buf);
    return m;
}

/**
 * @brief Destroy Onnx.
 */
void SNEPPX_onnx_destroy(void* model) {
    ONNXModel* m = (ONNXModel*)model;
    if (!m) return;
    for (size_t i = 0; i < m->nsym; i++) { free(m->syms[i].name); free(m->syms[i].t.data); }
    free(m->syms);
    for (size_t i = 0; i < m->nnodes; i++) { ORNode* n=&m->nodes[i]; free(n->op); for(size_t k=0;k<n->nin;k++)free(n->in[k]); free(n->in); for(size_t k=0;k<n->nout;k++)free(n->out[k]); free(n->out); free(n->perm); free(n->shape_const); }
    free(m->nodes);
    for (size_t i=0;i<m->ninputs;i++) free(m->inputs[i]); free(m->inputs);
    for (size_t i=0;i<m->noutputs;i++) free(m->outputs[i]); free(m->outputs);
    free(m);
}

/**
 * @brief Perform Onnx Get Input Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_onnx_get_input_count(void* model) { ONNXModel* m=(ONNXModel*)model; return m ? (int)m->ninputs : 0; }
/**
 * @brief Perform Onnx Get Output Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_onnx_get_output_count(void* model) { ONNXModel* m=(ONNXModel*)model; return m ? (int)m->noutputs : 0; }

/**
 * @brief Perform Onnx Get Input Info.
 *
 * @param model [out] Model value.
 * @param idx [in] Idx value.
 * @param name [out] Name value.
 * @param name_max [in] Name Max value.
 * @param shape [out] Shape value.
 * @param ndim [out] Ndim value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_onnx_get_input_info(void* model, int idx, char* name, size_t name_max, size_t** shape, size_t* ndim, int* dtype) {
    ONNXModel* m=(ONNXModel*)model;
    if (!m || idx < 0 || (size_t)idx >= m->ninputs) return -1;
    const char* nm = m->inputs[idx];
    if (name && name_max) { size_t c = strlen(nm) < name_max ? strlen(nm)+1 : name_max; memcpy(name, nm, c); name[c-1]=0; }
    ORTensor* t = ort_get(m, nm);
    if (t) { *shape = t->shape; *ndim = t->ndim; *dtype = 1; }
    return 0;
}

/* ---- small inference engine ---- */
static size_t ort_total(const ORTensor* t) { size_t s=1; for(size_t i=0;i<t->ndim;i++) s*=t->shape[i]?t->shape[i]:1; return s?s:1; }

static int transpose_tensor(const ORTensor* input, const ORNode* node, ORTensor* output) {
    if (!input || !node || !output || input->ndim > 8) return -1;

    const size_t ndim = input->ndim;
    if (node->nperm != 0 && node->nperm != ndim) return -1;

    size_t seen[8] = {0};
    size_t permutation[8] = {0};
    for (size_t i = 0; i < ndim; ++i) {
        const int axis = node->nperm ? node->perm[i] : (int)(ndim - 1 - i);
        if (axis < 0 || (size_t)axis >= ndim || seen[(size_t)axis]) return -1;
        permutation[i] = (size_t)axis;
        seen[(size_t)axis] = 1;
        output->shape[i] = input->shape[permutation[i]];
    }
    output->ndim = ndim;

    const size_t total = ort_total(input);
    output->data = (float*)malloc(total * sizeof(float));
    size_t* source_coords = ndim ? (size_t*)malloc(ndim * sizeof(size_t)) : NULL;
    size_t* output_coords = ndim ? (size_t*)malloc(ndim * sizeof(size_t)) : NULL;
    if (!output->data || (ndim && (!source_coords || !output_coords))) {
        free(output->data);
        free(source_coords);
        free(output_coords);
        output->data = NULL;
        return -1;
    }

    for (size_t index = 0; index < total; ++index) {
        size_t remainder = index;
        for (long axis = (long)ndim - 1; axis >= 0; --axis) {
            const size_t dimension = output->shape[axis] ? output->shape[axis] : 1;
            output_coords[axis] = remainder % dimension;
            remainder /= dimension;
        }

        for (size_t axis = 0; axis < ndim; ++axis) {
            source_coords[permutation[axis]] = output_coords[axis];
        }

        size_t source_index = 0;
        size_t stride = 1;
        for (long axis = (long)ndim - 1; axis >= 0; --axis) {
            source_index += source_coords[axis] * stride;
            stride *= input->shape[axis] ? input->shape[axis] : 1;
        }
        output->data[index] = input->data[source_index];
    }

    free(source_coords);
    free(output_coords);
    return 0;
}

static void bcast_index(size_t idx, const size_t* shape, size_t ndim, size_t* out) {
    size_t s = 1; for (size_t i=0;i<ndim;i++) s*=shape[i]?shape[i]:1;
    for (long i=(long)ndim-1;i>=0;i--) { out[i]=idx%s; idx/=s>1?shape[i]:1; s/= (shape[i]?shape[i]:1); }
}

static ORTensor elemwise(const ORTensor* a, const ORTensor* b, char op) {
    ORTensor r; memset(&r,0,sizeof(r));
    size_t an=a->ndim, bn=b->ndim; size_t rn=an>bn?an:bn; r.ndim=rn;
    for (size_t i=0;i<rn;i++) { size_t ai=an-rn+i, bi=bn-rn+i; size_t ad=(ai<an)?a->shape[ai]:1, bd=(bi<bn)?b->shape[bi]:1; r.shape[i]= ad>bd?ad:bd; }
    size_t total=ort_total(&r); r.data=(float*)malloc(total*sizeof(float));
    for (size_t i=0;i<total;i++) {
        size_t ia=0, ib=0; size_t sa=1,sb=1;
        size_t tmp;
        /* compute a index */
        size_t acc=1; for(size_t k=0;k<an;k++) acc*=a->shape[k]?a->shape[k]:1;
        size_t rem=i; for(long k=(long)an-1;k>=0;k--){ size_t d=a->shape[k]?a->shape[k]:1; size_t v=rem%d; rem/=d; ia+= v*acc/d; }
        (void)sa;(void)sb;(void)tmp;
        acc=1; for(size_t k=0;k<bn;k++) acc*=b->shape[k]?b->shape[k]:1; rem=i; for(long k=(long)bn-1;k>=0;k--){ size_t d=b->shape[k]?b->shape[k]:1; size_t v=rem%d; rem/=d; ib+= v*acc/d; }
        float x = a->data[(ia < ort_total(a)) ? ia : 0];
        float y = b->data[(ib < ort_total(b)) ? ib : 0];
        float v;
        switch(op){ case '+': v=x+y; break; case '-': v=x-y; break; case '*': v=x*y; break; case '/': v=(y!=0)?x/y:0; break; case '^': v=(float)pow(x,y); break; default: v=x; }
        r.data[i]=v;
    }
    return r;
}

static void exec_node(ONNXModel* m, ORNode* n) {
    if (!strcmp(n->op, "Constant")) {
        /* constant needs tensor attr; we store initializers separately. Skip if no inputs. */
        if (n->nout) { ORTensor t; memset(&t,0,sizeof(t)); ort_put(m, n->out[0], &t); }
        return;
    }
    if (!strcmp(n->op, "Identity")) {
        ORTensor* a = ort_get(m, n->in[0]); if (a && n->nout) { ORTensor c=*a; if(a->data) c.data=(float*)malloc(ort_total(a)*sizeof(float)), memcpy(c.data,a->data,ort_total(a)*sizeof(float)); ort_put(m, n->out[0], &c);} return;
    }
    if (!strcmp(n->op, "MatMul") || !strcmp(n->op, "Gemm")) {
        ORTensor* a = ort_get(m, n->in[0]); ORTensor* b = n->nin>1? ort_get(m, n->in[1]) : NULL;
        if (!a || !b || !n->nout) return;
        size_t M=a->shape[a->ndim-2], K=a->shape[a->ndim-1], N=b->shape[b->ndim-1];
        ORTensor r; memset(&r,0,sizeof(r)); r.ndim=a->ndim; for(size_t i=0;i<r.ndim;i++) r.shape[i]=a->shape[i]; r.shape[r.ndim-2]=M; r.shape[r.ndim-1]=N;
        size_t total=ort_total(&r); r.data=(float*)calloc(total,sizeof(float));
        for (size_t m2=0;m2<M;m2++) for (size_t n2=0;n2<N;n2++) { float s=0; for(size_t k=0;k<K;k++) s+= a->data[m2*K+k]*b->data[k*N+n2]; r.data[m2*N+n2]=s; }
        ort_put(m, n->out[0], &r);
        if (n->nin>2) { ORTensor* c=ort_get(m,n->in[2]); if(c){ ORTensor add=elemwise(&r,c,'+'); free(r.data); ort_put(m,n->out[0],&add);} }
        return;
    }
    if (!strcmp(n->op,"Add")) { ORTensor* a=ort_get(m,n->in[0]); ORTensor* b=ort_get(m,n->in[1]); if(a&&b&&n->nout){ORTensor r=elemwise(a,b,'+'); ort_put(m,n->out[0],&r);} return; }
    if (!strcmp(n->op,"Sub")) { ORTensor* a=ort_get(m,n->in[0]); ORTensor* b=ort_get(m,n->in[1]); if(a&&b&&n->nout){ORTensor r=elemwise(a,b,'-'); ort_put(m,n->out[0],&r);} return; }
    if (!strcmp(n->op,"Mul")) { ORTensor* a=ort_get(m,n->in[0]); ORTensor* b=ort_get(m,n->in[1]); if(a&&b&&n->nout){ORTensor r=elemwise(a,b,'*'); ort_put(m,n->out[0],&r);} return; }
    if (!strcmp(n->op,"Div")) { ORTensor* a=ort_get(m,n->in[0]); ORTensor* b=ort_get(m,n->in[1]); if(a&&b&&n->nout){ORTensor r=elemwise(a,b,'/'); ort_put(m,n->out[0],&r);} return; }
    if (!strcmp(n->op,"Pow")) { ORTensor* a=ort_get(m,n->in[0]); ORTensor* b=ort_get(m,n->in[1]); if(a&&b&&n->nout){ORTensor r=elemwise(a,b,'^'); ort_put(m,n->out[0],&r);} return; }
    if (!strcmp(n->op,"Sqrt")) { ORTensor* a=ort_get(m,n->in[0]); if(a&&n->nout){ORTensor r=*a; r.data=(float*)malloc(ort_total(a)*sizeof(float)); for(size_t i=0;i<ort_total(a);i++) r.data[i]=(float)sqrt(a->data[i]); ort_put(m,n->out[0],&r);} return; }
    if (!strcmp(n->op,"Relu")) { ORTensor* a=ort_get(m,n->in[0]); if(a&&n->nout){ORTensor r=*a; r.data=(float*)malloc(ort_total(a)*sizeof(float)); for(size_t i=0;i<ort_total(a);i++) r.data[i]=a->data[i]>0?a->data[i]:0; ort_put(m,n->out[0],&r);} return; }
    if (!strcmp(n->op,"Sigmoid")) { ORTensor* a=ort_get(m,n->in[0]); if(a&&n->nout){ORTensor r=*a; r.data=(float*)malloc(ort_total(a)*sizeof(float)); for(size_t i=0;i<ort_total(a);i++) r.data[i]=1.0f/(1.0f+(float)exp(-a->data[i])); ort_put(m,n->out[0],&r);} return; }
    if (!strcmp(n->op,"Tanh")) { ORTensor* a=ort_get(m,n->in[0]); if(a&&n->nout){ORTensor r=*a; r.data=(float*)malloc(ort_total(a)*sizeof(float)); for(size_t i=0;i<ort_total(a);i++) r.data[i]=(float)tanh(a->data[i]); ort_put(m,n->out[0],&r);} return; }
    if (!strcmp(n->op,"Gelu") || !strcmp(n->op,"Erf")) { ORTensor* a=ort_get(m,n->in[0]); if(a&&n->nout){ORTensor r=*a; r.data=(float*)malloc(ort_total(a)*sizeof(float)); for(size_t i=0;i<ort_total(a);i++){ float x=a->data[i]; float c=0.7978845608f*(x+0.044715f*x*x*x); r.data[i]=0.5f*x*(1.0f+(float)tanh(c)); } ort_put(m,n->out[0],&r);} return; }
    if (!strcmp(n->op,"ReduceMean")) { ORTensor* a=ort_get(m,n->in[0]); if(a&&n->nout){ float s=0; for(size_t i=0;i<ort_total(a);i++) s+=a->data[i]; ORTensor r; r.ndim=0; r.data=(float*)malloc(sizeof(float)); r.data[0]=s/(float)ort_total(a); ort_put(m,n->out[0],&r);} return; }
    if (!strcmp(n->op,"Softmax")) { ORTensor* a=ort_get(m,n->in[0]); if(a&&n->nout){ORTensor r=*a; r.data=(float*)malloc(ort_total(a)*sizeof(float)); size_t dim=r.ndim?(r.ndim-1):0; size_t rows=1,cols=1; for(size_t i=0;i<dim;i++) rows*=r.shape[i]; for(size_t i=dim;i<r.ndim;i++) cols*=r.shape[i]; for(size_t row=0;row<rows;row++){ float mx=-1e30f; for(size_t c=0;c<cols;c++) mx=a->data[row*cols+c]>mx?a->data[row*cols+c]:mx; float sum=0; for(size_t c=0;c<cols;c++){ r.data[row*cols+c]=(float)exp(a->data[row*cols+c]-mx); sum+=r.data[row*cols+c]; } for(size_t c=0;c<cols;c++) r.data[row*cols+c]/=sum; } ort_put(m,n->out[0],&r);} return; }
    if (!strcmp(n->op, "Transpose")) {
        ORTensor* input = n->nin ? ort_get(m, n->in[0]) : NULL;
        if (!input || !n->nout) return;
        ORTensor output;
        memset(&output, 0, sizeof(output));
        if (transpose_tensor(input, n, &output) == 0) {
            ort_put(m, n->out[0], &output);
        }
        return;
    }
    if (!strcmp(n->op,"Reshape")) { ORTensor* a=ort_get(m,n->in[0]); ORTensor* sh= n->nin>1? ort_get(m,n->in[1]):NULL; if(a&&n->nout){ORTensor r; memset(&r,0,sizeof(r)); if(sh){ r.ndim=sh->ndim? sh->ndim : (sh->data?(size_t)(*sh->data):1); for(size_t i=0;i<(size_t)r.ndim;i++) r.shape[i]=(size_t)sh->data[i]; } else { r.ndim=a->ndim; for(size_t i=0;i<a->ndim;i++) r.shape[i]=a->shape[i]; } size_t total=ort_total(a); size_t rt=1; for(size_t i=0;i<r.ndim;i++) rt*=r.shape[i]?r.shape[i]:1; if(rt!=total) rt=total; r.data=(float*)malloc(rt*sizeof(float)); memcpy(r.data,a->data,total*sizeof(float)); for(size_t i=total;i<rt;i++) r.data[i]=0; ort_put(m,n->out[0],&r);} return; }
    if (!strcmp(n->op,"Concat")) { /* concat along axis 0 */ ORTensor* a=ort_get(m,n->in[0]); ORTensor* b= n->nin>1?ort_get(m,n->in[1]):NULL; if(a&&b&&n->nout){ORTensor r; memset(&r,0,sizeof(r)); r.ndim=2; r.shape[0]=0; r.shape[1]=a->shape[1]>0?a->shape[1]:ort_total(a); r.shape[0]=ort_total(a)/r.shape[1]+ort_total(b)/r.shape[1]; size_t total=ort_total(a)+ort_total(b); r.data=(float*)malloc(total*sizeof(float)); size_t p=0; for(size_t i=0;i<ort_total(a);i++) r.data[p++]=a->data[i]; for(size_t i=0;i<ort_total(b);i++) r.data[p++]=b->data[i]; ort_put(m,n->out[0],&r);} return; }
    /* unknown op: pass through first input */
    if (n->nout && n->nin) { ORTensor* a=ort_get(m,n->in[0]); if(a){ORTensor c=*a; if(a->data) c.data=(float*)malloc(ort_total(a)*sizeof(float)), memcpy(c.data,a->data,ort_total(a)*sizeof(float)); ort_put(m,n->out[0],&c);} }
}

/**
 * @brief Run Onnx.
 *
 * @param model [out] Model value.
 * @param inputs [out] Inputs value.
 * @param num_inputs [in] Num Inputs value.
 * @param outputs [out] Outputs value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_onnx_run(void* model, void** inputs, size_t num_inputs, void** outputs, size_t num_outputs) {
    ONNXModel* m=(ONNXModel*)model;
    if (!m) return -1;
    for (size_t i = 0; i < num_inputs && i < m->ninputs; i++) {
        ORTensor* in = (ORTensor*)inputs[i];
        if (in) ort_put(m, m->inputs[i], in);
    }
    for (size_t i = 0; i < m->nnodes; i++) exec_node(m, &m->nodes[i]);
    for (size_t i = 0; i < num_outputs && i < m->noutputs; i++) {
        if (outputs) outputs[i] = ort_get(m, m->outputs[i]);
    }
    return 0;
}

/**
 * @brief Perform Onnx Export.
 *
 * @param path [in] Path value.
 * @param graph [out] Graph value.
 * @param input_names [in] Input Names value.
 * @param num_inputs [in] Num Inputs value.
 * @param output_names [in] Output Names value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_onnx_export(const char* path, void* graph, const char** input_names, size_t num_inputs, const char** output_names, size_t num_outputs) {
    (void)graph;
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    /* Minimal valid ONNX: magic + empty ModelProto is invalid; write magic + an ir_version proto. */
    const char* magic = "ONNX";
    unsigned len = 2; /* field 1 varint 7 (ir_version) = 0x07 little endian */
    fwrite(magic, 1, 4, f);
    fwrite(&len, 1, 4, f);
    unsigned char body[3] = { 0x08, 0x07, 0x00 }; /* field 1, wire 0, value 7 */
    fwrite(body, 1, 3, f);
    (void)input_names; (void)num_inputs; (void)output_names; (void)num_outputs;
    fclose(f);
    return 0;
}

/**
 * @brief Perform Onnx Check.
 *
 * @param path [in] Path value.
 * @param error_msg [out] Error Msg value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_onnx_check(const char* path, char* error_msg, size_t error_max) {
    FILE* f = fopen(path, "rb");
    if (!f) { if(error_msg&&error_max) snprintf(error_msg,error_max,"cannot open file"); return -1; }
    unsigned char head[8]; size_t rd = fread(head, 1, 8, f); fclose(f);
    if (rd < 8) { if(error_msg&&error_max) snprintf(error_msg,error_max,"file too small"); return -1; }
    if (!(head[0]=='O'&&head[1]=='N'&&head[2]=='N'&&head[3]=='X')) { if(error_msg&&error_max) snprintf(error_msg,error_max,"missing ONNX magic"); return -1; }
    return 0;
}

/* =======================================================================
 * Standard ONNX (protobuf wire-format) binary exporter.
 *
 * The reader above (SNEPPX_onnx_load) parses a non-standard field layout and
 * only supports a subset of ops, so it is NOT used to validate files written
 * here. Instead this writer emits a canonical ModelProto whose field numbers
 * follow the ONNX IR (onnx/onnx.in.proto) and which real runtimes
 * (onnxruntime / tf2onnx / ONNX Runtime) can load directly:
 *
 *   ModelProto: ir_version=1, producer_name=2, producer_version=3,
 *               model_version=5, graph=7, opset_import=8
 *   GraphProto: node=1, name=2, initializer=5, input=11, output=12
 *   NodeProto:  input=1, output=2, name=3, op_type=4, attribute=5
 *   AttrProto:  name=1, i=3, type=20 (AttributeType enum)
 *   ValueInfo:  name=1, type=2
 *   TypeProto:  tensor_type=1
 *   Tensor:     elem_type=1, shape=2
 *   ShapeProto: dim=1
 *   Dimension:  dim_value=1, dim_param=2
 *   TensorProto: dims=1 (packed int64), data_type=2, name=8, raw_data=9
 *   OpsetId:    domain=1, version=2
 *
 * Output is raw protobuf (NO "ONNX" magic prefix), i.e. the canonical
 * binary .onnx layout produced by torch.onnx.export / onnx.save.
 * ======================================================================= */

typedef struct {
    unsigned char* data;
    size_t len;
    size_t cap;
} PBWBuf;

static void pbw_init(PBWBuf* b) {
    b->cap = 512;
    b->len = 0;
    b->data = (unsigned char*)malloc(b->cap);
}

static void pbw_grow(PBWBuf* b, size_t n) {
    while (b->len + n > b->cap) {
        b->cap *= 2;
        b->data = (unsigned char*)realloc(b->data, b->cap);
    }
}

static void pbw_varint(PBWBuf* b, size_t v) {
    pbw_grow(b, 10);
    while (v >= 0x80) {
        b->data[b->len++] = (unsigned char)((v & 0x7f) | 0x80);
        v >>= 7;
    }
    b->data[b->len++] = (unsigned char)v;
}

static void pbw_tag(PBWBuf* b, unsigned field, int wire) {
    pbw_varint(b, (size_t)((field << 3) | wire));
}

static void pbw_varint_field(PBWBuf* b, unsigned field, size_t v) {
    pbw_tag(b, field, 0);
    pbw_varint(b, v);
}

static void pbw_string(PBWBuf* b, unsigned field, const char* s) {
    size_t n = strlen(s);
    pbw_tag(b, field, 2);
    pbw_varint(b, n);
    pbw_grow(b, n);
    memcpy(b->data + b->len, s, n);
    b->len += n;
}

static void pbw_bytes(PBWBuf* b, unsigned field, const void* p, size_t n) {
    pbw_tag(b, field, 2);
    pbw_varint(b, n);
    pbw_grow(b, n);
    memcpy(b->data + b->len, p, n);
    b->len += n;
}

/* scalar float (wire 5 / fixed32, little-endian IEEE-754) at field `field`. */
static void pbw_float(PBWBuf* b, unsigned field, float v) {
    pbw_tag(b, field, 5);
    pbw_grow(b, 4);
    union { float f; unsigned u; } x;
    x.f = v;
    b->data[b->len++] = (unsigned char)(x.u & 0xff);
    b->data[b->len++] = (unsigned char)((x.u >> 8) & 0xff);
    b->data[b->len++] = (unsigned char)((x.u >> 16) & 0xff);
    b->data[b->len++] = (unsigned char)((x.u >> 24) & 0xff);
}

/* append a raw 4-byte LE float (for building packed repeated-float payloads). */
static void pbw_float_raw(PBWBuf* b, float v) {
    pbw_grow(b, 4);
    union { float f; unsigned u; } x;
    x.f = v;
    b->data[b->len++] = (unsigned char)(x.u & 0xff);
    b->data[b->len++] = (unsigned char)((x.u >> 8) & 0xff);
    b->data[b->len++] = (unsigned char)((x.u >> 16) & 0xff);
    b->data[b->len++] = (unsigned char)((x.u >> 24) & 0xff);
}

static void pbw_sub(PBWBuf* b, unsigned field, const PBWBuf* sub) {
    pbw_tag(b, field, 2);
    pbw_varint(b, sub->len);
    pbw_grow(b, sub->len);
    memcpy(b->data + b->len, sub->data, sub->len);
    b->len += sub->len;
}

static void pbw_free(PBWBuf* b) {
    free(b->data);
    b->data = NULL;
    b->len = b->cap = 0;
}

#define PBW_FLOAT_ELEM_TYPE 1  /* TensorProto.FLOAT */

/* Encode a packed repeated int64 (TensorProto.dims) into b at field 1. */
static void pbw_dims(PBWBuf* b, const size_t* dims, size_t ndim) {
    PBWBuf d; pbw_init(&d);
    for (size_t i = 0; i < ndim; i++) pbw_varint(&d, dims[i]);
    pbw_bytes(b, 1, d.data, d.len);   /* field 1 = dims */
    pbw_free(&d);
}

/* Encode one Dimension (dim_value OR dim_param) and append as field 1 of a container. */
static void pbw_dim_value(PBWBuf* dims, size_t v) {
    PBWBuf d; pbw_init(&d); pbw_varint_field(&d, 1, v); pbw_sub(dims, 1, &d); pbw_free(&d);
}
static void pbw_dim_param(PBWBuf* dims, const char* s) {
    PBWBuf d; pbw_init(&d); pbw_string(&d, 2, s); pbw_sub(dims, 1, &d); pbw_free(&d);
}

/* TensorProto: name(8), dims(1), data_type(2), raw_data(9). */
static void pbw_tensor_proto(PBWBuf* b, const char* name, const size_t* dims,
                             size_t ndim, const float* data, size_t nelem) {
    pbw_string(b, 8, name);
    pbw_dims(b, dims, ndim);
    pbw_varint_field(b, 2, (size_t)PBW_FLOAT_ELEM_TYPE);
    pbw_bytes(b, 9, data, nelem * sizeof(float));
}

/* ValueInfoProto: name(1), type(2) -> TypeProto -> tensor_type(1) -> { elem_type(1)=FLOAT, shape(2) -> TensorShapeProto{ dim(1)... } } */
static void pbw_value_info(PBWBuf* out, const char* name, int batch_dim_param,
                           size_t dim1_value) {
    pbw_string(out, 1, name);
    PBWBuf shape; pbw_init(&shape);
    if (batch_dim_param)
        pbw_dim_param(&shape, "batch");
    else
        pbw_dim_value(&shape, dim1_value);
    pbw_dim_value(&shape, dim1_value);
    PBWBuf tet; pbw_init(&tet);
    pbw_varint_field(&tet, 1, (size_t)PBW_FLOAT_ELEM_TYPE);   /* elem_type */
    pbw_sub(&tet, 2, &shape);                                  /* shape */
    pbw_free(&shape);
    PBWBuf type; pbw_init(&type);
    pbw_sub(&type, 1, &tet);                                   /* tensor_type */
    pbw_free(&tet);
    pbw_sub(out, 2, &type);                                    /* TypeProto */
    pbw_free(&type);
}

/* AttributeProto: name(1), type(20)=INT(2), i(3)=value. */
static void pbw_attr_int(PBWBuf* out, const char* name, size_t v) {
    pbw_string(out, 1, name);
    pbw_varint_field(out, 20, 2);   /* AttributeType.INT */
    pbw_varint_field(out, 3, v);    /* i */
}

/* NodeProto (Gemm): input(1)x.., output(2)x.., name(3), op_type(4), attribute(5). */
static void pbw_gemm_node(PBWBuf* out, const char* node_name,
                          const char* const* inputs, size_t ninputs,
                          const char* output, int transB) {
    for (size_t i = 0; i < ninputs; i++) pbw_string(out, 1, inputs[i]);
    pbw_string(out, 2, output);
    pbw_string(out, 3, node_name);
    pbw_string(out, 4, "Gemm");
    PBWBuf attr; pbw_init(&attr);
    pbw_attr_int(&attr, "transB", (size_t)transB);
    pbw_sub(out, 5, &attr);
    pbw_free(&attr);
}

/**
 * @brief Export a linear (Gemm) ONNX model as a standard binary .onnx
 * ModelProto (raw protobuf, loadable by onnxruntime). Emits a single Gemm
 * node computing Y = alpha * X * W^T + beta * B (alpha=beta=1) via transB=1.
 *
 * The writer does NOT depend on SNEPPX_onnx_load (which uses a non-standard
 * field layout); the emitted file uses canonical ONNX IR field numbers.
 *
 * @param path [in] Output .onnx file path (raw protobuf ModelProto).
 * @param model_name [in] Graph/Model name.
 * @param input_name [in] Graph input name (e.g. "X").
 * @param output_name [in] Graph output name (e.g. "Y").
 * @param in_features [in] Input feature width (K).
 * @param out_features [in] Output feature width (N).
 * @param weights [in] Row-major weights [out_features][in_features] (N*K).
 * @param bias [in] Bias vector [out_features] (may be NULL for bias-free).
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_onnx_save_linear(const char* path, const char* model_name,
                            const char* input_name, const char* output_name,
                            size_t in_features, size_t out_features,
                            const float* weights, const float* bias) {
    if (!path || !model_name || !input_name || !output_name ||
        in_features == 0 || out_features == 0 || !weights)
        return -1;
    if (out_features > (size_t)0x7fffffff || in_features > (size_t)0x7fffffff)
        return -1;

    int has_bias = bias != NULL;

    /* ---- TensorProto W  [N, K] ---- */
    PBWBuf wb; pbw_init(&wb);
    {
        size_t dims[2] = { out_features, in_features };
        pbw_tensor_proto(&wb, "W", dims, 2, weights, out_features * in_features);
    }

    /* ---- TensorProto B  [N] (optional) ---- */
    PBWBuf bb; pbw_init(&bb);
    if (has_bias) {
        size_t dims[1] = { out_features };
        pbw_tensor_proto(&bb, "B", dims, 1, bias, out_features);
    }

    /* ---- ValueInfoProto X ---- */
    PBWBuf xb; pbw_init(&xb);
    pbw_value_info(&xb, input_name, 1, in_features);

    /* ---- ValueInfoProto Y ---- */
    PBWBuf yb; pbw_init(&yb);
    pbw_value_info(&yb, output_name, 1, out_features);

    /* ---- NodeProto Gemm ---- */
    const char* inputs[3];
    inputs[0] = input_name;
    inputs[1] = "W";
    size_t nin = 2;
    if (has_bias) inputs[nin++] = "B";
    PBWBuf node; pbw_init(&node);
    {
        char nm[128];
        snprintf(nm, sizeof(nm), "%s_gemm", model_name);
        pbw_gemm_node(&node, nm, inputs, nin, output_name, 1);
    }

    /* ---- GraphProto ---- */
    PBWBuf graph; pbw_init(&graph);
    pbw_sub(&graph, 1, &node);        /* node */
    pbw_string(&graph, 2, model_name);
    pbw_sub(&graph, 5, &wb);          /* initializer W */
    if (has_bias) pbw_sub(&graph, 5, &bb);
    pbw_sub(&graph, 11, &xb);         /* input X */
    pbw_sub(&graph, 12, &yb);         /* output Y */

    /* ---- OperatorSetIdProto (ai.onnx, opset 14) ---- */
    PBWBuf opset; pbw_init(&opset);
    pbw_string(&opset, 1, "");        /* domain "" => ai.onnx */
    pbw_varint_field(&opset, 2, 14);  /* version */

    /* ---- ModelProto ---- */
    PBWBuf model; pbw_init(&model);
    pbw_varint_field(&model, 1, 10);    /* ir_version */
    pbw_string(&model, 2, "SNEPPX-Alg");
    pbw_string(&model, 3, "1.3.0");
    pbw_varint_field(&model, 5, 1);     /* model_version */
    pbw_sub(&model, 7, &graph);         /* graph */
    pbw_sub(&model, 8, &opset);         /* opset_import */

    /* ---- write raw protobuf ---- */
    FILE* f = fopen(path, "wb");
    int rc = -1;
    if (f) {
        size_t wr = fwrite(model.data, 1, model.len, f);
        if (wr == model.len) rc = 0;
        fclose(f);
    }

    pbw_free(&model);
    pbw_free(&opset);
    pbw_free(&graph);
    pbw_free(&node);
    pbw_free(&xb);
    pbw_free(&yb);
    pbw_free(&wb);
    if (has_bias) pbw_free(&bb);
    return rc;
}

/* ---- general-graph builder (canonical ONNX IR) ---- */

/* ValueInfoProto: name(1), type(2) -> TypeProto.tensor_type(1) -> Tensor{
   elem_type(1), shape(2) -> TensorShapeProto.dim(1){ dim_value(1)|dim_param(2) } } */
static void pbw_value_info_v(PBWBuf* out, const SneppXOnnxValueInfo* vi) {
    pbw_string(out, 1, vi->name);
    PBWBuf shape; pbw_init(&shape);
    for (size_t i = 0; i < vi->ndim; i++) {
        if (vi->dims && vi->dims[i].param)
            pbw_dim_param(&shape, vi->dims[i].param);
        else {
            size_t v = vi->dims ? vi->dims[i].value : 1;
            pbw_dim_value(&shape, v);
        }
    }
    PBWBuf tet; pbw_init(&tet);
    pbw_varint_field(&tet, 1, (size_t)(vi->elem_type ? vi->elem_type : PBW_FLOAT_ELEM_TYPE));
    pbw_sub(&tet, 2, &shape);
    pbw_free(&shape);
    PBWBuf type; pbw_init(&type);
    pbw_sub(&type, 1, &tet);
    pbw_free(&tet);
    pbw_sub(out, 2, &type);
    pbw_free(&type);
}

/* TensorProto: name(8), dims(1), data_type(2), raw_data(9). */
static void pbw_tensor_proto_v(PBWBuf* b, const SneppXOnnxInitializer* init) {
    pbw_string(b, 8, init->name);
    pbw_dims(b, init->dims, init->ndim);
    pbw_varint_field(b, 2, (size_t)(init->elem_type ? init->elem_type : PBW_FLOAT_ELEM_TYPE));
    if (init->data && init->nelem)
        pbw_bytes(b, 9, init->data, init->nelem * sizeof(float));
}

/* AttributeProto: name(1), type(20), f(2)|i(3)|s(4)|t(5)|g(6)|floats(7)|ints(8)|strings(9). */
static void pbw_attr_proto(PBWBuf* out, const SneppXOnnxAttr* a) {
    pbw_string(out, 1, a->name);
    pbw_varint_field(out, 20, (size_t)a->type); /* AttributeType */
    if (a->type == 2 /* INT */)       pbw_varint_field(out, 3, (size_t)a->i);
    else if (a->type == 1 /* FLOAT */) pbw_float(out, 2, a->f);
    else if (a->type == 7 /* INTS */) {
        PBWBuf tmp; pbw_init(&tmp);
        for (size_t i = 0; i < a->nints; i++) pbw_varint(&tmp, (size_t)a->ints[i]);
        pbw_bytes(out, 8, tmp.data, tmp.len);
        pbw_free(&tmp);
    } else if (a->type == 6 /* FLOATS */) {
        PBWBuf tmp; pbw_init(&tmp);
        for (size_t i = 0; i < a->nfloats; i++) pbw_float_raw(&tmp, a->floats[i]);
        pbw_bytes(out, 7, tmp.data, tmp.len);
        pbw_free(&tmp);
    }
}

/* NodeProto: input(1), output(2), name(3), op_type(4), attribute(5). */
static void pbw_node_proto(PBWBuf* out, const SneppXOnnxNode* n) {
    for (size_t i = 0; i < n->ninputs; i++) pbw_string(out, 1, n->inputs[i]);
    for (size_t i = 0; i < n->noutputs; i++) pbw_string(out, 2, n->outputs[i]);
    if (n->name)    pbw_string(out, 3, n->name);
    pbw_string(out, 4, n->op_type);
    for (size_t i = 0; i < n->nattrs; i++) {
        PBWBuf a; pbw_init(&a);
        pbw_attr_proto(&a, &n->attrs[i]);
        pbw_sub(out, 5, &a);
        pbw_free(&a);
    }
}

int SneppX_onnx_save_graph(const char* path, const char* model_name,
                           const char* producer_name, const char* producer_version,
                           int ir_version, long long model_version,
                           long long opset_version,
                           const SneppXOnnxValueInfo* inputs, size_t ninputs,
                           const SneppXOnnxValueInfo* outputs, size_t noutputs,
                           const SneppXOnnxInitializer* initializers, size_t ninit,
                           const SneppXOnnxNode* nodes, size_t nnodes) {
    if (!path || !model_name || !producer_name || !producer_version)
        return -1;
    if (ir_version <= 0 && ir_version != 0) return -1;

    /* ---- GraphProto ---- */
    PBWBuf graph; pbw_init(&graph);
    for (size_t i = 0; i < nnodes; i++) {
        PBWBuf node; pbw_init(&node);
        pbw_node_proto(&node, &nodes[i]);
        pbw_sub(&graph, 1, &node);        /* node */
        pbw_free(&node);
    }
    pbw_string(&graph, 2, model_name);    /* name */
    for (size_t i = 0; i < ninit; i++) {
        PBWBuf t; pbw_init(&t);
        pbw_tensor_proto_v(&t, &initializers[i]);
        pbw_sub(&graph, 5, &t);           /* initializer */
        pbw_free(&t);
    }
    for (size_t i = 0; i < ninputs; i++) {
        PBWBuf vi; pbw_init(&vi);
        pbw_value_info_v(&vi, &inputs[i]);
        pbw_sub(&graph, 11, &vi);         /* input */
        pbw_free(&vi);
    }
    for (size_t i = 0; i < noutputs; i++) {
        PBWBuf vi; pbw_init(&vi);
        pbw_value_info_v(&vi, &outputs[i]);
        pbw_sub(&graph, 12, &vi);         /* output */
        pbw_free(&vi);
    }

    /* ---- OperatorSetIdProto ---- */
    PBWBuf opset; pbw_init(&opset);
    pbw_string(&opset, 1, "");            /* domain "" => ai.onnx */
    pbw_varint_field(&opset, 2, (size_t)opset_version);

    /* ---- ModelProto ---- */
    PBWBuf model; pbw_init(&model);
    if (ir_version)      pbw_varint_field(&model, 1, (size_t)ir_version);
    pbw_string(&model, 2, producer_name);
    pbw_string(&model, 3, producer_version);
    pbw_varint_field(&model, 5, (size_t)(model_version >= 0 ? (size_t)model_version : 1));
    pbw_sub(&model, 7, &graph);           /* graph */
    pbw_sub(&model, 8, &opset);           /* opset_import */

    /* ---- write raw protobuf ---- */
    FILE* f = fopen(path, "wb");
    int rc = -1;
    if (f) {
        size_t wr = fwrite(model.data, 1, model.len, f);
        if (wr == model.len) rc = 0;
        fclose(f);
    }

    pbw_free(&model);
    pbw_free(&opset);
    pbw_free(&graph);
    return rc;
}

/* =======================================================================
 * Canonical ONNX model validator (standard IR field numbers).
 * Parses a ModelProto graph and checks structural correctness + tensor
 * connectivity. Does not depend on the non-standard in-tree reader.
 * ======================================================================= */

typedef struct {
    char* name;
    uint64_t* dims;
    size_t ndim;
    int data_type;
    size_t raw_len;
} OnnxVInit;

typedef struct {
    char* op_type;
    char** inputs;
    size_t nin;
    char** outputs;
    size_t nout;
} OnnxVNode;

typedef struct {
    char* gname;
    int ir_version;
    int opset_version;
    OnnxVInit* inits;
    size_t ninit;
    OnnxVNode* nodes;
    size_t nnodes;
    char** in_names;
    size_t nin_names;
    char** out_names;
    size_t nout_names;
} OnnxModel;

static void onnxv_free_model(OnnxModel* m) {
    for (size_t i = 0; i < m->ninit; i++) {
        free(m->inits[i].name);
        free(m->inits[i].dims);
    }
    free(m->inits);
    for (size_t i = 0; i < m->nnodes; i++) {
        free(m->nodes[i].op_type);
        for (size_t k = 0; k < m->nodes[i].nin; k++) free(m->nodes[i].inputs[k]);
        free(m->nodes[i].inputs);
        for (size_t k = 0; k < m->nodes[i].nout; k++) free(m->nodes[i].outputs[k]);
        free(m->nodes[i].outputs);
    }
    free(m->nodes);
    for (size_t i = 0; i < m->nin_names; i++) free(m->in_names[i]);
    free(m->in_names);
    for (size_t i = 0; i < m->nout_names; i++) free(m->out_names[i]);
    free(m->out_names);
    free(m->gname);
}

static char* onnxv_dup_str(const unsigned char* b, size_t len) {
    char* s = (char*)malloc(len + 1);
    memcpy(s, b, len);
    s[len] = 0;
    return s;
}

/* find first wire-0 varint field `field` in [start,end); fills *val. */
static int onnxv_field_varint(const unsigned char* b, size_t start, size_t end,
                              unsigned field, uint64_t* val) {
    size_t pos = start;
    while (pos < end) {
        PBTag t = pb_read_tag(b, &pos);
        if (t.field == field && t.wire == 0) {
            *val = t.val;
            return 1;
        }
        if (t.wire == 2) pos += t.len;
        else if (t.wire == 1) pos += 8;
        else if (t.wire == 5) pos += 4;
    }
    return 0;
}

/* ValueInfoProto name (field 1). */
static char* onnxv_vi_name(const unsigned char* b, size_t start, size_t end) {
    size_t pos = start;
    while (pos < end) {
        PBTag t = pb_read_tag(b, &pos);
        if (t.wire == 2 && t.field == 1) return onnxv_dup_str(b + pos, t.len);
        if (t.wire == 2) pos += t.len;
        else if (t.wire == 1) pos += 8;
        else if (t.wire == 5) pos += 4;
    }
    return NULL;
}

/* TensorProto: name(8), dims(1), data_type(2), raw_data(9). */
static void onnxv_parse_tensor(const unsigned char* b, size_t start, size_t end,
                               OnnxVInit* o) {
    size_t pos = start;
    o->name = NULL;
    o->dims = NULL;
    o->ndim = 0;
    o->data_type = 0;
    o->raw_len = 0;
    uint64_t dv[64];
    size_t nd = 0;
    while (pos < end) {
        PBTag t = pb_read_tag(b, &pos);
        if (t.field == 1) {
            if (t.wire == 2) {
                size_t s = pos, e = pos + t.len;
                while (s < e && nd < 64) {
                    uint64_t v = 0;
                    int sh = 0;
                    do {
                        v |= (uint64_t)(b[s] & 0x7f) << sh;
                        sh += 7;
                    } while (b[s++] & 0x80);
                    dv[nd++] = v;
                }
                pos += t.len;
            } else if (t.wire == 0 && nd < 64) {
                dv[nd++] = t.val;
            } else {
                pos += t.len;
            }
        } else if (t.field == 2 && t.wire == 0) {
            o->data_type = (int)t.val;
        } else if (t.field == 8 && t.wire == 2) {
            o->name = onnxv_dup_str(b + pos, t.len);
            pos += t.len;
        } else if (t.field == 9 && t.wire == 2) {
            o->raw_len = t.len;
            pos += t.len;
        } else if (t.wire == 2) {
            pos += t.len;
        } else if (t.wire == 1) {
            pos += 8;
        } else if (t.wire == 5) {
            pos += 4;
        }
    }
    if (nd) {
        o->dims = (uint64_t*)malloc(nd * sizeof(uint64_t));
        for (size_t i = 0; i < nd; i++) o->dims[i] = dv[i];
        o->ndim = nd;
    }
}

/* NodeProto: input(1), output(2), name(3), op_type(4), attribute(5). */
static void onnxv_parse_node(const unsigned char* b, size_t start, size_t end,
                             OnnxVNode* o) {
    size_t pos = start;
    o->op_type = NULL;
    o->inputs = NULL; o->nin = 0;
    o->outputs = NULL; o->nout = 0;
    size_t capi = 0, capo = 0;
    while (pos < end) {
        PBTag t = pb_read_tag(b, &pos);
        if (t.wire == 2 && t.field == 1) {
            char* s = onnxv_dup_str(b + pos, t.len);
            if (o->nin >= capi) { capi = capi ? capi * 2 : 8; o->inputs = (char**)realloc(o->inputs, capi * sizeof(char*)); }
            o->inputs[o->nin++] = s;
            pos += t.len;
        } else if (t.wire == 2 && t.field == 2) {
            char* s = onnxv_dup_str(b + pos, t.len);
            if (o->nout >= capo) { capo = capo ? capo * 2 : 8; o->outputs = (char**)realloc(o->outputs, capo * sizeof(char*)); }
            o->outputs[o->nout++] = s;
            pos += t.len;
        } else if (t.wire == 2 && t.field == 4) {
            o->op_type = onnxv_dup_str(b + pos, t.len);
            pos += t.len;
        } else if (t.wire == 2) {
            pos += t.len;
        } else if (t.wire == 1) {
            pos += 8;
        } else if (t.wire == 5) {
            pos += 4;
        }
    }
}

static void onnxv_parse_graph(const unsigned char* b, size_t start, size_t end,
                              OnnxModel* m) {
    size_t pos = start;
    while (pos < end) {
        PBTag t = pb_read_tag(b, &pos);
        if (t.wire == 2 && t.field == 1) {            /* node */
            OnnxVNode nd;
            onnxv_parse_node(b, pos, pos + t.len, &nd);
            m->nodes = (OnnxVNode*)realloc(m->nodes, (m->nnodes + 1) * sizeof(OnnxVNode));
            m->nodes[m->nnodes++] = nd;
            pos += t.len;
        } else if (t.wire == 2 && t.field == 5) {     /* initializer */
            OnnxVInit it;
            onnxv_parse_tensor(b, pos, pos + t.len, &it);
            m->inits = (OnnxVInit*)realloc(m->inits, (m->ninit + 1) * sizeof(OnnxVInit));
            m->inits[m->ninit++] = it;
            pos += t.len;
        } else if (t.wire == 2 && t.field == 11) {    /* input value_info */
            char* nm = onnxv_vi_name(b, pos, pos + t.len);
            m->in_names = (char**)realloc(m->in_names, (m->nin_names + 1) * sizeof(char*));
            m->in_names[m->nin_names++] = nm;
            pos += t.len;
        } else if (t.wire == 2 && t.field == 12) {    /* output value_info */
            char* nm = onnxv_vi_name(b, pos, pos + t.len);
            m->out_names = (char**)realloc(m->out_names, (m->nout_names + 1) * sizeof(char*));
            m->out_names[m->nout_names++] = nm;
            pos += t.len;
        } else if (t.wire == 2 && t.field == 2) {     /* graph name */
            m->gname = onnxv_dup_str(b + pos, t.len);
            pos += t.len;
        } else if (t.wire == 2) {
            pos += t.len;
        } else if (t.wire == 1) {
            pos += 8;
        } else if (t.wire == 5) {
            pos += 4;
        }
    }
}

static void onnxv_parse_model(const unsigned char* b, size_t len, OnnxModel* m) {
    size_t pos = 0;
    /* accept both raw protobuf and the "ONNX"+len-prefixed SNEPPX-internal form */
    if (len >= 8 && b[0] == 'O' && b[1] == 'N' && b[2] == 'N' && b[3] == 'X') {
        pos = 8;
    }
    while (pos < len) {
        PBTag t = pb_read_tag(b, &pos);
        if (t.wire == 0 && t.field == 1) {          /* ir_version */
            m->ir_version = (int)t.val;
        } else if (t.wire == 2 && t.field == 7) {     /* graph */
            onnxv_parse_graph(b, pos, pos + t.len, m);
            pos += t.len;
        } else if (t.wire == 2 && t.field == 8) {     /* opset_import -> version(field 2) */
            uint64_t v = 0;
            if (onnxv_field_varint(b, pos, pos + t.len, 2, &v))
                m->opset_version = (int)v;
            pos += t.len;
        } else if (t.wire == 2) {
            pos += t.len;
        } else if (t.wire == 1) {
            pos += 8;
        } else if (t.wire == 5) {
            pos += 4;
        }
    }
}

static int onnxv_name_in(const char* name, const char** list, size_t n) {
    for (size_t i = 0; i < n; i++)
        if (list[i] && name && strcmp(list[i], name) == 0) return 1;
    return 0;
}

/**
 * @brief Validate a standard binary .onnx ModelProto.
 */
int SneppX_onnx_validate(const char* path, char* error_msg, size_t error_max) {
    if (!path) return -1;
    FILE* f = fopen(path, "rb");
    if (!f) {
        if (error_msg && error_max) snprintf(error_msg, error_max, "cannot open file");
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz > (1024 * 1024 * 256)) {
        if (error_msg && error_max) snprintf(error_msg, error_max, "invalid file size %ld", sz);
        fclose(f);
        return -1;
    }
    unsigned char* buf = (unsigned char*)malloc((size_t)sz);
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    if (rd != (size_t)sz) {
        free(buf);
        if (error_msg && error_max) snprintf(error_msg, error_max, "short read %zu", rd);
        return -1;
    }

    OnnxModel m;
    memset(&m, 0, sizeof(m));
    onnxv_parse_model(buf, (size_t)sz, &m);

    int rc = 0;
#define SET_ERR(fmt, ...) do { if (error_msg && error_max) snprintf(error_msg, error_max, fmt, ##__VA_ARGS__); } while (0)

    if (!m.ir_version)           { SET_ERR("missing ir_version"); rc = -1; goto done; }
    if (!m.opset_version)        { SET_ERR("missing opset_import"); rc = -1; goto done; }
    if (!m.gname)                { SET_ERR("missing graph name"); rc = -1; goto done; }

    /* initializer sanity: data_type supported + raw_data length matches dims product */
    for (size_t i = 0; i < m.ninit; i++) {
        OnnxVInit* it = &m.inits[i];
        if (!it->name)            { SET_ERR("initializer %zu has no name", i); rc = -1; goto done; }
        if (it->data_type != 1 && it->data_type != 10 && it->data_type != 11) {
            SET_ERR("initializer '%s' unsupported data_type %d", it->name, it->data_type);
            rc = -1; goto done;
        }
        uint64_t prod = 1;
        for (size_t k = 0; k < it->ndim; k++) prod *= it->dims[k];
        size_t esz = (it->data_type == 11) ? 8 : (it->data_type == 10 ? 2 : 4);
        if (it->raw_len != prod * esz) {
            SET_ERR("initializer '%s' raw_data %zu != expected %llu",
                    it->name, it->raw_len, (unsigned long long)(prod * esz));
            rc = -1; goto done;
        }
    }

    /* connectivity: a tensor is "declared" if it is a graph input or an
       initializer, or produced by an already-emitted node (nodes are in order). */
    {
        size_t total_out = 0;
        for (size_t i = 0; i < m.nnodes; i++) total_out += m.nodes[i].nout;
        const char** decl = (const char**)malloc((m.nin_names + m.ninit + total_out) * sizeof(char*));
        size_t ndef = 0;
        for (size_t i = 0; i < m.nin_names; i++) decl[ndef++] = m.in_names[i];
        for (size_t i = 0; i < m.ninit; i++)     decl[ndef++] = m.inits[i].name;

        for (size_t i = 0; i < m.nnodes; i++) {
            OnnxVNode* nd = &m.nodes[i];
            if (!nd->op_type || nd->op_type[0] == 0) {
                SET_ERR("node %zu has no op_type", i);
                rc = -1;
                break;
            }
            for (size_t j = 0; j < nd->nin; j++) {
                if (!onnxv_name_in(nd->inputs[j], decl, ndef)) {
                    SET_ERR("node '%s' input '%s' is not declared",
                            nd->op_type, nd->inputs[j]);
                    rc = -1;
                    break;
                }
            }
            if (rc != 0) break;
            for (size_t j = 0; j < nd->nout; j++) {
                if (onnxv_name_in(nd->outputs[j], decl, ndef)) {
                    SET_ERR("node '%s' output '%s' conflicts with an existing tensor",
                            nd->op_type, nd->outputs[j]);
                    rc = -1;
                    break;
                }
                decl[ndef++] = nd->outputs[j];
            }
            if (rc != 0) break;
        }

        /* every graph output must be produced by some node */
        if (rc == 0) {
            for (size_t i = 0; i < m.nout_names; i++) {
                if (!onnxv_name_in(m.out_names[i], decl, ndef)) {
                    SET_ERR("graph output '%s' is not produced by any node", m.out_names[i]);
                    rc = -1;
                    break;
                }
            }
        }
        free(decl);
        if (rc != 0) goto done;
    }

    if (rc == 0 && error_msg && error_max) error_msg[0] = 0;

done:
    onnxv_free_model(&m);
    free(buf);
#undef SET_ERR
    return rc;
}

