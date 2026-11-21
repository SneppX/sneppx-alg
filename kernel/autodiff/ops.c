/* ===== Zero-gradient helper for comparison ops (no gradient through comparisons) ===== */
#include "automatic_differentiation_framework.h"
#include "multidimensional_tensor_engine.h"
#include "multi_head_attention_module.h"
#include "polymorphic_memory_allocator.h"
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
/*
 * SNEPPX - Kernel Module
 *
 * WHAT
 *   Kernel Module.
 *
 * CONCEPT
 *   Kernel Module implementation.
 *
 * ROLE
 *   Core kernel module used throughout the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal kernel module).
 */



/* ---------- internal helpers ---------- */
static int requires_grad(SNEPPXVariable* a, SNEPPXVariable* b) {
    return (a && a->requires_grad) || (b && b->requires_grad);
}
static int requires_grad1(SNEPPXVariable* a) {
    return a && a->requires_grad;
}
static void grad_accum(SNEPPXTensor** grad, SNEPPXTensor* new_grad) {
    if (!new_grad) return;
    if (*grad) {
        float* gd = (float*)(*grad)->data;
        float* ng = (float*)new_grad->data;
        size_t n = (*grad)->size < new_grad->size ? (*grad)->size : new_grad->size;
        for (size_t i = 0; i < n; i++) gd[i] += ng[i];
        SNEPPX_tensor_destroy(new_grad);
    } else {
        *grad = new_grad;
    }
}
static void set_parents(SNEPPXVariable* var, SNEPPXVariable** parents, size_t n) {
    if (!var || !parents || n == 0) return;
    var->parents = (SNEPPXVariable**)SNEPPX_malloc(n * sizeof(SNEPPXVariable*), 64);
    if (!var->parents) return;
    memcpy(var->parents, parents, n * sizeof(SNEPPXVariable*));
    var->num_parents = n;
}
static void reduce_grad_to_shape(SNEPPXTensor** grad_ptr, const SNEPPXTensor* target) {
    SNEPPXTensor* g = *grad_ptr;
    if (!g) return;
    if (g->ndim == target->ndim && g->size == target->size) return;
    size_t max_ndim = g->ndim > target->ndim ? g->ndim : target->ndim;
    size_t* g_shape = (size_t*)calloc(max_ndim, sizeof(size_t));
    size_t* t_shape = (size_t*)calloc(max_ndim, sizeof(size_t));
    if (!g_shape || !t_shape) { SNEPPX_free(g_shape, max_ndim * sizeof(size_t)); SNEPPX_free(t_shape, max_ndim * sizeof(size_t)); return; }
    for (size_t i = 0; i < g->ndim; i++) g_shape[max_ndim - 1 - i] = g->shape[g->ndim - 1 - i];
    for (size_t i = 0; i < target->ndim; i++) t_shape[max_ndim - 1 - i] = target->shape[target->ndim - 1 - i];
    int broadcast = 0;
    for (size_t i = 0; i < max_ndim; i++) { if (g_shape[i] != t_shape[i]) { broadcast = 1; break; } }
    if (!broadcast) { free(g_shape); free(t_shape); return; }
    size_t flat_size = 1;
    for (size_t i = 0; i < target->ndim; i++) flat_size *= target->shape[i];
    float* reduced = (float*)calloc(flat_size, sizeof(float));
    if (!reduced) { free(g_shape); free(t_shape); return; }
    size_t* strides_g = (size_t*)calloc(max_ndim, sizeof(size_t));
    size_t* strides_t = (size_t*)calloc(target->ndim, sizeof(size_t));
    if (!strides_g || !strides_t) { free(g_shape); free(t_shape); free(reduced); free(strides_g); free(strides_t); return; }
    strides_t[target->ndim - 1] = 1;
    for (size_t i = target->ndim; i > 1; i--) strides_t[i - 2] = strides_t[i - 1] * target->shape[i - 1];
    strides_g[max_ndim - 1] = 1;
    for (size_t i = max_ndim; i > 1; i--) strides_g[i - 2] = strides_g[i - 1] * g_shape[i - 1];
    for (size_t i = 0; i < flat_size; i++) {
        size_t g_idx = 0; size_t tmp = i;
        for (size_t d = 0; d < target->ndim; d++) {
            size_t t_dim_idx = tmp / strides_t[d];
            tmp %= strides_t[d];
            size_t g_dim = max_ndim - target->ndim + d;
            size_t g_dim_idx = g_shape[g_dim] == 1 ? 0 : t_dim_idx;
            g_idx += g_dim_idx * strides_g[g_dim];
        }
        reduced[i] = ((float*)g->data)[g_idx];
    }
    memcpy(g->data, reduced, flat_size * sizeof(float));
    g->size = flat_size;
    size_t new_ndim = target->ndim;
    size_t* new_shape = (size_t*)SNEPPX_malloc(new_ndim * sizeof(size_t), 64);
    if (new_shape) { memcpy(new_shape, target->shape, new_ndim * sizeof(size_t)); }
    SNEPPX_free(g->shape, g->ndim * sizeof(size_t));
    g->shape = new_shape;
    g->ndim = new_ndim;
    free(g_shape); free(t_shape); free(reduced); free(strides_g); free(strides_t);
}

typedef SNEPPXTensor* (*sneppx_tensor_bin_fn_t)(const SNEPPXTensor*, const SNEPPXTensor*);

static SNEPPXVariable* op_zero_grad(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b, sneppx_tensor_bin_fn_t tensor_fn) {
    if (!a || !a->data || !b || !b->data) return NULL;
    int rg = 0;
    SNEPPXTensor* result = tensor_fn(a->data, b->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (tape) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== Comparison ops backward (zero gradient - no gradient through comparisons) ===== */
/**
 * @brief Perform Eq.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_eq(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b) {
    return op_zero_grad(tape, a, b, SNEPPX_tensor_eq);
}
/**
 * @brief Perform Ne.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_ne(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b) {
    return op_zero_grad(tape, a, b, SNEPPX_tensor_ne);
}
/**
 * @brief Perform Lt.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_lt(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b) {
    return op_zero_grad(tape, a, b, SNEPPX_tensor_lt);
}
/**
 * @brief Perform Le.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_le(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b) {
    return op_zero_grad(tape, a, b, SNEPPX_tensor_le);
}
/**
 * @brief Perform Gt.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_gt(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b) {
    return op_zero_grad(tape, a, b, SNEPPX_tensor_gt);
}
/**
 * @brief Perform Ge.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_ge(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b) {
    return op_zero_grad(tape, a, b, SNEPPX_tensor_ge);
}

/* ===== conv1d backward ===== */
typedef struct { SNEPPXVariable *input, *kernel; size_t stride, padding; } Conv1DCtx;
static void free_ctx_Conv1DCtx(void* p) { SNEPPX_free(p, sizeof(Conv1DCtx)); }
static void* recompute_Conv1DCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    Conv1DCtx* ctx = (Conv1DCtx*)SNEPPX_malloc(sizeof(Conv1DCtx), 64);
    if (ctx) {
        ctx->input = var->parents[0]; ctx->kernel = var->parents[1];
        ctx->stride = n > 0 ? params[0] : 1;
        ctx->padding = n > 1 ? params[1] : 0;
        var->free_ctx = free_ctx_Conv1DCtx;
    }
    return ctx;
}
static void backward_conv1d(void* ctx, SNEPPXTensor* grad_output) {
    Conv1DCtx* c = (Conv1DCtx*)ctx;
    float* go = (float*)grad_output->data;
    size_t N = c->input->data->shape[0], C = c->input->data->shape[1], L = c->input->data->shape[2];
    size_t K = c->kernel->data->shape[0];
    size_t KH = c->kernel->data->shape[2];
    size_t OH = (L + 2 * c->padding - KH) / c->stride + 1;

    if (c->input->requires_grad) {
        SNEPPXTensor* gi = SNEPPX_tensor_zeros(c->input->data->shape, c->input->data->ndim, SNEPPX_FLOAT32);
        float* gid = (float*)gi->data;
        float* kd = (float*)c->kernel->data->data;
        for (size_t n = 0; n < N; n++)
            for (size_t k = 0; k < K; k++)
                for (size_t oh = 0; oh < OH; oh++) {
                    float grad_val = go[n * K * OH + k * OH + oh];
                    for (size_t kh = 0; kh < KH; kh++) {
                        size_t ih = oh * c->stride + kh - c->padding;
                        if (ih < L)
                            for (size_t cc = 0; cc < C; cc++)
                                gid[n * C * L + cc * L + ih] += kd[k * C * KH + cc * KH + kh] * grad_val;
                    }
                }
        grad_accum(&c->input->grad, gi);
    }
    if (c->kernel->requires_grad) {
        SNEPPXTensor* gk = SNEPPX_tensor_zeros(c->kernel->data->shape, c->kernel->data->ndim, SNEPPX_FLOAT32);
        float* gkd = (float*)gk->data;
        float* id = (float*)c->input->data->data;
        for (size_t n = 0; n < N; n++)
            for (size_t k = 0; k < K; k++)
                for (size_t oh = 0; oh < OH; oh++) {
                    float grad_val = go[n * K * OH + k * OH + oh];
                    for (size_t kh = 0; kh < KH; kh++) {
                        size_t ih = oh * c->stride + kh - c->padding;
                        if (ih < L)
                            for (size_t cc = 0; cc < C; cc++)
                                gkd[k * C * KH + cc * KH + kh] += id[n * C * L + cc * L + ih] * grad_val;
                    }
                }
        grad_accum(&c->kernel->grad, gk);
    }
}

/**
 * @brief Perform Conv1d.
 *
 * @param tape [out] Tape value.
 * @param input [out] Input value.
 * @param kernel [out] Kernel value.
 * @param stride [in] Stride value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_conv1d(SNEPPXTape* tape, SNEPPXVariable* input, SNEPPXVariable* kernel, size_t stride, size_t padding) {
    int rg = requires_grad(input, kernel);
    if (!input || !kernel || !input->data || !kernel->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_conv1d(input->data, kernel->data, stride, padding);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        Conv1DCtx* ctx = (Conv1DCtx*)SNEPPX_malloc(sizeof(Conv1DCtx), 64);
        if (ctx) {
            ctx->input = input; ctx->kernel = kernel; ctx->stride = stride; ctx->padding = padding;
            var->backward_fn = backward_conv1d; var->backward_ctx = ctx;
            var->free_ctx = free_ctx_Conv1DCtx; var->recompute_ctx = recompute_Conv1DCtx;
            var->params[0] = stride; var->params[1] = padding; var->param_count = 2;
        }
        SNEPPXVariable* pars[2]; pars[0] = input; pars[1] = kernel;
        set_parents(var, pars, 2);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== cumsum backward ===== */
typedef struct { SNEPPXVariable* a; size_t dim; } CumsumCtx;
static void free_ctx_CumsumCtx(void* p) { SNEPPX_free(p, sizeof(CumsumCtx)); }
static void* recompute_CumsumCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    CumsumCtx* ctx = (CumsumCtx*)SNEPPX_malloc(sizeof(CumsumCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; ctx->dim = n > 0 ? params[0] : 0; var->free_ctx = free_ctx_CumsumCtx; }
    return ctx;
}
static void backward_cumsum(void* ctx, SNEPPXTensor* grad_output) {
    CumsumCtx* c = (CumsumCtx*)ctx;
    if (!c->a->requires_grad) return;
    float* go = (float*)grad_output->data;
    size_t outer = 1, inner = 1;
    for (size_t i = 0; i < c->dim; i++) outer *= c->a->data->shape[i];
    size_t dim_size = c->a->data->shape[c->dim];
    for (size_t i = c->dim + 1; i < c->a->data->ndim; i++) inner *= c->a->data->shape[i];

    SNEPPXTensor* g = SNEPPX_tensor_zeros(c->a->data->shape, c->a->data->ndim, SNEPPX_FLOAT32);
    float* gd = (float*)g->data;

    for (size_t o = 0; o < outer; o++) {
        for (size_t i = 0; i < inner; i++) {
            float cum = 0.0f;
            for (size_t d = dim_size; d > 0; d--) {
                size_t idx = o * dim_size * inner + (d - 1) * inner + i;
                cum += go[o * dim_size * inner + (d - 1) * inner + i];
                gd[idx] = cum;
            }
        }
    }
    grad_accum(&c->a->grad, g);
}
/**
 * @brief Perform Cumsum.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_cumsum(SNEPPXTape* tape, SNEPPXVariable* a, size_t dim) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_cumsum(a->data, dim);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        CumsumCtx* ctx = (CumsumCtx*)SNEPPX_malloc(sizeof(CumsumCtx), 64);
        if (ctx) { ctx->a = a; ctx->dim = dim; var->backward_fn = backward_cumsum; var->backward_ctx = ctx; var->free_ctx = free_ctx_CumsumCtx; var->recompute_ctx = recompute_CumsumCtx; var->params[0] = dim; var->param_count = 1; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== cumprod backward ===== */
typedef struct { SNEPPXVariable* a; size_t dim; } CumprodCtx;
static void free_ctx_CumprodCtx(void* p) { SNEPPX_free(p, sizeof(CumprodCtx)); }
static void* recompute_CumprodCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    CumprodCtx* ctx = (CumprodCtx*)SNEPPX_malloc(sizeof(CumprodCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; ctx->dim = n > 0 ? params[0] : 0; var->free_ctx = free_ctx_CumprodCtx; }
    return ctx;
}
static void backward_cumprod(void* ctx, SNEPPXTensor* grad_output) {
    CumprodCtx* c = (CumprodCtx*)ctx;
    if (!c->a->requires_grad) return;
    size_t outer = 1, inner = 1;
    for (size_t i = 0; i < c->dim; i++) outer *= c->a->data->shape[i];
    size_t dim_size = c->a->data->shape[c->dim];
    for (size_t i = c->dim + 1; i < c->a->data->ndim; i++) inner *= c->a->data->shape[i];

    float* xd = (float*)c->a->data->data;
    float* go = (float*)grad_output->data;
    SNEPPXTensor* g = SNEPPX_tensor_zeros(c->a->data->shape, c->a->data->ndim, SNEPPX_FLOAT32);
    float* gd = (float*)g->data;

    for (size_t o = 0; o < outer; o++) {
        for (size_t i = 0; i < inner; i++) {
            float cumprod = 1.0f;
            for (size_t d = dim_size; d > 0; d--) {
                size_t idx = o * dim_size * inner + (d - 1) * inner + i;
                cumprod *= xd[idx];
                gd[idx] = go[o * dim_size * inner + (d - 1) * inner + i] * (cumprod / (xd[idx] + 1e-12f));
            }
        }
    }
    grad_accum(&c->a->grad, g);
}
/**
 * @brief Perform Cumprod.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_cumprod(SNEPPXTape* tape, SNEPPXVariable* a, size_t dim) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_cumprod(a->data, dim);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        CumprodCtx* ctx = (CumprodCtx*)SNEPPX_malloc(sizeof(CumprodCtx), 64);
        if (ctx) { ctx->a = a; ctx->dim = dim; var->backward_fn = backward_cumprod; var->backward_ctx = ctx; var->free_ctx = free_ctx_CumprodCtx; var->recompute_ctx = recompute_CumprodCtx; var->params[0] = dim; var->param_count = 1; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== add forward/backward ===== */
typedef struct { SNEPPXVariable* a; SNEPPXVariable* b; } AddCtx;
static void free_ctx_AddCtx(void* p) { SNEPPX_free(p, sizeof(AddCtx)); }
static void* recompute_AddCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    AddCtx* ctx = (AddCtx*)SNEPPX_malloc(sizeof(AddCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; ctx->b = var->parents[1]; var->free_ctx = free_ctx_AddCtx; }
    return ctx;
}
static void backward_add(void* ctx, SNEPPXTensor* grad_output) {
    AddCtx* c = (AddCtx*)ctx;
    if (c->a->requires_grad) {
        SNEPPXTensor* g = SNEPPX_tensor_copy(grad_output);
        reduce_grad_to_shape(&g, c->a->data);
        grad_accum(&c->a->grad, g);
    }
    if (c->b->requires_grad) {
        SNEPPXTensor* g = SNEPPX_tensor_copy(grad_output);
        reduce_grad_to_shape(&g, c->b->data);
        grad_accum(&c->b->grad, g);
    }
}
/**
 * @brief Add Add.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_add(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b) {
    int rg = requires_grad(a, b);
    if (!a || !b || !a->data || !b->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_add(a->data, b->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        AddCtx* ctx = (AddCtx*)SNEPPX_malloc(sizeof(AddCtx), 64);
        if (ctx) { ctx->a = a; ctx->b = b; var->backward_fn = backward_add; var->backward_ctx = ctx; var->free_ctx = free_ctx_AddCtx; var->recompute_ctx = recompute_AddCtx; }
        SNEPPXVariable* pars[2]; pars[0] = a; pars[1] = b;
        set_parents(var, pars, 2);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== sub forward/backward ===== */
typedef struct { SNEPPXVariable* a; SNEPPXVariable* b; } SubCtx;
static void free_ctx_SubCtx(void* p) { SNEPPX_free(p, sizeof(SubCtx)); }
static void* recompute_SubCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    SubCtx* ctx = (SubCtx*)SNEPPX_malloc(sizeof(SubCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; ctx->b = var->parents[1]; var->free_ctx = free_ctx_SubCtx; }
    return ctx;
}
static void backward_sub(void* ctx, SNEPPXTensor* grad_output) {
    SubCtx* c = (SubCtx*)ctx;
    if (c->a->requires_grad) {
        SNEPPXTensor* g = SNEPPX_tensor_copy(grad_output);
        reduce_grad_to_shape(&g, c->a->data);
        grad_accum(&c->a->grad, g);
    }
    if (c->b->requires_grad) {
        SNEPPXTensor* g = SNEPPX_tensor_neg(grad_output);
        reduce_grad_to_shape(&g, c->b->data);
        grad_accum(&c->b->grad, g);
    }
}
/**
 * @brief Perform Sub.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_sub(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b) {
    int rg = requires_grad(a, b);
    if (!a || !b || !a->data || !b->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_sub(a->data, b->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        SubCtx* ctx = (SubCtx*)SNEPPX_malloc(sizeof(SubCtx), 64);
        if (ctx) { ctx->a = a; ctx->b = b; var->backward_fn = backward_sub; var->backward_ctx = ctx; var->free_ctx = free_ctx_SubCtx; var->recompute_ctx = recompute_SubCtx; }
        SNEPPXVariable* pars[2]; pars[0] = a; pars[1] = b;
        set_parents(var, pars, 2);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== mul forward/backward ===== */
typedef struct { SNEPPXVariable* a; SNEPPXVariable* b; } MulCtx;
static void free_ctx_MulCtx(void* p) { SNEPPX_free(p, sizeof(MulCtx)); }
static void* recompute_MulCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    MulCtx* ctx = (MulCtx*)SNEPPX_malloc(sizeof(MulCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; ctx->b = var->parents[1]; var->free_ctx = free_ctx_MulCtx; }
    return ctx;
}
static void backward_mul(void* ctx, SNEPPXTensor* grad_output) {
    MulCtx* c = (MulCtx*)ctx;
    if (c->a->requires_grad) {
        SNEPPXTensor* ga = SNEPPX_tensor_mul(c->b->data, grad_output);
        reduce_grad_to_shape(&ga, c->a->data);
        grad_accum(&c->a->grad, ga);
    }
    if (c->b->requires_grad) {
        SNEPPXTensor* gb = SNEPPX_tensor_mul(c->a->data, grad_output);
        reduce_grad_to_shape(&gb, c->b->data);
        grad_accum(&c->b->grad, gb);
    }
}
/**
 * @brief Perform Mul.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_mul(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b) {
    int rg = requires_grad(a, b);
    if (!a || !b || !a->data || !b->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_mul(a->data, b->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        MulCtx* ctx = (MulCtx*)SNEPPX_malloc(sizeof(MulCtx), 64);
        if (ctx) { ctx->a = a; ctx->b = b; var->backward_fn = backward_mul; var->backward_ctx = ctx; var->free_ctx = free_ctx_MulCtx; var->recompute_ctx = recompute_MulCtx; }
        SNEPPXVariable* pars[2]; pars[0] = a; pars[1] = b;
        set_parents(var, pars, 2);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== div forward/backward ===== */
typedef struct { SNEPPXVariable* a; SNEPPXVariable* b; } DivCtx;
static void free_ctx_DivCtx(void* p) { SNEPPX_free(p, sizeof(DivCtx)); }
static void* recompute_DivCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    DivCtx* ctx = (DivCtx*)SNEPPX_malloc(sizeof(DivCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; ctx->b = var->parents[1]; var->free_ctx = free_ctx_DivCtx; }
    return ctx;
}
static void backward_div(void* ctx, SNEPPXTensor* grad_output) {
    DivCtx* c = (DivCtx*)ctx;
    if (c->a->requires_grad) {
        SNEPPXTensor* ga = SNEPPX_tensor_div(grad_output, c->b->data);
        reduce_grad_to_shape(&ga, c->a->data);
        grad_accum(&c->a->grad, ga);
    }
    if (c->b->requires_grad) {
        SNEPPXTensor* b2 = SNEPPX_tensor_mul(c->b->data, c->b->data);
        SNEPPXTensor* a_neg = SNEPPX_tensor_neg(c->a->data);
        SNEPPXTensor* num = SNEPPX_tensor_mul(a_neg, grad_output);
        SNEPPXTensor* gb = SNEPPX_tensor_div(num, b2);
        SNEPPX_tensor_destroy(b2);
        SNEPPX_tensor_destroy(a_neg);
        SNEPPX_tensor_destroy(num);
        reduce_grad_to_shape(&gb, c->b->data);
        grad_accum(&c->b->grad, gb);
    }
}
/**
 * @brief Perform Div.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_div(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b) {
    int rg = requires_grad(a, b);
    if (!a || !b || !a->data || !b->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_div(a->data, b->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        DivCtx* ctx = (DivCtx*)SNEPPX_malloc(sizeof(DivCtx), 64);
        if (ctx) { ctx->a = a; ctx->b = b; var->backward_fn = backward_div; var->backward_ctx = ctx; var->free_ctx = free_ctx_DivCtx; var->recompute_ctx = recompute_DivCtx; }
        SNEPPXVariable* pars[2]; pars[0] = a; pars[1] = b;
        set_parents(var, pars, 2);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== pow forward/backward ===== */
typedef struct { SNEPPXVariable* a; SNEPPXVariable* b; SNEPPXTensor* result; } PowCtx;
static void free_ctx_PowCtx(void* p) { PowCtx* cp = (PowCtx*)p; if (cp->result) SNEPPX_tensor_destroy(cp->result); SNEPPX_free(p, sizeof(PowCtx)); }
static void* recompute_PowCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    PowCtx* ctx = (PowCtx*)SNEPPX_malloc(sizeof(PowCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; ctx->b = var->parents[1]; ctx->result = NULL; var->free_ctx = free_ctx_PowCtx; }
    return ctx;
}
static void backward_pow(void* ctx, SNEPPXTensor* grad_output) {
    PowCtx* c = (PowCtx*)ctx;
    if (c->a->requires_grad) {
        SNEPPXTensor* b_minus_1 = NULL;
        if (c->b->data->size == 1) {
            b_minus_1 = SNEPPX_tensor_copy(c->b->data);
            float bv = ((float*)b_minus_1->data)[0];
            ((float*)b_minus_1->data)[0] = bv - 1.0f;
        } else {
            SNEPPXTensor* ones = SNEPPX_tensor_ones(c->b->data->shape, c->b->data->ndim, SNEPPX_FLOAT32);
            b_minus_1 = SNEPPX_tensor_sub(c->b->data, ones);
            SNEPPX_tensor_destroy(ones);
        }
        SNEPPXTensor* a_pow_bm1 = SNEPPX_tensor_pow(c->a->data, b_minus_1);
        SNEPPXTensor* da = SNEPPX_tensor_mul(c->b->data, a_pow_bm1);
        SNEPPXTensor* ga = SNEPPX_tensor_mul(grad_output, da);
        SNEPPX_tensor_destroy(b_minus_1);
        SNEPPX_tensor_destroy(a_pow_bm1);
        SNEPPX_tensor_destroy(da);
        reduce_grad_to_shape(&ga, c->a->data);
        grad_accum(&c->a->grad, ga);
    }
    if (c->b->requires_grad && c->result) {
        SNEPPXTensor* log_a = SNEPPX_tensor_log(c->a->data);
        SNEPPXTensor* db = SNEPPX_tensor_mul(c->result, log_a);
        SNEPPXTensor* gb = SNEPPX_tensor_mul(grad_output, db);
        SNEPPX_tensor_destroy(log_a);
        SNEPPX_tensor_destroy(db);
        reduce_grad_to_shape(&gb, c->b->data);
        grad_accum(&c->b->grad, gb);
    }
}
/**
 * @brief Perform Pow.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_pow(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b) {
    int rg = requires_grad(a, b);
    if (!a || !b || !a->data || !b->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_pow(a->data, b->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        PowCtx* ctx = (PowCtx*)SNEPPX_malloc(sizeof(PowCtx), 64);
        if (ctx) { ctx->a = a; ctx->b = b; ctx->result = SNEPPX_tensor_copy(result); var->backward_fn = backward_pow; var->backward_ctx = ctx; var->free_ctx = free_ctx_PowCtx; var->recompute_ctx = recompute_PowCtx; }
        SNEPPXVariable* pars[2]; pars[0] = a; pars[1] = b;
        set_parents(var, pars, 2);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== neg forward/backward ===== */
typedef struct { SNEPPXVariable* a; } NegCtx;
static void free_ctx_NegCtx(void* p) { SNEPPX_free(p, sizeof(NegCtx)); }
static void* recompute_NegCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    NegCtx* ctx = (NegCtx*)SNEPPX_malloc(sizeof(NegCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; var->free_ctx = free_ctx_NegCtx; }
    return ctx;
}
static void backward_neg(void* ctx, SNEPPXTensor* grad_output) {
    NegCtx* c = (NegCtx*)ctx;
    if (c->a->requires_grad) grad_accum(&c->a->grad, SNEPPX_tensor_neg(grad_output));
}
/**
 * @brief Perform Neg.
 *
 * @param tape [out] Tape value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_neg(SNEPPXTape* tape, SNEPPXVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_neg(a->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        NegCtx* ctx = (NegCtx*)SNEPPX_malloc(sizeof(NegCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_neg; var->backward_ctx = ctx; var->free_ctx = free_ctx_NegCtx; var->recompute_ctx = recompute_NegCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== matmul forward/backward ===== */
typedef struct { SNEPPXVariable* a; SNEPPXVariable* b; } BinopCtx;
static void free_ctx_BinopCtx(void* p) { SNEPPX_free(p, sizeof(BinopCtx)); }
static void* recompute_BinopCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    BinopCtx* ctx = (BinopCtx*)SNEPPX_malloc(sizeof(BinopCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; ctx->b = var->parents[1]; var->free_ctx = free_ctx_BinopCtx; }
    return ctx;
}
static void backward_matmul(void* ctx, SNEPPXTensor* grad_output) {
    BinopCtx* c = (BinopCtx*)ctx;
    size_t m = grad_output->shape[0], n = grad_output->shape[1];
    size_t k = c->b->data->shape[0];
    float* go = (float*)grad_output->data;
    if (c->a->requires_grad) {
        float* bd = (float*)c->b->data->data;
        SNEPPXTensor* ga = SNEPPX_tensor_zeros(c->a->data->shape, c->a->data->ndim, SNEPPX_FLOAT32);
        float* gad = (float*)ga->data;
        for (size_t i = 0; i < m; i++)
            for (size_t j = 0; j < k; j++)
                for (size_t l = 0; l < n; l++)
                    gad[i * k + j] += go[i * n + l] * bd[j * n + l];
        grad_accum(&c->a->grad, ga);
    }
    if (c->b->requires_grad) {
        float* ad = (float*)c->a->data->data;
        SNEPPXTensor* gb = SNEPPX_tensor_zeros(c->b->data->shape, c->b->data->ndim, SNEPPX_FLOAT32);
        float* gbd = (float*)gb->data;
        for (size_t i = 0; i < m; i++)
            for (size_t j = 0; j < k; j++)
                for (size_t l = 0; l < n; l++)
                    gbd[j * n + l] += ad[i * k + j] * go[i * n + l];
        grad_accum(&c->b->grad, gb);
    }
}
/**
 * @brief Perform Matmul.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_matmul(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b) {
    int rg = requires_grad(a, b);
    if (!a || !b || !a->data || !b->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_matmul(a->data, b->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        BinopCtx* ctx = (BinopCtx*)SNEPPX_malloc(sizeof(BinopCtx), 64);
        if (ctx) { ctx->a = a; ctx->b = b; var->backward_fn = backward_matmul; var->backward_ctx = ctx; var->free_ctx = free_ctx_BinopCtx; var->recompute_ctx = recompute_BinopCtx; }
        SNEPPXVariable* pars[2]; pars[0] = a; pars[1] = b;
        set_parents(var, pars, 2);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== mse_loss forward/backward ===== */
typedef struct { SNEPPXVariable* pred; SNEPPXVariable* target; } MseLossCtx;
static void free_ctx_MseLossCtx(void* p) { SNEPPX_free(p, sizeof(MseLossCtx)); }
static void* recompute_MseLossCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    MseLossCtx* ctx = (MseLossCtx*)SNEPPX_malloc(sizeof(MseLossCtx), 64);
    if (ctx) { ctx->pred = var->parents[0]; ctx->target = var->parents[1]; var->free_ctx = free_ctx_MseLossCtx; }
    return ctx;
}
static void backward_mse_loss(void* ctx, SNEPPXTensor* grad_output) {
    MseLossCtx* c = (MseLossCtx*)ctx;
    if (!c->pred->requires_grad) return;
    SNEPPXTensor* diff = SNEPPX_tensor_sub(c->pred->data, c->target->data);
    float* dd = (float*)diff->data;
    float* go = (float*)grad_output->data;
    float n = (float)(diff->size);
    float scale = 2.0f / n;
    for (size_t i = 0; i < diff->size; i++) dd[i] *= scale * go[i % grad_output->size];
    grad_accum(&c->pred->grad, diff);
}
/**
 * @brief Perform Mse Loss.
 *
 * @param tape [out] Tape value.
 * @param pred [out] Pred value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_mse_loss(SNEPPXTape* tape, SNEPPXVariable* pred, SNEPPXVariable* target) {
    int rg = requires_grad1(pred);
    if (!pred || !target || !pred->data || !target->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_mse_loss(pred->data, target->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        MseLossCtx* ctx = (MseLossCtx*)SNEPPX_malloc(sizeof(MseLossCtx), 64);
        if (ctx) { ctx->pred = pred; ctx->target = target; var->backward_fn = backward_mse_loss; var->backward_ctx = ctx; var->free_ctx = free_ctx_MseLossCtx; var->recompute_ctx = recompute_MseLossCtx; }
        SNEPPXVariable* pars[2]; pars[0] = pred; pars[1] = target;
        set_parents(var, pars, 2);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== relu forward/backward ===== */
typedef struct { SNEPPXVariable* a; } ReluCtx;
static void free_ctx_ReluCtx(void* p) { SNEPPX_free(p, sizeof(ReluCtx)); }
static void* recompute_ReluCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    ReluCtx* ctx = (ReluCtx*)SNEPPX_malloc(sizeof(ReluCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; var->free_ctx = free_ctx_ReluCtx; }
    return ctx;
}
static void backward_relu(void* ctx, SNEPPXTensor* grad_output) {
    ReluCtx* c = (ReluCtx*)ctx;
    if (!c->a->requires_grad) return;
    SNEPPXTensor* ga = SNEPPX_tensor_copy(grad_output);
    float* gad = (float*)ga->data;
    float* ad = (float*)c->a->data->data;
    for (size_t i = 0; i < ga->size; i++)
        if (ad[i] <= 0.0f) gad[i] = 0.0f;
    grad_accum(&c->a->grad, ga);
}
/**
 * @brief Perform Relu.
 *
 * @param tape [out] Tape value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_relu(SNEPPXTape* tape, SNEPPXVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_relu(a->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        ReluCtx* ctx = (ReluCtx*)SNEPPX_malloc(sizeof(ReluCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_relu; var->backward_ctx = ctx; var->free_ctx = free_ctx_ReluCtx; var->recompute_ctx = recompute_ReluCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== fake_quant forward/backward (QAT, straight-through estimator) ===== */
typedef struct { SNEPPXVariable* a; float scale; int bits; } FakeQuantCtx;
static void free_ctx_FakeQuantCtx(void* p) { SNEPPX_free(p, sizeof(FakeQuantCtx)); }
static void* recompute_FakeQuantCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    FakeQuantCtx* ctx = (FakeQuantCtx*)SNEPPX_malloc(sizeof(FakeQuantCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; ctx->scale = 0.0f; ctx->bits = 0; var->free_ctx = free_ctx_FakeQuantCtx; }
    return ctx;
}
static void backward_fake_quant(void* ctx, SNEPPXTensor* grad_output) {
    FakeQuantCtx* c = (FakeQuantCtx*)ctx;
    if (!c->a->requires_grad) return;
    SNEPPXTensor* ga = SNEPPX_tensor_copy(grad_output);
    grad_accum(&c->a->grad, ga);
}
/**
 * @brief Perform Fake Quant (symmetric affine, STE backward).
 *
 * Forward quantizes the input tensor to `bits` with `scale` and de-quantizes;
 * backward is the straight-through estimator (unit gradient w.r.t. the input).
 *
 * @param tape [out] Tape value.
 * @param a [in] Input value.
 * @param scale [in] Scale value (> 0).
 * @param bits [in] Bit width (2..16).
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_fake_quant(SNEPPXTape* tape, SNEPPXVariable* a, float scale, int bits) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_fake_quant(a->data, scale, bits);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        FakeQuantCtx* ctx = (FakeQuantCtx*)SNEPPX_malloc(sizeof(FakeQuantCtx), 64);
        if (ctx) { ctx->a = a; ctx->scale = scale; ctx->bits = bits; var->backward_fn = backward_fake_quant; var->backward_ctx = ctx; var->free_ctx = free_ctx_FakeQuantCtx; var->recompute_ctx = recompute_FakeQuantCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== gelu forward/backward ===== */
typedef struct { SNEPPXVariable* a; } GeluCtx;
static void free_ctx_GeluCtx(void* p) { SNEPPX_free(p, sizeof(GeluCtx)); }
static void* recompute_GeluCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    GeluCtx* ctx = (GeluCtx*)SNEPPX_malloc(sizeof(GeluCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; var->free_ctx = free_ctx_GeluCtx; }
    return ctx;
}
static void backward_gelu(void* ctx, SNEPPXTensor* grad_output) {
    GeluCtx* c = (GeluCtx*)ctx;
    if (!c->a->requires_grad) return;
    float* ad = (float*)c->a->data->data;
    SNEPPXTensor* ga = SNEPPX_tensor_copy(grad_output);
    float* gad = (float*)ga->data;
    float sqrt_2_over_pi = 0.7978845608028654f;
    for (size_t i = 0; i < ga->size; i++) {
        float x = ad[i];
        float x3 = x * x * x;
        float tanh_in = sqrt_2_over_pi * (x + 0.044715f * x3);
        float th = tanhf(tanh_in);
        float sech2 = 1.0f - th * th;
        float d_gelu = 0.5f * (1.0f + th) + 0.5f * x * sech2 * sqrt_2_over_pi * (1.0f + 3.0f * 0.044715f * x * x);
        gad[i] *= d_gelu;
    }
    grad_accum(&c->a->grad, ga);
}
/**
 * @brief Perform Gelu.
 *
 * @param tape [out] Tape value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_gelu(SNEPPXTape* tape, SNEPPXVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_gelu(a->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        GeluCtx* ctx = (GeluCtx*)SNEPPX_malloc(sizeof(GeluCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_gelu; var->backward_ctx = ctx; var->free_ctx = free_ctx_GeluCtx; var->recompute_ctx = recompute_GeluCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== silu forward/backward ===== */
typedef struct { SNEPPXVariable* a; } SiluCtx;
static void free_ctx_SiluCtx(void* p) { SNEPPX_free(p, sizeof(SiluCtx)); }
static void* recompute_SiluCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    SiluCtx* ctx = (SiluCtx*)SNEPPX_malloc(sizeof(SiluCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; var->free_ctx = free_ctx_SiluCtx; }
    return ctx;
}
static void backward_silu(void* ctx, SNEPPXTensor* grad_output) {
    SiluCtx* c = (SiluCtx*)ctx;
    if (!c->a->requires_grad) return;
    float* ad = (float*)c->a->data->data;
    SNEPPXTensor* ga = SNEPPX_tensor_copy(grad_output);
    float* gad = (float*)ga->data;
    for (size_t i = 0; i < ga->size; i++) {
        float x = ad[i];
        float s = 1.0f / (1.0f + expf(-x));
        float ds = s * (1.0f + x * (1.0f - s));
        gad[i] *= ds;
    }
    grad_accum(&c->a->grad, ga);
}
/**
 * @brief Perform Silu.
 *
 * @param tape [out] Tape value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_silu(SNEPPXTape* tape, SNEPPXVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_silu(a->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        SiluCtx* ctx = (SiluCtx*)SNEPPX_malloc(sizeof(SiluCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_silu; var->backward_ctx = ctx; var->free_ctx = free_ctx_SiluCtx; var->recompute_ctx = recompute_SiluCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== sigmoid forward/backward ===== */
typedef struct { SNEPPXVariable* a; } SigmoidCtx;
static void free_ctx_SigmoidCtx(void* p) { SNEPPX_free(p, sizeof(SigmoidCtx)); }
static void* recompute_SigmoidCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    SigmoidCtx* ctx = (SigmoidCtx*)SNEPPX_malloc(sizeof(SigmoidCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; var->free_ctx = free_ctx_SigmoidCtx; }
    return ctx;
}
static void backward_sigmoid(void* ctx, SNEPPXTensor* grad_output) {
    SigmoidCtx* c = (SigmoidCtx*)ctx;
    if (!c->a->requires_grad) return;
    float* ad = (float*)c->a->data->data;
    SNEPPXTensor* ga = SNEPPX_tensor_copy(grad_output);
    float* gad = (float*)ga->data;
    for (size_t i = 0; i < ga->size; i++) {
        float s = 1.0f / (1.0f + expf(-ad[i]));
        gad[i] *= s * (1.0f - s);
    }
    grad_accum(&c->a->grad, ga);
}
/**
 * @brief Perform Sigmoid.
 *
 * @param tape [out] Tape value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_sigmoid(SNEPPXTape* tape, SNEPPXVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_sigmoid(a->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        SigmoidCtx* ctx = (SigmoidCtx*)SNEPPX_malloc(sizeof(SigmoidCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_sigmoid; var->backward_ctx = ctx; var->free_ctx = free_ctx_SigmoidCtx; var->recompute_ctx = recompute_SigmoidCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== tanh forward/backward ===== */
typedef struct { SNEPPXVariable* a; } TanhCtx;
static void free_ctx_TanhCtx(void* p) { SNEPPX_free(p, sizeof(TanhCtx)); }
static void* recompute_TanhCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    TanhCtx* ctx = (TanhCtx*)SNEPPX_malloc(sizeof(TanhCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; var->free_ctx = free_ctx_TanhCtx; }
    return ctx;
}
static void backward_tanh(void* ctx, SNEPPXTensor* grad_output) {
    TanhCtx* c = (TanhCtx*)ctx;
    if (!c->a->requires_grad) return;
    float* ad = (float*)c->a->data->data;
    SNEPPXTensor* ga = SNEPPX_tensor_copy(grad_output);
    float* gad = (float*)ga->data;
    for (size_t i = 0; i < ga->size; i++) {
        float t = tanhf(ad[i]);
        gad[i] *= 1.0f - t * t;
    }
    grad_accum(&c->a->grad, ga);
}
/**
 * @brief Perform Tanh.
 *
 * @param tape [out] Tape value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_tanh(SNEPPXTape* tape, SNEPPXVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_tanh(a->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        TanhCtx* ctx = (TanhCtx*)SNEPPX_malloc(sizeof(TanhCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_tanh; var->backward_ctx = ctx; var->free_ctx = free_ctx_TanhCtx; var->recompute_ctx = recompute_TanhCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== softmax forward/backward ===== */
typedef struct { SNEPPXVariable* a; size_t dim; } SoftmaxCtx;
static void free_ctx_SoftmaxCtx(void* p) { SNEPPX_free(p, sizeof(SoftmaxCtx)); }
static void* recompute_SoftmaxCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    SoftmaxCtx* ctx = (SoftmaxCtx*)SNEPPX_malloc(sizeof(SoftmaxCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; ctx->dim = n > 0 ? params[0] : 0; var->free_ctx = free_ctx_SoftmaxCtx; }
    return ctx;
}
static void backward_softmax(void* ctx, SNEPPXTensor* grad_output) {
    SoftmaxCtx* c = (SoftmaxCtx*)ctx;
    if (!c->a->requires_grad) return;
    SNEPPXTensor* s = SNEPPX_tensor_softmax(c->a->data, c->dim);
    SNEPPXTensor* s_grad = SNEPPX_tensor_mul(s, grad_output);
    SNEPPXTensor* sum_s_grad = SNEPPX_tensor_sum(s_grad, c->dim);

    size_t* expand_shape = (size_t*)SNEPPX_malloc(c->a->data->ndim * sizeof(size_t), 64);
    if (!expand_shape) { SNEPPX_tensor_destroy(s); SNEPPX_tensor_destroy(s_grad); SNEPPX_tensor_destroy(sum_s_grad); return; }
    for (size_t i = 0; i < c->a->data->ndim; i++)
        expand_shape[i] = i == c->dim ? c->a->data->shape[c->dim] : 1;
    SNEPPXTensor* expanded = SNEPPX_tensor_expand(sum_s_grad, expand_shape, c->a->data->ndim);
    SNEPPX_free(expand_shape, c->a->data->ndim * sizeof(size_t));
    if (!expanded) { SNEPPX_tensor_destroy(s); SNEPPX_tensor_destroy(s_grad); SNEPPX_tensor_destroy(sum_s_grad); return; }

    SNEPPXTensor* sub = SNEPPX_tensor_sub(grad_output, expanded);
    SNEPPXTensor* ga = SNEPPX_tensor_mul(s, sub);
    grad_accum(&c->a->grad, ga);
    SNEPPX_tensor_destroy(s);
    SNEPPX_tensor_destroy(s_grad);
    SNEPPX_tensor_destroy(sum_s_grad);
    SNEPPX_tensor_destroy(expanded);
    SNEPPX_tensor_destroy(sub);
}
/**
 * @brief Perform Softmax.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_softmax(SNEPPXTape* tape, SNEPPXVariable* a, size_t dim) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_softmax(a->data, dim);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        SoftmaxCtx* ctx = (SoftmaxCtx*)SNEPPX_malloc(sizeof(SoftmaxCtx), 64);
        if (ctx) { ctx->a = a; ctx->dim = dim; var->backward_fn = backward_softmax; var->backward_ctx = ctx; var->free_ctx = free_ctx_SoftmaxCtx; var->recompute_ctx = recompute_SoftmaxCtx; var->params[0] = dim; var->param_count = 1; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== exp forward/backward ===== */
typedef struct { SNEPPXVariable* a; } ExpCtx;
static void free_ctx_ExpCtx(void* p) { SNEPPX_free(p, sizeof(ExpCtx)); }
static void* recompute_ExpCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    ExpCtx* ctx = (ExpCtx*)SNEPPX_malloc(sizeof(ExpCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; var->free_ctx = free_ctx_ExpCtx; }
    return ctx;
}
static void backward_exp(void* ctx, SNEPPXTensor* grad_output) {
    ExpCtx* c = (ExpCtx*)ctx;
    if (!c->a->requires_grad) return;
    SNEPPXTensor* e = SNEPPX_tensor_exp(c->a->data);
    SNEPPXTensor* ga = SNEPPX_tensor_mul(e, grad_output);
    grad_accum(&c->a->grad, ga);
    SNEPPX_tensor_destroy(e);
}
/**
 * @brief Perform Exp.
 *
 * @param tape [out] Tape value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_exp(SNEPPXTape* tape, SNEPPXVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_exp(a->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        ExpCtx* ctx = (ExpCtx*)SNEPPX_malloc(sizeof(ExpCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_exp; var->backward_ctx = ctx; var->free_ctx = free_ctx_ExpCtx; var->recompute_ctx = recompute_ExpCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== log forward/backward ===== */
typedef struct { SNEPPXVariable* a; } LogCtx;
static void free_ctx_LogCtx(void* p) { SNEPPX_free(p, sizeof(LogCtx)); }
static void* recompute_LogCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    LogCtx* ctx = (LogCtx*)SNEPPX_malloc(sizeof(LogCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; var->free_ctx = free_ctx_LogCtx; }
    return ctx;
}
static void backward_log(void* ctx, SNEPPXTensor* grad_output) {
    LogCtx* c = (LogCtx*)ctx;
    if (!c->a->requires_grad) return;
    SNEPPXTensor* ga = SNEPPX_tensor_div(grad_output, c->a->data);
    grad_accum(&c->a->grad, ga);
}
/**
 * @brief Perform Log.
 *
 * @param tape [out] Tape value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_log(SNEPPXTape* tape, SNEPPXVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_log(a->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        LogCtx* ctx = (LogCtx*)SNEPPX_malloc(sizeof(LogCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_log; var->backward_ctx = ctx; var->free_ctx = free_ctx_LogCtx; var->recompute_ctx = recompute_LogCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== sum forward/backward ===== */
typedef struct { SNEPPXVariable* a; size_t dim; } SumCtx;
static void free_ctx_SumCtx(void* p) { SNEPPX_free(p, sizeof(SumCtx)); }
static void* recompute_SumCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    SumCtx* ctx = (SumCtx*)SNEPPX_malloc(sizeof(SumCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; ctx->dim = n > 0 ? params[0] : 0; var->free_ctx = free_ctx_SumCtx; }
    return ctx;
}
static void backward_sum(void* ctx, SNEPPXTensor* grad_output) {
    SumCtx* c = (SumCtx*)ctx;
    if (!c->a->requires_grad) return;
    SNEPPXTensor* g = SNEPPX_tensor_zeros(c->a->data->shape, c->a->data->ndim, SNEPPX_FLOAT32);
    float* gd = (float*)g->data;
    float* go = (float*)grad_output->data;
    size_t outer = 1, inner = 1;
    for (size_t i = 0; i < c->dim; i++) outer *= c->a->data->shape[i];
    size_t dim_size = c->a->data->shape[c->dim];
    for (size_t i = c->dim + 1; i < c->a->data->ndim; i++) inner *= c->a->data->shape[i];
    for (size_t o = 0; o < outer; o++)
        for (size_t d = 0; d < dim_size; d++)
            for (size_t i = 0; i < inner; i++)
                gd[o * dim_size * inner + d * inner + i] = go[o * inner + i];
    grad_accum(&c->a->grad, g);
}
/**
 * @brief Perform Sum.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_sum(SNEPPXTape* tape, SNEPPXVariable* a, size_t dim) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_sum(a->data, dim);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        SumCtx* ctx = (SumCtx*)SNEPPX_malloc(sizeof(SumCtx), 64);
        if (ctx) { ctx->a = a; ctx->dim = dim; var->backward_fn = backward_sum; var->backward_ctx = ctx; var->free_ctx = free_ctx_SumCtx; var->recompute_ctx = recompute_SumCtx; var->params[0] = dim; var->param_count = 1; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== mean forward/backward ===== */
typedef struct { SNEPPXVariable* a; size_t dim; } MeanCtx;
static void free_ctx_MeanCtx(void* p) { SNEPPX_free(p, sizeof(MeanCtx)); }
static void* recompute_MeanCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    MeanCtx* ctx = (MeanCtx*)SNEPPX_malloc(sizeof(MeanCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; ctx->dim = n > 0 ? params[0] : 0; var->free_ctx = free_ctx_MeanCtx; }
    return ctx;
}
static void backward_mean(void* ctx, SNEPPXTensor* grad_output) {
    MeanCtx* c = (MeanCtx*)ctx;
    if (!c->a->requires_grad) return;
    size_t dim_size = c->a->data->shape[c->dim];
    float inv_dim = 1.0f / (float)dim_size;
    SNEPPXTensor* go = SNEPPX_tensor_copy(grad_output);
    float* god = (float*)go->data;
    for (size_t i = 0; i < go->size; i++) god[i] *= inv_dim;
    size_t* expand_shape = (size_t*)SNEPPX_malloc(c->a->data->ndim * sizeof(size_t), 64);
    if (!expand_shape) { SNEPPX_tensor_destroy(go); return; }
    for (size_t i = 0; i < c->a->data->ndim; i++)
        expand_shape[i] = i == c->dim ? c->a->data->shape[c->dim] : 1;
    SNEPPXTensor* expanded = SNEPPX_tensor_expand(go, expand_shape, c->a->data->ndim);
    SNEPPX_free(expand_shape, c->a->data->ndim * sizeof(size_t));
    if (expanded) {
        grad_accum(&c->a->grad, expanded);
    }
    SNEPPX_tensor_destroy(go);
}
/**
 * @brief Perform Mean.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_mean(SNEPPXTape* tape, SNEPPXVariable* a, size_t dim) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_mean(a->data, dim);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        MeanCtx* ctx = (MeanCtx*)SNEPPX_malloc(sizeof(MeanCtx), 64);
        if (ctx) { ctx->a = a; ctx->dim = dim; var->backward_fn = backward_mean; var->backward_ctx = ctx; var->free_ctx = free_ctx_MeanCtx; var->recompute_ctx = recompute_MeanCtx; var->params[0] = dim; var->param_count = 1; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== transpose forward/backward ===== */
typedef struct { SNEPPXVariable* a; size_t dim1; size_t dim2; } TransposeCtx;
static void free_ctx_TransposeCtx(void* p) { SNEPPX_free(p, sizeof(TransposeCtx)); }
static void* recompute_TransposeCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    TransposeCtx* ctx = (TransposeCtx*)SNEPPX_malloc(sizeof(TransposeCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; ctx->dim1 = n > 0 ? params[0] : 0; ctx->dim2 = n > 1 ? params[1] : 1; var->free_ctx = free_ctx_TransposeCtx; }
    return ctx;
}
static void backward_transpose(void* ctx, SNEPPXTensor* grad_output) {
    TransposeCtx* c = (TransposeCtx*)ctx;
    if (!c->a->requires_grad) return;
    SNEPPXTensor* ga = SNEPPX_tensor_transpose(grad_output, c->dim1, c->dim2);
    grad_accum(&c->a->grad, ga);
}
/**
 * @brief Perform Transpose.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 * @param dim1 [in] Dim1 value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_transpose(SNEPPXTape* tape, SNEPPXVariable* a, size_t dim1, size_t dim2) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_transpose(a->data, dim1, dim2);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        TransposeCtx* ctx = (TransposeCtx*)SNEPPX_malloc(sizeof(TransposeCtx), 64);
        if (ctx) { ctx->a = a; ctx->dim1 = dim1; ctx->dim2 = dim2; var->backward_fn = backward_transpose; var->backward_ctx = ctx; var->free_ctx = free_ctx_TransposeCtx; var->recompute_ctx = recompute_TransposeCtx; var->params[0] = dim1; var->params[1] = dim2; var->param_count = 2; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== reshape forward/backward ===== */
typedef struct { SNEPPXVariable* a; } ReshapeCtx;
static void free_ctx_ReshapeCtx(void* p) { SNEPPX_free(p, sizeof(ReshapeCtx)); }
static void* recompute_ReshapeCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    ReshapeCtx* ctx = (ReshapeCtx*)SNEPPX_malloc(sizeof(ReshapeCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; var->free_ctx = free_ctx_ReshapeCtx; }
    return ctx;
}
static void backward_reshape(void* ctx, SNEPPXTensor* grad_output) {
    ReshapeCtx* c = (ReshapeCtx*)ctx;
    if (!c->a->requires_grad) return;
    SNEPPXTensor* ga = SNEPPX_tensor_reshape(grad_output, c->a->data->shape, c->a->data->ndim);
    grad_accum(&c->a->grad, ga);
}
/**
 * @brief Perform Reshape.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 * @param shape [in] Shape value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_reshape(SNEPPXTape* tape, SNEPPXVariable* a, const size_t* shape, size_t ndim) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_reshape(a->data, shape, ndim);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        ReshapeCtx* ctx = (ReshapeCtx*)SNEPPX_malloc(sizeof(ReshapeCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_reshape; var->backward_ctx = ctx; var->free_ctx = free_ctx_ReshapeCtx; var->recompute_ctx = recompute_ReshapeCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== dropout forward/backward ===== */
typedef struct { SNEPPXVariable* a; float rate; unsigned int seed; SNEPPXTensor* mask; } DropoutCtx;
static void free_ctx_DropoutCtx(void* p) {
    DropoutCtx* c = (DropoutCtx*)p;
    if (c->mask) SNEPPX_tensor_destroy(c->mask);
    SNEPPX_free(p, sizeof(DropoutCtx));
}
static void* recompute_DropoutCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    DropoutCtx* ctx = (DropoutCtx*)SNEPPX_malloc(sizeof(DropoutCtx), 64);
    if (ctx) {
        ctx->a = var->parents[0];
        ctx->rate = n > 0 ? *(float*)&params[0] : 0.5f;
        ctx->seed = n > 1 ? (unsigned int)params[1] : 0;
        ctx->mask = NULL;
        var->free_ctx = free_ctx_DropoutCtx;
    }
    return ctx;
}
static void backward_dropout(void* ctx, SNEPPXTensor* grad_output) {
    DropoutCtx* c = (DropoutCtx*)ctx;
    if (!c->a->requires_grad) return;
    float scale = 1.0f / (1.0f - c->rate);
    if (!c->mask) {
        unsigned long s = c->seed ? (unsigned long)c->seed : 123456789;
        c->mask = SNEPPX_tensor_zeros(c->a->data->shape, c->a->data->ndim, SNEPPX_FLOAT32);
        if (!c->mask) return;
        float* md = (float*)c->mask->data;
        for (size_t i = 0; i < c->mask->size; i++) {
            s = s * 1103515245UL + 12345UL;
            float r = (float)((s >> 16) & 0x7FFF) / 32767.0f;
            md[i] = (r >= c->rate) ? scale : 0.0f;
        }
    }
    SNEPPXTensor* ga = SNEPPX_tensor_mul(c->mask, grad_output);
    grad_accum(&c->a->grad, ga);
}
/**
 * @brief Perform Dropout.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 * @param rate [in] Rate value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_dropout(SNEPPXTape* tape, SNEPPXVariable* a, float rate, unsigned int seed) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_dropout(a->data, rate, seed);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        DropoutCtx* ctx = (DropoutCtx*)SNEPPX_malloc(sizeof(DropoutCtx), 64);
        if (ctx) {
            ctx->a = a; ctx->rate = rate; ctx->seed = seed; ctx->mask = NULL;
            var->backward_fn = backward_dropout; var->backward_ctx = ctx;
            var->free_ctx = free_ctx_DropoutCtx; var->recompute_ctx = recompute_DropoutCtx;
            *(float*)&var->params[0] = rate; var->params[1] = (size_t)seed; var->param_count = 2;
        }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== layer_norm forward/backward ===== */
typedef struct { SNEPPXVariable* a; SNEPPXVariable* gamma; SNEPPXVariable* beta; float eps; } LayerNormCtx;
static void free_ctx_LayerNormCtx(void* p) { SNEPPX_free(p, sizeof(LayerNormCtx)); }
static void* recompute_LayerNormCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    LayerNormCtx* ctx = (LayerNormCtx*)SNEPPX_malloc(sizeof(LayerNormCtx), 64);
    if (ctx) {
        ctx->a = var->parents[0]; ctx->gamma = var->parents[1]; ctx->beta = var->parents[2];
        ctx->eps = n > 0 ? *(float*)&params[0] : 1e-5f;
        var->free_ctx = free_ctx_LayerNormCtx;
    }
    return ctx;
}
static void backward_layer_norm(void* ctx, SNEPPXTensor* grad_output) {
    LayerNormCtx* c = (LayerNormCtx*)ctx;
    if (!c->a->requires_grad && !(c->gamma && c->gamma->requires_grad) && !(c->beta && c->beta->requires_grad)) return;
    size_t d = c->a->data->shape[c->a->data->ndim - 1];
    size_t n = c->a->data->size / d;
    float* xd = (float*)c->a->data->data;
    float* gd = c->gamma ? (float*)c->gamma->data->data : NULL;
    float* go = (float*)grad_output->data;
    float eps = c->eps;
    if (c->a->requires_grad) {
        SNEPPXTensor* gx = SNEPPX_tensor_zeros(c->a->data->shape, c->a->data->ndim, SNEPPX_FLOAT32);
        if (gx) {
            float* gxd = (float*)gx->data;
            for (size_t i = 0; i < n; i++) {
                double sum = 0.0, sum2 = 0.0;
                for (size_t j = 0; j < d; j++) { float v = xd[i * d + j]; sum += v; sum2 += (double)v * v; }
                double mean = sum / (double)d;
                double var = sum2 / (double)d - mean * mean;
                double inv_std = 1.0 / sqrt(var + (double)eps);
                for (size_t j = 0; j < d; j++) {
                    double x_hat = (xd[i * d + j] - mean) * inv_std;
                    double g = gd ? gd[j] : 1.0;
                    gxd[i * d + j] = (float)(g * go[i * d + j]);
                }
                double dx_norm_sum = 0.0, dx_norm_x_hat_sum = 0.0;
                for (size_t j = 0; j < d; j++) {
                    dx_norm_sum += gxd[i * d + j];
                    dx_norm_x_hat_sum += gxd[i * d + j] * (xd[i * d + j] - mean) * inv_std;
                }
                for (size_t j = 0; j < d; j++) {
                    double x_hat = (xd[i * d + j] - mean) * inv_std;
                    double grad = gxd[i * d + j] - dx_norm_sum / (double)d - x_hat * dx_norm_x_hat_sum / (double)d;
                    gxd[i * d + j] = (float)(grad * inv_std);
                }
            }
            grad_accum(&c->a->grad, gx);
        }
    }
    if (c->gamma && c->gamma->requires_grad) {
        SNEPPXTensor* gg = SNEPPX_tensor_zeros(c->gamma->data->shape, c->gamma->data->ndim, SNEPPX_FLOAT32);
        if (gg) {
            float* ggd = (float*)gg->data;
            for (size_t i = 0; i < n; i++) {
                double sum = 0.0, sum2 = 0.0;
                for (size_t j = 0; j < d; j++) { float v = xd[i * d + j]; sum += v; sum2 += (double)v * v; }
                double mean = sum / (double)d;
                double var = sum2 / (double)d - mean * mean;
                double inv_std = 1.0 / sqrt(var + (double)eps);
                for (size_t j = 0; j < d; j++) {
                    double x_hat = (xd[i * d + j] - mean) * inv_std;
                    ggd[j] += (float)(x_hat * go[i * d + j]);
                }
            }
            grad_accum(&c->gamma->grad, gg);
        }
    }
    if (c->beta && c->beta->requires_grad) {
        SNEPPXTensor* gb = SNEPPX_tensor_zeros(c->beta->data->shape, c->beta->data->ndim, SNEPPX_FLOAT32);
        if (gb) {
            float* gbd = (float*)gb->data;
            for (size_t i = 0; i < n; i++)
                for (size_t j = 0; j < d; j++)
                    gbd[j] += go[i * d + j];
            grad_accum(&c->beta->grad, gb);
        }
    }
}
/**
 * @brief Perform Layer Norm.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 * @param gamma [out] Gamma value.
 * @param beta [out] Beta value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_layer_norm(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* gamma, SNEPPXVariable* beta, float eps) {
    int rg = requires_grad(a, gamma) || (beta && beta->requires_grad);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_layer_norm(a->data, gamma ? gamma->data : NULL, beta ? beta->data : NULL, eps);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        LayerNormCtx* ctx = (LayerNormCtx*)SNEPPX_malloc(sizeof(LayerNormCtx), 64);
        if (ctx) {
            ctx->a = a; ctx->gamma = gamma; ctx->beta = beta; ctx->eps = eps;
            var->backward_fn = backward_layer_norm; var->backward_ctx = ctx;
            var->free_ctx = free_ctx_LayerNormCtx; var->recompute_ctx = recompute_LayerNormCtx;
            *(float*)&var->params[0] = eps; var->param_count = 1;
        }
        SNEPPXVariable* pars[3]; pars[0] = a; pars[1] = gamma; pars[2] = beta;
        set_parents(var, pars, 3);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== conv2d forward/backward ===== */
typedef struct { SNEPPXVariable *input, *kernel; size_t stride_h, stride_w, pad_h, pad_w; } Conv2DCtx;
static void free_ctx_Conv2DCtx(void* p) { SNEPPX_free(p, sizeof(Conv2DCtx)); }
static void* recompute_Conv2DCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    Conv2DCtx* ctx = (Conv2DCtx*)SNEPPX_malloc(sizeof(Conv2DCtx), 64);
    if (ctx) {
        ctx->input = var->parents[0]; ctx->kernel = var->parents[1];
        ctx->stride_h = n > 0 ? params[0] : 1; ctx->stride_w = n > 1 ? params[1] : 1;
        ctx->pad_h = n > 2 ? params[2] : 0; ctx->pad_w = n > 3 ? params[3] : 0;
        var->free_ctx = free_ctx_Conv2DCtx;
    }
    return ctx;
}
static void backward_conv2d(void* ctx, SNEPPXTensor* grad_output) {
    Conv2DCtx* c = (Conv2DCtx*)ctx;
    float* go = (float*)grad_output->data;
    size_t N = c->input->data->shape[0], C = c->input->data->shape[1];
    size_t H = c->input->data->shape[2], W = c->input->data->shape[3];
    size_t K = c->kernel->data->shape[0];
    size_t KH = c->kernel->data->shape[2], KW = c->kernel->data->shape[3];
    size_t OH = (H + 2 * c->pad_h - KH) / c->stride_h + 1;
    size_t OW = (W + 2 * c->pad_w - KW) / c->stride_w + 1;

    if (c->input->requires_grad) {
        SNEPPXTensor* gi = SNEPPX_tensor_zeros(c->input->data->shape, c->input->data->ndim, SNEPPX_FLOAT32);
        float* gid = (float*)gi->data;
        float* kd = (float*)c->kernel->data->data;
        for (size_t n = 0; n < N; n++)
            for (size_t k = 0; k < K; k++)
                for (size_t oh = 0; oh < OH; oh++)
                    for (size_t ow = 0; ow < OW; ow++) {
                        float grad_val = go[n * K * OH * OW + k * OH * OW + oh * OW + ow];
                        for (size_t kh = 0; kh < KH; kh++) {
                            size_t ih = oh * c->stride_h + kh - c->pad_h;
                            if (ih >= H) continue;
                            for (size_t kw = 0; kw < KW; kw++) {
                                size_t iw = ow * c->stride_w + kw - c->pad_w;
                                if (iw >= W) continue;
                                for (size_t cc = 0; cc < C; cc++)
                                    gid[n * C * H * W + cc * H * W + ih * W + iw] +=
                                        kd[k * C * KH * KW + cc * KH * KW + kh * KW + kw] * grad_val;
                            }
                        }
                    }
        grad_accum(&c->input->grad, gi);
    }
    if (c->kernel->requires_grad) {
        SNEPPXTensor* gk = SNEPPX_tensor_zeros(c->kernel->data->shape, c->kernel->data->ndim, SNEPPX_FLOAT32);
        float* gkd = (float*)gk->data;
        float* id = (float*)c->input->data->data;
        for (size_t n = 0; n < N; n++)
            for (size_t k = 0; k < K; k++)
                for (size_t oh = 0; oh < OH; oh++)
                    for (size_t ow = 0; ow < OW; ow++) {
                        float grad_val = go[n * K * OH * OW + k * OH * OW + oh * OW + ow];
                        for (size_t kh = 0; kh < KH; kh++) {
                            size_t ih = oh * c->stride_h + kh - c->pad_h;
                            if (ih >= H) continue;
                            for (size_t kw = 0; kw < KW; kw++) {
                                size_t iw = ow * c->stride_w + kw - c->pad_w;
                                if (iw >= W) continue;
                                for (size_t cc = 0; cc < C; cc++)
                                    gkd[k * C * KH * KW + cc * KH * KW + kh * KW + kw] +=
                                        id[n * C * H * W + cc * H * W + ih * W + iw] * grad_val;
                            }
                        }
                    }
        grad_accum(&c->kernel->grad, gk);
    }
}
/**
 * @brief Perform Conv2d.
 *
 * @param tape [out] Tape value.
 * @param input [out] Input value.
 * @param kernel [out] Kernel value.
 * @param stride_h [in] Stride H value.
 * @param stride_w [in] Stride W value.
 * @param pad_h [in] Pad H value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_conv2d(SNEPPXTape* tape, SNEPPXVariable* input, SNEPPXVariable* kernel, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w) {
    int rg = requires_grad(input, kernel);
    if (!input || !kernel || !input->data || !kernel->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_conv2d(input->data, kernel->data, stride_h, stride_w, pad_h, pad_w);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        Conv2DCtx* ctx = (Conv2DCtx*)SNEPPX_malloc(sizeof(Conv2DCtx), 64);
        if (ctx) {
            ctx->input = input; ctx->kernel = kernel;
            ctx->stride_h = stride_h; ctx->stride_w = stride_w;
            ctx->pad_h = pad_h; ctx->pad_w = pad_w;
            var->backward_fn = backward_conv2d; var->backward_ctx = ctx;
            var->free_ctx = free_ctx_Conv2DCtx; var->recompute_ctx = recompute_Conv2DCtx;
            var->params[0] = stride_h; var->params[1] = stride_w;
            var->params[2] = pad_h; var->params[3] = pad_w; var->param_count = 4;
        }
        SNEPPXVariable* pars[2]; pars[0] = input; pars[1] = kernel;
        set_parents(var, pars, 2);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== concat forward/backward ===== */
typedef struct { SNEPPXVariable** vars; size_t num_vars; size_t dim; size_t* sizes; } ConcatCtx;
static void free_ctx_ConcatCtx(void* p) {
    ConcatCtx* c = (ConcatCtx*)p;
    if (c->sizes) SNEPPX_free(c->sizes, c->num_vars * sizeof(size_t));
    if (c->vars) SNEPPX_free(c->vars, c->num_vars * sizeof(SNEPPXVariable*));
    SNEPPX_free(p, sizeof(ConcatCtx));
}
static void* recompute_ConcatCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    ConcatCtx* ctx = (ConcatCtx*)SNEPPX_malloc(sizeof(ConcatCtx), 64);
    if (ctx) {
        ctx->num_vars = var->num_parents;
        ctx->vars = var->parents;
        ctx->dim = n > 0 ? params[0] : 0;
        ctx->sizes = NULL;
        var->free_ctx = free_ctx_ConcatCtx;
    }
    return ctx;
}
static void backward_concat(void* ctx, SNEPPXTensor* grad_output) {
    ConcatCtx* c = (ConcatCtx*)ctx;
    if (!c) return;
    size_t offset = 0;
    for (size_t i = 0; i < c->num_vars; i++) {
        if (!c->vars[i]) break;
        if (!c->vars[i]->requires_grad) { offset += c->sizes ? c->sizes[i] : c->vars[i]->data->shape[c->dim]; continue; }
        size_t sz = c->sizes ? c->sizes[i] : c->vars[i]->data->shape[c->dim];
        SNEPPXTensor* slice = SNEPPX_tensor_narrow(grad_output, c->dim, offset, sz);
        if (slice) {
            grad_accum(&c->vars[i]->grad, slice);
        }
        offset += sz;
    }
}
/**
 * @brief Perform Concat.
 *
 * @param tape [out] Tape value.
 * @param vars [out] Vars value.
 * @param num_vars [in] Num Vars value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_concat(SNEPPXTape* tape, SNEPPXVariable** vars, size_t num_vars, size_t dim) {
    if (!vars || num_vars == 0) return NULL;
    int rg = 0;
    for (size_t i = 0; i < num_vars; i++) {
        if (!vars[i] || !vars[i]->data) return NULL;
        if (vars[i]->requires_grad) rg = 1;
    }
    SNEPPXTensor** t_vars = (SNEPPXTensor**)SNEPPX_malloc(num_vars * sizeof(SNEPPXTensor*), 64);
    if (!t_vars) return NULL;
    for (size_t i = 0; i < num_vars; i++) t_vars[i] = vars[i]->data;
    SNEPPXTensor* result = SNEPPX_tensor_concat((const SNEPPXTensor**)t_vars, num_vars, dim);
    SNEPPX_free(t_vars, num_vars * sizeof(SNEPPXTensor*));
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        ConcatCtx* ctx = (ConcatCtx*)SNEPPX_malloc(sizeof(ConcatCtx), 64);
        if (ctx) {
            ctx->vars = (SNEPPXVariable**)SNEPPX_malloc(num_vars * sizeof(SNEPPXVariable*), 64);
            if (ctx->vars) memcpy(ctx->vars, vars, num_vars * sizeof(SNEPPXVariable*));
            ctx->num_vars = num_vars; ctx->dim = dim;
            ctx->sizes = (size_t*)SNEPPX_malloc(num_vars * sizeof(size_t), 64);
            if (ctx->sizes)
                for (size_t i = 0; i < num_vars; i++) ctx->sizes[i] = vars[i]->data->shape[dim];
            var->backward_fn = backward_concat; var->backward_ctx = ctx;
            var->free_ctx = free_ctx_ConcatCtx; var->recompute_ctx = recompute_ConcatCtx;
            var->params[0] = dim; var->param_count = 1;
        }
        set_parents(var, vars, num_vars);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== embedding forward/backward ===== */
typedef struct { SNEPPXVariable* weight; SNEPPXVariable* indices; } EmbeddingCtx;
static void free_ctx_EmbeddingCtx(void* p) { SNEPPX_free(p, sizeof(EmbeddingCtx)); }
static void* recompute_EmbeddingCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    EmbeddingCtx* ctx = (EmbeddingCtx*)SNEPPX_malloc(sizeof(EmbeddingCtx), 64);
    if (ctx) { ctx->weight = var->parents[0]; ctx->indices = var->parents[1]; var->free_ctx = free_ctx_EmbeddingCtx; }
    return ctx;
}
static void backward_embedding(void* ctx, SNEPPXTensor* grad_output) {
    EmbeddingCtx* c = (EmbeddingCtx*)ctx;
    if (!c->weight->requires_grad) return;
    int* idx = (int*)c->indices->data->data;
    size_t embed_dim = c->weight->data->shape[1];
    size_t num_indices = c->indices->data->size;
    float* go = (float*)grad_output->data;
    SNEPPXTensor* gw_t = SNEPPX_tensor_zeros(c->weight->data->shape, c->weight->data->ndim, SNEPPX_FLOAT32);
    if (!gw_t) return;
    float* gwd = (float*)gw_t->data;
    size_t vocab_size = c->weight->data->shape[0];
    for (size_t i = 0; i < num_indices; i++) {
        int ix = idx[i];
        if (ix >= 0 && (size_t)ix < vocab_size) {
            for (size_t j = 0; j < embed_dim; j++)
                gwd[(size_t)ix * embed_dim + j] += go[i * embed_dim + j];
        }
    }
    grad_accum(&c->weight->grad, gw_t);
}
/**
 * @brief Perform Embedding.
 *
 * @param tape [out] Tape value.
 * @param weight [out] Weight value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_embedding(SNEPPXTape* tape, SNEPPXVariable* weight, SNEPPXVariable* indices) {
    int rg = requires_grad1(weight);
    if (!weight || !indices || !weight->data || !indices->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_embedding(weight->data, indices->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        EmbeddingCtx* ctx = (EmbeddingCtx*)SNEPPX_malloc(sizeof(EmbeddingCtx), 64);
        if (ctx) { ctx->weight = weight; ctx->indices = indices; var->backward_fn = backward_embedding; var->backward_ctx = ctx; var->free_ctx = free_ctx_EmbeddingCtx; var->recompute_ctx = recompute_EmbeddingCtx; }
        SNEPPXVariable* pars[2]; pars[0] = weight; pars[1] = indices;
        set_parents(var, pars, 2);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== cross_entropy forward/backward ===== */
typedef struct { SNEPPXVariable* pred; SNEPPXVariable* target; } CrossEntropyCtx;
static void free_ctx_CrossEntropyCtx(void* p) { SNEPPX_free(p, sizeof(CrossEntropyCtx)); }
static void* recompute_CrossEntropyCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    CrossEntropyCtx* ctx = (CrossEntropyCtx*)SNEPPX_malloc(sizeof(CrossEntropyCtx), 64);
    if (ctx) { ctx->pred = var->parents[0]; ctx->target = var->parents[1]; var->free_ctx = free_ctx_CrossEntropyCtx; }
    return ctx;
}
static void backward_cross_entropy(void* ctx, SNEPPXTensor* grad_output) {
    CrossEntropyCtx* c = (CrossEntropyCtx*)ctx;
    if (!c->pred->requires_grad) return;
    size_t N = c->pred->data->shape[0];
    size_t C = c->pred->data->shape[1];
    SNEPPXTensor* sm = SNEPPX_tensor_softmax(c->pred->data, 1);
    if (!sm) return;
    float* smd = (float*)sm->data;
    float* go = (float*)grad_output->data;
    float inv_N = 1.0f / (float)N;
    float scale = go[0] * inv_N;
    int* tidx = (int*)c->target->data->data;
    for (size_t i = 0; i < N * C; i++) smd[i] *= scale;
    for (size_t i = 0; i < N; i++) {
        int cl = tidx[i];
        if (cl >= 0 && (size_t)cl < C) smd[i * C + cl] -= scale;
    }
    grad_accum(&c->pred->grad, sm);
}
/**
 * @brief Perform Cross Entropy.
 *
 * @param tape [out] Tape value.
 * @param pred [out] Pred value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_cross_entropy(SNEPPXTape* tape, SNEPPXVariable* pred, SNEPPXVariable* target) {
    int rg = requires_grad1(pred);
    if (!pred || !target || !pred->data || !target->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_cross_entropy(pred->data, target->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        CrossEntropyCtx* ctx = (CrossEntropyCtx*)SNEPPX_malloc(sizeof(CrossEntropyCtx), 64);
        if (ctx) { ctx->pred = pred; ctx->target = target; var->backward_fn = backward_cross_entropy; var->backward_ctx = ctx; var->free_ctx = free_ctx_CrossEntropyCtx; var->recompute_ctx = recompute_CrossEntropyCtx; }
        SNEPPXVariable* pars[2]; pars[0] = pred; pars[1] = target;
        set_parents(var, pars, 2);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== sqrt forward/backward ===== */
typedef struct { SNEPPXVariable* a; } SqrtCtx;
static void free_ctx_SqrtCtx(void* p) { SNEPPX_free(p, sizeof(SqrtCtx)); }
static void* recompute_SqrtCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    SqrtCtx* ctx = (SqrtCtx*)SNEPPX_malloc(sizeof(SqrtCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; var->free_ctx = free_ctx_SqrtCtx; }
    return ctx;
}
static void backward_sqrt(void* ctx, SNEPPXTensor* grad_output) {
    SqrtCtx* c = (SqrtCtx*)ctx;
    if (!c->a->requires_grad) return;
    SNEPPXTensor* ga = SNEPPX_tensor_copy(grad_output);
    if (!ga) return;
    float* gad = (float*)ga->data;
    float* ad = (float*)c->a->data->data;
    for (size_t i = 0; i < ga->size; i++) gad[i] *= 0.5f / sqrtf(ad[i]);
    grad_accum(&c->a->grad, ga);
}
/**
 * @brief Perform Sqrt.
 *
 * @param tape [out] Tape value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_sqrt(SNEPPXTape* tape, SNEPPXVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_sqrt(a->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        SqrtCtx* ctx = (SqrtCtx*)SNEPPX_malloc(sizeof(SqrtCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_sqrt; var->backward_ctx = ctx; var->free_ctx = free_ctx_SqrtCtx; var->recompute_ctx = recompute_SqrtCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== abs forward/backward ===== */
typedef struct { SNEPPXVariable* a; } AbsCtx;
static void free_ctx_AbsCtx(void* p) { SNEPPX_free(p, sizeof(AbsCtx)); }
static void* recompute_AbsCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    AbsCtx* ctx = (AbsCtx*)SNEPPX_malloc(sizeof(AbsCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; var->free_ctx = free_ctx_AbsCtx; }
    return ctx;
}
static void backward_abs(void* ctx, SNEPPXTensor* grad_output) {
    AbsCtx* c = (AbsCtx*)ctx;
    if (!c->a->requires_grad) return;
    SNEPPXTensor* s = SNEPPX_tensor_sign(c->a->data);
    if (!s) return;
    SNEPPXTensor* ga = SNEPPX_tensor_mul(s, grad_output);
    if (ga) grad_accum(&c->a->grad, ga);
    SNEPPX_tensor_destroy(s);
}
/**
 * @brief Perform Abs.
 *
 * @param tape [out] Tape value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_abs(SNEPPXTape* tape, SNEPPXVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_abs(a->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        AbsCtx* ctx = (AbsCtx*)SNEPPX_malloc(sizeof(AbsCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_abs; var->backward_ctx = ctx; var->free_ctx = free_ctx_AbsCtx; var->recompute_ctx = recompute_AbsCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== log_softmax forward/backward ===== */
typedef struct { SNEPPXVariable* a; size_t dim; } LogSoftmaxCtx;
static void free_ctx_LogSoftmaxCtx(void* p) { SNEPPX_free(p, sizeof(LogSoftmaxCtx)); }
static void* recompute_LogSoftmaxCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    LogSoftmaxCtx* ctx = (LogSoftmaxCtx*)SNEPPX_malloc(sizeof(LogSoftmaxCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; ctx->dim = n > 0 ? params[0] : 0; var->free_ctx = free_ctx_LogSoftmaxCtx; }
    return ctx;
}
static void backward_log_softmax(void* ctx, SNEPPXTensor* grad_output) {
    LogSoftmaxCtx* c = (LogSoftmaxCtx*)ctx;
    if (!c->a->requires_grad) return;
    SNEPPXTensor* sm = SNEPPX_tensor_softmax(c->a->data, c->dim);
    if (!sm) return;
    SNEPPXTensor* sum_go = SNEPPX_tensor_sum(grad_output, c->dim);
    if (!sum_go) { SNEPPX_tensor_destroy(sm); return; }
    size_t* expand_shape = (size_t*)SNEPPX_malloc(c->a->data->ndim * sizeof(size_t), 64);
    if (!expand_shape) { SNEPPX_tensor_destroy(sm); SNEPPX_tensor_destroy(sum_go); return; }
    for (size_t i = 0; i < c->a->data->ndim; i++)
        expand_shape[i] = i == c->dim ? c->a->data->shape[c->dim] : 1;
    SNEPPXTensor* expanded = SNEPPX_tensor_expand(sum_go, expand_shape, c->a->data->ndim);
    SNEPPX_free(expand_shape, c->a->data->ndim * sizeof(size_t));
    if (!expanded) { SNEPPX_tensor_destroy(sm); SNEPPX_tensor_destroy(sum_go); return; }
    SNEPPXTensor* term = SNEPPX_tensor_mul(sm, expanded);
    if (!term) { SNEPPX_tensor_destroy(sm); SNEPPX_tensor_destroy(sum_go); SNEPPX_tensor_destroy(expanded); return; }
    SNEPPXTensor* ga = SNEPPX_tensor_sub(grad_output, term);
    if (ga) grad_accum(&c->a->grad, ga);
    SNEPPX_tensor_destroy(sm);
    SNEPPX_tensor_destroy(sum_go);
    SNEPPX_tensor_destroy(expanded);
    SNEPPX_tensor_destroy(term);
}
/**
 * @brief Perform Log Softmax.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_log_softmax(SNEPPXTape* tape, SNEPPXVariable* a, size_t dim) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_log_softmax(a->data, dim);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        LogSoftmaxCtx* ctx = (LogSoftmaxCtx*)SNEPPX_malloc(sizeof(LogSoftmaxCtx), 64);
        if (ctx) { ctx->a = a; ctx->dim = dim; var->backward_fn = backward_log_softmax; var->backward_ctx = ctx; var->free_ctx = free_ctx_LogSoftmaxCtx; var->recompute_ctx = recompute_LogSoftmaxCtx; var->params[0] = dim; var->param_count = 1; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== minimum forward/backward ===== */
typedef struct { SNEPPXVariable* a; SNEPPXVariable* b; } MinCtx;
static void free_ctx_MinCtx(void* p) { SNEPPX_free(p, sizeof(MinCtx)); }
static void* recompute_MinCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    MinCtx* ctx = (MinCtx*)SNEPPX_malloc(sizeof(MinCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; ctx->b = var->parents[1]; var->free_ctx = free_ctx_MinCtx; }
    return ctx;
}
static void backward_min(void* ctx, SNEPPXTensor* grad_output) {
    MinCtx* c = (MinCtx*)ctx;
    float* ad = (float*)c->a->data->data;
    float* bd = (float*)c->b->data->data;
    (void)grad_output;
    if (c->a->requires_grad) {
        SNEPPXTensor* ga = SNEPPX_tensor_copy(grad_output);
        if (ga) {
            float* gad = (float*)ga->data;
            for (size_t i = 0; i < ga->size; i++) {
                if (ad[i] > bd[i]) gad[i] = 0.0f;
                else if (ad[i] == bd[i]) gad[i] *= 0.5f;
            }
            grad_accum(&c->a->grad, ga);
        }
    }
    if (c->b->requires_grad) {
        SNEPPXTensor* gb = SNEPPX_tensor_copy(grad_output);
        if (gb) {
            float* gbd = (float*)gb->data;
            for (size_t i = 0; i < gb->size; i++) {
                if (ad[i] < bd[i]) gbd[i] = 0.0f;
                else if (ad[i] == bd[i]) gbd[i] *= 0.5f;
            }
            grad_accum(&c->b->grad, gb);
        }
    }
}
/**
 * @brief Perform Minimum.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_minimum(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b) {
    int rg = requires_grad(a, b);
    if (!a || !b || !a->data || !b->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_minimum(a->data, b->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        MinCtx* ctx = (MinCtx*)SNEPPX_malloc(sizeof(MinCtx), 64);
        if (ctx) { ctx->a = a; ctx->b = b; var->backward_fn = backward_min; var->backward_ctx = ctx; var->free_ctx = free_ctx_MinCtx; var->recompute_ctx = recompute_MinCtx; }
        SNEPPXVariable* pars[2]; pars[0] = a; pars[1] = b;
        set_parents(var, pars, 2);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== maximum forward/backward ===== */
typedef struct { SNEPPXVariable* a; SNEPPXVariable* b; } MaxCtx;
static void free_ctx_MaxCtx(void* p) { SNEPPX_free(p, sizeof(MaxCtx)); }
static void* recompute_MaxCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    MaxCtx* ctx = (MaxCtx*)SNEPPX_malloc(sizeof(MaxCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; ctx->b = var->parents[1]; var->free_ctx = free_ctx_MaxCtx; }
    return ctx;
}
static void backward_max(void* ctx, SNEPPXTensor* grad_output) {
    MaxCtx* c = (MaxCtx*)ctx;
    float* ad = (float*)c->a->data->data;
    float* bd = (float*)c->b->data->data;
    (void)grad_output;
    if (c->a->requires_grad) {
        SNEPPXTensor* ga = SNEPPX_tensor_copy(grad_output);
        if (ga) {
            float* gad = (float*)ga->data;
            for (size_t i = 0; i < ga->size; i++) {
                if (ad[i] < bd[i]) gad[i] = 0.0f;
                else if (ad[i] == bd[i]) gad[i] *= 0.5f;
            }
            grad_accum(&c->a->grad, ga);
        }
    }
    if (c->b->requires_grad) {
        SNEPPXTensor* gb = SNEPPX_tensor_copy(grad_output);
        if (gb) {
            float* gbd = (float*)gb->data;
            for (size_t i = 0; i < gb->size; i++) {
                if (ad[i] > bd[i]) gbd[i] = 0.0f;
                else if (ad[i] == bd[i]) gbd[i] *= 0.5f;
            }
            grad_accum(&c->b->grad, gb);
        }
    }
}
/**
 * @brief Perform Maximum.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_maximum(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b) {
    int rg = requires_grad(a, b);
    if (!a || !b || !a->data || !b->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_maximum(a->data, b->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        MaxCtx* ctx = (MaxCtx*)SNEPPX_malloc(sizeof(MaxCtx), 64);
        if (ctx) { ctx->a = a; ctx->b = b; var->backward_fn = backward_max; var->backward_ctx = ctx; var->free_ctx = free_ctx_MaxCtx; var->recompute_ctx = recompute_MaxCtx; }
        SNEPPXVariable* pars[2]; pars[0] = a; pars[1] = b;
        set_parents(var, pars, 2);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== rope forward/backward ===== */
typedef struct { SNEPPXVariable* a; SNEPPXTensor* cos_table; } RopeCtx;
static void free_ctx_RopeCtx(void* p) { SNEPPX_free(p, sizeof(RopeCtx)); }
static void* recompute_RopeCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    RopeCtx* ctx = (RopeCtx*)SNEPPX_malloc(sizeof(RopeCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; ctx->cos_table = NULL; var->free_ctx = free_ctx_RopeCtx; }
    return ctx;
}
static void backward_rope(void* ctx, SNEPPXTensor* grad_output) {
    RopeCtx* c = (RopeCtx*)ctx;
    if (!c->a->requires_grad || !c->cos_table) return;
    SNEPPXTensor* inv_table = SNEPPX_tensor_copy(c->cos_table);
    if (!inv_table) return;
    float* td = (float*)inv_table->data;
    for (size_t i = 1; i < inv_table->size; i += 2) td[i] = -td[i];
    SNEPPXTensor* ga = SNEPPX_tensor_rope(grad_output, inv_table);
    if (ga) grad_accum(&c->a->grad, ga);
    SNEPPX_tensor_destroy(inv_table);
}
/**
 * @brief Perform Rope.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_rope(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXTensor* cos_table) {
    int rg = requires_grad1(a);
    if (!a || !a->data || !cos_table) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_rope(a->data, cos_table);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        RopeCtx* ctx = (RopeCtx*)SNEPPX_malloc(sizeof(RopeCtx), 64);
        if (ctx) { ctx->a = a; ctx->cos_table = cos_table; var->backward_fn = backward_rope; var->backward_ctx = ctx; var->free_ctx = free_ctx_RopeCtx; var->recompute_ctx = recompute_RopeCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== var (variance along dim) forward/backward ===== */
typedef struct { SNEPPXVariable* a; size_t dim; } VarCtx;
static void free_ctx_VarCtx(void* p) { SNEPPX_free(p, sizeof(VarCtx)); }
static void* recompute_VarCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    VarCtx* ctx = (VarCtx*)SNEPPX_malloc(sizeof(VarCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; ctx->dim = n > 0 ? params[0] : 0; var->free_ctx = free_ctx_VarCtx; }
    return ctx;
}
static void backward_var(void* ctx, SNEPPXTensor* grad_output) {
    VarCtx* c = (VarCtx*)ctx;
    if (!c->a->requires_grad) return;
    size_t N = c->a->data->shape[c->dim];
    float inv_n1 = 1.0f / (float)(N > 1 ? N - 1 : 1);
    SNEPPXTensor* mean_t = SNEPPX_tensor_mean(c->a->data, c->dim);
    if (!mean_t) return;
    SNEPPXTensor* centered = SNEPPX_tensor_sub(c->a->data, mean_t);
    SNEPPX_tensor_destroy(mean_t);
    if (!centered) return;
    SNEPPXTensor* ga = SNEPPX_tensor_mul(centered, grad_output);
    if (!ga) { SNEPPX_tensor_destroy(centered); return; }
    float* gad = (float*)ga->data;
    for (size_t i = 0; i < ga->size; i++) gad[i] *= 2.0f * inv_n1;
    grad_accum(&c->a->grad, ga);
    SNEPPX_tensor_destroy(centered);
}
/**
 * @brief Perform Var.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_var(SNEPPXTape* tape, SNEPPXVariable* a, size_t dim) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_var(a->data, dim);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        VarCtx* ctx = (VarCtx*)SNEPPX_malloc(sizeof(VarCtx), 64);
        if (ctx) { ctx->a = a; ctx->dim = dim; var->backward_fn = backward_var; var->backward_ctx = ctx; var->free_ctx = free_ctx_VarCtx; var->recompute_ctx = recompute_VarCtx; var->params[0] = dim; var->param_count = 1; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== std (standard deviation along dim) forward/backward ===== */
typedef struct { SNEPPXVariable* a; size_t dim; } StdCtx;
static void free_ctx_StdCtx(void* p) { SNEPPX_free(p, sizeof(StdCtx)); }
static void* recompute_StdCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    StdCtx* ctx = (StdCtx*)SNEPPX_malloc(sizeof(StdCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; ctx->dim = n > 0 ? params[0] : 0; var->free_ctx = free_ctx_StdCtx; }
    return ctx;
}
static void backward_std(void* ctx, SNEPPXTensor* grad_output) {
    StdCtx* c = (StdCtx*)ctx;
    if (!c->a->requires_grad) return;
    size_t N = c->a->data->shape[c->dim];
    float inv_n1 = 1.0f / (float)(N > 1 ? N - 1 : 1);
    SNEPPXTensor* mean_t = SNEPPX_tensor_mean(c->a->data, c->dim);
    if (!mean_t) return;
    SNEPPXTensor* centered = SNEPPX_tensor_sub(c->a->data, mean_t);
    SNEPPX_tensor_destroy(mean_t);
    if (!centered) return;
    SNEPPXTensor* std_t = SNEPPX_tensor_std(c->a->data, c->dim);
    if (!std_t) { SNEPPX_tensor_destroy(centered); return; }
    float* stdd = (float*)std_t->data;
    SNEPPXTensor* ga = SNEPPX_tensor_mul(centered, grad_output);
    if (!ga) { SNEPPX_tensor_destroy(centered); SNEPPX_tensor_destroy(std_t); return; }
    float* gad = (float*)ga->data;
    for (size_t i = 0; i < ga->size; i++) gad[i] *= inv_n1 / (stdd[i % std_t->size] + 1e-12f);
    grad_accum(&c->a->grad, ga);
    SNEPPX_tensor_destroy(centered);
    SNEPPX_tensor_destroy(std_t);
}
/**
 * @brief Perform Std.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_std(SNEPPXTape* tape, SNEPPXVariable* a, size_t dim) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_std(a->data, dim);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        StdCtx* ctx = (StdCtx*)SNEPPX_malloc(sizeof(StdCtx), 64);
        if (ctx) { ctx->a = a; ctx->dim = dim; var->backward_fn = backward_std; var->backward_ctx = ctx; var->free_ctx = free_ctx_StdCtx; var->recompute_ctx = recompute_StdCtx; var->params[0] = dim; var->param_count = 1; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== sin forward/backward ===== */
typedef struct { SNEPPXVariable* a; } SinCtx;
static void free_ctx_SinCtx(void* p) { SNEPPX_free(p, sizeof(SinCtx)); }
static void* recompute_SinCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    SinCtx* ctx = (SinCtx*)SNEPPX_malloc(sizeof(SinCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; var->free_ctx = free_ctx_SinCtx; }
    return ctx;
}
static void backward_sin(void* ctx, SNEPPXTensor* grad_output) {
    SinCtx* c = (SinCtx*)ctx;
    if (!c->a->requires_grad) return;
    SNEPPXTensor* cos_t = SNEPPX_tensor_cos(c->a->data);
    if (!cos_t) return;
    SNEPPXTensor* ga = SNEPPX_tensor_mul(cos_t, grad_output);
    if (ga) grad_accum(&c->a->grad, ga);
    SNEPPX_tensor_destroy(cos_t);
}
/**
 * @brief Perform Sin.
 *
 * @param tape [out] Tape value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_sin(SNEPPXTape* tape, SNEPPXVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_sin(a->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        SinCtx* ctx = (SinCtx*)SNEPPX_malloc(sizeof(SinCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_sin; var->backward_ctx = ctx; var->free_ctx = free_ctx_SinCtx; var->recompute_ctx = recompute_SinCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== cos forward/backward ===== */
typedef struct { SNEPPXVariable* a; } CosCtx;
static void free_ctx_CosCtx(void* p) { SNEPPX_free(p, sizeof(CosCtx)); }
static void* recompute_CosCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    CosCtx* ctx = (CosCtx*)SNEPPX_malloc(sizeof(CosCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; var->free_ctx = free_ctx_CosCtx; }
    return ctx;
}
static void backward_cos(void* ctx, SNEPPXTensor* grad_output) {
    CosCtx* c = (CosCtx*)ctx;
    if (!c->a->requires_grad) return;
    SNEPPXTensor* sin_t = SNEPPX_tensor_sin(c->a->data);
    if (!sin_t) return;
    float* sd = (float*)sin_t->data;
    for (size_t i = 0; i < sin_t->size; i++) sd[i] = -sd[i];
    SNEPPXTensor* ga = SNEPPX_tensor_mul(sin_t, grad_output);
    if (ga) grad_accum(&c->a->grad, ga);
    SNEPPX_tensor_destroy(sin_t);
}
/**
 * @brief Perform Cos.
 *
 * @param tape [out] Tape value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_cos(SNEPPXTape* tape, SNEPPXVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_cos(a->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        CosCtx* ctx = (CosCtx*)SNEPPX_malloc(sizeof(CosCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_cos; var->backward_ctx = ctx; var->free_ctx = free_ctx_CosCtx; var->recompute_ctx = recompute_CosCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== tan forward/backward ===== */
typedef struct { SNEPPXVariable* a; } TanCtx;
static void free_ctx_TanCtx(void* p) { SNEPPX_free(p, sizeof(TanCtx)); }
static void* recompute_TanCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    TanCtx* ctx = (TanCtx*)SNEPPX_malloc(sizeof(TanCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; var->free_ctx = free_ctx_TanCtx; }
    return ctx;
}
static void backward_tan(void* ctx, SNEPPXTensor* grad_output) {
    TanCtx* c = (TanCtx*)ctx;
    if (!c->a->requires_grad) return;
    SNEPPXTensor* cos_t = SNEPPX_tensor_cos(c->a->data);
    if (!cos_t) return;
    float* cd = (float*)cos_t->data;
    for (size_t i = 0; i < cos_t->size; i++) cd[i] = 1.0f / (cd[i] * cd[i] + 1e-12f);
    SNEPPXTensor* ga = SNEPPX_tensor_mul(cos_t, grad_output);
    if (ga) grad_accum(&c->a->grad, ga);
    SNEPPX_tensor_destroy(cos_t);
}
/**
 * @brief Perform Tan.
 *
 * @param tape [out] Tape value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_tan(SNEPPXTape* tape, SNEPPXVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_tan(a->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        TanCtx* ctx = (TanCtx*)SNEPPX_malloc(sizeof(TanCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_tan; var->backward_ctx = ctx; var->free_ctx = free_ctx_TanCtx; var->recompute_ctx = recompute_TanCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== asin forward/backward ===== */
typedef struct { SNEPPXVariable* a; } AsinCtx;
static void free_ctx_AsinCtx(void* p) { SNEPPX_free(p, sizeof(AsinCtx)); }
static void* recompute_AsinCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    AsinCtx* ctx = (AsinCtx*)SNEPPX_malloc(sizeof(AsinCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; var->free_ctx = free_ctx_AsinCtx; }
    return ctx;
}
static void backward_asin(void* ctx, SNEPPXTensor* grad_output) {
    AsinCtx* c = (AsinCtx*)ctx;
    if (!c->a->requires_grad) return;
    SNEPPXTensor* ga = SNEPPX_tensor_copy(grad_output);
    if (!ga) return;
    float* gad = (float*)ga->data;
    float* ad = (float*)c->a->data->data;
    for (size_t i = 0; i < ga->size; i++) gad[i] /= sqrtf(1.0f - ad[i] * ad[i] + 1e-12f);
    grad_accum(&c->a->grad, ga);
}
/**
 * @brief Perform Asin.
 *
 * @param tape [out] Tape value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_asin(SNEPPXTape* tape, SNEPPXVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_asin(a->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        AsinCtx* ctx = (AsinCtx*)SNEPPX_malloc(sizeof(AsinCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_asin; var->backward_ctx = ctx; var->free_ctx = free_ctx_AsinCtx; var->recompute_ctx = recompute_AsinCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== acos forward/backward ===== */
typedef struct { SNEPPXVariable* a; } AcosCtx;
static void free_ctx_AcosCtx(void* p) { SNEPPX_free(p, sizeof(AcosCtx)); }
static void* recompute_AcosCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    AcosCtx* ctx = (AcosCtx*)SNEPPX_malloc(sizeof(AcosCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; var->free_ctx = free_ctx_AcosCtx; }
    return ctx;
}
static void backward_acos(void* ctx, SNEPPXTensor* grad_output) {
    AcosCtx* c = (AcosCtx*)ctx;
    if (!c->a->requires_grad) return;
    SNEPPXTensor* ga = SNEPPX_tensor_copy(grad_output);
    if (!ga) return;
    float* gad = (float*)ga->data;
    float* ad = (float*)c->a->data->data;
    for (size_t i = 0; i < ga->size; i++) gad[i] /= -sqrtf(1.0f - ad[i] * ad[i] + 1e-12f);
    grad_accum(&c->a->grad, ga);
}
/**
 * @brief Perform Acos.
 *
 * @param tape [out] Tape value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_acos(SNEPPXTape* tape, SNEPPXVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_acos(a->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        AcosCtx* ctx = (AcosCtx*)SNEPPX_malloc(sizeof(AcosCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_acos; var->backward_ctx = ctx; var->free_ctx = free_ctx_AcosCtx; var->recompute_ctx = recompute_AcosCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== atan forward/backward ===== */
typedef struct { SNEPPXVariable* a; } AtanCtx;
static void free_ctx_AtanCtx(void* p) { SNEPPX_free(p, sizeof(AtanCtx)); }
static void* recompute_AtanCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    AtanCtx* ctx = (AtanCtx*)SNEPPX_malloc(sizeof(AtanCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; var->free_ctx = free_ctx_AtanCtx; }
    return ctx;
}
static void backward_atan(void* ctx, SNEPPXTensor* grad_output) {
    AtanCtx* c = (AtanCtx*)ctx;
    if (!c->a->requires_grad) return;
    SNEPPXTensor* ga = SNEPPX_tensor_copy(grad_output);
    if (!ga) return;
    float* gad = (float*)ga->data;
    float* ad = (float*)c->a->data->data;
    for (size_t i = 0; i < ga->size; i++) gad[i] /= 1.0f + ad[i] * ad[i];
    grad_accum(&c->a->grad, ga);
}
/**
 * @brief Perform Atan.
 *
 * @param tape [out] Tape value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_atan(SNEPPXTape* tape, SNEPPXVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_atan(a->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        AtanCtx* ctx = (AtanCtx*)SNEPPX_malloc(sizeof(AtanCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_atan; var->backward_ctx = ctx; var->free_ctx = free_ctx_AtanCtx; var->recompute_ctx = recompute_AtanCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== sinh forward/backward ===== */
typedef struct { SNEPPXVariable* a; } SinhCtx;
static void free_ctx_SinhCtx(void* p) { SNEPPX_free(p, sizeof(SinhCtx)); }
static void* recompute_SinhCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    SinhCtx* ctx = (SinhCtx*)SNEPPX_malloc(sizeof(SinhCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; var->free_ctx = free_ctx_SinhCtx; }
    return ctx;
}
static void backward_sinh(void* ctx, SNEPPXTensor* grad_output) {
    SinhCtx* c = (SinhCtx*)ctx;
    if (!c->a->requires_grad) return;
    SNEPPXTensor* cosh_t = SNEPPX_tensor_cosh(c->a->data);
    if (!cosh_t) return;
    SNEPPXTensor* ga = SNEPPX_tensor_mul(cosh_t, grad_output);
    if (ga) grad_accum(&c->a->grad, ga);
    SNEPPX_tensor_destroy(cosh_t);
}
/**
 * @brief Perform Sinh.
 *
 * @param tape [out] Tape value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_sinh(SNEPPXTape* tape, SNEPPXVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_sinh(a->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        SinhCtx* ctx = (SinhCtx*)SNEPPX_malloc(sizeof(SinhCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_sinh; var->backward_ctx = ctx; var->free_ctx = free_ctx_SinhCtx; var->recompute_ctx = recompute_SinhCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== cosh forward/backward ===== */
typedef struct { SNEPPXVariable* a; } CoshCtx;
static void free_ctx_CoshCtx(void* p) { SNEPPX_free(p, sizeof(CoshCtx)); }
static void* recompute_CoshCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    CoshCtx* ctx = (CoshCtx*)SNEPPX_malloc(sizeof(CoshCtx), 64);
    if (ctx) { ctx->a = var->parents[0]; var->free_ctx = free_ctx_CoshCtx; }
    return ctx;
}
static void backward_cosh(void* ctx, SNEPPXTensor* grad_output) {
    CoshCtx* c = (CoshCtx*)ctx;
    if (!c->a->requires_grad) return;
    SNEPPXTensor* sinh_t = SNEPPX_tensor_sinh(c->a->data);
    if (!sinh_t) return;
    SNEPPXTensor* ga = SNEPPX_tensor_mul(sinh_t, grad_output);
    if (ga) grad_accum(&c->a->grad, ga);
    SNEPPX_tensor_destroy(sinh_t);
}
/**
 * @brief Perform Cosh.
 *
 * @param tape [out] Tape value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_cosh(SNEPPXTape* tape, SNEPPXVariable* a) {
    int rg = requires_grad1(a);
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_cosh(a->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        CoshCtx* ctx = (CoshCtx*)SNEPPX_malloc(sizeof(CoshCtx), 64);
        if (ctx) { ctx->a = a; var->backward_fn = backward_cosh; var->backward_ctx = ctx; var->free_ctx = free_ctx_CoshCtx; var->recompute_ctx = recompute_CoshCtx; }
        set_parents(var, &a, 1);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== sign (non-differentiable, zero gradient) ===== */
/**
 * @brief Sign Sign.
 *
 * @param tape [out] Tape value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_sign(SNEPPXTape* tape, SNEPPXVariable* a) {
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_sign(a->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, 0);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (tape) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== floor (non-differentiable, zero gradient) ===== */
/**
 * @brief Perform Floor.
 *
 * @param tape [out] Tape value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_floor(SNEPPXTape* tape, SNEPPXVariable* a) {
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_floor(a->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, 0);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (tape) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== ceil (non-differentiable, zero gradient) ===== */
/**
 * @brief Perform Ceil.
 *
 * @param tape [out] Tape value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_ceil(SNEPPXTape* tape, SNEPPXVariable* a) {
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_ceil(a->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, 0);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (tape) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== round (non-differentiable, zero gradient) ===== */
/**
 * @brief Perform Round.
 *
 * @param tape [out] Tape value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_round(SNEPPXTape* tape, SNEPPXVariable* a) {
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_round(a->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, 0);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (tape) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== trunc (non-differentiable, zero gradient) ===== */
/**
 * @brief Perform Trunc.
 *
 * @param tape [out] Tape value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_trunc(SNEPPXTape* tape, SNEPPXVariable* a) {
    if (!a || !a->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_trunc(a->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, 0);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (tape) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== nll_loss forward/backward ===== */
typedef struct { SNEPPXVariable* pred; SNEPPXVariable* target; } NllLossCtx;
static void free_ctx_NllLossCtx(void* p) { SNEPPX_free(p, sizeof(NllLossCtx)); }
static void* recompute_NllLossCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    NllLossCtx* ctx = (NllLossCtx*)SNEPPX_malloc(sizeof(NllLossCtx), 64);
    if (ctx) { ctx->pred = var->parents[0]; ctx->target = var->parents[1]; var->free_ctx = free_ctx_NllLossCtx; }
    return ctx;
}
static void backward_nll_loss(void* ctx, SNEPPXTensor* grad_output) {
    NllLossCtx* c = (NllLossCtx*)ctx;
    if (!c->pred->requires_grad) return;
    size_t N = c->pred->data->shape[0];
    size_t C = c->pred->data->shape[1];
    float inv_N = -1.0f / (float)N;
    float scale = grad_output ? ((float*)grad_output->data)[0] * inv_N : inv_N;
    int* tidx = (int*)c->target->data->data;
    SNEPPXTensor* g = SNEPPX_tensor_zeros(c->pred->data->shape, c->pred->data->ndim, SNEPPX_FLOAT32);
    if (!g) return;
    float* gd = (float*)g->data;
    for (size_t i = 0; i < N; i++) {
        int cl = tidx[i];
        if (cl >= 0 && (size_t)cl < C) gd[i * C + cl] = scale;
    }
    grad_accum(&c->pred->grad, g);
}
/**
 * @brief Perform Nll Loss.
 *
 * @param tape [out] Tape value.
 * @param pred [out] Pred value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_nll_loss(SNEPPXTape* tape, SNEPPXVariable* pred, SNEPPXVariable* target) {
    int rg = requires_grad1(pred);
    if (!pred || !target || !pred->data || !target->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_nll_loss(pred->data, target->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        NllLossCtx* ctx = (NllLossCtx*)SNEPPX_malloc(sizeof(NllLossCtx), 64);
        if (ctx) { ctx->pred = pred; ctx->target = target; var->backward_fn = backward_nll_loss; var->backward_ctx = ctx; var->free_ctx = free_ctx_NllLossCtx; var->recompute_ctx = recompute_NllLossCtx; }
        SNEPPXVariable* pars[2]; pars[0] = pred; pars[1] = target;
        set_parents(var, pars, 2);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== bce_loss forward/backward ===== */
typedef struct { SNEPPXVariable* pred; SNEPPXVariable* target; } BceLossCtx;
static void free_ctx_BceLossCtx(void* p) { SNEPPX_free(p, sizeof(BceLossCtx)); }
static void* recompute_BceLossCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    (void)params; (void)n;
    BceLossCtx* ctx = (BceLossCtx*)SNEPPX_malloc(sizeof(BceLossCtx), 64);
    if (ctx) { ctx->pred = var->parents[0]; ctx->target = var->parents[1]; var->free_ctx = free_ctx_BceLossCtx; }
    return ctx;
}
static void backward_bce_loss(void* ctx, SNEPPXTensor* grad_output) {
    BceLossCtx* c = (BceLossCtx*)ctx;
    if (!c->pred->requires_grad) return;
    float* pd = (float*)c->pred->data->data;
    float* td = (float*)c->target->data->data;
    float n = (float)c->pred->data->size;
    float scale = grad_output ? ((float*)grad_output->data)[0] / n : 1.0f / n;
    SNEPPXTensor* g = SNEPPX_tensor_zeros(c->pred->data->shape, c->pred->data->ndim, SNEPPX_FLOAT32);
    if (!g) return;
    float* gd = (float*)g->data;
    for (size_t i = 0; i < g->size; i++)
        gd[i] = scale * (pd[i] - td[i]);
    grad_accum(&c->pred->grad, g);
}
/**
 * @brief Perform Bce Loss.
 *
 * @param tape [out] Tape value.
 * @param pred [out] Pred value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_bce_loss(SNEPPXTape* tape, SNEPPXVariable* pred, SNEPPXVariable* target) {
    int rg = requires_grad1(pred);
    if (!pred || !target || !pred->data || !target->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_binary_cross_entropy(pred->data, target->data);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        BceLossCtx* ctx = (BceLossCtx*)SNEPPX_malloc(sizeof(BceLossCtx), 64);
        if (ctx) { ctx->pred = pred; ctx->target = target; var->backward_fn = backward_bce_loss; var->backward_ctx = ctx; var->free_ctx = free_ctx_BceLossCtx; var->recompute_ctx = recompute_BceLossCtx; }
        SNEPPXVariable* pars[2]; pars[0] = pred; pars[1] = target;
        set_parents(var, pars, 2);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}

/* ===== batch_norm forward/backward ===== */
typedef struct { SNEPPXVariable *a, *gamma, *beta, *running_mean, *running_var; float eps; } BatchNormCtx;
static void free_ctx_BatchNormCtx(void* p) { SNEPPX_free(p, sizeof(BatchNormCtx)); }
static void* recompute_BatchNormCtx(SNEPPXVariable* var, size_t* params, size_t n) {
    BatchNormCtx* ctx = (BatchNormCtx*)SNEPPX_malloc(sizeof(BatchNormCtx), 64);
    if (ctx) {
        ctx->a = var->parents[0]; ctx->gamma = var->parents[1]; ctx->beta = var->parents[2];
        ctx->running_mean = NULL; ctx->running_var = NULL;
        ctx->eps = n > 0 ? *(float*)&params[0] : 1e-5f;
        var->free_ctx = free_ctx_BatchNormCtx;
    }
    return ctx;
}
static void backward_batch_norm(void* ctx, SNEPPXTensor* grad_output) {
    BatchNormCtx* c = (BatchNormCtx*)ctx;
    int need_a = c->a->requires_grad;
    int need_gamma = c->gamma && c->gamma->requires_grad;
    int need_beta = c->beta && c->beta->requires_grad;
    if (!need_a && !need_gamma && !need_beta) return;
    float* xd = (float*)c->a->data->data;
    float* go = (float*)grad_output->data;
    float* gd = c->gamma ? (float*)c->gamma->data->data : NULL;
    float eps = c->eps;
    size_t N = c->a->data->shape[0];
    size_t C = c->a->data->shape[1];
    size_t H = c->a->data->ndim > 2 ? c->a->data->shape[2] : 1;
    size_t W = c->a->data->ndim > 3 ? c->a->data->shape[3] : 1;
    size_t spatial = H * W;
    size_t per_chn = N * spatial;
    float inv_N = 1.0f / (float)(N * spatial);

    if (need_a) {
        SNEPPXTensor* gx = SNEPPX_tensor_zeros(c->a->data->shape, c->a->data->ndim, SNEPPX_FLOAT32);
        if (gx) {
            float* gxd = (float*)gx->data;
                for (size_t ch = 0; ch < C; ch++) {
                    double sum_x = 0.0, sum_x2 = 0.0;
                    for (size_t i = 0; i < per_chn; i++) {
                        float v = xd[ch * per_chn + i];
                        sum_x += v; sum_x2 += (double)v * v;
                    }
                    double mean = sum_x * (double)inv_N;
                    double var = sum_x2 * (double)inv_N - mean * mean;
                    double inv_std = 1.0 / sqrt(var + (double)eps);
                    double gsum = 0.0, gxsum = 0.0, gamma_val = gd ? (double)gd[ch] : 1.0;
                    for (size_t i = 0; i < per_chn; i++) {
                        size_t idx = ch * per_chn + i;
                        double x_hat = (xd[idx] - mean) * inv_std;
                        gsum += (double)go[idx] * gamma_val;
                        gxsum += (double)go[idx] * gamma_val * x_hat;
                    }
                    for (size_t i = 0; i < per_chn; i++) {
                        size_t idx = ch * per_chn + i;
                        double x_hat = (xd[idx] - mean) * inv_std;
                        gxd[idx] = (float)(gamma_val * inv_std * (go[idx] - (gsum * inv_N) - x_hat * (gxsum * inv_N)));
                    }
            }
            grad_accum(&c->a->grad, gx);
        }
    }
    if (need_gamma) {
        SNEPPXTensor* gg = SNEPPX_tensor_zeros(c->gamma->data->shape, c->gamma->data->ndim, SNEPPX_FLOAT32);
        if (gg) {
            float* ggd = (float*)gg->data;
            for (size_t ch = 0; ch < C; ch++) {
                double sum_x = 0.0, sum_x2 = 0.0;
                for (size_t i = 0; i < per_chn; i++) {
                    float v = xd[ch * per_chn + i];
                    sum_x += v; sum_x2 += (double)v * v;
                }
                double mean = sum_x * (double)inv_N;
                double var = sum_x2 * (double)inv_N - mean * mean;
                double inv_std = 1.0 / sqrt(var + (double)eps);
                double accum = 0.0;
                for (size_t i = 0; i < per_chn; i++) {
                    size_t idx = ch * per_chn + i;
                    double x_hat = (xd[idx] - mean) * inv_std;
                    accum += (double)go[idx] * x_hat;
                }
                ggd[ch] = (float)accum;
            }
            grad_accum(&c->gamma->grad, gg);
        }
    }
    if (need_beta) {
        SNEPPXTensor* gb = SNEPPX_tensor_zeros(c->beta->data->shape, c->beta->data->ndim, SNEPPX_FLOAT32);
        if (gb) {
            float* gbd = (float*)gb->data;
            for (size_t ch = 0; ch < C; ch++) {
                double accum = 0.0;
                for (size_t i = 0; i < per_chn; i++)
                    accum += (double)go[ch * per_chn + i];
                gbd[ch] = (float)accum;
            }
            grad_accum(&c->beta->grad, gb);
        }
    }
}
/**
 * @brief Perform Batch Norm.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 * @param gamma [out] Gamma value.
 * @param beta [out] Beta value.
 * @param running_mean [out] Running Mean value.
 * @param running_var [out] Running Var value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_batch_norm(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* gamma, SNEPPXVariable* beta, SNEPPXVariable* running_mean, SNEPPXVariable* running_var, float eps) {
    int rg = requires_grad(a, gamma) || (beta && beta->requires_grad);
    if (!a || !a->data || !gamma || !gamma->data || !beta || !beta->data || !running_mean || !running_mean->data || !running_var || !running_var->data) return NULL;
    SNEPPXTensor* result = SNEPPX_tensor_batch_norm(a->data, gamma->data, beta->data, running_mean->data, running_var->data, eps);
    if (!result) return NULL;
    SNEPPXVariable* var = SNEPPX_variable_create(result, rg);
    if (!var) { SNEPPX_tensor_destroy(result); return NULL; }
    if (rg && !SNEPPX_no_grad_is_active()) {
        BatchNormCtx* ctx = (BatchNormCtx*)SNEPPX_malloc(sizeof(BatchNormCtx), 64);
        if (ctx) {
            ctx->a = a; ctx->gamma = gamma; ctx->beta = beta;
            ctx->running_mean = running_mean; ctx->running_var = running_var; ctx->eps = eps;
            var->backward_fn = backward_batch_norm; var->backward_ctx = ctx;
            var->free_ctx = free_ctx_BatchNormCtx; var->recompute_ctx = recompute_BatchNormCtx;
            *(float*)&var->params[0] = eps; var->param_count = 1;
        }
        SNEPPXVariable* pars[3]; pars[0] = a; pars[1] = gamma; pars[2] = beta;
        set_parents(var, pars, 3);
    }
    if (tape && !SNEPPX_no_grad_is_active()) SNEPPX_tape_record(tape, var);
    return var;
}
