#include "tensor_expr.h"
#include "simd_gemm.h"
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



/* =========================================================================
 * Lazy Tensor Expression Engine
 * ========================================================================= */

static int g_next_id = 1;

/**
 * @brief Create Expr Graph.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXExprGraph* SNEPPX_expr_graph_create(int enable_fusion) {
    SNEPPXExprGraph* g = (SNEPPXExprGraph*)calloc(1, sizeof(SNEPPXExprGraph));
    if (!g) return NULL;
    g->capacity = 1024;
    g->nodes = (SNEPPXExprNode**)calloc(g->capacity, sizeof(SNEPPXExprNode*));
    if (!g->nodes) { free(g); return NULL; }
    g->num_nodes = 0;
    g->next_id = 1;
    g->enable_fusion = enable_fusion;
    g->fused_kernel_count = 0;
    return g;
}

/**
 * @brief Destroy Expr Graph.
 */
void SNEPPX_expr_graph_destroy(SNEPPXExprGraph* g) {
    if (!g) return;
    for (int i = 0; i < g->num_nodes; i++) {
        SNEPPXExprNode* n = g->nodes[i];
        if (n) {
            free(n->inputs);
            if (n->materialized && n->payload) free(n->payload);
            free(n);
        }
    }
    free(g->nodes);
    free(g);
}

static SNEPPXExprNode* new_node(SNEPPXExprGraph* g, SNEPPXExprOp op, int n_inputs) {
    if (g->num_nodes >= g->capacity) {
        g->capacity *= 2;
        g->nodes = (SNEPPXExprNode**)realloc(g->nodes, g->capacity * sizeof(SNEPPXExprNode*));
    }
    SNEPPXExprNode* n = (SNEPPXExprNode*)calloc(1, sizeof(SNEPPXExprNode));
    if (!n) return NULL;
    n->op = op;
    n->id = g->next_id++;
    n->n_inputs = n_inputs;
    n->inputs = n_inputs > 0 ? (SNEPPXExprNode**)calloc(n_inputs, sizeof(SNEPPXExprNode*)) : NULL;
    n->fusion_group = -1;
    n->fuseable = 0;
    n->materialized = 0;
    g->nodes[g->num_nodes++] = n;
    return n;
}

static int shape_size(const int* shape, int ndim) {
    int s = 1;
    for (int i = 0; i < ndim; i++) s *= (shape[i] > 0 ? shape[i] : 1);
    return s;
}

static void carry_shape(SNEPPXExprNode* dst, const int* shape, int ndim) {
    dst->ndim = ndim;
    for (int i = 0; i < ndim; i++) dst->shape[i] = shape[i];
}

/**
 * @brief Perform Expr Input.
 *
 * @param g [out] G value.
 * @param shape [in] Shape value.
 * @param ndim [in] Ndim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXExprNode* SNEPPX_expr_input(SNEPPXExprGraph* g, const int* shape, int ndim, void* data) {
    SNEPPXExprNode* n = new_node(g, SNEPPX_EXPR_INPUT, 0);
    if (!n) return NULL;
    carry_shape(n, shape, ndim);
    n->payload = data;
    n->materialized = (data != NULL);
    n->fuseable = 0;
    return n;
}

/**
 * @brief Perform Expr Const.
 *
 * @param g [out] G value.
 * @param value [in] Value value.
 * @param shape [in] Shape value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXExprNode* SNEPPX_expr_const(SNEPPXExprGraph* g, float value, const int* shape, int ndim) {
    SNEPPXExprNode* n = new_node(g, SNEPPX_EXPR_CONST, 0);
    if (!n) return NULL;
    carry_shape(n, shape, ndim);
    n->scalar = value;
    n->fuseable = 1;
    return n;
}

/**
 * @brief Perform Expr Binary.
 *
 * @param g [out] G value.
 * @param op [in] Op value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXExprNode* SNEPPX_expr_binary(SNEPPXExprGraph* g, SNEPPXExprOp op,
                                    SNEPPXExprNode* a, SNEPPXExprNode* b) {
    SNEPPXExprNode* n = new_node(g, op, 2);
    if (!n) return NULL;
    n->inputs[0] = a;
    n->inputs[1] = b;
    carry_shape(n, a->shape, a->ndim);
    n->fuseable = (op == SNEPPX_EXPR_ADD || op == SNEPPX_EXPR_SUB ||
                   op == SNEPPX_EXPR_MUL || op == SNEPPX_EXPR_DIV);
    return n;
}

/**
 * @brief Perform Expr Unary.
 *
 * @param g [out] G value.
 * @param op [in] Op value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXExprNode* SNEPPX_expr_unary(SNEPPXExprGraph* g, SNEPPXExprOp op, SNEPPXExprNode* a) {
    SNEPPXExprNode* n = new_node(g, op, 1);
    if (!n) return NULL;
    n->inputs[0] = a;
    carry_shape(n, a->shape, a->ndim);
    n->fuseable = (op == SNEPPX_EXPR_RELU || op == SNEPPX_EXPR_GELU ||
                   op == SNEPPX_EXPR_SILU || op == SNEPPX_EXPR_TANH ||
                   op == SNEPPX_EXPR_SIGMOID || op == SNEPPX_EXPR_EXP ||
                   op == SNEPPX_EXPR_LOG || op == SNEPPX_EXPR_SQRT ||
                   op == SNEPPX_EXPR_CLAMP);
    return n;
}

/**
 * @brief Perform Expr Matmul.
 *
 * @param g [out] G value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXExprNode* SNEPPX_expr_matmul(SNEPPXExprGraph* g, SNEPPXExprNode* a, SNEPPXExprNode* b) {
    SNEPPXExprNode* n = new_node(g, SNEPPX_EXPR_MATMUL, 2);
    if (!n) return NULL;
    n->inputs[0] = a;
    n->inputs[1] = b;
    n->ndim = 2;
    n->shape[0] = a->shape[0];
    n->shape[1] = b->shape[1];
    n->fuseable = 0;
    return n;
}

/**
 * @brief Perform Expr Activation.
 *
 * @param g [out] G value.
 * @param op [in] Op value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXExprNode* SNEPPX_expr_activation(SNEPPXExprGraph* g, SNEPPXExprOp op,
                                       SNEPPXExprNode* a, int act_kind) {
    SNEPPXExprNode* n = SNEPPX_expr_unary(g, op, a);
    if (n) n->act_kind = act_kind;
    return n;
}

/**
 * @brief Perform Expr Reduction.
 *
 * @param g [out] G value.
 * @param op [in] Op value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXExprNode* SNEPPX_expr_reduction(SNEPPXExprGraph* g, SNEPPXExprOp op,
                                       SNEPPXExprNode* a, int axis) {
    SNEPPXExprNode* n = new_node(g, op, 1);
    if (!n) return NULL;
    n->inputs[0] = a;
    (void)axis;
    if (op == SNEPPX_EXPR_SUM || op == SNEPPX_EXPR_MEAN) {
        /* reduce to scalar (2D -> 1) */
        n->ndim = 1;
        n->shape[0] = 1;
    } else {
        carry_shape(n, a->shape, a->ndim);
    }
    n->fuseable = 0;
    return n;
}

/**
 * @brief Perform Expr Fused Linear Bias.
 *
 * @param g [out] G value.
 * @param a [out] A value.
 * @param w [out] W value.
 * @param bias [out] Bias value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXExprNode* SNEPPX_expr_fused_linear_bias(SNEPPXExprGraph* g,
                                              SNEPPXExprNode* a, SNEPPXExprNode* w,
                                              SNEPPXExprNode* bias, int act_kind) {
    int n_in = (bias ? 3 : 2);
    SNEPPXExprNode* n = new_node(g, SNEPPX_EXPR_BIAS_ADD, n_in);
    if (!n) return NULL;
    n->inputs[0] = a;
    n->inputs[1] = w;
    if (bias) n->inputs[2] = bias;
    n->ndim = 2;
    n->shape[0] = a->shape[0];
    n->shape[1] = w->shape[1];
    n->act_kind = act_kind;
    n->fuseable = 0;
    return n;
}

/* =========================================================================
 * Fusion planner: greedily group chains of fuseable elementwise ops
 * ========================================================================= */

/**
 * @brief Perform Expr Plan Fusions.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_expr_plan_fusions(SNEPPXExprGraph* g) {
    if (!g->enable_fusion) { g->fused_kernel_count = g->num_nodes; return g->num_nodes; }
    int group = 0;
    for (int i = 0; i < g->num_nodes; i++) {
        SNEPPXExprNode* n = g->nodes[i];
        if (n->fusion_group >= 0) continue;
        if (!n->fuseable) { n->fusion_group = group++; continue; }
        /* start a fusion group */
        n->fusion_group = group;
        /* absorb downstream consumers that are also fuseable and single-use */
        for (int j = i + 1; j < g->num_nodes; j++) {
            SNEPPXExprNode* m = g->nodes[j];
            if (m->fuseable && m->fusion_group < 0) {
                int single_use = 1;
                for (int k = 0; k < g->num_nodes; k++) {
                    for (int ii = 0; ii < g->nodes[k]->n_inputs; ii++) {
                        if (g->nodes[k]->inputs[ii] == m && g->nodes[k] != m) {
                            single_use = 0;
                        }
                    }
                }
                if (single_use) m->fusion_group = group;
            }
        }
        group++;
    }
    g->fused_kernel_count = group;
    return group;
}

/**
 * @brief Perform Expr Print Plan.
 */
void SNEPPX_expr_print_plan(const SNEPPXExprGraph* g) {
    if (!g) return;
    printf("ExprGraph: %d nodes, fusion=%d -> %d fused kernels\n",
           g->num_nodes, g->enable_fusion, g->fused_kernel_count);
    for (int i = 0; i < g->num_nodes; i++) {
        SNEPPXExprNode* n = g->nodes[i];
        const char* opname = "unknown";
        switch (n->op) {
            case SNEPPX_EXPR_INPUT: opname = "input"; break;
            case SNEPPX_EXPR_CONST: opname = "const"; break;
            case SNEPPX_EXPR_ADD: opname = "add"; break;
            case SNEPPX_EXPR_MUL: opname = "mul"; break;
            case SNEPPX_EXPR_MATMUL: opname = "matmul"; break;
            case SNEPPX_EXPR_RELU: opname = "relu"; break;
            case SNEPPX_EXPR_GELU: opname = "gelu"; break;
            case SNEPPX_EXPR_SILU: opname = "silu"; break;
            case SNEPPX_EXPR_TANH: opname = "tanh"; break;
            case SNEPPX_EXPR_SOFTMAX: opname = "softmax"; break;
            case SNEPPX_EXPR_LAYERNORM: opname = "layernorm"; break;
            case SNEPPX_EXPR_RMSNORM: opname = "rmsnorm"; break;
            default: break;
        }
        printf("  [%d] op=%-10s group=%d fuseable=%d shape=[%d,%d]\n",
               n->id, opname, n->fusion_group, n->fuseable,
               n->shape[0], n->ndim > 1 ? n->shape[1] : 1);
    }
}

/* =========================================================================
 * Evaluation
 * ========================================================================= */

static float* alloc_payload(SNEPPXExprNode* n) {
    int sz = shape_size(n->shape, n->ndim);
    float* p = (float*)malloc((size_t)sz * sizeof(float));
    n->payload = p;
    n->materialized = 1;
    return p;
}

static void eval_node(SNEPPXExprGraph* g, SNEPPXExprNode* n);

static void eval_unary(SNEPPXExprGraph* g, SNEPPXExprNode* n, float* out) {
    eval_node(g, n->inputs[0]);
    float* in = (float*)n->inputs[0]->payload;
    int sz = shape_size(n->shape, n->ndim);
    for (int i = 0; i < sz; i++) {
        float x = in[i];
        switch (n->op) {
            case SNEPPX_EXPR_RELU: out[i] = x > 0 ? x : 0; break;
            case SNEPPX_EXPR_GELU: { float c=0.79788456f; out[i]=0.5f*x*(1+tanhf(c*(x+0.044715f*x*x*x))); break; }
            case SNEPPX_EXPR_SILU: out[i] = x/(1+expf(-x)); break;
            case SNEPPX_EXPR_TANH: out[i] = tanhf(x); break;
            case SNEPPX_EXPR_SIGMOID: out[i] = 1/(1+expf(-x)); break;
            case SNEPPX_EXPR_EXP: out[i] = expf(x); break;
            case SNEPPX_EXPR_LOG: out[i] = logf(x > 1e-12f ? x : 1e-12f); break;
            case SNEPPX_EXPR_SQRT: out[i] = sqrtf(x > 0 ? x : 0); break;
            case SNEPPX_EXPR_CLAMP: out[i] = x < 0 ? 0 : (x > 1 ? 1 : x); break;
            default: out[i] = x; break;
        }
    }
}

static void eval_binary(SNEPPXExprGraph* g, SNEPPXExprNode* n, float* out) {
    eval_node(g, n->inputs[0]);
    eval_node(g, n->inputs[1]);
    float* a = (float*)n->inputs[0]->payload;
    float* b = (float*)n->inputs[1]->payload;
    int sz = shape_size(n->shape, n->ndim);
    for (int i = 0; i < sz; i++) {
        float av = a[i];
        float bv = (n->inputs[1]->op == SNEPPX_EXPR_CONST) ? n->inputs[1]->scalar : b[i];
        switch (n->op) {
            case SNEPPX_EXPR_ADD: out[i] = av + bv; break;
            case SNEPPX_EXPR_SUB: out[i] = av - bv; break;
            case SNEPPX_EXPR_MUL: out[i] = av * bv; break;
            case SNEPPX_EXPR_DIV: out[i] = av / (bv != 0 ? bv : 1e-12f); break;
            default: out[i] = av; break;
        }
    }
}

static void eval_node(SNEPPXExprGraph* g, SNEPPXExprNode* n) {
    if (n->materialized) return;
    float* out = alloc_payload(n);
    switch (n->op) {
        case SNEPPX_EXPR_INPUT:
        case SNEPPX_EXPR_CONST:
            if (n->op == SNEPPX_EXPR_CONST) {
                int sz = shape_size(n->shape, n->ndim);
                for (int i = 0; i < sz; i++) out[i] = n->scalar;
            }
            break;
        case SNEPPX_EXPR_ADD:
        case SNEPPX_EXPR_SUB:
        case SNEPPX_EXPR_MUL:
        case SNEPPX_EXPR_DIV:
            eval_binary(g, n, out);
            break;
        case SNEPPX_EXPR_RELU:
        case SNEPPX_EXPR_GELU:
        case SNEPPX_EXPR_SILU:
        case SNEPPX_EXPR_TANH:
        case SNEPPX_EXPR_SIGMOID:
        case SNEPPX_EXPR_EXP:
        case SNEPPX_EXPR_LOG:
        case SNEPPX_EXPR_SQRT:
        case SNEPPX_EXPR_CLAMP:
            eval_unary(g, n, out);
            break;
        case SNEPPX_EXPR_MATMUL: {
            eval_node(g, n->inputs[0]);
            eval_node(g, n->inputs[1]);
            float* a = (float*)n->inputs[0]->payload;
            float* b = (float*)n->inputs[1]->payload;
            SNEPPX_gemm(a, b, out, n->shape[0], n->shape[1],
                        n->inputs[0]->shape[1], 1.0f, 0.0f);
            break;
        }
        case SNEPPX_EXPR_BIAS_ADD: {
            eval_node(g, n->inputs[0]);
            eval_node(g, n->inputs[1]);
            float* a = (float*)n->inputs[0]->payload;
            float* w = (float*)n->inputs[1]->payload;
            float* bias = (n->n_inputs > 2) ? (float*)n->inputs[2]->payload : NULL;
            SNEPPX_fused_linear_bias_avx2(a, w, bias, out,
                                          n->shape[0], n->shape[1],
                                          n->inputs[0]->shape[1], n->act_kind);
            break;
        }
        case SNEPPX_EXPR_SUM:
        case SNEPPX_EXPR_MEAN: {
            eval_node(g, n->inputs[0]);
            float* in = (float*)n->inputs[0]->payload;
            int sz = shape_size(n->inputs[0]->shape, n->inputs[0]->ndim);
            float s = 0;
            for (int i = 0; i < sz; i++) s += in[i];
            if (n->op == SNEPPX_EXPR_MEAN) s /= (sz > 0 ? sz : 1);
            out[0] = s;
            break;
        }
        default:
            break;
    }
}

/**
 * @brief Perform Expr Eval.
 *
 * @param g [out] G value.
 * @param node [out] Node value.
 * @param out_data [out] Out Data value.
 * @param out_shape [out] Out Shape value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_expr_eval(SNEPPXExprGraph* g, SNEPPXExprNode* node, float** out_data,
                     int* out_shape, int* out_ndim) {
    if (!g || !node) return -1;
    eval_node(g, node);
    *out_data = (float*)node->payload;
    for (int i = 0; i < node->ndim; i++) out_shape[i] = node->shape[i];
    *out_ndim = node->ndim;
    return 0;
}

/**
 * @brief Perform Expr Toposort.
 *
 * @param g [in] G value.
 * @param order [out] Order value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_expr_toposort(const SNEPPXExprGraph* g, int* order, int max_order) {
    if (!g) return 0;
    int count = 0;
    /* Simple: already added in dependency order (inputs first) */
    for (int i = 0; i < g->num_nodes && count < max_order; i++) {
        order[count++] = g->nodes[i]->id;
    }
    return count;
}

/**
 * @brief Perform Expr Mem Estimate.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_expr_mem_estimate(const SNEPPXExprGraph* g) {
    if (!g) return 0;
    size_t total = 0;
    for (int i = 0; i < g->num_nodes; i++) {
        SNEPPXExprNode* n = g->nodes[i];
        total += (size_t)shape_size(n->shape, n->ndim) * sizeof(float);
    }
    return total;
}
