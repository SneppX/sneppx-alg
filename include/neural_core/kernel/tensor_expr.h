#ifndef SNEPPX_TENSOR_EXPR_H
#define SNEPPX_TENSOR_EXPR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
/*
 * SNEPPX - Tensor Expr
 *
 * WHAT
 *   Tensor Expr.
 *
 * CONCEPT
 *   Provides tensor operations.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif

/* =========================================================================
 * Lazy Tensor Expression Engine + Operator Fusion Planner
 *
 * A deferred-execution IR: tensor operations build a DAG of ExprNodes
 * instead of materializing results immediately. The fusion planner
 * groups eligible elementwise/reduction ops into fused kernels and
 * schedules matmuls/GEMMs on the SIMD/CPU backend, enabling torch.compile
 * style "graph capture -> fusion -> execute" semantics.
 * ========================================================================= */

typedef enum {
    SNEPPX_EXPR_INPUT = 0,      /* leaf: a materialized tensor */
    SNEPPX_EXPR_CONST,          /* scalar constant broadcast */
    SNEPPX_EXPR_ADD,
    SNEPPX_EXPR_SUB,
    SNEPPX_EXPR_MUL,
    SNEPPX_EXPR_DIV,
    SNEPPX_EXPR_MATMUL,
    SNEPPX_EXPR_RELU,
    SNEPPX_EXPR_GELU,
    SNEPPX_EXPR_SILU,
    SNEPPX_EXPR_TANH,
    SNEPPX_EXPR_SIGMOID,
    SNEPPX_EXPR_SOFTMAX,
    SNEPPX_EXPR_LAYERNORM,
    SNEPPX_EXPR_RMSNORM,
    SNEPPX_EXPR_DROPOUT,
    SNEPPX_EXPR_RESHAPE,
    SNEPPX_EXPR_TRANSPOSE,
    SNEPPX_EXPR_CONCAT,
    SNEPPX_EXPR_SUM,
    SNEPPX_EXPR_MEAN,
    SNEPPX_EXPR_BIAS_ADD,        /* fused linear bias */
    SNEPPX_EXPR_CLAMP,
    SNEPPX_EXPR_POW,
    SNEPPX_EXPR_SQRT,
    SNEPPX_EXPR_EXP,
    SNEPPX_EXPR_LOG,
    SNEPPX_EXPR_GATHER,
    SNEPPX_EXPR_SLICE
} SNEPPXExprOp;

typedef struct SNEPPXExprNode SNEPPXExprNode;

struct SNEPPXExprNode {
    SNEPPXExprOp op;
    int id;
    int n_inputs;
    SNEPPXExprNode** inputs;
    /* Shape (4D max for simplicity, -1 = dynamic) */
    int shape[4];
    int ndim;
    float scalar;            /* for CONST / POW exponent */
    int act_kind;            /* for fused activations */
    void* payload;           /* optional materialized data */
    int materialized;        /* 1 if payload holds output */
    /* fusion metadata */
    int fusion_group;        /* group id for kernel fusion */
    int fuseable;            /* elementwise/reduction eligible */
};

typedef struct {
    SNEPPXExprNode** nodes;
    int num_nodes;
    int capacity;
    int next_id;
    int enable_fusion;       /* 1 = plan fusions */
    int fused_kernel_count;  /* number of fused kernels after plan() */
} SNEPPXExprGraph;

/* Graph lifecycle */
/**
 * @brief Create Expr Graph.
 *
 * @param enable_fusion [in] Enable Fusion value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXExprGraph* SNEPPX_expr_graph_create(int enable_fusion);
/**
 * @brief Destroy Expr Graph.
 *
 * @param g [out] G value.
 */
void SNEPPX_expr_graph_destroy(SNEPPXExprGraph* g);

/* Node constructors */
/**
 * @brief Perform Expr Input.
 *
 * @param g [out] G value.
 * @param shape [in] Shape value.
 * @param ndim [in] Ndim value.
 * @param data [out] Data value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXExprNode* SNEPPX_expr_input(SNEPPXExprGraph* g, const int* shape, int ndim, void* data);
/**
 * @brief Perform Expr Const.
 *
 * @param g [out] G value.
 * @param value [in] Value value.
 * @param shape [in] Shape value.
 * @param ndim [in] Ndim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXExprNode* SNEPPX_expr_const(SNEPPXExprGraph* g, float value, const int* shape, int ndim);
/**
 * @brief Perform Expr Binary.
 *
 * @param g [out] G value.
 * @param op [in] Op value.
 * @param a [out] A value.
 * @param b [out] B value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXExprNode* SNEPPX_expr_binary(SNEPPXExprGraph* g, SNEPPXExprOp op,
                                    SNEPPXExprNode* a, SNEPPXExprNode* b);
/**
 * @brief Perform Expr Unary.
 *
 * @param g [out] G value.
 * @param op [in] Op value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXExprNode* SNEPPX_expr_unary(SNEPPXExprGraph* g, SNEPPXExprOp op, SNEPPXExprNode* a);
/**
 * @brief Perform Expr Matmul.
 *
 * @param g [out] G value.
 * @param a [out] A value.
 * @param b [out] B value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXExprNode* SNEPPX_expr_matmul(SNEPPXExprGraph* g, SNEPPXExprNode* a, SNEPPXExprNode* b);
/**
 * @brief Perform Expr Activation.
 *
 * @param g [out] G value.
 * @param op [in] Op value.
 * @param a [out] A value.
 * @param act_kind [in] Act Kind value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXExprNode* SNEPPX_expr_activation(SNEPPXExprGraph* g, SNEPPXExprOp op,
                                       SNEPPXExprNode* a, int act_kind);
/**
 * @brief Perform Expr Reduction.
 *
 * @param g [out] G value.
 * @param op [in] Op value.
 * @param a [out] A value.
 * @param axis [in] Axis value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXExprNode* SNEPPX_expr_reduction(SNEPPXExprGraph* g, SNEPPXExprOp op,
                                       SNEPPXExprNode* a, int axis);
/**
 * @brief Perform Expr Fused Linear Bias.
 *
 * @param g [out] G value.
 * @param a [out] A value.
 * @param w [out] W value.
 * @param bias [out] Bias value.
 * @param act_kind [in] Act Kind value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXExprNode* SNEPPX_expr_fused_linear_bias(SNEPPXExprGraph* g,
                                              SNEPPXExprNode* a, SNEPPXExprNode* w,
                                              SNEPPXExprNode* bias, int act_kind);

/* Fusion planner: assigns fusion_group ids, returns count of fused kernels */
/**
 * @brief Perform Expr Plan Fusions.
 *
 * @param g [out] G value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_expr_plan_fusions(SNEPPXExprGraph* g);

/* Print a textual graph + fusion plan (for debugging) */
/**
 * @brief Perform Expr Print Plan.
 *
 * @param g [in] G value.
 */
void SNEPPX_expr_print_plan(const SNEPPXExprGraph* g);

/* Materialize a node (recursively evaluates; fused groups run as one pass) */
/**
 * @brief Perform Expr Eval.
 *
 * @param g [out] G value.
 * @param node [out] Node value.
 * @param out_data [out] Out Data value.
 * @param out_shape [out] Out Shape value.
 * @param out_ndim [out] Out Ndim value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_expr_eval(SNEPPXExprGraph* g, SNEPPXExprNode* node, float** out_data,
                     int* out_shape, int* out_ndim);

/* Topologically sort nodes (fill `order` with node ids, returns count) */
/**
 * @brief Perform Expr Toposort.
 *
 * @param g [in] G value.
 * @param order [out] Order value.
 * @param max_order [in] Max Order value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_expr_toposort(const SNEPPXExprGraph* g, int* order, int max_order);

/* Memory estimate (bytes) for materializing the full graph */
/**
 * @brief Perform Expr Mem Estimate.
 *
 * @param g [in] G value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_expr_mem_estimate(const SNEPPXExprGraph* g);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_TENSOR_EXPR_H */
