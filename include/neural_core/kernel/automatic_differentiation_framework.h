#ifndef SNEPPX_AUTODIFF_H
#define SNEPPX_AUTODIFF_H

#ifdef __cplusplus
extern "C" {
#endif

#include "multidimensional_tensor_engine.h"
#include <stddef.h>

/*
 * SNEPPX - Automatic Differentiation Framework
 *
 * WHAT
 *   Automatic Differentiation Framework.
 *
 * CONCEPT
 *   Provides the Automatic Differentiation Framework.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef void (*BackwardFn)(void* ctx, SNEPPXTensor* grad_output);

typedef struct SNEPPXVariable SNEPPXVariable;

typedef void* (*RecomputeCtxFn)(SNEPPXVariable* var, size_t* params, size_t param_count);

typedef struct SNEPPXVariable {
    SNEPPXTensor* data;
    SNEPPXTensor* grad;
    int requires_grad;
    int checkpointed;
    BackwardFn backward_fn;
    void* backward_ctx;
    void (*free_ctx)(void*);
    RecomputeCtxFn recompute_ctx;
    int ref_count;
    struct SNEPPXVariable** parents;
    size_t num_parents;
    size_t params[8];
    size_t param_count;
} SNEPPXVariable;

typedef struct {
    SNEPPXVariable** vars;
    size_t num_vars;
    size_t capacity;
    int checkpointing;
} SNEPPXTape;

/**
 * @brief Create Variable.
 *
 * @param data [out] Data value.
 * @param requires_grad [in] Requires Grad value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_variable_create(SNEPPXTensor* data, int requires_grad);
/**
 * @brief Destroy Variable.
 *
 * @param var [out] Var value.
 */
void          SNEPPX_variable_destroy(SNEPPXVariable* var);
/**
 * @brief Perform Variable Set Requires Grad.
 *
 * @param var [out] Var value.
 * @param requires_grad [in] Requires Grad value.
 */
void          SNEPPX_variable_set_requires_grad(SNEPPXVariable* var, int requires_grad);
/**
 * @brief Perform Variable Detach.
 *
 * @param var [out] Var value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_variable_detach(SNEPPXVariable* var);
/**
 * @brief Perform Variable Copy.
 *
 * @param var [out] Var value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_variable_copy(SNEPPXVariable* var);
/**
 * @brief Perform Variable Zero Grad.
 *
 * @param var [out] Var value.
 */
void          SNEPPX_variable_zero_grad(SNEPPXVariable* var);
/**
 * @brief Perform Variable Item.
 *
 * @param var [out] Var value.
 *
 * @return The result value, or 0 on error.
 */
float         SNEPPX_variable_item(SNEPPXVariable* var);
/**
 * @brief Perform Variable Numel.
 *
 * @param var [out] Var value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t        SNEPPX_variable_numel(SNEPPXVariable* var);

/**
 * @brief Create Tape.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTape* SNEPPX_tape_create(void);
/**
 * @brief Destroy Tape.
 *
 * @param tape [out] Tape value.
 */
void      SNEPPX_tape_destroy(SNEPPXTape* tape);
/**
 * @brief Perform Tape Record.
 *
 * @param tape [out] Tape value.
 * @param var [out] Var value.
 */
void      SNEPPX_tape_record(SNEPPXTape* tape, SNEPPXVariable* var);
/**
 * @brief Run the backward pass for Tape.
 *
 * @param tape [out] Tape value.
 * @param loss [out] Loss value.
 */
void      SNEPPX_tape_backward(SNEPPXTape* tape, SNEPPXVariable* loss);
/**
 * @brief Perform Tape Zero Grad.
 *
 * @param tape [out] Tape value.
 */
void      SNEPPX_tape_zero_grad(SNEPPXTape* tape);
/**
 * @brief Perform Tape Global Norm.
 *
 * @param tape [out] Tape value.
 *
 * @return The result value, or 0 on error.
 */
float     SNEPPX_tape_global_norm(SNEPPXTape* tape);
/**
 * @brief Perform Tape Clip Grad Norm.
 *
 * @param tape [out] Tape value.
 * @param max_norm [in] Max Norm value.
 */
void      SNEPPX_tape_clip_grad_norm(SNEPPXTape* tape, float max_norm);

/**
 * @brief Perform No Grad Enter.
 */
void  SNEPPX_no_grad_enter(void);
/**
 * @brief Perform No Grad Exit.
 */
void  SNEPPX_no_grad_exit(void);
/**
 * @brief Perform No Grad Is Active.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_no_grad_is_active(void);

/**
 * @brief Perform Tape Checkpoint Begin.
 *
 * @param tape [out] Tape value.
 */
void  SNEPPX_tape_checkpoint_begin(SNEPPXTape* tape);
/**
 * @brief Perform Tape Checkpoint End.
 *
 * @param tape [out] Tape value.
 */
void  SNEPPX_tape_checkpoint_end(SNEPPXTape* tape);

/**
 * @brief Add Add.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 * @param b [out] B value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_add(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b);
/**
 * @brief Perform Sub.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 * @param b [out] B value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_sub(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b);
/**
 * @brief Perform Mul.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 * @param b [out] B value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_mul(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b);
/**
 * @brief Perform Div.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 * @param b [out] B value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_div(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b);
/**
 * @brief Perform Pow.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 * @param b [out] B value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_pow(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b);
/**
 * @brief Perform Neg.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_neg(SNEPPXTape* tape, SNEPPXVariable* a);
/**
 * @brief Perform Matmul.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 * @param b [out] B value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_matmul(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b);
/**
 * @brief Perform Mse Loss.
 *
 * @param tape [out] Tape value.
 * @param pred [out] Pred value.
 * @param target [out] Target value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_mse_loss(SNEPPXTape* tape, SNEPPXVariable* pred, SNEPPXVariable* target);
/**
 * @brief Perform Relu.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_relu(SNEPPXTape* tape, SNEPPXVariable* a);
/**
 * @brief Perform Fake Quant (QAT).
 *
 * Forward quantizes the input to `bits` with `scale` and de-quantizes; the
 * backward is the straight-through estimator (unit gradient w.r.t. input).
 *
 * @param tape [out] Tape value.
 * @param a [in] Input value.
 * @param scale [in] Scale value (> 0).
 * @param bits [in] Bit width (2..16).
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_fake_quant(SNEPPXTape* tape, SNEPPXVariable* a, float scale, int bits);
/**
 * @brief Perform Gelu.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_gelu(SNEPPXTape* tape, SNEPPXVariable* a);
/**
 * @brief Perform Silu.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_silu(SNEPPXTape* tape, SNEPPXVariable* a);
/**
 * @brief Perform Sigmoid.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_sigmoid(SNEPPXTape* tape, SNEPPXVariable* a);
/**
 * @brief Perform Tanh.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_tanh(SNEPPXTape* tape, SNEPPXVariable* a);
/**
 * @brief Perform Softmax.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 * @param dim [in] Dim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_softmax(SNEPPXTape* tape, SNEPPXVariable* a, size_t dim);
/**
 * @brief Perform Exp.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_exp(SNEPPXTape* tape, SNEPPXVariable* a);
/**
 * @brief Perform Log.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_log(SNEPPXTape* tape, SNEPPXVariable* a);
/**
 * @brief Perform Sum.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 * @param dim [in] Dim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_sum(SNEPPXTape* tape, SNEPPXVariable* a, size_t dim);
/**
 * @brief Perform Mean.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 * @param dim [in] Dim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_mean(SNEPPXTape* tape, SNEPPXVariable* a, size_t dim);
/**
 * @brief Perform Transpose.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 * @param dim1 [in] Dim1 value.
 * @param dim2 [in] Dim2 value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_transpose(SNEPPXTape* tape, SNEPPXVariable* a, size_t dim1, size_t dim2);
/**
 * @brief Perform Reshape.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 * @param shape [in] Shape value.
 * @param ndim [in] Ndim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_reshape(SNEPPXTape* tape, SNEPPXVariable* a, const size_t* shape, size_t ndim);
/**
 * @brief Perform Rope.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 * @param cos_table [out] Cos Table value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_rope(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXTensor* cos_table);
/**
 * @brief Perform Dropout.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 * @param rate [in] Rate value.
 * @param seed [in] Seed value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_dropout(SNEPPXTape* tape, SNEPPXVariable* a, float rate, unsigned int seed);
/**
 * @brief Perform Layer Norm.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 * @param gamma [out] Gamma value.
 * @param beta [out] Beta value.
 * @param eps [in] Eps value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_layer_norm(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* gamma, SNEPPXVariable* beta, float eps);
/**
 * @brief Perform Conv2d.
 *
 * @param tape [out] Tape value.
 * @param input [out] Input value.
 * @param kernel [out] Kernel value.
 * @param stride_h [in] Stride H value.
 * @param stride_w [in] Stride W value.
 * @param pad_h [in] Pad H value.
 * @param pad_w [in] Pad W value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_conv2d(SNEPPXTape* tape, SNEPPXVariable* input, SNEPPXVariable* kernel, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w);
/**
 * @brief Perform Concat.
 *
 * @param tape [out] Tape value.
 * @param vars [out] Vars value.
 * @param num_vars [in] Num Vars value.
 * @param dim [in] Dim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_concat(SNEPPXTape* tape, SNEPPXVariable** vars, size_t num_vars, size_t dim);

/**
 * @brief Perform Minimum.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 * @param b [out] B value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_minimum(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b);
/**
 * @brief Perform Maximum.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 * @param b [out] B value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_maximum(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b);
/**
 * @brief Perform Sqrt.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_sqrt(SNEPPXTape* tape, SNEPPXVariable* a);
/**
 * @brief Perform Abs.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_abs(SNEPPXTape* tape, SNEPPXVariable* a);
/**
 * @brief Perform Sin.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_sin(SNEPPXTape* tape, SNEPPXVariable* a);
/**
 * @brief Perform Cos.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_cos(SNEPPXTape* tape, SNEPPXVariable* a);
/**
 * @brief Perform Tan.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_tan(SNEPPXTape* tape, SNEPPXVariable* a);
/**
 * @brief Perform Asin.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_asin(SNEPPXTape* tape, SNEPPXVariable* a);
/**
 * @brief Perform Acos.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_acos(SNEPPXTape* tape, SNEPPXVariable* a);
/**
 * @brief Perform Atan.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_atan(SNEPPXTape* tape, SNEPPXVariable* a);
/**
 * @brief Perform Sinh.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_sinh(SNEPPXTape* tape, SNEPPXVariable* a);
/**
 * @brief Perform Cosh.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_cosh(SNEPPXTape* tape, SNEPPXVariable* a);
/**
 * @brief Perform Var.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 * @param dim [in] Dim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_var(SNEPPXTape* tape, SNEPPXVariable* a, size_t dim);
/**
 * @brief Perform Std.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 * @param dim [in] Dim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_std(SNEPPXTape* tape, SNEPPXVariable* a, size_t dim);
/**
 * @brief Perform Cross Entropy.
 *
 * @param tape [out] Tape value.
 * @param pred [out] Pred value.
 * @param target [out] Target value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_cross_entropy(SNEPPXTape* tape, SNEPPXVariable* pred, SNEPPXVariable* target);
/**
 * @brief Perform Nll Loss.
 *
 * @param tape [out] Tape value.
 * @param pred [out] Pred value.
 * @param target [out] Target value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_nll_loss(SNEPPXTape* tape, SNEPPXVariable* pred, SNEPPXVariable* target);
/**
 * @brief Perform Bce Loss.
 *
 * @param tape [out] Tape value.
 * @param pred [out] Pred value.
 * @param target [out] Target value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_bce_loss(SNEPPXTape* tape, SNEPPXVariable* pred, SNEPPXVariable* target);
/**
 * @brief Perform Embedding.
 *
 * @param tape [out] Tape value.
 * @param weight [out] Weight value.
 * @param indices [out] Indices value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_embedding(SNEPPXTape* tape, SNEPPXVariable* weight, SNEPPXVariable* indices);

/**
 * @brief Perform Log Softmax.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 * @param dim [in] Dim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_log_softmax(SNEPPXTape* tape, SNEPPXVariable* a, size_t dim);
/**
 * @brief Sign Sign.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_sign(SNEPPXTape* tape, SNEPPXVariable* a);
/**
 * @brief Perform Floor.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_floor(SNEPPXTape* tape, SNEPPXVariable* a);
/**
 * @brief Perform Ceil.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_ceil(SNEPPXTape* tape, SNEPPXVariable* a);
/**
 * @brief Perform Round.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_round(SNEPPXTape* tape, SNEPPXVariable* a);
/**
 * @brief Perform Trunc.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_trunc(SNEPPXTape* tape, SNEPPXVariable* a);
/**
 * @brief Perform Batch Norm.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 * @param gamma [out] Gamma value.
 * @param beta [out] Beta value.
 * @param running_mean [out] Running Mean value.
 * @param running_var [out] Running Var value.
 * @param eps [in] Eps value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_batch_norm(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* gamma, SNEPPXVariable* beta, SNEPPXVariable* running_mean, SNEPPXVariable* running_var, float eps);

/* ---------- comparison ops (zero gradient) ---------- */
/**
 * @brief Perform Eq.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 * @param b [out] B value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_eq(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b);
/**
 * @brief Perform Ne.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 * @param b [out] B value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_ne(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b);
/**
 * @brief Perform Lt.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 * @param b [out] B value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_lt(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b);
/**
 * @brief Perform Le.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 * @param b [out] B value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_le(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b);
/**
 * @brief Perform Gt.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 * @param b [out] B value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_gt(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b);
/**
 * @brief Perform Ge.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 * @param b [out] B value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_ge(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b);

/* ---------- convolution / scan ---------- */
/**
 * @brief Perform Conv1d.
 *
 * @param tape [out] Tape value.
 * @param input [out] Input value.
 * @param kernel [out] Kernel value.
 * @param stride [in] Stride value.
 * @param padding [in] Padding value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_conv1d(SNEPPXTape* tape, SNEPPXVariable* input, SNEPPXVariable* kernel, size_t stride, size_t padding);

/* ---------- cumulative ops ---------- */
/**
 * @brief Perform Cumsum.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 * @param dim [in] Dim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_cumsum(SNEPPXTape* tape, SNEPPXVariable* a, size_t dim);
/**
 * @brief Perform Cumprod.
 *
 * @param tape [out] Tape value.
 * @param a [out] A value.
 * @param dim [in] Dim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_cumprod(SNEPPXTape* tape, SNEPPXVariable* a, size_t dim);


#ifdef __cplusplus
}
#endif
#endif
