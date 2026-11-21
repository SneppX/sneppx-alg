#include "gradient_optimization_suite.h"
#include "multidimensional_tensor_engine.h"
#include <stdlib.h>
#include <string.h>
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



/* Real gradient-optimizer implementation. Supports SGD (with momentum and
 * weight decay, optionally Nesterov) and Adam / AdamW (with bias correction).
 * Buffers are kept per-parameter and index-aligned across step() calls. */

typedef struct SNEPPXGradientOptimizer {
    SNEPPXOptimizerType type;
    double lr;
    double weight_decay;
    double momentum;
    double beta1;
    double beta2;
    double epsilon;
    int nesterov;
    size_t step_count;
    size_t num_params;
    float* buf;   /* momentum (SGD) or m (Adam) */
    float* buf2;  /* v (Adam) */
} SNEPPXGradientOptimizer;

static float get_flat(const SNEPPXTensor* t, size_t flat) {
    size_t idx[8]; size_t tmp = flat;
    for (long i = (long)t->ndim - 1; i >= 0; i--) {
        idx[i] = (t->strides[i] > 0) ? (tmp / t->strides[i]) % t->shape[i] : 0;
        if (t->strides[i] > 0) tmp %= t->strides[i];
    }
    return SNEPPX_tensor_get_f32(t, idx);
}

static void set_flat(SNEPPXTensor* t, size_t flat, float v) {
    size_t idx[8]; size_t tmp = flat;
    for (long i = (long)t->ndim - 1; i >= 0; i--) {
        idx[i] = (t->strides[i] > 0) ? (tmp / t->strides[i]) % t->shape[i] : 0;
        if (t->strides[i] > 0) tmp %= t->strides[i];
    }
    SNEPPX_tensor_set_f32(t, idx, v);
}

static void ensure_buffers(SNEPPXGradientOptimizer* opt, size_t num_params) {
    if (num_params != opt->num_params) {
        opt->buf = (float*)realloc(opt->buf, num_params * sizeof(float));
        opt->buf2 = (float*)realloc(opt->buf2, num_params * sizeof(float));
        if (opt->buf) memset(opt->buf, 0, num_params * sizeof(float));
        if (opt->buf2) memset(opt->buf2, 0, num_params * sizeof(float));
        opt->num_params = num_params;
    }
}

/**
 * @brief Create Grad Opt.
 *
 * @param type [in] Type value.
 * @param lr [in] Lr value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXGradientOptimizer* SNEPPX_grad_opt_create(SNEPPXOptimizerType type, double lr, double weight_decay) {
    SNEPPXGradientOptimizer* opt = (SNEPPXGradientOptimizer*)calloc(1, sizeof(SNEPPXGradientOptimizer));
    if (!opt) return NULL;
    opt->type = type;
    opt->lr = lr;
    opt->weight_decay = weight_decay;
    opt->momentum = 0.0;
    opt->beta1 = 0.9;
    opt->beta2 = 0.999;
    opt->epsilon = 1e-8;
    opt->nesterov = 0;
    opt->step_count = 0;
    opt->num_params = 0;
    opt->buf = NULL;
    opt->buf2 = NULL;
    return opt;
}

/**
 * @brief Destroy Grad Opt.
 */
void SNEPPX_grad_opt_destroy(SNEPPXGradientOptimizer* opt) {
    if (!opt) return;
    free(opt->buf);
    free(opt->buf2);
    free(opt);
}

/**
 * @brief Perform Grad Opt Step.
 *
 * @param opt [out] Opt value.
 * @param params [out] Params value.
 * @param grads [out] Grads value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_grad_opt_step(SNEPPXGradientOptimizer* opt, SNEPPXTensor** params, SNEPPXTensor** grads, size_t num_params) {
    if (!opt || !params || !grads) return -1;
    ensure_buffers(opt, num_params);
    opt->step_count++;
    int is_adam = (opt->type == SNEPPX_OPTIMIZER_ADAM || opt->type == SNEPPX_OPTIMIZER_ADAMW || opt->type == SNEPPX_OPTIMIZER_RADAM || opt->type == SNEPPX_OPTIMIZER_SF_ADAMW);
    double lr = opt->lr;
    double wd = opt->weight_decay;
    for (size_t p = 0; p < num_params; p++) {
        SNEPPXTensor* param = params[p];
        SNEPPXTensor* grad = grads[p];
        if (!param || !grad) return -1;
        size_t n = param->size < grad->size ? param->size : grad->size;
        for (size_t i = 0; i < n; i++) {
            float value = get_flat(param, i);
            float g = get_flat(grad, i);
            if (wd != 0.0) g += (float)(wd * value);
            if (is_adam) {
                float* m = &opt->buf[p];
                float* v = &opt->buf2[p];
                *m = (float)(opt->beta1 * (*m) + (1.0 - opt->beta1) * g);
                *v = (float)(opt->beta2 * (*v) + (1.0 - opt->beta2) * g * g);
                double bc1 = 1.0 - pow(opt->beta1, (double)opt->step_count);
                double bc2 = 1.0 - pow(opt->beta2, (double)opt->step_count);
                float mhat = (float)(*m / bc1);
                float vhat = (float)(*v / bc2);
                float denom = (float)sqrt((double)vhat) + (float)opt->epsilon;
                float update = mhat / denom;
                if (opt->type == SNEPPX_OPTIMIZER_ADAMW || opt->type == SNEPPX_OPTIMIZER_SF_ADAMW)
                    value -= (float)(lr * (wd * value + update));
                else
                    value -= (float)(lr * update);
            } else {
                float g2 = g;
                if (opt->momentum != 0.0) {
                    float prev = opt->buf[p];
                    float cur = (float)(opt->momentum * prev + g2);
                    if (opt->nesterov) g2 = (float)(opt->momentum * prev + g2);
                    else g2 = cur;
                    opt->buf[p] = cur;
                }
                value -= (float)(lr * g2);
            }
            set_flat(param, i, value);
        }
    }
    return 0;
}

/**
 * @brief Perform Grad Opt Zero Grad.
 *
 * @param opt [out] Opt value.
 * @param params [out] Params value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_grad_opt_zero_grad(SNEPPXGradientOptimizer* opt, SNEPPXTensor** params, size_t num_params) {
    (void)opt;
    if (!params) return -1;
    for (size_t p = 0; p < num_params; p++) {
        if (!params[p]) continue;
        float* d = (float*)params[p]->data;
        for (size_t i = 0; i < params[p]->size; i++) d[i] = 0.0f;
    }
    return 0;
}

/**
 * @brief Perform Grad Opt Set Lr.
 *
 * @param opt [out] Opt value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_grad_opt_set_lr(SNEPPXGradientOptimizer* opt, double lr) {
    if (!opt) return -1;
    opt->lr = lr;
    return 0;
}

/**
 * @brief Perform Grad Opt Get Lr.
 *
 * @return The result value, or 0 on error.
 */
double SNEPPX_grad_opt_get_lr(const SNEPPXGradientOptimizer* opt) {
    return opt ? opt->lr : 0.001;
}

/**
 * @brief Perform Grad Opt Add Param Group.
 *
 * @param opt [out] Opt value.
 * @param params [out] Params value.
 * @param num_params [in] Num Params value.
 * @param lr [in] Lr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_grad_opt_add_param_group(SNEPPXGradientOptimizer* opt, SNEPPXTensor** params, size_t num_params, double lr, double weight_decay) {
    (void)params; (void)num_params;
    if (!opt) return -1;
    opt->lr = lr;
    opt->weight_decay = weight_decay;
    return 0;
}

/**
 * @brief Perform Grad Opt Clip Grad Norm.
 *
 * @param grads [out] Grads value.
 * @param num_grads [in] Num Grads value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_grad_opt_clip_grad_norm(SNEPPXTensor** grads, size_t num_grads, double max_norm) {
    if (!grads) return -1;
    double total = 0.0;
    for (size_t g = 0; g < num_grads; g++) {
        if (!grads[g]) continue;
        float* d = (float*)grads[g]->data;
        for (size_t i = 0; i < grads[g]->size; i++) total += (double)d[i] * d[i];
    }
    double norm = sqrt(total);
    if (norm > max_norm && norm > 0.0) {
        double scale = max_norm / norm;
        for (size_t g = 0; g < num_grads; g++) {
            if (!grads[g]) continue;
            float* d = (float*)grads[g]->data;
            for (size_t i = 0; i < grads[g]->size; i++) d[i] = (float)(d[i] * scale);
        }
    }
    return 0;
}

/**
 * @brief Perform Grad Opt Clip Grad Value.
 *
 * @param grads [out] Grads value.
 * @param num_grads [in] Num Grads value.
 * @param min_val [in] Min Val value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_grad_opt_clip_grad_value(SNEPPXTensor** grads, size_t num_grads, double min_val, double max_val) {
    if (!grads) return -1;
    for (size_t g = 0; g < num_grads; g++) {
        if (!grads[g]) continue;
        float* d = (float*)grads[g]->data;
        for (size_t i = 0; i < grads[g]->size; i++) {
            if (d[i] < min_val) d[i] = (float)min_val;
            else if (d[i] > max_val) d[i] = (float)max_val;
        }
    }
    return 0;
}

/* The LR-scheduler symbols (SNEPPX_lr_scheduler_*) are declared in
 * gradient_optimization_suite.h and implemented for real in optimizer.c;
 * they are intentionally NOT redefined here so the real versions are linked. */

