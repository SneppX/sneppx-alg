#include "gradient_optimization_suite.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
/*
 * SNEPPX - Optimizer Library
 *
 * WHAT
 *   Optimizer Library.
 *
 * CONCEPT
 *   Optimizer Library implementation.
 *
 * ROLE
 *   Core kernel module used throughout the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal kernel module).
 */



/**
 * @brief Perform Optimizer Config Default.
 *
 * @return The result value, or 0 on error.
 */
SNEPPXOptimizerConfig SNEPPX_optimizer_config_default(void) {
    SNEPPXOptimizerConfig cfg;
    cfg.learning_rate = 0.01f;
    cfg.momentum = 0.9f;
    cfg.weight_decay = 1e-4f;
    cfg.grad_clip = 1.0f;
    cfg.type = SNEPPX_OPTIMIZER_SGD;
    cfg.beta1 = 0.9f;
    cfg.beta2 = 0.999f;
    cfg.epsilon = 1e-8f;
    cfg.dampening = 0.0f;
    cfg.nesterov = 0;
    cfg.rho = 0.9f;
    return cfg;
}

static SNEPPXTensor** alloc_bufs(size_t n, size_t sz, size_t ndim, const size_t* shape) {
    SNEPPXTensor** bufs = (SNEPPXTensor**)SNEPPX_malloc(n * sizeof(SNEPPXTensor*), 64);
    if (!bufs) return NULL;
    memset(bufs, 0, n * sizeof(SNEPPXTensor*));
    for (size_t i = 0; i < n; i++) {
        bufs[i] = SNEPPX_tensor_zeros(shape, ndim, SNEPPX_FLOAT32);
    }
    return bufs;
}

static void free_bufs(SNEPPXTensor** bufs, size_t n) {
    if (!bufs) return;
    for (size_t i = 0; i < n; i++)
        if (bufs[i]) SNEPPX_tensor_destroy(bufs[i]);
    SNEPPX_free(bufs, n * sizeof(SNEPPXTensor*));
}

/**
 * @brief Create Optimizer.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXOptimizer* SNEPPX_optimizer_create(const SNEPPXOptimizerConfig* config) {
    if (!config) return NULL;
    SNEPPXOptimizer* opt = (SNEPPXOptimizer*)SNEPPX_malloc(sizeof(SNEPPXOptimizer), 64);
    if (!opt) return NULL;
    memset(opt, 0, sizeof(SNEPPXOptimizer));
    opt->learning_rate = config->learning_rate;
    opt->momentum = config->momentum;
    opt->weight_decay = config->weight_decay;
    opt->grad_clip = config->grad_clip;
    opt->type = config->type;
    opt->beta1 = config->beta1;
    opt->beta2 = config->beta2;
    opt->epsilon = config->epsilon;
    opt->dampening = config->dampening;
    opt->nesterov = config->nesterov;
    opt->rho = config->rho;
    opt->num_params = 0;
    opt->momentum_buffers = NULL;
    opt->state_buf2 = NULL;
    opt->state_buf3 = NULL;
    opt->step_count = 0;
    return opt;
}

/**
 * @brief Destroy Optimizer.
 */
void SNEPPX_optimizer_destroy(SNEPPXOptimizer* opt) {
    if (!opt) return;
    free_bufs(opt->momentum_buffers, opt->num_params);
    free_bufs(opt->state_buf2, opt->num_params);
    free_bufs(opt->state_buf3, opt->num_params);
    SNEPPX_free(opt, sizeof(SNEPPXOptimizer));
}

static float clip_grad(float g, float clip) {
    if (g > clip) return clip;
    if (g < -clip) return -clip;
    return g;
}

static void ensure_bufs(SNEPPXOptimizer* opt, SNEPPXTensor** params, size_t n) {
    if (opt->num_params == n) return;
    free_bufs(opt->momentum_buffers, opt->num_params);
    free_bufs(opt->state_buf2, opt->num_params);
    free_bufs(opt->state_buf3, opt->num_params);
    opt->momentum_buffers = NULL;
    opt->state_buf2 = NULL;
    opt->state_buf3 = NULL;
    opt->num_params = 0;

    opt->momentum_buffers = (SNEPPXTensor**)SNEPPX_malloc(n * sizeof(SNEPPXTensor*), 64);
    if (!opt->momentum_buffers) return;
    memset(opt->momentum_buffers, 0, n * sizeof(SNEPPXTensor*));

    if (opt->type == SNEPPX_OPTIMIZER_ADAM || opt->type == SNEPPX_OPTIMIZER_ADAMW ||
        opt->type == SNEPPX_OPTIMIZER_ADAMAX || opt->type == SNEPPX_OPTIMIZER_RMSPROP ||
        opt->type == SNEPPX_OPTIMIZER_ADAGRAD || opt->type == SNEPPX_OPTIMIZER_ADADELTA) {
        opt->state_buf2 = (SNEPPXTensor**)SNEPPX_malloc(n * sizeof(SNEPPXTensor*), 64);
        if (opt->state_buf2) memset(opt->state_buf2, 0, n * sizeof(SNEPPXTensor*));
    }
    if (opt->type == SNEPPX_OPTIMIZER_ADADELTA) {
        opt->state_buf3 = (SNEPPXTensor**)SNEPPX_malloc(n * sizeof(SNEPPXTensor*), 64);
        if (opt->state_buf3) memset(opt->state_buf3, 0, n * sizeof(SNEPPXTensor*));
    }

    for (size_t i = 0; i < n; i++) {
        if (!params[i]) continue;
        opt->momentum_buffers[i] = SNEPPX_tensor_zeros(params[i]->shape, params[i]->ndim, SNEPPX_FLOAT32);
        if (opt->state_buf2) opt->state_buf2[i] = SNEPPX_tensor_zeros(params[i]->shape, params[i]->ndim, SNEPPX_FLOAT32);
        if (opt->state_buf3) opt->state_buf3[i] = SNEPPX_tensor_zeros(params[i]->shape, params[i]->ndim, SNEPPX_FLOAT32);
    }
    opt->num_params = n;
}

static void step_sgd(SNEPPXOptimizer* opt, SNEPPXTensor** params, SNEPPXTensor** grads, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (!params[i] || !grads[i]) continue;
        float* pd = (float*)params[i]->data;
        float* gd = (float*)grads[i]->data;
        float lr = opt->learning_rate, wd = opt->weight_decay, clip = opt->grad_clip;
        float mu = opt->momentum, dp = opt->dampening;
        float* md = opt->momentum_buffers[i] ? (float*)opt->momentum_buffers[i]->data : NULL;
        size_t sz = params[i]->size;

        if (md && mu > 0) {
            for (size_t j = 0; j < sz; j++) {
                float g = clip_grad(gd[j], clip);
                g += wd * pd[j];
                md[j] = mu * md[j] - lr * g;
                pd[j] += md[j];
            }
        } else {
            for (size_t j = 0; j < sz; j++) {
                float g = clip_grad(gd[j], clip);
                g += wd * pd[j];
                pd[j] -= lr * g;
            }
        }
    }
}

static void step_adam(SNEPPXOptimizer* opt, SNEPPXTensor** params, SNEPPXTensor** grads, size_t n, int decoupled_wd) {
    float lr = opt->learning_rate, b1 = opt->beta1, b2 = opt->beta2;
    float eps = opt->epsilon, wd = opt->weight_decay, clip = opt->grad_clip;
    float bias1 = 1.0f, bias2 = 1.0f;
    for (size_t t = 0; t < opt->step_count; t++) { bias1 *= b1; bias2 *= b2; }
    bias1 = 1.0f - bias1;
    bias2 = 1.0f - bias2;

    for (size_t i = 0; i < n; i++) {
        if (!params[i] || !grads[i]) continue;
        float* pd = (float*)params[i]->data;
        float* gd = (float*)grads[i]->data;
        float* m1 = (float*)opt->momentum_buffers[i]->data;
        float* m2 = (float*)opt->state_buf2[i]->data;
        size_t sz = params[i]->size;

        for (size_t j = 0; j < sz; j++) {
            float g = clip_grad(gd[j], clip);
            m1[j] = b1 * m1[j] + (1.0f - b1) * g;
            m2[j] = b2 * m2[j] + (1.0f - b2) * g * g;
            float m1_hat = m1[j] / bias1;
            float m2_hat = m2[j] / bias2;
            float step = lr * m1_hat / (sqrtf(m2_hat) + eps);
            if (decoupled_wd) step += lr * wd * pd[j];
            pd[j] -= step;
            if (!decoupled_wd) pd[j] -= lr * wd * pd[j];
        }
    }
}

static void step_adamax(SNEPPXOptimizer* opt, SNEPPXTensor** params, SNEPPXTensor** grads, size_t n) {
    float lr = opt->learning_rate, b1 = opt->beta1, b2 = opt->beta2;
    float eps = opt->epsilon, wd = opt->weight_decay, clip = opt->grad_clip;
    float bias1 = 1.0f;
    for (size_t t = 0; t < opt->step_count; t++) bias1 *= b1;
    bias1 = 1.0f - bias1;

    for (size_t i = 0; i < n; i++) {
        if (!params[i] || !grads[i]) continue;
        float* pd = (float*)params[i]->data;
        float* gd = (float*)grads[i]->data;
        float* m1 = (float*)opt->momentum_buffers[i]->data;
        float* m2 = (float*)opt->state_buf2[i]->data;
        size_t sz = params[i]->size;

        for (size_t j = 0; j < sz; j++) {
            float g = clip_grad(gd[j], clip);
            m1[j] = b1 * m1[j] + (1.0f - b1) * g;
            m2[j] = fmaxf(b2 * m2[j], fabsf(g));
            pd[j] -= lr * m1[j] / (bias1 * (m2[j] + eps)) + lr * wd * pd[j];
        }
    }
}

static void step_rmsprop(SNEPPXOptimizer* opt, SNEPPXTensor** params, SNEPPXTensor** grads, size_t n) {
    float lr = opt->learning_rate, alpha = opt->beta1, eps = opt->epsilon;
    float wd = opt->weight_decay, clip = opt->grad_clip, mu = opt->momentum;

    for (size_t i = 0; i < n; i++) {
        if (!params[i] || !grads[i]) continue;
        float* pd = (float*)params[i]->data;
        float* gd = (float*)grads[i]->data;
        float* v = (float*)opt->state_buf2[i]->data;
        float* m = opt->momentum_buffers[i] ? (float*)opt->momentum_buffers[i]->data : NULL;
        size_t sz = params[i]->size;

        for (size_t j = 0; j < sz; j++) {
            float g = clip_grad(gd[j], clip);
            g += wd * pd[j];
            v[j] = alpha * v[j] + (1.0f - alpha) * g * g;
            float step = lr * g / (sqrtf(v[j]) + eps);
            if (m && mu > 0) {
                m[j] = mu * m[j] + step;
                pd[j] -= m[j];
            } else {
                pd[j] -= step;
            }
        }
    }
}

static void step_adagrad(SNEPPXOptimizer* opt, SNEPPXTensor** params, SNEPPXTensor** grads, size_t n) {
    float lr = opt->learning_rate, eps = opt->epsilon, wd = opt->weight_decay, clip = opt->grad_clip;

    for (size_t i = 0; i < n; i++) {
        if (!params[i] || !grads[i]) continue;
        float* pd = (float*)params[i]->data;
        float* gd = (float*)grads[i]->data;
        float* v = (float*)opt->state_buf2[i]->data;
        size_t sz = params[i]->size;

        for (size_t j = 0; j < sz; j++) {
            float g = clip_grad(gd[j], clip);
            g += wd * pd[j];
            v[j] += g * g;
            pd[j] -= lr * g / (sqrtf(v[j]) + eps);
        }
    }
}

static void step_adadelta(SNEPPXOptimizer* opt, SNEPPXTensor** params, SNEPPXTensor** grads, size_t n) {
    float rho = opt->rho, eps = opt->epsilon, wd = opt->weight_decay, clip = opt->grad_clip;

    for (size_t i = 0; i < n; i++) {
        if (!params[i] || !grads[i]) continue;
        float* pd = (float*)params[i]->data;
        float* gd = (float*)grads[i]->data;
        float* v = (float*)opt->state_buf2[i]->data;
        float* d = (float*)opt->state_buf3[i]->data;
        size_t sz = params[i]->size;

        for (size_t j = 0; j < sz; j++) {
            float g = clip_grad(gd[j], clip);
            g += wd * pd[j];
            v[j] = rho * v[j] + (1.0f - rho) * g * g;
            float delta = g * sqrtf(d[j] + eps) / sqrtf(v[j] + eps);
            d[j] = rho * d[j] + (1.0f - rho) * delta * delta;
            pd[j] -= delta;
        }
    }
}

/**
 * @brief Perform Optimizer Step.
 *
 * @param opt [out] Opt value.
 * @param params [out] Params value.
 * @param grads [out] Grads value.
 */
void SNEPPX_optimizer_step(SNEPPXOptimizer* opt, SNEPPXTensor** params, SNEPPXTensor** grads, size_t num_params) {
    if (!opt || !params || !grads || num_params == 0) return;
    ensure_bufs(opt, params, num_params);
    if (opt->num_params == 0) return;
    opt->step_count++;

    switch (opt->type) {
        case SNEPPX_OPTIMIZER_SGD:      step_sgd(opt, params, grads, num_params); break;
        case SNEPPX_OPTIMIZER_ADAM:     step_adam(opt, params, grads, num_params, 0); break;
        case SNEPPX_OPTIMIZER_ADAMW:    step_adam(opt, params, grads, num_params, 1); break;
        case SNEPPX_OPTIMIZER_ADAMAX:   step_adamax(opt, params, grads, num_params); break;
        case SNEPPX_OPTIMIZER_RMSPROP:  step_rmsprop(opt, params, grads, num_params); break;
        case SNEPPX_OPTIMIZER_ADAGRAD:  step_adagrad(opt, params, grads, num_params); break;
        case SNEPPX_OPTIMIZER_ADADELTA: step_adadelta(opt, params, grads, num_params); break;
    }
}

/**
 * @brief Perform Lr Scheduler Step Lr.
 *
 * @param lr_ptr [out] Lr Ptr value.
 * @param gamma [in] Gamma value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXLRScheduler* SNEPPX_lr_scheduler_step_lr(float* lr_ptr, float gamma, size_t step_size) {
    if (!lr_ptr) return NULL;
    SNEPPXLRScheduler* s = (SNEPPXLRScheduler*)SNEPPX_malloc(sizeof(SNEPPXLRScheduler), 64);
    if (!s) return NULL;
    memset(s, 0, sizeof(SNEPPXLRScheduler));
    s->type = SNEPPX_LR_STEP;
    s->lr_ptr = lr_ptr;
    s->gamma = gamma;
    s->step_size = step_size;
    s->mode_min = 1;
    return s;
}

/**
 * @brief Perform Lr Scheduler Exponential.
 *
 * @param lr_ptr [out] Lr Ptr value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXLRScheduler* SNEPPX_lr_scheduler_exponential(float* lr_ptr, float gamma) {
    if (!lr_ptr) return NULL;
    SNEPPXLRScheduler* s = (SNEPPXLRScheduler*)SNEPPX_malloc(sizeof(SNEPPXLRScheduler), 64);
    if (!s) return NULL;
    memset(s, 0, sizeof(SNEPPXLRScheduler));
    s->type = SNEPPX_LR_EXPONENTIAL;
    s->lr_ptr = lr_ptr;
    s->gamma = gamma;
    s->mode_min = 1;
    return s;
}

/**
 * @brief Perform Lr Scheduler Cosine.
 *
 * @param lr_ptr [out] Lr Ptr value.
 * @param min_lr [in] Min Lr value.
 * @param max_lr [in] Max Lr value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXLRScheduler* SNEPPX_lr_scheduler_cosine(float* lr_ptr, float min_lr, float max_lr, size_t total_steps) {
    if (!lr_ptr) return NULL;
    SNEPPXLRScheduler* s = (SNEPPXLRScheduler*)SNEPPX_malloc(sizeof(SNEPPXLRScheduler), 64);
    if (!s) return NULL;
    memset(s, 0, sizeof(SNEPPXLRScheduler));
    s->type = SNEPPX_LR_COSINE;
    s->lr_ptr = lr_ptr;
    s->min_lr = min_lr;
    s->max_lr = max_lr;
    s->total_steps = total_steps;
    s->mode_min = 1;
    return s;
}

/**
 * @brief Perform Lr Scheduler Reduce On Plateau.
 *
 * @param lr_ptr [out] Lr Ptr value.
 * @param factor [in] Factor value.
 * @param patience [in] Patience value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXLRScheduler* SNEPPX_lr_scheduler_reduce_on_plateau(float* lr_ptr, float factor, size_t patience, int mode_min) {
    if (!lr_ptr) return NULL;
    SNEPPXLRScheduler* s = (SNEPPXLRScheduler*)SNEPPX_malloc(sizeof(SNEPPXLRScheduler), 64);
    if (!s) return NULL;
    memset(s, 0, sizeof(SNEPPXLRScheduler));
    s->type = SNEPPX_LR_REDUCE_ON_PLATEAU;
    s->lr_ptr = lr_ptr;
    s->factor = factor;
    s->patience = patience;
    s->mode_min = mode_min;
    s->best_loss = mode_min ? 1e30f : -1e30f;
    return s;
}

/**
 * @brief Destroy Lr Scheduler.
 */
void SNEPPX_lr_scheduler_destroy(SNEPPXLRScheduler* sched) {
    SNEPPX_free(sched, sizeof(SNEPPXLRScheduler));
}

/**
 * @brief Create Sgd.
 *
 * @param lr [in] Lr value.
 * @param momentum [in] Momentum value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXOptimizer* SNEPPX_sgd_create(float lr, float momentum, float weight_decay) {
    SNEPPXOptimizerConfig cfg = SNEPPX_optimizer_config_default();
    cfg.learning_rate = lr; cfg.type = SNEPPX_OPTIMIZER_SGD;
    cfg.momentum = momentum; cfg.weight_decay = weight_decay;
    return SNEPPX_optimizer_create(&cfg);
}

/**
 * @brief Create Adam.
 *
 * @param lr [in] Lr value.
 * @param beta1 [in] Beta1 value.
 * @param beta2 [in] Beta2 value.
 * @param eps [in] Eps value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXOptimizer* SNEPPX_adam_create(float lr, float beta1, float beta2, float eps, float weight_decay) {
    SNEPPXOptimizerConfig cfg = SNEPPX_optimizer_config_default();
    cfg.learning_rate = lr; cfg.type = SNEPPX_OPTIMIZER_ADAM;
    cfg.beta1 = beta1; cfg.beta2 = beta2; cfg.epsilon = eps; cfg.weight_decay = weight_decay;
    return SNEPPX_optimizer_create(&cfg);
}

/**
 * @brief Create Adamw.
 *
 * @param lr [in] Lr value.
 * @param beta1 [in] Beta1 value.
 * @param beta2 [in] Beta2 value.
 * @param eps [in] Eps value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXOptimizer* SNEPPX_adamw_create(float lr, float beta1, float beta2, float eps, float weight_decay) {
    SNEPPXOptimizerConfig cfg = SNEPPX_optimizer_config_default();
    cfg.learning_rate = lr; cfg.type = SNEPPX_OPTIMIZER_ADAMW;
    cfg.beta1 = beta1; cfg.beta2 = beta2; cfg.epsilon = eps; cfg.weight_decay = weight_decay;
    return SNEPPX_optimizer_create(&cfg);
}

/**
 * @brief Create Rmsprop.
 *
 * @param lr [in] Lr value.
 * @param alpha [in] Alpha value.
 * @param eps [in] Eps value.
 * @param momentum [in] Momentum value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXOptimizer* SNEPPX_rmsprop_create(float lr, float alpha, float eps, float momentum, float weight_decay) {
    SNEPPXOptimizerConfig cfg = SNEPPX_optimizer_config_default();
    cfg.learning_rate = lr; cfg.type = SNEPPX_OPTIMIZER_RMSPROP;
    cfg.beta1 = alpha; cfg.epsilon = eps; cfg.momentum = momentum; cfg.weight_decay = weight_decay;
    return SNEPPX_optimizer_create(&cfg);
}

/**
 * @brief Create Adagrad.
 *
 * @param lr [in] Lr value.
 * @param eps [in] Eps value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXOptimizer* SNEPPX_adagrad_create(float lr, float eps, float weight_decay) {
    SNEPPXOptimizerConfig cfg = SNEPPX_optimizer_config_default();
    cfg.learning_rate = lr; cfg.type = SNEPPX_OPTIMIZER_ADAGRAD;
    cfg.epsilon = eps; cfg.weight_decay = weight_decay;
    return SNEPPX_optimizer_create(&cfg);
}

/**
 * @brief Perform Lr Scheduler Step.
 *
 * @param sched [out] Sched value.
 */
void SNEPPX_lr_scheduler_step(SNEPPXLRScheduler* sched, float current_loss) {
    if (!sched || !sched->lr_ptr) return;
    sched->last_epoch++;

    switch (sched->type) {
        case SNEPPX_LR_STEP:
            if (sched->step_size > 0 && sched->last_epoch % sched->step_size == 0)
                *sched->lr_ptr *= sched->gamma;
            break;
        case SNEPPX_LR_EXPONENTIAL:
            *sched->lr_ptr *= sched->gamma;
            break;
        case SNEPPX_LR_COSINE: {
            float progress = (float)sched->last_epoch / (float)(sched->total_steps > 0 ? sched->total_steps : 1);
            if (progress > 1.0f) progress = 1.0f;
            *sched->lr_ptr = sched->min_lr + 0.5f * (sched->max_lr - sched->min_lr) * (1.0f + cosf((float)3.14159265f * progress));
            break;
        }
        case SNEPPX_LR_REDUCE_ON_PLATEAU: {
            int improved = sched->mode_min ? (current_loss < sched->best_loss) : (current_loss > sched->best_loss);
            if (improved) {
                sched->best_loss = current_loss;
                sched->wait_count = 0;
            } else {
                sched->wait_count++;
                if (sched->wait_count >= (int)sched->patience && sched->patience > 0) {
                    *sched->lr_ptr *= sched->factor;
                    sched->wait_count = 0;
                }
            }
            break;
        }
    }
}
