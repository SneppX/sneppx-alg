#include "optimizer_impl.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * SNEPPX - Optimizer Impl
 *
 * WHAT
 *   Optimizer Impl.
 *
 * CONCEPT
 *   Provides optimizer implementations.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/**
 * @brief Initialize Optimizer State.
 *
 * @param state [out] State value.
 * @param num_params [in] Num Params value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_optimizer_state_init(SNEPPXOptimizerState* state, size_t num_params, size_t param_size) {
    if (!state) return -1;
    memset(state, 0, sizeof(*state));
    state->num_params = num_params;
    state->param_size = param_size;
    state->lr = 0.001f;
    state->beta1 = 0.9f;
    state->beta2 = 0.999f;
    state->eps = 1e-8f;
    state->weight_decay = 0.0f;
    state->step = 0;
    if (num_params > 0) {
        state->param_data = (void**)calloc(num_params, sizeof(void*));
        state->grad_data = (void**)calloc(num_params, sizeof(void*));
        state->momentum_buf = (void**)calloc(num_params, sizeof(void*));
        state->velocity_buf = (void**)calloc(num_params, sizeof(void*));
        if (!state->param_data || !state->grad_data || !state->momentum_buf || !state->velocity_buf)
            return -1;
    }
    return 0;
}

/**
 * @brief Destroy Optimizer State.
 */
void SNEPPX_optimizer_state_destroy(SNEPPXOptimizerState* state) {
    if (!state) return;
    for (size_t i = 0; i < state->num_params; i++) {
        free(state->momentum_buf[i]);
        free(state->velocity_buf[i]);
    }
    free(state->param_data);
    free(state->grad_data);
    free(state->momentum_buf);
    free(state->velocity_buf);
    memset(state, 0, sizeof(*state));
}

static float* get_or_create(void*** buf, size_t idx, size_t param_size) {
    if (!(*buf)[idx]) {
        (*buf)[idx] = calloc(1, param_size);
    }
    return (float*)(*buf)[idx];
}

/**
 * @brief Perform Optimizer Sgd Step.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_optimizer_sgd_step(SNEPPXOptimizerState* state) {
    if (!state) return -1;
    float lr = state->lr;
    float wd = state->weight_decay;
    size_t n = state->param_size / sizeof(float);
    for (size_t i = 0; i < state->num_params; i++) {
        float* p = (float*)state->param_data[i];
        float* g = (float*)state->grad_data[i];
        if (!p || !g) continue;
        for (size_t j = 0; j < n; j++) {
            float grad = g[j];
            if (wd != 0.0f) grad += wd * p[j];
            p[j] -= lr * grad;
        }
    }
    state->step++;
    return 0;
}

/**
 * @brief Perform Optimizer Adam Step.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_optimizer_adam_step(SNEPPXOptimizerState* state) {
    if (!state) return -1;
    float lr = state->lr;
    float b1 = state->beta1, b2 = state->beta2;
    float ep = state->eps, wd = state->weight_decay;
    uint64_t t = state->step + 1;
    float bias_corr1 = 1.0f / (1.0f - powf(b1, (float)t));
    float bias_corr2 = 1.0f / (1.0f - powf(b2, (float)t));
    float adj_lr = lr * bias_corr1;
    size_t n = state->param_size / sizeof(float);
    for (size_t i = 0; i < state->num_params; i++) {
        float* p = (float*)state->param_data[i];
        float* g = (float*)state->grad_data[i];
        float* m = get_or_create(&state->momentum_buf, i, state->param_size);
        float* v = get_or_create(&state->velocity_buf, i, state->param_size);
        if (!p || !g || !m || !v) continue;
        for (size_t j = 0; j < n; j++) {
            float grad = g[j];
            if (wd != 0.0f) grad += wd * p[j];
            m[j] = b1 * m[j] + (1.0f - b1) * grad;
            v[j] = b2 * v[j] + (1.0f - b2) * grad * grad;
            float m_hat = m[j] * bias_corr1;
            float v_hat = v[j] * bias_corr2;
            p[j] -= adj_lr * m_hat / (sqrtf(v_hat) + ep);
        }
    }
    state->step++;
    return 0;
}

/**
 * @brief Perform Optimizer Adamw Step.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_optimizer_adamw_step(SNEPPXOptimizerState* state) {
    if (!state) return -1;
    float lr = state->lr;
    float b1 = state->beta1, b2 = state->beta2;
    float ep = state->eps, wd = state->weight_decay;
    uint64_t t = state->step + 1;
    float bias_corr1 = 1.0f / (1.0f - powf(b1, (float)t));
    float bias_corr2 = 1.0f / (1.0f - powf(b2, (float)t));
    float adj_lr = lr * bias_corr1;
    size_t n = state->param_size / sizeof(float);
    for (size_t i = 0; i < state->num_params; i++) {
        float* p = (float*)state->param_data[i];
        float* g = (float*)state->grad_data[i];
        float* m = get_or_create(&state->momentum_buf, i, state->param_size);
        float* v = get_or_create(&state->velocity_buf, i, state->param_size);
        if (!p || !g || !m || !v) continue;
        for (size_t j = 0; j < n; j++) {
            if (wd != 0.0f) p[j] -= lr * wd * p[j];
            m[j] = b1 * m[j] + (1.0f - b1) * g[j];
            v[j] = b2 * v[j] + (1.0f - b2) * g[j] * g[j];
            float m_hat = m[j] * bias_corr1;
            float v_hat = v[j] * bias_corr2;
            p[j] -= adj_lr * m_hat / (sqrtf(v_hat) + ep);
        }
    }
    state->step++;
    return 0;
}

/**
 * @brief Perform Optimizer Lamb Step.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_optimizer_lamb_step(SNEPPXOptimizerState* state) {
    if (!state) return -1;
    float lr = state->lr;
    float b1 = state->beta1, b2 = state->beta2;
    float ep = state->eps, wd = state->weight_decay;
    uint64_t t = state->step + 1;
    float bias_corr1 = 1.0f / (1.0f - powf(b1, (float)t));
    float bias_corr2 = 1.0f / (1.0f - powf(b2, (float)t));
    size_t n = state->param_size / sizeof(float);
    for (size_t i = 0; i < state->num_params; i++) {
        float* p = (float*)state->param_data[i];
        float* g = (float*)state->grad_data[i];
        float* m = get_or_create(&state->momentum_buf, i, state->param_size);
        float* v = get_or_create(&state->velocity_buf, i, state->param_size);
        if (!p || !g || !m || !v) continue;
        float grad_norm = 0.0f, param_norm = 0.0f;
        for (size_t j = 0; j < n; j++) {
            float grad = g[j];
            if (wd != 0.0f) grad += wd * p[j];
            m[j] = b1 * m[j] + (1.0f - b1) * grad;
            v[j] = b2 * v[j] + (1.0f - b2) * grad * grad;
            float m_hat = m[j] * bias_corr1;
            float v_hat = v[j] * bias_corr2;
            float update = m_hat / (sqrtf(v_hat) + ep);
            grad_norm += update * update;
            param_norm += p[j] * p[j];
            p[j] -= lr * update;
        }
        float gn = sqrtf(grad_norm);
        float pn = sqrtf(param_norm);
        if (gn > 0.0f && pn > 0.0f) {
            float trust_ratio = (pn > 0.0f) ? (pn / gn) : 1.0f;
            if (trust_ratio > 10.0f) trust_ratio = 10.0f;
            for (size_t j = 0; j < n; j++) {
                float grad = g[j];
                if (wd != 0.0f) grad += wd * p[j];
                float m_hat = m[j] * bias_corr1;
                float v_hat = v[j] * bias_corr2;
                p[j] += lr * trust_ratio * m_hat / (sqrtf(v_hat) + ep) - lr * m_hat / (sqrtf(v_hat) + ep);
                p[j] -= lr * trust_ratio * m_hat / (sqrtf(v_hat) + ep);
            }
        }
    }
    state->step++;
    return 0;
}

/**
 * @brief Perform Gradient Clip Norm.
 *
 * @param grads [out] Grads value.
 * @param num_params [in] Num Params value.
 * @param param_size [in] Param Size value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_gradient_clip_norm(void** grads, size_t num_params, size_t param_size, float max_norm) {
    if (!grads || num_params == 0 || param_size == 0) return 0;
    float total_norm = 0.0f;
    size_t n = param_size / sizeof(float);
    for (size_t i = 0; i < num_params; i++) {
        float* g = (float*)grads[i];
        if (!g) continue;
        for (size_t j = 0; j < n; j++) total_norm += g[j] * g[j];
    }
    total_norm = sqrtf(total_norm);
    if (total_norm <= max_norm || total_norm == 0.0f) return 0;
    float scale = max_norm / total_norm;
    for (size_t i = 0; i < num_params; i++) {
        float* g = (float*)grads[i];
        if (!g) continue;
        for (size_t j = 0; j < n; j++) g[j] *= scale;
    }
    return 0;
}

/**
 * @brief Perform Gradient Clip Value.
 *
 * @param grads [out] Grads value.
 * @param num_params [in] Num Params value.
 * @param param_size [in] Param Size value.
 * @param min_val [in] Min Val value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_gradient_clip_value(void** grads, size_t num_params, size_t param_size, float min_val, float max_val) {
    if (!grads || num_params == 0 || param_size == 0) return 0;
    size_t n = param_size / sizeof(float);
    for (size_t i = 0; i < num_params; i++) {
        float* g = (float*)grads[i];
        if (!g) continue;
        for (size_t j = 0; j < n; j++) {
            if (g[j] < min_val) g[j] = min_val;
            else if (g[j] > max_val) g[j] = max_val;
        }
    }
    return 0;
}

/**
 * @brief Initialize Lr Scheduler.
 *
 * @param sched [out] Sched value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_lr_scheduler_init(SNEPPXLRScheduler* sched, SNEPPXLRSchedule type) {
    if (!sched) return -1;
    memset(sched, 0, sizeof(*sched));
    sched->type = type;
    sched->base_lr = 0.001f;
    sched->min_lr = 0.0f;
    sched->warmup_steps = 0;
    sched->total_steps = 100000;
    sched->step_count = 0;
    sched->current_lr = sched->base_lr;
    return 0;
}

/**
 * @brief Perform Impl Lr Scheduler Step.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_impl_lr_scheduler_step(SNEPPXLRScheduler* sched) {
    if (!sched) return 0.0f;
    size_t step = sched->step_count;
    float progress = (sched->total_steps > 0) ? (float)step / (float)sched->total_steps : 0.0f;
    if (step < sched->warmup_steps && sched->warmup_steps > 0) {
        sched->current_lr = sched->base_lr * (float)(step + 1) / (float)sched->warmup_steps;
    } else {
        switch (sched->type) {
            case SNEPPX_SCHEDULE_CONSTANT:
                sched->current_lr = sched->base_lr;
                break;
            case SNEPPX_SCHEDULE_COSINE:
                sched->current_lr = sched->min_lr + 0.5f * (sched->base_lr - sched->min_lr) * (1.0f + cosf((float)3.14159265f * progress));
                break;
            case SNEPPX_SCHEDULE_LINEAR:
                sched->current_lr = sched->base_lr + (sched->min_lr - sched->base_lr) * progress;
                break;
            case SNEPPX_SCHEDULE_STEP_DECAY: {
                size_t milestone = sched->total_steps / 3;
                if (milestone == 0) milestone = 1;
                size_t decay_count = step / milestone;
                float decay_factor = 1.0f;
                for (size_t d = 0; d < decay_count && d < 3; d++) decay_factor *= 0.1f;
                sched->current_lr = sched->base_lr * decay_factor;
                break;
            }
            case SNEPPX_SCHEDULE_WARMUP_COSINE:
                progress = (float)(step - sched->warmup_steps) / (float)(sched->total_steps - sched->warmup_steps + 1);
                if (progress > 1.0f) progress = 1.0f;
                sched->current_lr = sched->min_lr + 0.5f * (sched->base_lr - sched->min_lr) * (1.0f + cosf((float)3.14159265f * progress));
                break;
            default:
                sched->current_lr = sched->base_lr;
                break;
        }
    }
    sched->step_count++;
    return sched->current_lr;
}

/**
 * @brief Reset Lr Scheduler.
 */
void SNEPPX_lr_scheduler_reset(SNEPPXLRScheduler* sched) {
    if (sched) { sched->step_count = 0; sched->current_lr = sched->base_lr; }
}
