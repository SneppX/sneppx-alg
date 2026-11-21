#include "optimizer_radam.h"
#include "polymorphic_memory_allocator.h"
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



/**
 * @brief Initialize Radam.
 *
 * @param state [out] State value.
 * @param size [in] Size value.
 * @param lr [in] Lr value.
 * @param beta1 [in] Beta1 value.
 * @param beta2 [in] Beta2 value.
 */
void SNEPPX_radam_init(SNEPPXRAdamState* state, size_t size, float lr, float beta1, float beta2, float eps) {
    if (!state) return;
    memset(state, 0, sizeof(SNEPPXRAdamState));
    state->m = (float*)SNEPPX_malloc(size * sizeof(float), 64);
    state->v = (float*)SNEPPX_malloc(size * sizeof(float), 64);
    if (state->m) memset(state->m, 0, size * sizeof(float));
    if (state->v) memset(state->v, 0, size * sizeof(float));
    state->beta1 = beta1;
    state->beta2 = beta2;
    state->eps = eps;
    state->lr = lr;
    state->t = 0;
    state->size = size;
}

/**
 * @brief Perform Radam Step.
 *
 * @param state [out] State value.
 * @param params [out] Params value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_radam_step(SNEPPXRAdamState* state, float* params, const float* grads) {
    if (!state || !params || !grads) return -1;
    if (!state->m || !state->v) return -1;
    size_t sz = state->size;
    float b1 = state->beta1, b2 = state->beta2;
    float eps = state->eps, lr = state->lr;
    float* m = state->m, *v = state->v;

    state->t++;
    int t = state->t;

    float b1_t = powf(b1, (float)t);
    float b2_t = powf(b2, (float)t);
    float rho_inf = 2.0f / (1.0f - b2) - 1.0f;
    float rho_t = rho_inf - 2.0f * (float)t * b2_t / (1.0f - b2_t);

    for (size_t i = 0; i < sz; i++) {
        float g = grads[i];
        m[i] = b1 * m[i] + (1.0f - b1) * g;
        v[i] = b2 * v[i] + (1.0f - b2) * g * g;
        float m_hat = m[i] / (1.0f - b1_t);
        float update;
        if (rho_t > 4.0f) {
            float l = sqrtf((1.0f - b2_t) / (v[i] + eps));
            float r_num = (rho_t - 4.0f) * (rho_t - 2.0f) * rho_inf;
            float r_den = (rho_inf - 4.0f) * (rho_inf - 2.0f) * rho_t;
            float r = sqrtf(r_num / r_den);
            update = m_hat * r * l;
        } else {
            update = m_hat;
        }
        params[i] -= lr * update;
    }
    return 0;
}

/**
 * @brief Destroy Radam.
 */
void SNEPPX_radam_destroy(SNEPPXRAdamState* state) {
    if (!state) return;
    if (state->m) SNEPPX_free(state->m, state->size * sizeof(float));
    if (state->v) SNEPPX_free(state->v, state->size * sizeof(float));
    memset(state, 0, sizeof(SNEPPXRAdamState));
}
