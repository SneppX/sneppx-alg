#include "optimizer_sf_adamw.h"
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
 * @brief Initialize Sf Adamw.
 *
 * @param state [out] State value.
 * @param size [in] Size value.
 * @param lr [in] Lr value.
 * @param beta1 [in] Beta1 value.
 * @param beta2 [in] Beta2 value.
 * @param eps [in] Eps value.
 */
void SNEPPX_sf_adamw_init(SNEPPXSFRAdamWState* state, size_t size, float lr, float beta1, float beta2, float eps, float weight_decay) {
    if (!state) return;
    memset(state, 0, sizeof(SNEPPXSFRAdamWState));
    state->z = (float*)SNEPPX_malloc(size * sizeof(float), 64);
    state->m = (float*)SNEPPX_malloc(size * sizeof(float), 64);
    state->v = (float*)SNEPPX_malloc(size * sizeof(float), 64);
    if (state->z) memset(state->z, 0, size * sizeof(float));
    if (state->m) memset(state->m, 0, size * sizeof(float));
    if (state->v) memset(state->v, 0, size * sizeof(float));
    state->beta1 = beta1;
    state->beta2 = beta2;
    state->eps = eps;
    state->lr = lr;
    state->weight_decay = weight_decay;
    state->t = 0;
    state->size = size;
}

/**
 * @brief Perform Sf Adamw Step.
 *
 * @param state [out] State value.
 * @param params [out] Params value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sf_adamw_step(SNEPPXSFRAdamWState* state, float* params, const float* grads) {
    if (!state || !params || !grads) return -1;
    if (!state->z || !state->m || !state->v) return -1;
    size_t sz = state->size;
    float b1 = state->beta1, b2 = state->beta2;
    float eps = state->eps, lr = state->lr, wd = state->weight_decay;
    float* z = state->z, *m = state->m, *v = state->v;

    state->t++;

    for (size_t i = 0; i < sz; i++) {
        float g = grads[i];
        m[i] = b1 * m[i] + (1.0f - b1) * g;
        v[i] = b2 * v[i] + (1.0f - b2) * g * g;
        z[i] = (1.0f - b1) * params[i] + b1 * z[i];
        float step = m[i] / (sqrtf(v[i]) + eps);
        params[i] = z[i] - lr * step - lr * wd * z[i];
    }
    return 0;
}

/**
 * @brief Destroy Sf Adamw.
 */
void SNEPPX_sf_adamw_destroy(SNEPPXSFRAdamWState* state) {
    if (!state) return;
    if (state->z) SNEPPX_free(state->z, state->size * sizeof(float));
    if (state->m) SNEPPX_free(state->m, state->size * sizeof(float));
    if (state->v) SNEPPX_free(state->v, state->size * sizeof(float));
    memset(state, 0, sizeof(SNEPPXSFRAdamWState));
}
