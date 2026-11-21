#ifndef SNEPPX_OPTIMIZER_RADAM_H
#define SNEPPX_OPTIMIZER_RADAM_H

#include <stddef.h>

/*
 * SNEPPX - Optimizer Radam
 *
 * WHAT
 *   Optimizer Radam.
 *
 * CONCEPT
 *   Provides optimizer implementations.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef struct {
    float* m;
    float* v;
    float beta1, beta2;
    float eps;
    float lr;
    int t;
    size_t size;
} SNEPPXRAdamState;

/**
 * @brief Initialize Radam.
 *
 * @param state [out] State value.
 * @param size [in] Size value.
 * @param lr [in] Lr value.
 * @param beta1 [in] Beta1 value.
 * @param beta2 [in] Beta2 value.
 * @param eps [in] Eps value.
 */
void SNEPPX_radam_init(SNEPPXRAdamState* state, size_t size, float lr, float beta1, float beta2, float eps);
/**
 * @brief Perform Radam Step.
 *
 * @param state [out] State value.
 * @param params [out] Params value.
 * @param grads [in] Grads value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_radam_step(SNEPPXRAdamState* state, float* params, const float* grads);
/**
 * @brief Destroy Radam.
 *
 * @param state [out] State value.
 */
void SNEPPX_radam_destroy(SNEPPXRAdamState* state);

#endif
