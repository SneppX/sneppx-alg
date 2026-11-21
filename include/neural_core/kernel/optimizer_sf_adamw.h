#ifndef SNEPPX_OPTIMIZER_SF_ADAMW_H
#define SNEPPX_OPTIMIZER_SF_ADAMW_H

#include <stddef.h>

/*
 * SNEPPX - Optimizer Sf Adamw
 *
 * WHAT
 *   Optimizer Sf Adamw.
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
    float* z;
    float* m;
    float* v;
    float beta1, beta2;
    float eps;
    float lr;
    float weight_decay;
    int t;
    size_t size;
} SNEPPXSFRAdamWState;

/**
 * @brief Initialize Sf Adamw.
 *
 * @param state [out] State value.
 * @param size [in] Size value.
 * @param lr [in] Lr value.
 * @param beta1 [in] Beta1 value.
 * @param beta2 [in] Beta2 value.
 * @param eps [in] Eps value.
 * @param weight_decay [in] Weight Decay value.
 */
void SNEPPX_sf_adamw_init(SNEPPXSFRAdamWState* state, size_t size, float lr, float beta1, float beta2, float eps, float weight_decay);
/**
 * @brief Perform Sf Adamw Step.
 *
 * @param state [out] State value.
 * @param params [out] Params value.
 * @param grads [in] Grads value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_sf_adamw_step(SNEPPXSFRAdamWState* state, float* params, const float* grads);
/**
 * @brief Destroy Sf Adamw.
 *
 * @param state [out] State value.
 */
void SNEPPX_sf_adamw_destroy(SNEPPXSFRAdamWState* state);

#endif
