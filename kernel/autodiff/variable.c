#include "automatic_differentiation_framework.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
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
 * @brief Create Variable.
 *
 * @param data [out] Data value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_variable_create(SNEPPXTensor* data, int requires_grad) {
    SNEPPXVariable* var = (SNEPPXVariable*)SNEPPX_malloc(sizeof(SNEPPXVariable), 64);
    if (!var) return NULL;
    memset(var, 0, sizeof(SNEPPXVariable));
    var->data = data;
    var->requires_grad = SNEPPX_no_grad_is_active() ? 0 : requires_grad;
    var->grad = NULL;
    var->backward_fn = NULL;
    var->backward_ctx = NULL;
    var->free_ctx = NULL;
    var->recompute_ctx = NULL;
    var->checkpointed = 0;
    var->ref_count = 0;
    var->parents = NULL;
    var->num_parents = 0;
    var->param_count = 0;
    return var;
}

/**
 * @brief Destroy Variable.
 */
void SNEPPX_variable_destroy(SNEPPXVariable* var) {
    if (!var) return;
    if (var->data) SNEPPX_tensor_destroy(var->data);
    if (var->grad) SNEPPX_tensor_destroy(var->grad);
    if (var->backward_ctx && var->free_ctx) var->free_ctx(var->backward_ctx);
    if (var->parents) SNEPPX_free(var->parents, var->num_parents * sizeof(SNEPPXVariable*));
    SNEPPX_free(var, sizeof(SNEPPXVariable));
}

/**
 * @brief Perform Variable Set Requires Grad.
 *
 * @param var [out] Var value.
 */
void SNEPPX_variable_set_requires_grad(SNEPPXVariable* var, int requires_grad) {
    if (!var) return;
    var->requires_grad = requires_grad;
}

/**
 * @brief Perform Variable Detach.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_variable_detach(SNEPPXVariable* var) {
    if (!var || !var->data) return NULL;
    SNEPPXVariable* out = SNEPPX_variable_create(
        SNEPPX_tensor_create(var->data->shape, var->data->ndim, SNEPPX_FLOAT32), 0);
    if (out && out->data) {
        memcpy(out->data->data, var->data->data, var->data->size * sizeof(float));
    }
    return out;
}

/**
 * @brief Perform Variable Copy.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVariable* SNEPPX_variable_copy(SNEPPXVariable* var) {
    if (!var || !var->data) return NULL;
    SNEPPXVariable* out = SNEPPX_variable_create(
        SNEPPX_tensor_create(var->data->shape, var->data->ndim, SNEPPX_FLOAT32), var->requires_grad);
    if (out && out->data) {
        memcpy(out->data->data, var->data->data, var->data->size * sizeof(float));
        if (var->grad) {
            out->grad = SNEPPX_tensor_create(var->grad->shape, var->grad->ndim, SNEPPX_FLOAT32);
            if (out->grad) memcpy(out->grad->data, var->grad->data, var->grad->size * sizeof(float));
        }
    }
    return out;
}

/**
 * @brief Perform Variable Zero Grad.
 */
void SNEPPX_variable_zero_grad(SNEPPXVariable* var) {
    if (!var) return;
    if (var->grad) { SNEPPX_tensor_destroy(var->grad); var->grad = NULL; }
}

/**
 * @brief Perform Variable Item.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_variable_item(SNEPPXVariable* var) {
    if (!var || !var->data || var->data->size == 0) return 0.0f;
    return ((float*)var->data->data)[0];
}

/**
 * @brief Perform Variable Numel.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_variable_numel(SNEPPXVariable* var) {
    if (!var || !var->data) return 0;
    return var->data->size;
}
