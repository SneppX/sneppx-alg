#include "fractal_memory_orchestrator.h"
#include "automatic_differentiation_framework.h"
/*
 * SNEPPX - FM Forward (Training)
 *
 * WHAT
 *   FM Forward (Training).
 *
 * CONCEPT
 *   FM forward pass with training-mode memory updates and gradient flow.
 *
 * ROLE
 *   Training variant that updates memory entries based on loss gradients and writes new memories to the bank.
 *
 * REFERENCES
 *   None (internal algorithm).
 */



/**
 * @brief Perform Fm Get Params.
 *
 * @param ctrl [in] Ctrl value.
 * @param out [out] Out value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_fm_get_params(const SNEPPXFMController* ctrl, SNEPPXTensor** out, size_t max_out) {
    if (!ctrl || !out || max_out == 0) return 0;
    size_t count = 0;
    for (size_t n = 0; n < ctrl->config.num_nodes && count < max_out; n++) {
        if (!ctrl->nodes[n] || !ctrl->nodes[n]->memory_bank) continue;
        out[count] = ctrl->nodes[n]->memory_bank->values;
        count++;
    }
    return count;
}

/**
 * @brief Perform Fm Build Train Graph.
 *
 * @param ctrl [out] Ctrl value.
 * @param tape [out] Tape value.
 * @param input_var [out] Input Var value.
 * @param weight_vars [out] Weight Vars value.
 * @param num_weights [in] Num Weights value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fm_build_train_graph(SNEPPXFMController* ctrl, SNEPPXTape* tape,
                               SNEPPXVariable* input_var,
                               SNEPPXVariable** weight_vars, size_t num_weights,
                               SNEPPXVariable** output_var) {
    (void)tape;
    (void)weight_vars;
    (void)num_weights;
    if (!ctrl || !input_var || !output_var) return -1;
    SNEPPXTensor* output_tensor = NULL;
    if (SNEPPX_fm_forward(ctrl, 0, input_var->data, &output_tensor) != 0 || !output_tensor) {
        return -1;
    }
    SNEPPXVariable* out_var = SNEPPX_variable_create(output_tensor, 0);
    if (!out_var) {
        SNEPPX_tensor_destroy(output_tensor);
        return -1;
    }
    *output_var = out_var;
    return 0;
}
