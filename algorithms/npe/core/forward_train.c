#include "neural_programming_engine.h"
#include "automatic_differentiation_framework.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
/*
 * SNEPPX - NPE Forward (Training)
 *
 * WHAT
 *   NPE Forward (Training).
 *
 * CONCEPT
 *   NPE forward pass with training mode and gradient flow through the VM.
 *
 * ROLE
 *   Training variant of the NPE forward pass that records VM execution trace for backpropagation through the neural program.
 *
 * REFERENCES
 *   None (internal algorithm).
 */



/**
 * @brief Perform Npe Get Params.
 *
 * @param prog [out] Prog value.
 * @param out [out] Out value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_npe_get_params(SNEPPXNPEProgram* prog, SNEPPXTensor** out, size_t max_out) {
    if (!prog || !prog->memory) return 0;

    size_t dim = 0, hidden_dim = 0;
    for (size_t i = 0; i < prog->num_instructions && i < 32; i++) {
        if (prog->instructions[i].opcode == SNEPPX_LOAD && prog->instructions[i].shape_a[0] > 0 && prog->instructions[i].shape_a[1] > 0) {
            dim = (size_t)prog->instructions[i].shape_a[0];
            hidden_dim = (size_t)prog->instructions[i].shape_a[1];
            break;
        }
    }
    if (dim == 0 || hidden_dim == 0) return 0;

    if (!prog->param_w1) {
        size_t w1_off = 0;
        size_t w1_shape[2] = {dim, hidden_dim};
        size_t w1_str[2] = {hidden_dim, 1};
        prog->param_w1 = SNEPPX_tensor_as_strided(prog->memory, w1_off, w1_shape, 2, w1_str);
    }
    if (!prog->param_b1) {
        size_t b1_off = dim * hidden_dim;
        size_t b1_shape[1] = {hidden_dim};
        size_t b1_str[1] = {1};
        prog->param_b1 = SNEPPX_tensor_as_strided(prog->memory, b1_off, b1_shape, 1, b1_str);
    }
    if (!prog->param_w2) {
        size_t w2_off = dim * hidden_dim + hidden_dim;
        size_t w2_shape[2] = {hidden_dim, dim};
        size_t w2_str[2] = {dim, 1};
        prog->param_w2 = SNEPPX_tensor_as_strided(prog->memory, w2_off, w2_shape, 2, w2_str);
    }
    if (!prog->param_b2) {
        size_t b2_off = dim * hidden_dim + hidden_dim + hidden_dim * dim;
        size_t b2_shape[1] = {dim};
        size_t b2_str[1] = {1};
        prog->param_b2 = SNEPPX_tensor_as_strided(prog->memory, b2_off, b2_shape, 1, b2_str);
    }

    if (out) {
        size_t idx = 0;
        if (idx < max_out && prog->param_w1) out[idx++] = prog->param_w1;
        if (idx < max_out && prog->param_b1) out[idx++] = prog->param_b1;
        if (idx < max_out && prog->param_w2) out[idx++] = prog->param_w2;
        if (idx < max_out && prog->param_b2) out[idx++] = prog->param_b2;
    }
    return 4;
}

/**
 * @brief Perform Npe Build Train Graph.
 *
 * @param prog [out] Prog value.
 * @param tape [out] Tape value.
 * @param input_var [out] Input Var value.
 * @param weight_vars [out] Weight Vars value.
 * @param num_weights [in] Num Weights value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_npe_build_train_graph(SNEPPXNPEProgram* prog, SNEPPXTape* tape,
                                SNEPPXVariable* input_var,
                                SNEPPXVariable** weight_vars, size_t num_weights,
                                SNEPPXVariable** output_var) {
    if (!prog || !tape || !input_var || !weight_vars || !output_var) return -1;
    if (num_weights < 4) return -1;

    SNEPPXVariable* w1 = weight_vars[0];
    SNEPPXVariable* b1 = weight_vars[1];
    SNEPPXVariable* w2 = weight_vars[2];
    SNEPPXVariable* b2 = weight_vars[3];

    SNEPPXVariable* current = SNEPPX_matmul(tape, input_var, w1);
    if (!current) return -1;
    current = SNEPPX_add(tape, current, b1);
    if (!current) return -1;
    current = SNEPPX_relu(tape, current);
    if (!current) return -1;
    current = SNEPPX_matmul(tape, current, w2);
    if (!current) return -1;
    current = SNEPPX_add(tape, current, b2);
    if (!current) return -1;

    *output_var = current;
    return 0;
}
