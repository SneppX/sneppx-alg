#include "hierarchical_state_space.h"
#include "automatic_differentiation_framework.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>
/*
 * SNEPPX - HSS Forward Pass (Training Mode)
 *
 * WHAT
 *   HSS Forward Pass (Training Mode).
 *
 * CONCEPT
 *   HSS forward pass with gradient flow through the selective scan for backpropagation.
 *
 * ROLE
 *   Training variant of the HSS forward pass that records intermediate values for reverse-mode autodiff.
 *
 * REFERENCES
 *   None (internal algorithm).
 */



/**
 * @brief Perform Hss Build Train Graph.
 *
 * @param model [out] Model value.
 * @param tape [out] Tape value.
 * @param input_var [out] Input Var value.
 * @param weight_vars [out] Weight Vars value.
 * @param num_weights [in] Num Weights value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hss_build_train_graph(SNEPPXHSSModel* model, SNEPPXTape* tape,
                               SNEPPXVariable* input_var,
                               SNEPPXVariable** weight_vars, size_t num_weights,
                               SNEPPXVariable** output_var) {
    if (!model || !tape || !input_var || !output_var) return -1;
    size_t seq_len = input_var->data->shape[0];
    size_t i_dim = input_var->data->shape[1];
    size_t o_dim = model->config.output_dim;
    size_t s_dim = model->config.state_dim;
    size_t wi = 0;

    SNEPPXVariable* layer_input = input_var;
    size_t cur_dim = i_dim;

    for (size_t l = 0; l < model->config.num_layers; l++) {
        if (wi + 9 > num_weights) return -1;
        SNEPPXVariable* A   = weight_vars[wi++];
        SNEPPXVariable* B   = weight_vars[wi++];
        SNEPPXVariable* C   = weight_vars[wi++];
        SNEPPXVariable* D   = weight_vars[wi++];
        SNEPPXVariable* dt  = weight_vars[wi++];
        SNEPPXVariable* xp  = weight_vars[wi++];
        SNEPPXVariable* xpb = weight_vars[wi++];
        SNEPPXVariable* ng  = weight_vars[wi++];
        SNEPPXVariable* nb  = weight_vars[wi++];

        size_t s1[] = {s_dim, 1};
        SNEPPXTensor* dt2_t = SNEPPX_tensor_zeros(s1, 2, SNEPPX_FLOAT32);
        if (!dt2_t) return -1;
        memcpy((float*)dt2_t->data, (float*)dt->data->data, s_dim * sizeof(float));
        SNEPPXVariable* dt2 = SNEPPX_variable_create(dt2_t, 1);
        SNEPPX_tape_record(tape, dt2);

        size_t or1[] = {1, s_dim};
        SNEPPXTensor* or1_t = SNEPPX_tensor_ones(or1, 2, SNEPPX_FLOAT32);
        if (!or1_t) return -1;
        SNEPPXVariable* or1_v = SNEPPX_variable_create(or1_t, 0);
        SNEPPX_tape_record(tape, or1_v);

        SNEPPXVariable* dt_mat = SNEPPX_matmul(tape, dt2, or1_v);

        SNEPPXTensor* eye_t = SNEPPX_tensor_eye(s_dim, SNEPPX_FLOAT32);
        if (!eye_t) return -1;
        SNEPPXVariable* eye_v = SNEPPX_variable_create(eye_t, 0);
        SNEPPX_tape_record(tape, eye_v);

        SNEPPXVariable* A_dt = SNEPPX_mul(tape, A, dt_mat);
        SNEPPXVariable* A_bar = SNEPPX_add(tape, eye_v, A_dt);

        size_t ori[] = {1, cur_dim};
        SNEPPXTensor* ori_t = SNEPPX_tensor_ones(ori, 2, SNEPPX_FLOAT32);
        if (!ori_t) return -1;
        SNEPPXVariable* ori_v = SNEPPX_variable_create(ori_t, 0);
        SNEPPX_tape_record(tape, ori_v);

        SNEPPXVariable* dt_mat2 = SNEPPX_matmul(tape, dt2, ori_v);
        SNEPPXVariable* B_bar = SNEPPX_mul(tape, B, dt_mat2);

        size_t hs[] = {s_dim, 1};
        SNEPPXTensor* h0 = SNEPPX_tensor_zeros(hs, 2, SNEPPX_FLOAT32);
        if (!h0) return -1;
        SNEPPXVariable* h = SNEPPX_variable_create(h0, 0);
        SNEPPX_tape_record(tape, h);

        SNEPPXVariable** yts = (SNEPPXVariable**)SNEPPX_malloc(seq_len * sizeof(SNEPPXVariable*), 64);
        if (!yts) return -1;
        memset(yts, 0, seq_len * sizeof(SNEPPXVariable*));

        for (size_t t = 0; t < seq_len; t++) {
            SNEPPXTensor* xs = SNEPPX_tensor_slice(layer_input->data, 0, (int)t, (int)(t + 1));
            if (!xs) { SNEPPX_free(yts, seq_len * sizeof(SNEPPXVariable*)); return -1; }
            SNEPPXVariable* xt_row = SNEPPX_variable_create(xs, 0);
            SNEPPX_tape_record(tape, xt_row);

            SNEPPXVariable* xt_col = SNEPPX_transpose(tape, xt_row, 0, 1);

            SNEPPXVariable* xp_out = SNEPPX_matmul(tape, xp, xt_col);
            SNEPPXVariable* xp_b = SNEPPX_add(tape, xp_out, xpb);
            SNEPPXVariable* xp_b_row = SNEPPX_transpose(tape, xp_b, 0, 1);
            SNEPPXVariable* xp_norm = SNEPPX_layer_norm(tape, xp_b_row, ng, nb, 1e-5f);
            SNEPPXVariable* xp_norm_col = SNEPPX_transpose(tape, xp_norm, 0, 1);

            size_t iz[] = {cur_dim, 1};
            SNEPPXTensor* zt = SNEPPX_tensor_zeros(iz, 2, SNEPPX_FLOAT32);
            if (!zt) { SNEPPX_free(yts, seq_len * sizeof(SNEPPXVariable*)); return -1; }
            SNEPPXVariable* zv = SNEPPX_variable_create(zt, 0);
            SNEPPX_tape_record(tape, zv);
            SNEPPXVariable* x_col2 = SNEPPX_add(tape, zv, xp_norm_col);

            SNEPPXVariable* h_new = SNEPPX_add(tape,
                SNEPPX_matmul(tape, A_bar, h),
                SNEPPX_matmul(tape, B_bar, x_col2));

            SNEPPXVariable* yt = SNEPPX_add(tape,
                SNEPPX_matmul(tape, C, h_new),
                SNEPPX_matmul(tape, D, x_col2));

            h = h_new;
            yts[t] = yt;
        }

        {
            SNEPPXVariable** yt_rows = (SNEPPXVariable**)SNEPPX_malloc(seq_len * sizeof(SNEPPXVariable*), 64);
            if (!yt_rows) { SNEPPX_free(yts, seq_len * sizeof(SNEPPXVariable*)); return -1; }
            for (size_t t = 0; t < seq_len; t++) {
                yt_rows[t] = SNEPPX_transpose(tape, yts[t], 0, 1);
            }
            SNEPPXVariable* layer_out = SNEPPX_concat(tape, yt_rows, seq_len, 0);
            SNEPPX_free(yt_rows, seq_len * sizeof(SNEPPXVariable*));
            SNEPPX_free(yts, seq_len * sizeof(SNEPPXVariable*));
            if (!layer_out) return -1;
            if (l == model->config.num_layers - 1) {
                *output_var = layer_out;
            } else {
                layer_input = layer_out;
                cur_dim = o_dim;
            }
        }
    }
    return 0;
}
