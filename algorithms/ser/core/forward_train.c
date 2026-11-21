#include "sparse_expert_routing.h"
#include "automatic_differentiation_framework.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <stdlib.h>
/*
 * SNEPPX - SER Layer Forward (Training)
 *
 * WHAT
 *   SER Layer Forward (Training).
 *
 * CONCEPT
 *   SER layer forward pass with training mode including auxiliary loss computation.
 *
 * ROLE
 *   Training variant that computes the load-balancing auxiliary loss alongside the main forward pass.
 *
 * REFERENCES
 *   None (internal algorithm).
 */



/**
 * @brief Perform Ser Build Train Graph.
 *
 * @param model [out] Model value.
 * @param tape [out] Tape value.
 * @param input_var [out] Input Var value.
 * @param weight_vars [out] Weight Vars value.
 * @param num_weights [in] Num Weights value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ser_build_train_graph(SNEPPXSERModel* model, SNEPPXTape* tape,
                               SNEPPXVariable* input_var,
                               SNEPPXVariable** weight_vars, size_t num_weights,
                               SNEPPXVariable** output_var) {
    if (!model || !tape || !input_var || !output_var) return -1;

    size_t B = input_var->data->shape[0];
    size_t D = model->config.input_dim;
    size_t E = model->config.expert_dim;
    size_t O = model->config.output_dim;
    size_t N = model->config.num_experts;
    size_t gater_extra = model->config.use_mlp_gater ? 4 : 0;
    size_t wp_layer = 2 + gater_extra + 4 * N;
    size_t total_layers = model->num_layers;

    if (num_weights < total_layers * wp_layer) return -1;

    size_t wi = 0;
    SNEPPXVariable* cur_input = input_var;
    size_t cur_dim = D;

    for (size_t l = 0; l < total_layers; l++) {
        SNEPPXVariable* router = weight_vars[wi++];
        SNEPPXVariable* router_bias = weight_vars[wi++];

        SNEPPXVariable* gw1 = NULL, *gb1 = NULL, *gw2 = NULL, *gb2 = NULL;
        if (model->config.use_mlp_gater) {
            gw1 = weight_vars[wi++];
            gb1 = weight_vars[wi++];
            gw2 = weight_vars[wi++];
            gb2 = weight_vars[wi++];
        }

        SNEPPXVariable** w1 = (SNEPPXVariable**)SNEPPX_malloc(N * sizeof(SNEPPXVariable*), 64);
        SNEPPXVariable** b1 = (SNEPPXVariable**)SNEPPX_malloc(N * sizeof(SNEPPXVariable*), 64);
        SNEPPXVariable** w2 = (SNEPPXVariable**)SNEPPX_malloc(N * sizeof(SNEPPXVariable*), 64);
        SNEPPXVariable** b2 = (SNEPPXVariable**)SNEPPX_malloc(N * sizeof(SNEPPXVariable*), 64);
        if (!w1 || !b1 || !w2 || !b2) {
            SNEPPX_free(w1, N * sizeof(SNEPPXVariable*));
            SNEPPX_free(b1, N * sizeof(SNEPPXVariable*));
            SNEPPX_free(w2, N * sizeof(SNEPPXVariable*));
            SNEPPX_free(b2, N * sizeof(SNEPPXVariable*));
            return -1;
        }

        for (size_t e = 0; e < N; e++) {
            w1[e] = weight_vars[wi++];
            b1[e] = weight_vars[wi++];
            w2[e] = weight_vars[wi++];
            b2[e] = weight_vars[wi++];
        }

        SNEPPXVariable* router_T = SNEPPX_transpose(tape, router, 0, 1);
        SNEPPXVariable* logits;
        if (model->config.use_mlp_gater) {
            SNEPPXVariable* hidden = SNEPPX_matmul(tape, cur_input, gw1);
            hidden = SNEPPX_add(tape, hidden, gb1);
            hidden = SNEPPX_relu(tape, hidden);
            SNEPPXVariable* gater_T = SNEPPX_transpose(tape, gw2, 0, 1);
            logits = SNEPPX_matmul(tape, hidden, gater_T);
            logits = SNEPPX_add(tape, logits, gb2);
        } else {
            logits = SNEPPX_matmul(tape, cur_input, router_T);
        }
        SNEPPXVariable* gate = SNEPPX_softmax(tape, logits, 1);

        size_t ones_shape[] = {1, O};
        SNEPPXTensor* ones_t = SNEPPX_tensor_ones(ones_shape, 2, SNEPPX_FLOAT32);
        SNEPPXVariable* ones_v = SNEPPX_variable_create(ones_t, 0);
        SNEPPX_tape_record(tape, ones_v);

        size_t acc_shape[] = {B, O};
        SNEPPXTensor* acc_t = SNEPPX_tensor_zeros(acc_shape, 2, SNEPPX_FLOAT32);
        SNEPPXVariable* acc = SNEPPX_variable_create(acc_t, 0);
        SNEPPX_tape_record(tape, acc);

        for (size_t e = 0; e < N; e++) {
            float* hot_data = (float*)SNEPPX_malloc(N * sizeof(float), 64);
            if (!hot_data) { SNEPPX_free(w1, N * sizeof(SNEPPXVariable*)); SNEPPX_free(b1, N * sizeof(SNEPPXVariable*)); SNEPPX_free(w2, N * sizeof(SNEPPXVariable*)); SNEPPX_free(b2, N * sizeof(SNEPPXVariable*)); return -1; }
            memset(hot_data, 0, N * sizeof(float));
            hot_data[e] = 1.0f;
            size_t hot_shape[] = {1, N};
            SNEPPXTensor* hot_t = SNEPPX_tensor_create(hot_shape, 2, SNEPPX_FLOAT32);
            if (!hot_t) { SNEPPX_free(hot_data, N * sizeof(float)); return -1; }
            memcpy((float*)hot_t->data, hot_data, N * sizeof(float));
            SNEPPX_free(hot_data, N * sizeof(float));

            SNEPPXVariable* hot_v = SNEPPX_variable_create(hot_t, 0);
            SNEPPX_tape_record(tape, hot_v);
            SNEPPXVariable* hot_T = SNEPPX_transpose(tape, hot_v, 0, 1);
            SNEPPXVariable* gw = SNEPPX_matmul(tape, gate, hot_T);
            SNEPPXVariable* gw_exp = SNEPPX_matmul(tape, gw, ones_v);

            SNEPPXVariable* w1_T = SNEPPX_transpose(tape, w1[e], 0, 1);
            SNEPPXVariable* hidden = SNEPPX_matmul(tape, cur_input, w1_T);
            hidden = SNEPPX_add(tape, hidden, b1[e]);
            hidden = SNEPPX_relu(tape, hidden);

            SNEPPXVariable* w2_T = SNEPPX_transpose(tape, w2[e], 0, 1);
            SNEPPXVariable* exp_out = SNEPPX_matmul(tape, hidden, w2_T);
            exp_out = SNEPPX_add(tape, exp_out, b2[e]);

            SNEPPXVariable* weighted = SNEPPX_mul(tape, exp_out, gw_exp);
            acc = SNEPPX_add(tape, acc, weighted);
        }

        SNEPPX_free(w1, N * sizeof(SNEPPXVariable*));
        SNEPPX_free(b1, N * sizeof(SNEPPXVariable*));
        SNEPPX_free(w2, N * sizeof(SNEPPXVariable*));
        SNEPPX_free(b2, N * sizeof(SNEPPXVariable*));

        if (l == total_layers - 1) {
            *output_var = SNEPPX_sum(tape, acc, 0);
        } else {
            cur_input = acc;
            cur_dim = O;
        }
    }

    return 0;
}
