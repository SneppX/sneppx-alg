#include "sparse_expert_routing.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
/*
 * SNEPPX - SER Layer Forward Pass
 *
 * WHAT
 *   SER Layer Forward Pass.
 *
 * CONCEPT
 *   SER layer forward pass: gating, expert dispatch, and combine.
 *
 * ROLE
 *   Core SER layer that gates input tokens to experts, dispatches them, computes expert outputs, and combines them.
 *
 * REFERENCES
 *   None (internal algorithm).
 */



/**
 * @brief Run the forward pass for Ser.
 *
 * @param layer [out] Layer value.
 * @param input [in] Input value.
 */
void SNEPPX_ser_forward(SNEPPXSERLayer* layer, const SNEPPXTensor* input, SNEPPXTensor** output) {
    fprintf(stderr, "[BENCH] SER layer forward start\n");
    size_t num_tokens = input->shape[0];
    size_t i_dim = input->shape[1];
    size_t n_exp = layer->config.num_experts;
    size_t n_act = layer->config.num_active;
    size_t o_dim = layer->config.output_dim;

    SNEPPXTensor* gate_weights = NULL;
    int* expert_indices = NULL;
    SNEPPX_ser_route(layer, input, &gate_weights, &expert_indices);
    if (!gate_weights || !expert_indices) return;

    size_t shape_out[] = {num_tokens, o_dim};
    *output = SNEPPX_tensor_zeros(shape_out, 2, SNEPPX_FLOAT32);
    if (!*output) {
        SNEPPX_tensor_destroy(gate_weights);
        SNEPPX_free(expert_indices, num_tokens * n_act * sizeof(int));
        return;
    }

    float* gw_data = (float*)gate_weights->data;
    float* acc_data = (float*)(*output)->data;
    float* in_data = (float*)input->data;

    for (size_t e = 0; e < n_exp; e++) {
        int count = 0;
        for (size_t t = 0; t < num_tokens; t++) {
            for (size_t k = 0; k < n_act; k++) {
                if (expert_indices[t * n_act + k] == (int)e) {
                    count++;
                    break;
                }
            }
        }
        if (count == 0) continue;

        float* tok_buf = (float*)malloc((size_t)count * i_dim * sizeof(float));
        float* out_buf = (float*)malloc((size_t)count * o_dim * sizeof(float));
        int* tok_positions = (int*)malloc((size_t)count * sizeof(int));
        float* tok_weights = (float*)malloc((size_t)count * sizeof(float));
        if (!tok_buf || !out_buf || !tok_positions || !tok_weights) {
            free(tok_buf); free(out_buf); free(tok_positions); free(tok_weights);
            continue;
        }

        int idx = 0;
        for (size_t t = 0; t < num_tokens; t++) {
            float w = 0.0f;
            int routed = 0;
            for (size_t k = 0; k < n_act; k++) {
                if (expert_indices[t * n_act + k] == (int)e) {
                    w += gw_data[t * n_act + k];
                    routed = 1;
                }
            }
            if (routed) {
                memcpy(tok_buf + (size_t)idx * i_dim, in_data + t * i_dim, i_dim * sizeof(float));
                tok_positions[idx] = (int)t;
                tok_weights[idx] = w;
                idx++;
            }
        }

        size_t shape_tok[] = {(size_t)count, i_dim};
        size_t shape_out_tok[] = {(size_t)count, o_dim};
        SNEPPXTensor in_tok;
        in_tok.data = tok_buf;
        in_tok.shape = shape_tok;
        in_tok.ndim = 2;
        in_tok.size = (size_t)count * i_dim;
        in_tok.item_size = sizeof(float);
        in_tok.dtype = SNEPPX_FLOAT32;
        in_tok.strides = NULL;

        SNEPPXTensor out_tok;
        out_tok.data = out_buf;
        out_tok.shape = shape_out_tok;
        out_tok.ndim = 2;
        out_tok.size = (size_t)count * o_dim;
        out_tok.item_size = sizeof(float);
        out_tok.dtype = SNEPPX_FLOAT32;
        out_tok.strides = NULL;

        SNEPPX_ser_expert_forward(layer->experts[e], &in_tok, &out_tok);

        for (int i = 0; i < count; i++) {
            size_t pos = (size_t)tok_positions[i];
            float w = tok_weights[i];
            for (size_t o = 0; o < o_dim; o++) {
                acc_data[pos * o_dim + o] += w * out_buf[(size_t)i * o_dim + o];
            }
        }

        free(tok_buf); free(out_buf); free(tok_positions); free(tok_weights);
    }

    if (layer->config.dropout_rate > 0.0f) {
        unsigned long state = 67890;
        for (size_t i = 0; i < num_tokens * o_dim; i++) {
            state = state * 1103515245UL + 12345UL;
            float r = (float)((state >> 16) & 0x7FFF) / 32767.0f;
            if (r < layer->config.dropout_rate) {
                acc_data[i] = 0.0f;
            } else {
                acc_data[i] /= (1.0f - layer->config.dropout_rate);
            }
        }
    }

    SNEPPX_tensor_destroy(gate_weights);
    SNEPPX_free(expert_indices, num_tokens * n_act * sizeof(int));
}
