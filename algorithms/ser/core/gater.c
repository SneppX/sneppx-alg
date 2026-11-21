#include "sparse_expert_routing.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <math.h>
/*
 * SNEPPX - SER Top-K Gating
 *
 * WHAT
 *   SER Top-K Gating.
 *
 * CONCEPT
 *   Softmax-based top-k gating with auxiliary load-balancing loss.
 *
 * ROLE
 *   The gating function computes a probability distribution over experts using softmax on the router logits, selects the top-k experts per token, and adds an auxiliary loss that encourages uniform expert utilization.
 *
 * REFERENCES
 *   None (internal algorithm).
 */



/**
 * @brief Perform Ser Route Mlp Gated.
 *
 * @param layer [out] Layer value.
 * @param input [in] Input value.
 * @param gate_weights [out] Gate Weights value.
 */
void SNEPPX_ser_route_mlp_gated(SNEPPXSERLayer* layer, const SNEPPXTensor* input,
                                SNEPPXTensor** gate_weights, int** expert_indices) {
    if (!layer || !input || !gate_weights || !expert_indices) return;
    size_t num_tokens = input->shape[0];
    size_t i_dim = input->shape[1];
    size_t h_dim = layer->config.gater_hidden_dim;
    size_t n_exp = layer->config.num_experts;
    size_t n_act = layer->config.num_active;

    float* in_d = (float*)input->data;
    float* w1_d = (float*)layer->gater_w1->data;
    float* b1_d = (float*)layer->gater_b1->data;
    float* w2_d = (float*)layer->gater_w2->data;
    float* b2_d = (float*)layer->gater_b2->data;

    float logits[1024];
    for (size_t t = 0; t < num_tokens; t++) {
        float hidden[256];
        for (size_t h = 0; h < h_dim; h++) {
            float sum = b1_d[h];
            for (size_t i = 0; i < i_dim; i++) {
                sum += w1_d[h * i_dim + i] * in_d[t * i_dim + i];
            }
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }
        for (size_t e = 0; e < n_exp; e++) {
            float sum = b2_d[e];
            for (size_t h = 0; h < h_dim; h++) {
                sum += w2_d[e * h_dim + h] * hidden[h];
            }
            logits[e * num_tokens + t] = sum;
        }
    }

    for (size_t t = 0; t < num_tokens; t++) {
        float max_val = logits[t];
        for (size_t e = 1; e < n_exp; e++) {
            if (logits[e * num_tokens + t] > max_val)
                max_val = logits[e * num_tokens + t];
        }
        float sum = 0.0f;
        for (size_t e = 0; e < n_exp; e++) {
            logits[e * num_tokens + t] = expf(logits[e * num_tokens + t] - max_val);
            sum += logits[e * num_tokens + t];
        }
        for (size_t e = 0; e < n_exp; e++) {
            logits[e * num_tokens + t] /= sum;
        }
    }

    size_t shape_gw[] = {num_tokens, n_act};
    *gate_weights = SNEPPX_tensor_create(shape_gw, 2, SNEPPX_FLOAT32);
    *expert_indices = (int*)SNEPPX_malloc(num_tokens * n_act * sizeof(int), 64);
    if (!*gate_weights || !*expert_indices) return;

    for (size_t t = 0; t < num_tokens; t++) {
        float scores[256];
        int idxs[256];
        for (size_t e = 0; e < n_exp; e++) {
            scores[e] = logits[e * num_tokens + t];
            idxs[e] = (int)e;
        }
        for (size_t i = 0; i < n_exp - 1; i++) {
            for (size_t j = 0; j < n_exp - 1 - i; j++) {
                if (scores[j] < scores[j + 1]) {
                    float tmp_s = scores[j];
                    scores[j] = scores[j + 1];
                    scores[j + 1] = tmp_s;
                    int tmp_i = idxs[j];
                    idxs[j] = idxs[j + 1];
                    idxs[j + 1] = tmp_i;
                }
            }
        }
        float* gw_d = (float*)(*gate_weights)->data;
        int* ei_d = *expert_indices;
        float sum_gw = 0.0f;
        for (size_t k = 0; k < n_act; k++) {
            gw_d[t * n_act + k] = scores[k];
            ei_d[t * n_act + k] = idxs[k];
            sum_gw += scores[k];
        }
        if (sum_gw > 0.0f) {
            for (size_t k = 0; k < n_act; k++) {
                gw_d[t * n_act + k] /= sum_gw;
            }
        }
    }
}
