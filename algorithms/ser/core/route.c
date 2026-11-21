#include "sparse_expert_routing.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
/*
 * SNEPPX - SER Expert Dispatch/Combine
 *
 * WHAT
 *   SER Expert Dispatch/Combine.
 *
 * CONCEPT
 *   All-to-all expert dispatch and combine for the SER architecture.
 *
 * ROLE
 *   Implements the routing pattern that scatters tokens to expert ranks and gathers results back, using an all-to-all communication pattern.
 *
 * REFERENCES
 *   None (internal algorithm).
 */



static unsigned long lcg_next(unsigned long state) {
    return state * 1103515245UL + 12345UL;
}

static void softmax(float* data, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; r++) {
        float max_val = data[r * cols];
        for (size_t c = 1; c < cols; c++) {
            if (data[r * cols + c] > max_val) max_val = data[r * cols + c];
        }
        float sum = 0.0f;
        for (size_t c = 0; c < cols; c++) {
            data[r * cols + c] = expf(data[r * cols + c] - max_val);
            sum += data[r * cols + c];
        }
        for (size_t c = 0; c < cols; c++) {
            data[r * cols + c] /= sum;
        }
    }
}

static int cmp_desc(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) ? -1 : (fa < fb) ? 1 : 0;
}

/**
 * @brief Perform Ser Route.
 *
 * @param layer [out] Layer value.
 * @param input [in] Input value.
 * @param gate_weights [out] Gate Weights value.
 */
void SNEPPX_ser_route(SNEPPXSERLayer* layer, const SNEPPXTensor* input, SNEPPXTensor** gate_weights, int** expert_indices) {
    size_t num_tokens = input->shape[0];
    size_t i_dim = input->shape[1];
    size_t n_exp = layer->config.num_experts;
    size_t n_act = layer->config.num_active;

    float* router_data = (float*)layer->router->data;
    float* bias_data = (float*)layer->router_bias->data;
    float* in_data = (float*)input->data;

    float* logits = (float*)SNEPPX_malloc(n_exp * num_tokens * sizeof(float), 16);
    if (!logits) return;

    for (size_t e = 0; e < n_exp; e++) {
        for (size_t t = 0; t < num_tokens; t++) {
            float sum = bias_data[e];
            for (size_t i = 0; i < i_dim; i++) {
                sum += router_data[e * i_dim + i] * in_data[t * i_dim + i];
            }
            logits[e * num_tokens + t] = sum;
        }
    }

    if (layer->config.top_k_method == SNEPPX_TOPK_NOISY) {
        unsigned long state = 12345;
        for (size_t i = 0; i < n_exp * num_tokens; i++) {
            state = lcg_next(state);
            float noise = ((state >> 16) & 0x7FFF) / 32767.0f * 0.1f;
            logits[i] += noise;
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
    *expert_indices = (int*)SNEPPX_malloc(num_tokens * n_act * sizeof(int), 16);
    if (!*gate_weights || !*expert_indices) {
        SNEPPX_free(logits, n_exp * num_tokens * sizeof(float));
        return;
    }

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

        float* gw_data = (float*)(*gate_weights)->data;
        int* ei_data = *expert_indices;
        float sum_gw = 0.0f;
        for (size_t k = 0; k < n_act; k++) {
            gw_data[t * n_act + k] = scores[k];
            ei_data[t * n_act + k] = idxs[k];
            sum_gw += scores[k];
        }
        if (sum_gw > 0.0f) {
            for (size_t k = 0; k < n_act; k++) {
                gw_data[t * n_act + k] /= sum_gw;
            }
        }
    }

    SNEPPX_free(logits, n_exp * num_tokens * sizeof(float));
}

// Gating forward with learned temperature scaling
// Returns pre-softmax logits for z-loss computation
/**
 * @brief Run the forward pass for Ser Gate.
 *
 * @param layer [in] Layer value.
 * @param input [in] Input value.
 * @param gate_weights [out] Gate Weights value.
 * @param expert_indices [out] Expert Indices value.
 * @param gate_logits [out] Gate Logits value.
 */
void SNEPPX_ser_gate_forward(const SNEPPXSERLayer* layer, const SNEPPXTensor* input,
                           SNEPPXTensor** gate_weights, int** expert_indices,
                           SNEPPXTensor** gate_logits, float temperature) {
    size_t num_tokens = input->shape[0];
    size_t i_dim = input->shape[1];
    size_t n_exp = layer->config.num_experts;
    size_t n_act = layer->config.num_active;

    float* router_data = (float*)layer->router->data;
    float* bias_data = (float*)layer->router_bias->data;
    float* in_data = (float*)input->data;

    float inv_temp = (temperature > 0.0f) ? (1.0f / temperature) : 1.0f;

    size_t logits_shape[] = {num_tokens, n_exp};
    *gate_logits = SNEPPX_tensor_create(logits_shape, 2, SNEPPX_FLOAT32);
    if (!*gate_logits) return;
    float* logits_data = (float*)(*gate_logits)->data;

    for (size_t t = 0; t < num_tokens; t++) {
        for (size_t e = 0; e < n_exp; e++) {
            float sum = bias_data[e];
            for (size_t i = 0; i < i_dim; i++) {
                sum += router_data[e * i_dim + i] * in_data[t * i_dim + i];
            }
            logits_data[t * n_exp + e] = sum * inv_temp;
        }
    }

    float* sm_data = (float*)SNEPPX_malloc(num_tokens * n_exp * sizeof(float), 16);
    if (!sm_data) { SNEPPX_tensor_destroy(*gate_logits); *gate_logits = NULL; return; }
    memcpy(sm_data, logits_data, num_tokens * n_exp * sizeof(float));
    softmax(sm_data, num_tokens, n_exp);

    size_t shape_gw[] = {num_tokens, n_act};
    *gate_weights = SNEPPX_tensor_create(shape_gw, 2, SNEPPX_FLOAT32);
    *expert_indices = (int*)SNEPPX_malloc(num_tokens * n_act * sizeof(int), 16);
    if (!*gate_weights || !*expert_indices) {
        SNEPPX_free(sm_data, num_tokens * n_exp * sizeof(float));
        SNEPPX_tensor_destroy(*gate_logits);
        *gate_logits = NULL;
        return;
    }

    for (size_t t = 0; t < num_tokens; t++) {
        float scores[256];
        int idxs[256];
        for (size_t e = 0; e < n_exp; e++) {
            scores[e] = sm_data[t * n_exp + e];
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
        float* gw_data = (float*)(*gate_weights)->data;
        int* ei_data = *expert_indices;
        float sum_gw = 0.0f;
        for (size_t k = 0; k < n_act; k++) {
            gw_data[t * n_act + k] = scores[k];
            ei_data[t * n_act + k] = idxs[k];
            sum_gw += scores[k];
        }
        if (sum_gw > 0.0f) {
            for (size_t k = 0; k < n_act; k++) {
                gw_data[t * n_act + k] /= sum_gw;
            }
        }
    }

    SNEPPX_free(sm_data, num_tokens * n_exp * sizeof(float));
}

// Z-loss: auxiliary loss that keeps gate logits near zero
// Computes mean(gate_logits^2) to penalize large logit magnitudes
/**
 * @brief Perform Ser Z Loss.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_ser_z_loss(const SNEPPXTensor* gate_logits) {
    size_t n = gate_logits->size;
    float* data = (float*)gate_logits->data;
    float sum_sq = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum_sq += data[i] * data[i];
    }
    return (n > 0) ? (sum_sq / (float)n) : 0.0f;
}

// Combined auxiliary loss: load balancing + z-loss
/**
 * @brief Perform Ser Aux Loss.
 *
 * @param gate_weights [in] Gate Weights value.
 * @param expert_indices [in] Expert Indices value.
 * @param gate_logits [in] Gate Logits value.
 * @param num_tokens [in] Num Tokens value.
 * @param load_balance_coef [in] Load Balance Coef value.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_ser_aux_loss(const SNEPPXTensor* gate_weights, const int* expert_indices,
                        const SNEPPXTensor* gate_logits, size_t num_tokens,
                        float load_balance_coef, float z_loss_coef) {
    float lb_loss = SNEPPX_ser_load_balance_loss(gate_weights, expert_indices, num_tokens);
    float zl_loss = SNEPPX_ser_z_loss(gate_logits);
    return load_balance_coef * lb_loss + z_loss_coef * zl_loss;
}

// Enforce expert capacity: drop lowest-weighted tokens from overloaded experts
/**
 * @brief Perform Ser Expert Capacity Balance.
 *
 * @param gate_weights [out] Gate Weights value.
 * @param expert_indices [out] Expert Indices value.
 * @param num_tokens [in] Num Tokens value.
 * @param num_active [in] Num Active value.
 */
void SNEPPX_ser_expert_capacity_balance(SNEPPXTensor* gate_weights, int* expert_indices,
                                      size_t num_tokens, size_t num_active,
                                      size_t expert_capacity) {
    if (expert_capacity == 0) return;

    int n_exp = 0;
    for (size_t i = 0; i < num_tokens * num_active; i++) {
        if (expert_indices[i] + 1 > n_exp) n_exp = expert_indices[i] + 1;
    }
    if (n_exp == 0) return;

    size_t* counts = (size_t*)calloc((size_t)n_exp, sizeof(size_t));
    if (!counts) return;

    float* gw_data = (float*)gate_weights->data;
    for (size_t t = 0; t < num_tokens; t++) {
        for (size_t k = 0; k < num_active; k++) {
            int e = expert_indices[t * num_active + k];
            if (e >= 0 && e < n_exp) counts[e]++;
        }
    }

    for (int e = 0; e < n_exp; e++) {
        if (counts[e] > expert_capacity) {
            typedef struct { float w; size_t idx; } WIdx;
            WIdx* pairs = (WIdx*)malloc(counts[e] * sizeof(WIdx));
            if (!pairs) continue;
            size_t cnt = 0;
            for (size_t t = 0; t < num_tokens; t++) {
                for (size_t k = 0; k < num_active; k++) {
                    size_t idx = t * num_active + k;
                    if (expert_indices[idx] == e) {
                        pairs[cnt].w = gw_data[idx];
                        pairs[cnt].idx = idx;
                        cnt++;
                    }
                }
            }
            if (cnt > 0) {
                for (size_t i = 0; i < cnt - 1; i++) {
                    for (size_t j = 0; j < cnt - 1 - i; j++) {
                        if (pairs[j].w < pairs[j + 1].w) {
                            WIdx tmp = pairs[j];
                            pairs[j] = pairs[j + 1];
                            pairs[j + 1] = tmp;
                        }
                    }
                }
                for (size_t i = expert_capacity; i < cnt; i++) {
                    gw_data[pairs[i].idx] = 0.0f;
                    expert_indices[pairs[i].idx] = -1;
                }
            }
            free(pairs);
        }
    }

    free(counts);
}
