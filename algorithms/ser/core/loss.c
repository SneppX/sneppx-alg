#include "sparse_expert_routing.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
/*
 * SNEPPX - SER Load-Balancing Loss
 *
 * WHAT
 *   SER Load-Balancing Loss.
 *
 * CONCEPT
 *   Auxiliary loss that encourages uniform expert utilization across tokens.
 *
 * ROLE
 *   Computes the load-balancing loss term that penalizes the router for sending too many tokens to the same expert, preventing expert collapse.
 *
 * REFERENCES
 *   None (internal algorithm).
 */



/**
 * @brief Perform Ser Load Balance Loss.
 *
 * @param gate_weights [in] Gate Weights value.
 * @param expert_indices [in] Expert Indices value.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_ser_load_balance_loss(const SNEPPXTensor* gate_weights, const int* expert_indices, size_t num_tokens) {
    size_t n_act = gate_weights->shape[1];
    int n_exp = 0;
    for (size_t i = 0; i < num_tokens * n_act; i++) {
        if (expert_indices[i] > n_exp) n_exp = expert_indices[i];
    }
    n_exp++;

    float* frac = (float*)calloc((size_t)n_exp, sizeof(float));
    float* p = (float*)calloc((size_t)n_exp, sizeof(float));
    if (!frac || !p) { free(frac); free(p); return 0.0f; }

    float* gw = (float*)gate_weights->data;
    for (size_t t = 0; t < num_tokens; t++) {
        for (size_t k = 0; k < n_act; k++) {
            int e = expert_indices[t * n_act + k];
            if (e >= 0 && e < n_exp) {
                frac[e] += 1.0f;
                p[e] += gw[t * n_act + k];
            }
        }
    }

    float loss = 0.0f;
    for (int e = 0; e < n_exp; e++) {
        float f_e = frac[e] / (float)num_tokens;
        float p_e = p[e] / (float)num_tokens;
        loss += f_e * p_e;
    }
    loss *= (float)n_exp;

    free(frac);
    free(p);
    return loss;
}

/**
 * @brief Perform Ser Importance Loss.
 *
 * @param gate_weights [in] Gate Weights value.
 * @param expert_indices [in] Expert Indices value.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_ser_importance_loss(const SNEPPXTensor* gate_weights, const int* expert_indices, int num_experts) {
    size_t n_act = gate_weights->shape[1];
    float* frac = (float*)calloc((size_t)num_experts, sizeof(float));
    float* p = (float*)calloc((size_t)num_experts, sizeof(float));
    if (!frac || !p) { free(frac); free(p); return 0.0f; }

    float* gw = (float*)gate_weights->data;
    for (size_t t = 0; t < gate_weights->shape[0]; t++) {
        for (size_t k = 0; k < n_act; k++) {
            int e = expert_indices[t * n_act + k];
            if (e >= 0 && e < num_experts) {
                frac[e] += 1.0f;
                p[e] += gw[t * n_act + k];
            }
        }
    }

    float loss = 0.0f;
    for (int e = 0; e < num_experts; e++) {
        float f_e = frac[e] / (float)gate_weights->shape[0];
        float p_e = p[e] / (float)gate_weights->shape[0];
        loss += f_e * p_e;
    }
    loss *= (float)num_experts;

    free(frac);
    free(p);
    return loss;
}

/**
 * @brief Perform Ser Aux Load Balance.
 *
 * @param gate_weights [in] Gate Weights value.
 * @param expert_indices [in] Expert Indices value.
 * @param num_experts [in] Num Experts value.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_ser_aux_load_balance(const SNEPPXTensor* gate_weights, const int* expert_indices,
                                  int num_experts, size_t num_tokens) {
    SNEPPXTensor dummy_logits = *gate_weights;  // Use gate_weights as dummy logits
    return SNEPPX_ser_aux_loss(gate_weights, expert_indices, &dummy_logits, num_tokens, 1.0f, 0.1f);
}
