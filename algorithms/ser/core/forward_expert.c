#include "sparse_expert_routing.h"
#include <math.h>
#include <stdlib.h>
/*
 * SNEPPX - SER Per-Expert Computation
 *
 * WHAT
 *   SER Per-Expert Computation.
 *
 * CONCEPT
 *   Per-expert computation dispatcher that routes tokens to experts and collects results.
 *
 * ROLE
 *   Implements the per-expert forward pass that applies the expert network to each token assigned to that expert.
 *
 * REFERENCES
 *   None (internal algorithm).
 */



static float gelu(float x) {
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x3)));
}

static float swish(float x) {
    return x / (1.0f + expf(-x));
}

/**
 * @brief Run the forward pass for Ser Expert.
 *
 * @param expert [in] Expert value.
 * @param input [in] Input value.
 */
void SNEPPX_ser_expert_forward(const SNEPPXExpert* expert, const SNEPPXTensor* input, SNEPPXTensor* output) {
    size_t num_tokens = input->shape[0];
    size_t i_dim = input->shape[1];
    size_t e_dim = expert->w1->shape[0];
    size_t o_dim = expert->w2->shape[0];

    float* w1_data = (float*)expert->w1->data;
    float* w2_data = (float*)expert->w2->data;
    float* b1_data = (float*)expert->b1->data;
    float* b2_data = (float*)expert->b2->data;
    float* in_data = (float*)input->data;
    float* out_data = (float*)output->data;

    for (size_t t = 0; t < num_tokens; t++) {
        float* in_t = in_data + t * i_dim;
        float* out_t = out_data + t * o_dim;

        float* hidden = (float*)malloc(e_dim * sizeof(float));
        if (!hidden) return;

        for (size_t e = 0; e < e_dim; e++) {
            float sum = b1_data[e];
            for (size_t i = 0; i < i_dim; i++) {
                sum += w1_data[e * i_dim + i] * in_t[i];
            }
            switch (expert->activation) {
                case SNEPPX_ACT_RELU: hidden[e] = (sum > 0.0f) ? sum : 0.0f; break;
                case SNEPPX_ACT_GELU: hidden[e] = gelu(sum); break;
                case SNEPPX_ACT_SWISH: hidden[e] = swish(sum); break;
                default: hidden[e] = sum; break;
            }
        }

        for (size_t o = 0; o < o_dim; o++) {
            float sum = b2_data[o];
            for (size_t e = 0; e < e_dim; e++) {
                sum += w2_data[o * e_dim + e] * hidden[e];
            }
            out_t[o] = sum;
        }

        free(hidden);
    }
}
