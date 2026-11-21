#include "hierarchical_state_space.h"
/*
 * SNEPPX - HSS Single Step
 *
 * WHAT
 *   HSS Single Step.
 *
 * CONCEPT
 *   Single-step HSS state transition for one time step.
 *
 * ROLE
 *   Implements one step of the HSS recurrence for use in autoregressive decoding and step-by-step inference.
 *
 * REFERENCES
 *   None (internal algorithm).
 */



/**
 * @brief Perform Hss Step.
 *
 * @param layer [in] Layer value.
 * @param x [in] X value.
 */
void SNEPPX_hss_step(const SNEPPXHSSLayer* layer, const SNEPPXTensor* x, SNEPPXTensor* h_next) {
    size_t s_dim = layer->A_bar->shape[0];
    size_t i_dim = x->size;

    float* A_bar = (float*)layer->A_bar->data;
    float* B_bar = (float*)layer->B_bar->data;
    float* h = (float*)layer->h->data;
    float* x_data = (float*)x->data;
    float* h_next_data = (float*)h_next->data;

    for (size_t i = 0; i < s_dim; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < s_dim; j++) {
            sum += A_bar[i * s_dim + j] * h[j];
        }
        for (size_t k = 0; k < i_dim; k++) {
            sum += B_bar[i * i_dim + k] * x_data[k];
        }
        h_next_data[i] = sum;
    }
}
