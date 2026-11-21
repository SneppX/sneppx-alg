#include "activations.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
/*
 * SNEPPX - Kernel Module
 *
 * WHAT
 *   Kernel Module.
 *
 * CONCEPT
 *   Kernel Module implementation.
 *
 * ROLE
 *   Core kernel module used throughout the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal kernel module).
 */



/**
 * @brief Run the forward pass for Gated Activation.
 *
 * @param x [in] X value.
 * @param gate [in] Gate value.
 * @param output [out] Output value.
 * @param act [in] Act value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_gated_activation_forward(const float* x, const float* gate, float* output, SNEPPXGatedActType act, size_t n) {
    if (!x || !gate || !output) return -1;
    for (size_t i = 0; i < n; i++) {
        float xv = x[i], gv = gate[i], act_val;
        switch (act) {
            case SNEPPX_GATED_SWIGLU:
                act_val = xv / (1.0f + expf(-xv));
                break;
            case SNEPPX_GATED_GEGLU: {
                float c = 0.7978845608028654f;
                float x3 = xv * xv * xv;
                act_val = 0.5f * xv * (1.0f + tanhf(c * (xv + 0.044715f * x3)));
                break;
            }
            case SNEPPX_GATED_REGLU:
                act_val = xv > 0.0f ? xv : 0.0f;
                break;
            default:
                act_val = xv;
        }
        output[i] = act_val * gv;
    }
    return 0;
}

/**
 * @brief Run the backward pass for Gated Activation.
 *
 * @param x [in] X value.
 * @param gate [in] Gate value.
 * @param grad_out [in] Grad Out value.
 * @param grad_x [out] Grad X value.
 * @param grad_gate [out] Grad Gate value.
 * @param act [in] Act value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_gated_activation_backward(const float* x, const float* gate, const float* grad_out, float* grad_x, float* grad_gate, SNEPPXGatedActType act, size_t n) {
    if (!x || !gate || !grad_out || !grad_x || !grad_gate) return -1;
    for (size_t i = 0; i < n; i++) {
        float xv = x[i], gv = gate[i], go = grad_out[i];
        float act_val, d_act;
        switch (act) {
            case SNEPPX_GATED_SWIGLU: {
                float s = 1.0f / (1.0f + expf(-xv));
                act_val = s * xv;
                d_act = s * (1.0f + xv * (1.0f - s));
                break;
            }
            case SNEPPX_GATED_GEGLU: {
                float c = 0.7978845608028654f;
                float x3 = xv * xv * xv;
                float t = tanhf(c * (xv + 0.044715f * x3));
                float sech2 = 1.0f - t * t;
                act_val = 0.5f * xv * (1.0f + t);
                d_act = 0.5f * (1.0f + t) + 0.5f * xv * c * (1.0f + 3.0f * 0.044715f * xv * xv) * sech2;
                break;
            }
            case SNEPPX_GATED_REGLU:
                act_val = xv > 0.0f ? xv : 0.0f;
                d_act = xv > 0.0f ? 1.0f : 0.0f;
                break;
            default:
                act_val = xv;
                d_act = 1.0f;
        }
        grad_x[i] = go * d_act * gv;
        grad_gate[i] = go * act_val;
    }
    return 0;
}

/**
 * @brief Run the forward pass for Gated Ffn.
 *
 * @param x [in] X value.
 * @param w1 [in] W1 value.
 * @param w2 [in] W2 value.
 * @param w3 [in] W3 value.
 * @param output [out] Output value.
 * @param act [in] Act value.
 * @param dim [in] Dim value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_gated_ffn_forward(const float* x, const float* w1, const float* w2, const float* w3, float* output, SNEPPXGatedActType act, size_t dim, size_t hidden_dim) {
    if (!x || !w1 || !w2 || !w3 || !output || dim == 0 || hidden_dim == 0) return -1;
    float* hidden = (float*)malloc((hidden_dim * 2) * sizeof(float));
    if (!hidden) return -1;
    float* gate = hidden + hidden_dim;
    for (size_t j = 0; j < hidden_dim; j++) {
        float s = 0.0f, g = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            s += x[i] * w1[j * dim + i];
            g += x[i] * w3[j * dim + i];
        }
        hidden[j] = s;
        gate[j] = g;
    }
    float* activated = (float*)malloc(hidden_dim * sizeof(float));
    if (!activated) { free(hidden); return -1; }
    SNEPPX_gated_activation_forward(hidden, gate, activated, act, hidden_dim);
    for (size_t j = 0; j < dim; j++) {
        float s = 0.0f;
        for (size_t i = 0; i < hidden_dim; i++)
            s += activated[i] * w2[j * hidden_dim + i];
        output[j] = s;
    }
    free(activated);
    free(hidden);
    return 0;
}
