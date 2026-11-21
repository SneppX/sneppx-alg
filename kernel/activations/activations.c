#include "activations.h"
#include <math.h>
#include <float.h>
/*
 * SNEPPX - Activation Functions
 *
 * WHAT
 *   Activation Functions.
 *
 * CONCEPT
 *   Activation Functions implementation.
 *
 * ROLE
 *   Core kernel module used throughout the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal kernel module).
 */



/**
 * @brief Run the forward pass for Act.
 *
 * @param x [in] X value.
 * @param output [out] Output value.
 * @param act [in] Act value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_act_forward(const float* x, float* output, SNEPPXActivationType act, size_t n) {
    if (!x || !output || n == 0) return -1;
    switch (act) {
        case SNEPPX_ACT_RELU:
            for (size_t i = 0; i < n; i++) output[i] = x[i] > 0.0f ? x[i] : 0.0f;
            return 0;
        case SNEPPX_ACT_LEAKY_RELU:
            return SNEPPX_act_leaky_relu(x, output, 0.01f, n);
        case SNEPPX_ACT_PRELU:
            return SNEPPX_act_prelu(x, output, x, n);
        case SNEPPX_ACT_ELU:
            return SNEPPX_act_elu(x, output, 1.0f, n);
        case SNEPPX_ACT_SELU:
            return SNEPPX_act_selu(x, output, n);
        case SNEPPX_ACT_GELU_TANH:
            return SNEPPX_act_gelu_tanh(x, output, n);
        case SNEPPX_ACT_GELU_ERF:
            return SNEPPX_act_gelu_erf(x, output, n);
        case SNEPPX_ACT_SILU:
            for (size_t i = 0; i < n; i++) output[i] = x[i] / (1.0f + expf(-x[i]));
            return 0;
        case SNEPPX_ACT_SIGMOID:
            for (size_t i = 0; i < n; i++) output[i] = 1.0f / (1.0f + expf(-x[i]));
            return 0;
        case SNEPPX_ACT_TANH:
            for (size_t i = 0; i < n; i++) output[i] = tanhf(x[i]);
            return 0;
        case SNEPPX_ACT_SOFTMAX:
            return SNEPPX_act_softmax_stable(x, output, n);
        case SNEPPX_ACT_LOG_SOFTMAX:
            return SNEPPX_act_log_softmax(x, output, n);
        default:
            return -1;
    }
}

/**
 * @brief Run the backward pass for Act.
 *
 * @param x [in] X value.
 * @param grad_out [in] Grad Out value.
 * @param grad_in [out] Grad In value.
 * @param act [in] Act value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_act_backward(const float* x, const float* grad_out, float* grad_in, SNEPPXActivationType act, size_t n) {
    if (!x || !grad_out || !grad_in || n == 0) return -1;
    for (size_t i = 0; i < n; i++) {
        float xv = x[i], g = grad_out[i];
        switch (act) {
            case SNEPPX_ACT_RELU:
                grad_in[i] = xv > 0.0f ? g : 0.0f;
                break;
            case SNEPPX_ACT_LEAKY_RELU:
                grad_in[i] = xv > 0.0f ? g : g * 0.01f;
                break;
            case SNEPPX_ACT_GELU_TANH: {
                float c = 0.7978845608028654f;
                float x3 = xv * xv * xv;
                float t = tanhf(c * (xv + 0.044715f * x3));
                float sech2 = 1.0f - t * t;
                float dg = 0.5f * (1.0f + t) + 0.5f * xv * c * (1.0f + 3.0f * 0.044715f * xv * xv) * sech2;
                grad_in[i] = g * dg;
                break;
            }
            case SNEPPX_ACT_SILU: {
                float s = 1.0f / (1.0f + expf(-xv));
                grad_in[i] = g * s * (1.0f + xv * (1.0f - s));
                break;
            }
            case SNEPPX_ACT_SIGMOID: {
                float s = 1.0f / (1.0f + expf(-xv));
                grad_in[i] = g * s * (1.0f - s);
                break;
            }
            case SNEPPX_ACT_TANH: {
                float t = tanhf(xv);
                grad_in[i] = g * (1.0f - t * t);
                break;
            }
            default:
                grad_in[i] = g;
        }
    }
    return 0;
}

/**
 * @brief Perform Act Softmax Stable.
 *
 * @param x [in] X value.
 * @param output [out] Output value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_act_softmax_stable(const float* x, float* output, size_t n) {
    if (!x || !output || n == 0) return -1;
    float maxv = -FLT_MAX;
    for (size_t i = 0; i < n; i++) if (x[i] > maxv) maxv = x[i];
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) { float e = expf(x[i] - maxv); output[i] = e; sum += e; }
    if (sum > 0.0f) for (size_t i = 0; i < n; i++) output[i] /= sum;
    return 0;
}

/**
 * @brief Perform Act Log Softmax.
 *
 * @param x [in] X value.
 * @param output [out] Output value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_act_log_softmax(const float* x, float* output, size_t n) {
    if (!x || !output || n == 0) return -1;
    if (SNEPPX_act_softmax_stable(x, output, n) != 0) return -1;
    for (size_t i = 0; i < n; i++) output[i] = logf(fmaxf(output[i], FLT_MIN));
    return 0;
}

/**
 * @brief Perform Act Leaky Relu.
 *
 * @param x [in] X value.
 * @param output [out] Output value.
 * @param alpha [in] Alpha value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_act_leaky_relu(const float* x, float* output, float alpha, size_t n) {
    if (!x || !output || n == 0) return -1;
    for (size_t i = 0; i < n; i++) output[i] = x[i] > 0.0f ? x[i] : alpha * x[i];
    return 0;
}

/**
 * @brief Perform Act Prelu.
 *
 * @param x [in] X value.
 * @param output [out] Output value.
 * @param alpha [in] Alpha value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_act_prelu(const float* x, float* output, const float* alpha, size_t n) {
    if (!x || !output || !alpha || n == 0) return -1;
    for (size_t i = 0; i < n; i++) output[i] = x[i] > 0.0f ? x[i] : alpha[i % n] * x[i];
    return 0;
}

/**
 * @brief Perform Act Elu.
 *
 * @param x [in] X value.
 * @param output [out] Output value.
 * @param alpha [in] Alpha value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_act_elu(const float* x, float* output, float alpha, size_t n) {
    if (!x || !output || n == 0) return -1;
    for (size_t i = 0; i < n; i++) output[i] = x[i] > 0.0f ? x[i] : alpha * (expf(x[i]) - 1.0f);
    return 0;
}

/**
 * @brief Perform Act Selu.
 *
 * @param x [in] X value.
 * @param output [out] Output value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_act_selu(const float* x, float* output, size_t n) {
    if (!x || !output || n == 0) return -1;
    const float lambda = 1.0507009873554805f;
    const float alpha = 1.6732632423543778f;
    for (size_t i = 0; i < n; i++)
        output[i] = lambda * (x[i] > 0.0f ? x[i] : alpha * (expf(x[i]) - 1.0f));
    return 0;
}

/**
 * @brief Perform Act Gelu Tanh.
 *
 * @param x [in] X value.
 * @param output [out] Output value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_act_gelu_tanh(const float* x, float* output, size_t n) {
    if (!x || !output || n == 0) return -1;
    float c = 0.7978845608028654f;
    for (size_t i = 0; i < n; i++) {
        float xv = x[i];
        output[i] = 0.5f * xv * (1.0f + tanhf(c * (xv + 0.044715f * xv * xv * xv)));
    }
    return 0;
}

/**
 * @brief Perform Act Gelu Erf.
 *
 * @param x [in] X value.
 * @param output [out] Output value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_act_gelu_erf(const float* x, float* output, size_t n) {
    if (!x || !output || n == 0) return -1;
    float rsqrt2 = 0.7071067811865475f;
    for (size_t i = 0; i < n; i++)
        output[i] = 0.5f * x[i] * (1.0f + erff(x[i] * rsqrt2));
    return 0;
}
