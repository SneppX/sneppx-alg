#ifndef SNEPPX_ACTIVATIONS_H
#define SNEPPX_ACTIVATIONS_H

#include <stddef.h>

#ifdef __cplusplus
/*
 * SNEPPX - Activations
 *
 * WHAT
 *   Activations.
 *
 * CONCEPT
 *   Provides activation functions.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif

typedef enum {
    SNEPPX_ACT_RELU,
    SNEPPX_ACT_LEAKY_RELU,
    SNEPPX_ACT_PRELU,
    SNEPPX_ACT_ELU,
    SNEPPX_ACT_SELU,
    SNEPPX_ACT_GELU_TANH,
    SNEPPX_ACT_GELU_ERF,
    SNEPPX_ACT_SILU,
    SNEPPX_ACT_SIGMOID,
    SNEPPX_ACT_TANH,
    SNEPPX_ACT_SOFTMAX,
    SNEPPX_ACT_LOG_SOFTMAX,
} SNEPPXActivationType;

/**
 * @brief Run the forward pass for Act.
 *
 * @param x [in] X value.
 * @param output [out] Output value.
 * @param act [in] Act value.
 * @param n [in] N value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_act_forward(const float* x, float* output, SNEPPXActivationType act, size_t n);
/**
 * @brief Run the backward pass for Act.
 *
 * @param x [in] X value.
 * @param grad_out [in] Grad Out value.
 * @param grad_in [out] Grad In value.
 * @param act [in] Act value.
 * @param n [in] N value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_act_backward(const float* x, const float* grad_out, float* grad_in, SNEPPXActivationType act, size_t n);

/**
 * @brief Perform Act Softmax Stable.
 *
 * @param x [in] X value.
 * @param output [out] Output value.
 * @param n [in] N value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_act_softmax_stable(const float* x, float* output, size_t n);
/**
 * @brief Perform Act Log Softmax.
 *
 * @param x [in] X value.
 * @param output [out] Output value.
 * @param n [in] N value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_act_log_softmax(const float* x, float* output, size_t n);

/**
 * @brief Perform Act Leaky Relu.
 *
 * @param x [in] X value.
 * @param output [out] Output value.
 * @param alpha [in] Alpha value.
 * @param n [in] N value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_act_leaky_relu(const float* x, float* output, float alpha, size_t n);
/**
 * @brief Perform Act Prelu.
 *
 * @param x [in] X value.
 * @param output [out] Output value.
 * @param alpha [in] Alpha value.
 * @param n [in] N value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_act_prelu(const float* x, float* output, const float* alpha, size_t n);
/**
 * @brief Perform Act Elu.
 *
 * @param x [in] X value.
 * @param output [out] Output value.
 * @param alpha [in] Alpha value.
 * @param n [in] N value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_act_elu(const float* x, float* output, float alpha, size_t n);
/**
 * @brief Perform Act Selu.
 *
 * @param x [in] X value.
 * @param output [out] Output value.
 * @param n [in] N value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_act_selu(const float* x, float* output, size_t n);
/**
 * @brief Perform Act Gelu Tanh.
 *
 * @param x [in] X value.
 * @param output [out] Output value.
 * @param n [in] N value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_act_gelu_tanh(const float* x, float* output, size_t n);
/**
 * @brief Perform Act Gelu Erf.
 *
 * @param x [in] X value.
 * @param output [out] Output value.
 * @param n [in] N value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_act_gelu_erf(const float* x, float* output, size_t n);

typedef enum {
    SNEPPX_GATED_SWIGLU,
    SNEPPX_GATED_GEGLU,
    SNEPPX_GATED_REGLU,
} SNEPPXGatedActType;

/**
 * @brief Run the forward pass for Gated Activation.
 *
 * @param x [in] X value.
 * @param gate [in] Gate value.
 * @param output [out] Output value.
 * @param act [in] Act value.
 * @param n [in] N value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_gated_activation_forward(const float* x, const float* gate, float* output, SNEPPXGatedActType act, size_t n);
/**
 * @brief Run the backward pass for Gated Activation.
 *
 * @param x [in] X value.
 * @param gate [in] Gate value.
 * @param grad_out [in] Grad Out value.
 * @param grad_x [out] Grad X value.
 * @param grad_gate [out] Grad Gate value.
 * @param act [in] Act value.
 * @param n [in] N value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_gated_activation_backward(const float* x, const float* gate, const float* grad_out, float* grad_x, float* grad_gate, SNEPPXGatedActType act, size_t n);
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
 * @param hidden_dim [in] Hidden Dim value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_gated_ffn_forward(const float* x, const float* w1, const float* w2, const float* w3, float* output, SNEPPXGatedActType act, size_t dim, size_t hidden_dim);

#ifdef __cplusplus
}
#endif

#endif
