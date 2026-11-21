#ifndef SNEPPX_ARC_H
#define SNEPPX_ARC_H

#ifdef __cplusplus
extern "C" {
#endif

#include "multidimensional_tensor_engine.h"
#include "automatic_differentiation_framework.h"
#include <stddef.h>
/*
 * SNEPPX - Adversarial Robustness Certification (ARC)
 *
 * WHAT
 *   Adversarial Robustness Certification (ARC).
 *
 * CONCEPT
 *   ARC model configuration, attack simulation, and verification API.
 *
 * ROLE
 *   Declares the ARC module API for adversarial robustness training and certification.
 *
 * REFERENCES
 *   None (internal module).
 */



typedef enum {
    SNEPPX_OBF_NONE = 0,
    SNEPPX_OBF_NOISE = 1,
    SNEPPX_OBF_CLAMP = 2,
    SNEPPX_OBF_MIXED = 3
} SNEPPXObfuscationMethod;

typedef enum {
    SNEPPX_ATTACK_FGSM = 1,
    SNEPPX_ATTACK_PGD = 2,
    SNEPPX_ATTACK_CW = 4
} SNEPPXAttackType;

typedef struct {
    float input_guard_strength;
    int gradient_obfuscation_method;
    float gradient_noise_scale;
    float gradient_clip_max;
    size_t output_verify_layers;
    float output_verify_threshold;
    int adversarial_training;
    int attack_simulation_types;
    float attack_epsilon;
} SNEPPXARCConfig;

typedef struct {
    SNEPPXTensor* projection_matrix;
    float anomaly_threshold;
    SNEPPXTensor* norm_stats_mean;
    SNEPPXTensor* norm_stats_var;
} SNEPPXInputGuard;

typedef struct {
    SNEPPXTensor* noise_buffer;
    SNEPPXTensor* clamp_mask;
} SNEPPXGradientObfuscator;

typedef struct {
    SNEPPXTensor** verification_weights;
    SNEPPXTensor** verification_biases;
    size_t num_layers;
    SNEPPXTensor* consistency_history;
    size_t history_idx;
    size_t history_filled;
} SNEPPXOutputVerifier;

typedef struct {
    SNEPPXInputGuard* input_guard;
    SNEPPXGradientObfuscator* gradient_obfuscator;
    SNEPPXOutputVerifier* output_verifier;
    SNEPPXARCConfig config;
    SNEPPXTensor* attack_buffer;
    size_t input_dim;
    size_t output_dim;
} SNEPPXARCLayer;

/**
 * @brief Perform Arc Config Default.
 *
 * @return The result value, or 0 on error.
 */
SNEPPXARCConfig SNEPPX_arc_config_default(void);
/**
 * @brief Create Input Guard.
 *
 * @param input_dim [in] Input Dim value.
 * @param output_dim [in] Output Dim value.
 * @param seed [in] Seed value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXInputGuard* SNEPPX_input_guard_create(size_t input_dim, size_t output_dim, unsigned int seed);
/**
 * @brief Destroy Input Guard.
 *
 * @param guard [out] Guard value.
 */
void SNEPPX_input_guard_destroy(SNEPPXInputGuard* guard);
/**
 * @brief Create Gradient Obfuscator.
 *
 * @param max_params [in] Max Params value.
 * @param seed [in] Seed value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXGradientObfuscator* SNEPPX_gradient_obfuscator_create(size_t max_params, unsigned int seed);
/**
 * @brief Destroy Gradient Obfuscator.
 *
 * @param obf [out] Obf value.
 */
void SNEPPX_gradient_obfuscator_destroy(SNEPPXGradientObfuscator* obf);
/**
 * @brief Create Arc Output Verifier.
 *
 * @param output_dim [in] Output Dim value.
 * @param num_layers [in] Num Layers value.
 * @param seed [in] Seed value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXOutputVerifier* SNEPPX_arc_output_verifier_create(size_t output_dim, size_t num_layers, unsigned int seed);
/**
 * @brief Destroy Arc Output Verifier.
 *
 * @param verifier [out] Verifier value.
 */
void SNEPPX_arc_output_verifier_destroy(SNEPPXOutputVerifier* verifier);
/**
 * @brief Create Arc Layer.
 *
 * @param config [in] Config value.
 * @param input_dim [in] Input Dim value.
 * @param output_dim [in] Output Dim value.
 * @param seed [in] Seed value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXARCLayer* SNEPPX_arc_layer_create(const SNEPPXARCConfig* config, size_t input_dim, size_t output_dim, unsigned int seed);
/**
 * @brief Destroy Arc Layer.
 *
 * @param layer [out] Layer value.
 */
void SNEPPX_arc_layer_destroy(SNEPPXARCLayer* layer);
/**
 * @brief Run the forward pass for Arc Input Guard.
 *
 * @param guard [out] Guard value.
 * @param input [in] Input value.
 * @param sanitized [out] Sanitized value.
 * @param anomaly_score [out] Anomaly Score value.
 */
void SNEPPX_arc_input_guard_forward(SNEPPXInputGuard* guard, const SNEPPXTensor* input, SNEPPXTensor** sanitized, float* anomaly_score);
/**
 * @brief Perform Arc Obfuscate Gradients.
 *
 * @param obf [out] Obf value.
 * @param gradients [out] Gradients value.
 * @param method [in] Method value.
 */
void SNEPPX_arc_obfuscate_gradients(SNEPPXGradientObfuscator* obf, SNEPPXTensor* gradients, int method);
/**
 * @brief Perform Arc Verify Output.
 *
 * @param verifier [out] Verifier value.
 * @param output [in] Output value.
 * @param verified_output [out] Verified Output value.
 * @param confidence [out] Confidence value.
 */
void SNEPPX_arc_verify_output(SNEPPXOutputVerifier* verifier, const SNEPPXTensor* output, SNEPPXTensor** verified_output, float* confidence);
/**
 * @brief Run the forward pass for Arc.
 *
 * @param layer [out] Layer value.
 * @param input [in] Input value.
 * @param output [out] Output value.
 * @param security_metrics [out] Security Metrics value.
 */
void SNEPPX_arc_forward(SNEPPXARCLayer* layer, const SNEPPXTensor* input, SNEPPXTensor** output, float* security_metrics);
/**
 * @brief Perform Arc Simulate Attack.
 *
 * @param clean_input [in] Clean Input value.
 * @param attack_type [in] Attack Type value.
 * @param epsilon [in] Epsilon value.
 * @param adversarial [out] Adversarial value.
 */
void SNEPPX_arc_simulate_attack(const SNEPPXTensor* clean_input, int attack_type, float epsilon, SNEPPXTensor** adversarial);
/**
 * @brief Perform Arc Get Params.
 *
 * @param layer [in] Layer value.
 * @param out_params [out] Out Params value.
 * @param max_params [in] Max Params value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_arc_get_params(const SNEPPXARCLayer* layer, SNEPPXTensor** out_params, size_t max_params);
/**
 * @brief Perform Arc Build Train Graph.
 *
 * @param layer [out] Layer value.
 * @param tape [out] Tape value.
 * @param input_var [out] Input Var value.
 * @param weight_vars [out] Weight Vars value.
 * @param num_weights [in] Num Weights value.
 * @param output_var [out] Output Var value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_arc_build_train_graph(SNEPPXARCLayer* layer, SNEPPXTape* tape,
                                SNEPPXVariable* input_var,
                                SNEPPXVariable** weight_vars, size_t num_weights,
                                SNEPPXVariable** output_var);
/**
 * @brief Perform Arc Build Adversarial Train Graph.
 *
 * @param layer [out] Layer value.
 * @param tape [out] Tape value.
 * @param input_var [out] Input Var value.
 * @param weight_vars [out] Weight Vars value.
 * @param num_weights [in] Num Weights value.
 * @param clean_output [out] Clean Output value.
 * @param adv_output [out] Adv Output value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_arc_build_adversarial_train_graph(SNEPPXARCLayer* layer, SNEPPXTape* tape,
                                            SNEPPXVariable* input_var,
                                            SNEPPXVariable** weight_vars, size_t num_weights,
                                            SNEPPXVariable** clean_output,
                                            SNEPPXVariable** adv_output);


#ifdef __cplusplus
}
#endif
#endif /* SNEPPX_ARC_H */
