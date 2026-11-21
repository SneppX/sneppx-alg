#include "adversarial_robustness_certification.h"
#include "automatic_differentiation_framework.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

/**
 * @brief Perform Arc Build Train Graph.
 *
 * @param layer [out] Layer value.
 * @param tape [out] Tape value.
 * @param input_var [out] Input Var value.
 * @param weight_vars [out] Weight Vars value.
 * @param num_weights [in] Num Weights value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_arc_build_train_graph(SNEPPXARCLayer* layer, SNEPPXTape* tape,
                                SNEPPXVariable* input_var,
                                SNEPPXVariable** weight_vars, size_t num_weights,
                                SNEPPXVariable** output_var) {
    if (!layer || !tape || !input_var || !weight_vars || !output_var) return -1;
    size_t n_verifier_layers = layer->output_verifier->num_layers;
    size_t expected = 1 + 2 * n_verifier_layers;
    if (num_weights < expected) return -1;

    size_t wi = 0;
    SNEPPXVariable* proj_w = weight_vars[wi++];

    /* Input guard: project through projection_matrix (always sanitize during training) */
    SNEPPXVariable* proj_w_t = SNEPPX_transpose(tape, proj_w, 0, 1);
    SNEPPXVariable* current = SNEPPX_matmul(tape, input_var, proj_w_t);
    
    /* Output verifier: MLP with ReLU */
    for (size_t l = 0; l < n_verifier_layers; l++) {
        SNEPPXVariable* w = weight_vars[wi++];
        SNEPPXVariable* b = weight_vars[wi++];
        SNEPPXVariable* w_t = SNEPPX_transpose(tape, w, 0, 1);
        SNEPPXVariable* out = SNEPPX_matmul(tape, current, w_t);
        out = SNEPPX_add(tape, out, b);
        current = SNEPPX_relu(tape, out);
    }

    *output_var = current;
    return 0;
}

/**
 * @brief Perform Arc Build Adversarial Train Graph.
 *
 * @param layer [out] Layer value.
 * @param tape [out] Tape value.
 * @param input_var [out] Input Var value.
 * @param weight_vars [out] Weight Vars value.
 * @param num_weights [in] Num Weights value.
 * @param clean_output [out] Clean Output value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_arc_build_adversarial_train_graph(SNEPPXARCLayer* layer, SNEPPXTape* tape,
                                            SNEPPXVariable* input_var,
                                            SNEPPXVariable** weight_vars, size_t num_weights,
                                            SNEPPXVariable** clean_output,
                                            SNEPPXVariable** adv_output) {
    if (!layer || !tape || !input_var || !weight_vars || !clean_output || !adv_output) return -1;

    int attack_type = layer->config.attack_simulation_types;
    float epsilon = layer->config.attack_epsilon;
    if (attack_type == 0) {
        return SNEPPX_arc_build_train_graph(layer, tape, input_var, weight_vars, num_weights, clean_output);
    }

    int first_type = 0;
    if (attack_type & SNEPPX_ATTACK_FGSM) first_type = SNEPPX_ATTACK_FGSM;
    else if (attack_type & SNEPPX_ATTACK_PGD) first_type = SNEPPX_ATTACK_PGD;
    else if (attack_type & SNEPPX_ATTACK_CW) first_type = SNEPPX_ATTACK_CW;

    SNEPPXTensor* adv_t = NULL;
    SNEPPX_arc_simulate_attack(input_var->data, first_type, epsilon, &adv_t);
    if (!adv_t) {
        return -1;
    }

    SNEPPXVariable* adv_var = SNEPPX_variable_create(adv_t, 0);
    SNEPPX_tape_record(tape, adv_var);

    int rc = SNEPPX_arc_build_train_graph(layer, tape, input_var, weight_vars, num_weights, clean_output);
    if (rc != 0) { SNEPPX_tensor_destroy(adv_t); return rc; }

    rc = SNEPPX_arc_build_train_graph(layer, tape, adv_var, weight_vars, num_weights, adv_output);
    if (rc != 0) { SNEPPX_tensor_destroy(adv_t); return -1; }

    return 0;
}
