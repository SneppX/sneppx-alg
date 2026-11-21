#include "adversarial_robustness_certification.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
/*
 * SNEPPX - ARC Layer Wrapper
 *
 * WHAT
 *   ARC Layer Wrapper.
 *
 * CONCEPT
 *   Initialization, forward, and destruction of the ARC layer object.
 *
 * ROLE
 *   Public API for creating and managing ARC layers.
 *
 * REFERENCES
 *   None (internal module).
 */



/**
 * @brief Create Arc Layer.
 *
 * @param config [in] Config value.
 * @param input_dim [in] Input Dim value.
 * @param output_dim [in] Output Dim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXARCLayer* SNEPPX_arc_layer_create(const SNEPPXARCConfig* config, size_t input_dim, size_t output_dim, unsigned int seed) {
    SNEPPXARCLayer* layer = (SNEPPXARCLayer*)SNEPPX_malloc(sizeof(SNEPPXARCLayer), 64);
    if (!layer) return NULL;
    memset(layer, 0, sizeof(SNEPPXARCLayer));

    layer->config = *config;
    layer->input_dim = input_dim;
    layer->output_dim = output_dim;

    layer->input_guard = SNEPPX_input_guard_create(input_dim, output_dim, seed);
    layer->gradient_obfuscator = SNEPPX_gradient_obfuscator_create(input_dim * output_dim, seed + 100);
    layer->output_verifier = SNEPPX_arc_output_verifier_create(output_dim, config->output_verify_layers, seed + 200);

    size_t shape_ab[] = {input_dim};
    layer->attack_buffer = SNEPPX_tensor_zeros(shape_ab, 1, SNEPPX_FLOAT32);

    return layer;
}

/**
 * @brief Destroy Arc Layer.
 */
void SNEPPX_arc_layer_destroy(SNEPPXARCLayer* layer) {
    if (!layer) return;
    if (layer->input_guard) SNEPPX_input_guard_destroy(layer->input_guard);
    if (layer->gradient_obfuscator) SNEPPX_gradient_obfuscator_destroy(layer->gradient_obfuscator);
    if (layer->output_verifier) SNEPPX_arc_output_verifier_destroy(layer->output_verifier);
    if (layer->attack_buffer) SNEPPX_tensor_destroy(layer->attack_buffer);
    SNEPPX_free(layer, sizeof(SNEPPXARCLayer));
}

/**
 * @brief Perform Arc Get Params.
 *
 * @param layer [in] Layer value.
 * @param out [out] Out value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_arc_get_params(const SNEPPXARCLayer* layer, SNEPPXTensor** out, size_t max_out) {
    if (!layer) return 0;
    size_t total = 1 + 2 * layer->output_verifier->num_layers;
    if (out) {
        size_t idx = 0;
        if (idx < max_out && layer->input_guard && layer->input_guard->projection_matrix)
            out[idx++] = layer->input_guard->projection_matrix;
        for (size_t l = 0; l < layer->output_verifier->num_layers && idx < max_out; l++) {
            if (layer->output_verifier->verification_weights[l])
                out[idx++] = layer->output_verifier->verification_weights[l];
            if (idx < max_out && layer->output_verifier->verification_biases[l])
                out[idx++] = layer->output_verifier->verification_biases[l];
        }
    }
    return total;
}
