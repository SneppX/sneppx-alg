#include "hierarchical_state_space.h"
#include "polymorphic_memory_allocator.h"
#include <stdlib.h>
#include <string.h>
/*
 * SNEPPX - HSS Model Assembly
 *
 * WHAT
 *   HSS Model Assembly.
 *
 * CONCEPT
 *   Assembles a complete HSS model from layers, parameters, and configuration.
 *
 * ROLE
 *   Public API for building a full HSS model from individual components.
 *
 * REFERENCES
 *   None (internal module).
 */



/**
 * @brief Create Hss Model.
 *
 * @param config [in] Config value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXHSSModel* SNEPPX_hss_model_create(const SNEPPXHSSConfig* config, unsigned int seed) {
    SNEPPXHSSModel* model = (SNEPPXHSSModel*)SNEPPX_malloc(sizeof(SNEPPXHSSModel), 64);
    if (!model) return NULL;
    memset(model, 0, sizeof(SNEPPXHSSModel));

    model->config = *config;

    model->layers = (SNEPPXHSSLayer**)SNEPPX_malloc(config->num_layers * sizeof(SNEPPXHSSLayer*), 64);
    if (!model->layers) {
        SNEPPX_free(model, sizeof(SNEPPXHSSModel));
        return NULL;
    }
    memset(model->layers, 0, config->num_layers * sizeof(SNEPPXHSSLayer*));

    model->norm_gamma = (SNEPPXTensor**)SNEPPX_malloc(config->num_layers * sizeof(SNEPPXTensor*), 64);
    model->norm_beta = (SNEPPXTensor**)SNEPPX_malloc(config->num_layers * sizeof(SNEPPXTensor*), 64);
    if (!model->norm_gamma || !model->norm_beta) {
        SNEPPX_free(model->layers, config->num_layers * sizeof(SNEPPXHSSLayer*));
        SNEPPX_free(model->norm_gamma, config->num_layers * sizeof(SNEPPXTensor*));
        SNEPPX_free(model->norm_beta, config->num_layers * sizeof(SNEPPXTensor*));
        SNEPPX_free(model, sizeof(SNEPPXHSSModel));
        return NULL;
    }
    memset(model->norm_gamma, 0, config->num_layers * sizeof(SNEPPXTensor*));
    memset(model->norm_beta, 0, config->num_layers * sizeof(SNEPPXTensor*));

    for (size_t i = 0; i < config->num_layers; i++) {
        model->layers[i] = SNEPPX_hss_layer_create(config, seed + (unsigned int)i);

        size_t shape_dim[] = {config->input_dim};
        model->norm_gamma[i] = SNEPPX_tensor_ones(shape_dim, 1, SNEPPX_FLOAT32);
        model->norm_beta[i] = SNEPPX_tensor_zeros(shape_dim, 1, SNEPPX_FLOAT32);
    }

    return model;
}

/**
 * @brief Perform Hss Get Params.
 *
 * @param model [in] Model value.
 * @param out [out] Out value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_hss_get_params(const SNEPPXHSSModel* model, SNEPPXTensor** out, size_t max_out) {
    if (!model) return 0;

    size_t count = 0;
    for (size_t l = 0; l < model->config.num_layers; l++) {
        SNEPPXHSSLayer* layer = model->layers[l];
        SNEPPXTensor* tensors[9] = {
            layer->A, layer->B, layer->C, layer->D, layer->dt,
            layer->x_proj, layer->x_proj_bias,
            model->norm_gamma[l], model->norm_beta[l]
        };
        for (int i = 0; i < 9; i++) {
            if (out && count < max_out) out[count] = tensors[i];
            count++;
        }
    }
    return count;
}

/**
 * @brief Destroy Hss Model.
 */
void SNEPPX_hss_model_destroy(SNEPPXHSSModel* model) {
    if (!model) return;
    for (size_t i = 0; i < model->config.num_layers; i++) {
        if (model->layers[i]) SNEPPX_hss_layer_destroy(model->layers[i]);
        if (model->norm_gamma[i]) SNEPPX_tensor_destroy(model->norm_gamma[i]);
        if (model->norm_beta[i]) SNEPPX_tensor_destroy(model->norm_beta[i]);
    }
    SNEPPX_free(model->layers, model->config.num_layers * sizeof(SNEPPXHSSLayer*));
    SNEPPX_free(model->norm_gamma, model->config.num_layers * sizeof(SNEPPXTensor*));
    SNEPPX_free(model->norm_beta, model->config.num_layers * sizeof(SNEPPXTensor*));
    SNEPPX_free(model, sizeof(SNEPPXHSSModel));
}
