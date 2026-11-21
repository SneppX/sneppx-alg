#include "sparse_expert_routing.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
/*
 * SNEPPX - SER Model Assembly
 *
 * WHAT
 *   SER Model Assembly.
 *
 * CONCEPT
 *   Assembles a complete SER model from layers, experts, and configuration.
 *
 * ROLE
 *   Public API for building a full SER model.
 *
 * REFERENCES
 *   None (internal module).
 */



/**
 * @brief Create Ser Model.
 *
 * @param config [in] Config value.
 * @param seed [in] Seed value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXSERModel* SNEPPX_ser_model_create(const SNEPPXSERConfig* config, unsigned int seed, size_t num_layers) {
    SNEPPXSERModel* model = (SNEPPXSERModel*)SNEPPX_malloc(sizeof(SNEPPXSERModel), 64);
    if (!model) return NULL;
    memset(model, 0, sizeof(SNEPPXSERModel));

    model->config = *config;
    model->num_layers = num_layers;

    model->layers = (SNEPPXSERLayer**)SNEPPX_malloc(num_layers * sizeof(SNEPPXSERLayer*), 64);
    if (!model->layers) {
        SNEPPX_free(model, sizeof(SNEPPXSERModel));
        return NULL;
    }
    memset(model->layers, 0, num_layers * sizeof(SNEPPXSERLayer*));

    for (size_t i = 0; i < num_layers; i++) {
        model->layers[i] = SNEPPX_ser_layer_create(config, seed + (unsigned int)i * 100);
    }

    return model;
}

/**
 * @brief Destroy Ser Model.
 */
void SNEPPX_ser_model_destroy(SNEPPXSERModel* model) {
    if (!model) return;
    for (size_t i = 0; i < model->num_layers; i++) {
        if (model->layers[i]) SNEPPX_ser_layer_destroy(model->layers[i]);
    }
    SNEPPX_free(model->layers, model->num_layers * sizeof(SNEPPXSERLayer*));
    SNEPPX_free(model, sizeof(SNEPPXSERModel));
}

/**
 * @brief Perform Ser Get Params.
 *
 * @param model [in] Model value.
 * @param out [out] Out value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_ser_get_params(const SNEPPXSERModel* model, SNEPPXTensor** out, size_t max_out) {
    if (!model) return 0;
    size_t count = 0;
    for (size_t l = 0; l < model->num_layers; l++) {
        SNEPPXSERLayer* layer = model->layers[l];
        if (out && count < max_out) out[count] = layer->router;
        count++;
        if (out && count < max_out) out[count] = layer->router_bias;
        count++;
        if (layer->config.use_mlp_gater) {
            if (out && count < max_out) out[count] = layer->gater_w1; count++;
            if (out && count < max_out) out[count] = layer->gater_b1; count++;
            if (out && count < max_out) out[count] = layer->gater_w2; count++;
            if (out && count < max_out) out[count] = layer->gater_b2; count++;
        }
        for (size_t e = 0; e < model->config.num_experts; e++) {
            SNEPPXExpert* exp = layer->experts[e];
            if (out && count < max_out) out[count] = exp->w1; count++;
            if (out && count < max_out) out[count] = exp->b1; count++;
            if (out && count < max_out) out[count] = exp->w2; count++;
            if (out && count < max_out) out[count] = exp->b2; count++;
        }
    }
    return count;
}
