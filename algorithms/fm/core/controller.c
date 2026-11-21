#include "fractal_memory_orchestrator.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <stdlib.h>
/*
 * SNEPPX - FM Fractal Memory Controller
 *
 * WHAT
 *   FM Fractal Memory Controller.
 *
 * CONCEPT
 *   Controls the fractal memory system: allocation, indexing, and lifecycle of memory nodes.
 *
 * ROLE
 *   The controller manages the hierarchical memory tree, coordinating reads and writes across the fractal memory bank.
 *
 * REFERENCES
 *   None (internal algorithm).
 */



/**
 * @brief Perform Fm Config Default.
 *
 * @return The result value, or 0 on error.
 */
SNEPPXFMConfig SNEPPX_fm_config_default(void) {
    SNEPPXFMConfig cfg;
    cfg.num_nodes = 4;
    cfg.memory_dim = 512;
    cfg.memory_capacity = 1024;
    cfg.sync_interval = 100;
    cfg.sync_method = SNEPPX_SYNC_ALL_REDUCE;
    cfg.compression_ratio = 0.1f;
    cfg.privacy_epsilon = 1.0f;
    cfg.catastrophic_forgetting_protection = 1;
    cfg.ewm_alpha = 0.9f;
    return cfg;
}

/**
 * @brief Create Fm Controller.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXFMController* SNEPPX_fm_controller_create(const SNEPPXFMConfig* config) {
    if (!config) return NULL;
    SNEPPXFMController* ctrl = (SNEPPXFMController*)SNEPPX_malloc(sizeof(SNEPPXFMController), 64);
    if (!ctrl) return NULL;
    memset(ctrl, 0, sizeof(SNEPPXFMController));

    ctrl->config = *config;
    ctrl->step_counter = 0;

    ctrl->nodes = (SNEPPXFMNode**)SNEPPX_malloc(config->num_nodes * sizeof(SNEPPXFMNode*), 64);
    if (!ctrl->nodes) {
        SNEPPX_free(ctrl, sizeof(SNEPPXFMController));
        return NULL;
    }
    memset(ctrl->nodes, 0, config->num_nodes * sizeof(SNEPPXFMNode*));

    for (size_t i = 0; i < config->num_nodes; i++) {
        ctrl->nodes[i] = SNEPPX_fm_node_create(i, config->memory_dim, config->memory_capacity);
        if (!ctrl->nodes[i]) {
            SNEPPX_fm_controller_destroy(ctrl);
            return NULL;
        }
    }

    size_t gm_shape[] = {config->memory_capacity, config->memory_dim};
    ctrl->sync_state.global_memory = SNEPPX_tensor_zeros(gm_shape, 2, SNEPPX_FLOAT32);
    if (!ctrl->sync_state.global_memory) {
        SNEPPX_fm_controller_destroy(ctrl);
        return NULL;
    }
    ctrl->sync_state.sync_round = 0;
    ctrl->sync_state.node_contributions = (float*)SNEPPX_malloc(config->num_nodes * sizeof(float), 64);
    if (ctrl->sync_state.node_contributions) {
        memset(ctrl->sync_state.node_contributions, 0, config->num_nodes * sizeof(float));
    }
    ctrl->sync_state.conflict_log = NULL;
    ctrl->sync_state.conflict_count = 0;

    return ctrl;
}

/**
 * @brief Destroy Fm Controller.
 */
void SNEPPX_fm_controller_destroy(SNEPPXFMController* ctrl) {
    if (!ctrl) return;
    if (ctrl->nodes) {
        for (size_t i = 0; i < ctrl->config.num_nodes; i++) {
            if (ctrl->nodes[i]) SNEPPX_fm_node_destroy(ctrl->nodes[i]);
        }
        SNEPPX_free(ctrl->nodes, ctrl->config.num_nodes * sizeof(SNEPPXFMNode*));
    }
    if (ctrl->sync_state.global_memory) SNEPPX_tensor_destroy(ctrl->sync_state.global_memory);
    if (ctrl->sync_state.node_contributions) SNEPPX_free(ctrl->sync_state.node_contributions, ctrl->config.num_nodes * sizeof(float));
    if (ctrl->sync_state.conflict_log) SNEPPX_free(ctrl->sync_state.conflict_log, ctrl->sync_state.conflict_count * sizeof(size_t));
    SNEPPX_free(ctrl, sizeof(SNEPPXFMController));
}
