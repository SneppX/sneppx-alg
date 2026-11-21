#ifndef SNEPPX_FM_H
#define SNEPPX_FM_H

#ifdef __cplusplus
extern "C" {
#endif

#include "multidimensional_tensor_engine.h"
#include "automatic_differentiation_framework.h"
#include <stddef.h>
#include <stdint.h>
/*
 * SNEPPX - Fractal Memory Orchestrator (FM)
 *
 * WHAT
 *   Fractal Memory Orchestrator (FM).
 *
 * CONCEPT
 *   FM controller, memory bank, and sync API declarations.
 *
 * ROLE
 *   Declares the FM module API for hierarchical memory management.
 *
 * REFERENCES
 *   None (internal module).
 */



typedef enum {
    SNEPPX_SYNC_ALL_REDUCE,
    SNEPPX_SYNC_GOSSIP,
    SNEPPX_SYNC_TOPOLOGY
} SNEPPXFMSyncMethod;

typedef struct {
    SNEPPXTensor* keys;
    SNEPPXTensor* values;
    SNEPPXTensor* timestamps;
    SNEPPXTensor* access_counts;
    size_t num_entries;
    size_t max_entries;
} SNEPPXFMMemoryBank;

typedef struct {
    size_t node_id;
    SNEPPXFMMemoryBank* memory_bank;
    SNEPPXTensor* gradient_accumulator;
    uint64_t last_sync_time;
    int is_online;
    float trust_score;
} SNEPPXFMNode;

typedef struct {
    size_t num_nodes;
    size_t memory_dim;
    size_t memory_capacity;
    size_t sync_interval;
    SNEPPXFMSyncMethod sync_method;
    float compression_ratio;
    float privacy_epsilon;
    int catastrophic_forgetting_protection;
    float ewm_alpha;
} SNEPPXFMConfig;

typedef struct {
    SNEPPXTensor* global_memory;
    size_t sync_round;
    float* node_contributions;
    size_t* conflict_log;
    size_t conflict_count;
} SNEPPXFMSyncState;

typedef struct {
    SNEPPXFMNode** nodes;
    SNEPPXFMConfig config;
    SNEPPXFMSyncState sync_state;
    size_t step_counter;
} SNEPPXFMController;

/**
 * @brief Perform Fm Config Default.
 *
 * @return The result value, or 0 on error.
 */
SNEPPXFMConfig SNEPPX_fm_config_default(void);
/**
 * @brief Create Fm Memory Bank.
 *
 * @param memory_dim [in] Memory Dim value.
 * @param capacity [in] Capacity value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXFMMemoryBank* SNEPPX_fm_memory_bank_create(size_t memory_dim, size_t capacity);
/**
 * @brief Destroy Fm Memory Bank.
 *
 * @param bank [out] Bank value.
 */
void SNEPPX_fm_memory_bank_destroy(SNEPPXFMMemoryBank* bank);
/**
 * @brief Write Fm Memory Bank.
 *
 * @param bank [out] Bank value.
 * @param key [in] Key value.
 * @param value [in] Value value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fm_memory_bank_write(SNEPPXFMMemoryBank* bank, const SNEPPXTensor* key, const SNEPPXTensor* value);
/**
 * @brief Read Fm Memory Bank.
 *
 * @param bank [out] Bank value.
 * @param key [in] Key value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_fm_memory_bank_read(SNEPPXFMMemoryBank* bank, const SNEPPXTensor* key);
/**
 * @brief Get Fm Memory Bank For.
 *
 * @param bank [out] Bank value.
 * @param forget_rate [in] Forget Rate value.
 */
void SNEPPX_fm_memory_bank_forget(SNEPPXFMMemoryBank* bank, float forget_rate);
/**
 * @brief Create Fm Node.
 *
 * @param node_id [in] Node Id value.
 * @param memory_dim [in] Memory Dim value.
 * @param capacity [in] Capacity value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXFMNode* SNEPPX_fm_node_create(size_t node_id, size_t memory_dim, size_t capacity);
/**
 * @brief Destroy Fm Node.
 *
 * @param node [out] Node value.
 */
void SNEPPX_fm_node_destroy(SNEPPXFMNode* node);
/**
 * @brief Create Fm Controller.
 *
 * @param config [in] Config value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXFMController* SNEPPX_fm_controller_create(const SNEPPXFMConfig* config);
/**
 * @brief Destroy Fm Controller.
 *
 * @param ctrl [out] Ctrl value.
 */
void SNEPPX_fm_controller_destroy(SNEPPXFMController* ctrl);
/**
 * @brief Perform Fm Sync All Reduce.
 *
 * @param ctrl [out] Ctrl value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fm_sync_all_reduce(SNEPPXFMController* ctrl);
typedef int (*SNEPPXFMSyncCallback)(void* data, size_t count, void* context);
/**
 * @brief Perform Fm Sync Nccl.
 *
 * @param ctrl [out] Ctrl value.
 * @param pg_allreduce [in] Pg Allreduce value.
 * @param context [out] Context value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fm_sync_nccl(SNEPPXFMController* ctrl, SNEPPXFMSyncCallback pg_allreduce, void* context);
/**
 * @brief Perform Fm Sync Gossip.
 *
 * @param ctrl [out] Ctrl value.
 * @param num_pairs [in] Num Pairs value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fm_sync_gossip(SNEPPXFMController* ctrl, size_t num_pairs);
/**
 * @brief Perform Fm Sync Topology.
 *
 * @param ctrl [out] Ctrl value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fm_sync_topology(SNEPPXFMController* ctrl);
/**
 * @brief Perform Fm Compress Gradients.
 *
 * @param gradients [in] Gradients value.
 * @param ratio [in] Ratio value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_fm_compress_gradients(const SNEPPXTensor* gradients, float ratio);
/**
 * @brief Perform Fm Add Privacy Noise.
 *
 * @param data [out] Data value.
 * @param epsilon [in] Epsilon value.
 */
void SNEPPX_fm_add_privacy_noise(SNEPPXTensor* data, float epsilon);
/**
 * @brief Run the forward pass for Fm.
 *
 * @param ctrl [out] Ctrl value.
 * @param node_id [in] Node Id value.
 * @param input [in] Input value.
 * @param output [out] Output value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fm_forward(SNEPPXFMController* ctrl, size_t node_id, const SNEPPXTensor* input, SNEPPXTensor** output);

// Training graph support (FM has no trainable params, pass-through only)
/**
 * @brief Perform Fm Get Params.
 *
 * @param ctrl [in] Ctrl value.
 * @param out_params [out] Out Params value.
 * @param max_params [in] Max Params value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_fm_get_params(const SNEPPXFMController* ctrl, SNEPPXTensor** out_params, size_t max_params);
/**
 * @brief Perform Fm Build Train Graph.
 *
 * @param ctrl [out] Ctrl value.
 * @param tape [out] Tape value.
 * @param input_var [out] Input Var value.
 * @param weight_vars [out] Weight Vars value.
 * @param num_weights [in] Num Weights value.
 * @param output_var [out] Output Var value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fm_build_train_graph(SNEPPXFMController* ctrl, SNEPPXTape* tape,
                               SNEPPXVariable* input_var,
                               SNEPPXVariable** weight_vars, size_t num_weights,
                               SNEPPXVariable** output_var);

// ── Error-compensated gradient compression (EF-SGD) ──────────────────────────

typedef struct {
    SNEPPXTensor* error_buffer;
    SNEPPXTensor* compressed_grad;
    float compression_ratio;
    size_t dim;
} SNEPPXFMErrorFeedback;

/**
 * @brief Create Fm Error Feedback.
 *
 * @param dim [in] Dim value.
 * @param ratio [in] Ratio value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXFMErrorFeedback* SNEPPX_fm_error_feedback_create(size_t dim, float ratio);
/**
 * @brief Destroy Fm Error Feedback.
 *
 * @param ef [out] Ef value.
 */
void SNEPPX_fm_error_feedback_destroy(SNEPPXFMErrorFeedback* ef);
/**
 * @brief Perform Fm Compress With Error.
 *
 * @param ef [out] Ef value.
 * @param gradient [in] Gradient value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_fm_compress_with_error(SNEPPXFMErrorFeedback* ef, const SNEPPXTensor* gradient);

// ── Exponential moving average (catastrophic forgetting) ─────────────────────

/**
 * @brief Update Fm Ewm.
 *
 * @param bank [out] Bank value.
 * @param alpha [in] Alpha value.
 */
void SNEPPX_fm_ewm_update(SNEPPXFMMemoryBank* bank, float alpha);

// ── Adaptive sync frequency ──────────────────────────────────────────────────

/**
 * @brief Perform Fm Compute Change Rate.
 *
 * @param bank [out] Bank value.
 * @param new_values [in] New Values value.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_fm_compute_change_rate(SNEPPXFMMemoryBank* bank, const SNEPPXTensor* new_values);
/**
 * @brief Perform Fm Adaptive Sync Interval.
 *
 * @param ctrl [out] Ctrl value.
 * @param base_interval [in] Base Interval value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_fm_adaptive_sync_interval(SNEPPXFMController* ctrl, float base_interval);

// ── Gradient send / receive ──────────────────────────────────────────────────

/**
 * @brief Perform Fm Send Gradients.
 *
 * @param ctrl [out] Ctrl value.
 * @param node_id [in] Node Id value.
 * @param gradients [in] Gradients value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fm_send_gradients(SNEPPXFMController* ctrl, size_t node_id, const SNEPPXTensor* gradients);
/**
 * @brief Perform Fm Receive Gradients.
 *
 * @param ctrl [out] Ctrl value.
 * @param node_id [in] Node Id value.
 * @param aggregated [out] Aggregated value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fm_receive_gradients(SNEPPXFMController* ctrl, size_t node_id, SNEPPXTensor* aggregated);


#ifdef __cplusplus
}
#endif
#endif /* SNEPPX_FM_H */
