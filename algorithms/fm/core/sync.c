#include "fractal_memory_orchestrator.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
/*
 * SNEPPX - FM Memory Bank Synchronization
 *
 * WHAT
 *   FM Memory Bank Synchronization.
 *
 * CONCEPT
 *   Ring/butterfly all-reduce for federated memory synchronization across ranks.
 *
 * ROLE
 *   Implements the synchronization pattern that aggregates memory updates across distributed ranks using ring or butterfly all-reduce.
 *
 * REFERENCES
 *   None (internal algorithm).
 */



static float uniform_01(void) {
    static unsigned long seed = 123456789;
    seed = seed * 1103515245UL + 12345UL;
    return (float)((seed >> 16) & 0x7FFF) / 32767.0f;
}

static float laplace_noise(float epsilon) {
    float u = uniform_01() - 0.5f;
    return -epsilon * (u > 0 ? 1.0f : -1.0f) * logf(1.0f - 2.0f * fabsf(u) + 1e-10f);
}

/**
 * @brief Perform Fm Add Privacy Noise.
 *
 * @param data [out] Data value.
 */
void SNEPPX_fm_add_privacy_noise(SNEPPXTensor* data, float epsilon) {
    if (!data || epsilon <= 0.0f) return;
    float* d = (float*)data->data;
    for (size_t i = 0; i < data->size; i++) {
        d[i] += laplace_noise(epsilon);
    }
}

typedef struct {
    float val;
    size_t idx;
} MagEntry;

static int cmp_mag(const void* a, const void* b) {
    float fa = fabsf(((const MagEntry*)a)->val);
    float fb = fabsf(((const MagEntry*)b)->val);
    if (fa < fb) return 1;
    if (fa > fb) return -1;
    return 0;
}

/**
 * @brief Perform Fm Compress Gradients.
 *
 * @param gradients [in] Gradients value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_fm_compress_gradients(const SNEPPXTensor* gradients, float ratio) {
    if (!gradients || ratio <= 0.0f) return NULL;
    if (ratio >= 1.0f) {
        SNEPPXTensor* copy = SNEPPX_tensor_create(gradients->shape, gradients->ndim, SNEPPX_FLOAT32);
        if (copy) memcpy((float*)copy->data, (float*)gradients->data, gradients->size * sizeof(float));
        return copy;
    }

    SNEPPXTensor* result = SNEPPX_tensor_create(gradients->shape, gradients->ndim, SNEPPX_FLOAT32);
    if (!result) return NULL;
    memcpy((float*)result->data, (float*)gradients->data, gradients->size * sizeof(float));

    size_t total = gradients->size;
    size_t keep = (size_t)(ratio * (float)total);
    if (keep == 0) keep = 1;
    if (keep > total) keep = total;

    MagEntry* entries = (MagEntry*)malloc(total * sizeof(MagEntry));
    if (!entries) { SNEPPX_tensor_destroy(result); return NULL; }

    float* src = (float*)gradients->data;
    for (size_t i = 0; i < total; i++) {
        entries[i].val = src[i];
        entries[i].idx = i;
    }

    qsort(entries, total, sizeof(MagEntry), cmp_mag);

    float* dst = (float*)result->data;
    int* keep_mask = (int*)calloc(total, sizeof(int));
    if (!keep_mask) { free(entries); SNEPPX_tensor_destroy(result); return NULL; }

    for (size_t i = 0; i < keep; i++) {
        keep_mask[entries[i].idx] = 1;
    }

    for (size_t i = 0; i < total; i++) {
        if (!keep_mask[i]) dst[i] = 0.0f;
    }

    free(keep_mask);
    free(entries);
    return result;
}

static void average_two_banks(SNEPPXFMMemoryBank* a, SNEPPXFMMemoryBank* b, float alpha) {
    size_t dim = a->keys->shape[1];
    size_t min_entries = a->num_entries < b->num_entries ? a->num_entries : b->num_entries;
    if (min_entries == 0) return;

    float* a_keys = (float*)a->keys->data;
    float* a_vals = (float*)a->values->data;
    float* b_keys = (float*)b->keys->data;
    float* b_vals = (float*)b->values->data;

    for (size_t i = 0; i < min_entries; i++) {
        for (size_t j = 0; j < dim; j++) {
            float avg_key = a_keys[i * dim + j] * alpha + b_keys[i * dim + j] * (1.0f - alpha);
            float avg_val = a_vals[i * dim + j] * alpha + b_vals[i * dim + j] * (1.0f - alpha);
            a_keys[i * dim + j] = avg_key;
            a_vals[i * dim + j] = avg_val;
            b_keys[i * dim + j] = avg_key;
            b_vals[i * dim + j] = avg_val;
        }
    }
}

/**
 * @brief Perform Fm Sync All Reduce.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fm_sync_all_reduce(SNEPPXFMController* ctrl) {
    if (!ctrl || ctrl->config.num_nodes == 0) return 1;
    size_t dim = ctrl->config.memory_dim;
    size_t cap = ctrl->config.memory_capacity;
    size_t n_nodes = ctrl->config.num_nodes;

    for (size_t slot = 0; slot < cap; slot++) {
        float sum_keys[512], sum_vals[512];
        float total_weight = 0.0f;
        memset(sum_keys, 0, dim * sizeof(float));
        memset(sum_vals, 0, dim * sizeof(float));

        for (size_t n = 0; n < n_nodes; n++) {
            if (!ctrl->nodes[n]->is_online) continue;
            SNEPPXFMMemoryBank* bank = ctrl->nodes[n]->memory_bank;
            if (slot >= bank->num_entries) continue;
            float w = ctrl->nodes[n]->trust_score;
            total_weight += w;
            float* kd = (float*)bank->keys->data + slot * dim;
            float* vd = (float*)bank->values->data + slot * dim;
            for (size_t j = 0; j < dim; j++) {
                sum_keys[j] += kd[j] * w;
                sum_vals[j] += vd[j] * w;
            }
        }

        if (total_weight < 1e-10f) continue;

        float variance = 0.0f;
        for (size_t n = 0; n < n_nodes; n++) {
            if (!ctrl->nodes[n]->is_online) continue;
            SNEPPXFMMemoryBank* bank = ctrl->nodes[n]->memory_bank;
            if (slot >= bank->num_entries) continue;
            float* vd = (float*)bank->values->data + slot * dim;
            for (size_t j = 0; j < dim; j++) {
                float diff = vd[j] - sum_vals[j] / total_weight;
                variance += diff * diff;
            }
        }
        variance /= (float)(n_nodes * dim);

        if (variance > 0.1f && ctrl->sync_state.conflict_log) {
            ctrl->sync_state.conflict_log[ctrl->sync_state.conflict_count++] = slot;
        }

        for (size_t n = 0; n < n_nodes; n++) {
            if (!ctrl->nodes[n]->is_online) continue;
            SNEPPXFMMemoryBank* bank = ctrl->nodes[n]->memory_bank;
            float* kd = (float*)bank->keys->data + slot * dim;
            float* vd = (float*)bank->values->data + slot * dim;
            for (size_t j = 0; j < dim; j++) {
                kd[j] = sum_keys[j] / total_weight;
                vd[j] = sum_vals[j] / total_weight;
            }
        }
    }

    SNEPPX_fm_add_privacy_noise(ctrl->sync_state.global_memory, ctrl->config.privacy_epsilon);
    ctrl->sync_state.sync_round++;

    for (size_t n = 0; n < n_nodes; n++) {
        if (!ctrl->nodes[n]->is_online) continue;
        ctrl->nodes[n]->last_sync_time = ctrl->step_counter;
        if (ctrl->sync_state.node_contributions) {
            ctrl->sync_state.node_contributions[n] += 0.1f;
        }
    }

    return 0;
}

/**
 * @brief Perform Fm Sync Gossip.
 *
 * @param ctrl [out] Ctrl value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fm_sync_gossip(SNEPPXFMController* ctrl, size_t num_pairs) {
    if (!ctrl || ctrl->config.num_nodes < 2) return 1;
    size_t n_nodes = ctrl->config.num_nodes;

    for (size_t p = 0; p < num_pairs; p++) {
        size_t a = (size_t)(uniform_01() * (float)n_nodes) % n_nodes;
        size_t b = (size_t)(uniform_01() * (float)n_nodes) % n_nodes;
        if (a == b) { b = (a + 1) % n_nodes; }
        if (!ctrl->nodes[a]->is_online || !ctrl->nodes[b]->is_online) continue;

        average_two_banks(ctrl->nodes[a]->memory_bank, ctrl->nodes[b]->memory_bank, 0.5f);
        ctrl->nodes[a]->last_sync_time = ctrl->step_counter;
        ctrl->nodes[b]->last_sync_time = ctrl->step_counter;
    }

    ctrl->sync_state.sync_round++;
    return 0;
}

/**
 * @brief Perform Fm Sync Topology.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fm_sync_topology(SNEPPXFMController* ctrl) {
    if (!ctrl || ctrl->config.num_nodes < 2) return 1;
    size_t n_nodes = ctrl->config.num_nodes;

    for (size_t i = 0; i < n_nodes; i++) {
        size_t next = (i + 1) % n_nodes;
        if (!ctrl->nodes[i]->is_online || !ctrl->nodes[next]->is_online) continue;
        average_two_banks(ctrl->nodes[i]->memory_bank, ctrl->nodes[next]->memory_bank, 0.5f);
        ctrl->nodes[i]->last_sync_time = ctrl->step_counter;
        ctrl->nodes[next]->last_sync_time = ctrl->step_counter;
    }

    ctrl->sync_state.sync_round++;
    return 0;
}

// ── Error-compensated gradient compression (EF-SGD) ──────────────────────────

/**
 * @brief Create Fm Error Feedback.
 *
 * @param dim [in] Dim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXFMErrorFeedback* SNEPPX_fm_error_feedback_create(size_t dim, float ratio) {
    if (dim == 0 || ratio <= 0.0f) return NULL;
    SNEPPXFMErrorFeedback* ef = (SNEPPXFMErrorFeedback*)SNEPPX_malloc(sizeof(SNEPPXFMErrorFeedback), 64);
    if (!ef) return NULL;
    memset(ef, 0, sizeof(SNEPPXFMErrorFeedback));

    size_t shape[] = {dim};
    ef->error_buffer = SNEPPX_tensor_zeros(shape, 1, SNEPPX_FLOAT32);
    ef->compressed_grad = SNEPPX_tensor_zeros(shape, 1, SNEPPX_FLOAT32);
    if (!ef->error_buffer || !ef->compressed_grad) {
        SNEPPX_fm_error_feedback_destroy(ef);
        return NULL;
    }
    ef->compression_ratio = ratio;
    ef->dim = dim;
    return ef;
}

/**
 * @brief Destroy Fm Error Feedback.
 */
void SNEPPX_fm_error_feedback_destroy(SNEPPXFMErrorFeedback* ef) {
    if (!ef) return;
    if (ef->error_buffer) SNEPPX_tensor_destroy(ef->error_buffer);
    if (ef->compressed_grad) SNEPPX_tensor_destroy(ef->compressed_grad);
    SNEPPX_free(ef, sizeof(SNEPPXFMErrorFeedback));
}

/**
 * @brief Perform Fm Compress With Error.
 *
 * @param ef [out] Ef value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_fm_compress_with_error(SNEPPXFMErrorFeedback* ef, const SNEPPXTensor* gradient) {
    if (!ef || !gradient) return NULL;

    SNEPPXTensor* g_eff = SNEPPX_tensor_add(gradient, ef->error_buffer);
    if (!g_eff) return NULL;

    SNEPPXTensor* g_comp = SNEPPX_fm_compress_gradients(g_eff, ef->compression_ratio);
    SNEPPX_tensor_destroy(g_eff);
    if (!g_comp) return NULL;

    SNEPPXTensor* diff = SNEPPX_tensor_sub(gradient, g_comp);
    if (diff) {
        SNEPPXTensor* new_error = SNEPPX_tensor_add(ef->error_buffer, diff);
        if (new_error) {
            SNEPPX_tensor_destroy(ef->error_buffer);
            ef->error_buffer = new_error;
        }
        SNEPPX_tensor_destroy(diff);
    }

    if (ef->compressed_grad) SNEPPX_tensor_destroy(ef->compressed_grad);
    ef->compressed_grad = SNEPPX_tensor_copy(g_comp);

    return g_comp;
}

// ── Exponential moving average for catastrophic forgetting protection ────────

/**
 * @brief Update Fm Ewm.
 *
 * @param bank [out] Bank value.
 */
void SNEPPX_fm_ewm_update(SNEPPXFMMemoryBank* bank, float alpha) {
    if (!bank || bank->num_entries == 0 || alpha < 0.0f || alpha > 1.0f) return;
    size_t dim = bank->keys->shape[1];
    size_t n = bank->num_entries;
    float* vals = (float*)bank->values->data;

    float* old_vals = (float*)malloc(n * dim * sizeof(float));
    if (!old_vals) return;
    memcpy(old_vals, vals, n * dim * sizeof(float));

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < dim; j++) {
            vals[i * dim + j] = alpha * old_vals[i * dim + j] + (1.0f - alpha) * vals[i * dim + j];
        }
    }
    free(old_vals);
}

// ── Adaptive sync frequency ──────────────────────────────────────────────────

/**
 * @brief Perform Fm Compute Change Rate.
 *
 * @param bank [out] Bank value.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_fm_compute_change_rate(SNEPPXFMMemoryBank* bank, const SNEPPXTensor* new_values) {
    if (!bank || !new_values || bank->num_entries == 0) return 0.0f;
    size_t dim = bank->keys->shape[1];
    float* vals = (float*)bank->values->data;
    float* nv = (float*)new_values->data;
    size_t n_entries = bank->num_entries;
    size_t nv_entries = new_values->size / dim;
    size_t n = n_entries < nv_entries ? n_entries : nv_entries;

    float total_diff = 0.0f;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < dim; j++) {
            float d = vals[i * dim + j] - nv[i * dim + j];
            total_diff += d * d;
        }
    }
    return sqrtf(total_diff / (float)(n * dim) + 1e-10f);
}

/**
 * @brief Perform Fm Adaptive Sync Interval.
 *
 * @param ctrl [out] Ctrl value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_fm_adaptive_sync_interval(SNEPPXFMController* ctrl, float base_interval) {
    if (!ctrl || base_interval < 1.0f) base_interval = 1.0f;
    float total_contrib = 0.0f;
    float contrib_sq = 0.0f;
    size_t n_online = 0;
    for (size_t i = 0; i < ctrl->config.num_nodes; i++) {
        if (ctrl->nodes[i]->is_online) {
            float c = ctrl->sync_state.node_contributions ? ctrl->sync_state.node_contributions[i] : 0.0f;
            total_contrib += c;
            contrib_sq += c * c;
            n_online++;
        }
    }
    if (n_online == 0) return (size_t)base_interval;

    float mean = total_contrib / (float)n_online;
    float variance = contrib_sq / (float)n_online - mean * mean;
    if (variance < 0.0f) variance = 0.0f;
    float stddev = sqrtf(variance + 1e-10f);

    float multiplier = 1.0f / (1.0f + stddev);
    if (multiplier < 0.1f) multiplier = 0.1f;
    if (multiplier > 2.0f) multiplier = 2.0f;

    return (size_t)(base_interval * multiplier + 0.5f);
}

// ── Gradient send / receive ──────────────────────────────────────────────────

/**
 * @brief Perform Fm Send Gradients.
 *
 * @param ctrl [out] Ctrl value.
 * @param node_id [in] Node Id value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fm_send_gradients(SNEPPXFMController* ctrl, size_t node_id, const SNEPPXTensor* gradients) {
    if (!ctrl || !gradients || node_id >= ctrl->config.num_nodes) return 1;
    SNEPPXFMNode* node = ctrl->nodes[node_id];
    if (!node->is_online) return 1;

    if (!node->gradient_accumulator) {
        node->gradient_accumulator = SNEPPX_tensor_copy(gradients);
        return node->gradient_accumulator ? 0 : 1;
    }

    SNEPPXTensor* updated = SNEPPX_tensor_add(node->gradient_accumulator, gradients);
    if (!updated) return 1;
    SNEPPX_tensor_destroy(node->gradient_accumulator);
    node->gradient_accumulator = updated;
    return 0;
}

/**
 * @brief Perform Fm Receive Gradients.
 *
 * @param ctrl [out] Ctrl value.
 * @param node_id [in] Node Id value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fm_receive_gradients(SNEPPXFMController* ctrl, size_t node_id, SNEPPXTensor* aggregated) {
    if (!ctrl || !aggregated || node_id >= ctrl->config.num_nodes) return 1;

    memset((float*)aggregated->data, 0, aggregated->size * sizeof(float));

    for (size_t i = 0; i < ctrl->config.num_nodes; i++) {
        if (!ctrl->nodes[i]->is_online) continue;
        if (i == node_id) continue;
        SNEPPXFMNode* node = ctrl->nodes[i];
        if (!node->gradient_accumulator) continue;

        float* grad_data = (float*)node->gradient_accumulator->data;
        float* agg_data = (float*)aggregated->data;
        size_t min_size = node->gradient_accumulator->size < aggregated->size
                              ? node->gradient_accumulator->size
                              : aggregated->size;
        for (size_t j = 0; j < min_size; j++) {
            agg_data[j] += grad_data[j];
        }
    }
    return 0;
}

/**
 * @brief Perform Fm Sync Nccl.
 *
 * @param ctrl [out] Ctrl value.
 * @param pg_allreduce [in] Pg Allreduce value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fm_sync_nccl(SNEPPXFMController* ctrl, SNEPPXFMSyncCallback pg_allreduce, void* context) {
    if (!ctrl) return 1;
    if (pg_allreduce) {
        size_t dim = ctrl->config.memory_dim;
        size_t n_nodes = ctrl->config.num_nodes;
        for (size_t n = 0; n < n_nodes; n++) {
            if (!ctrl->nodes[n]->is_online) continue;
            SNEPPXFMMemoryBank* bank = ctrl->nodes[n]->memory_bank;
            if (bank->num_entries == 0) continue;
            size_t count = bank->num_entries * dim;
            pg_allreduce((float*)bank->keys->data, count, context);
            pg_allreduce((float*)bank->values->data, count, context);
        }
    }
    return SNEPPX_fm_sync_all_reduce(ctrl);
}
