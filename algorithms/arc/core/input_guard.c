#include "adversarial_robustness_certification.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
/*
 * SNEPPX - ARC Input Sanitization Guard
 *
 * WHAT
 *   ARC Input Sanitization Guard.
 *
 * CONCEPT
 *   Sanitizes inputs before they reach the model to prevent adversarial exploitation.
 *
 * ROLE
 *   Applies input normalization, clipping, and perturbation detection to filter out adversarial inputs before inference.
 *
 * REFERENCES
 *   None (internal algorithm).
 */



/**
 * @brief Perform Arc Config Default.
 *
 * @return The result value, or 0 on error.
 */
SNEPPXARCConfig SNEPPX_arc_config_default(void) {
    SNEPPXARCConfig cfg;
    cfg.input_guard_strength = 0.7f;
    cfg.gradient_obfuscation_method = SNEPPX_OBF_MIXED;
    cfg.gradient_noise_scale = 0.01f;
    cfg.gradient_clip_max = 1.0f;
    cfg.output_verify_layers = 2;
    cfg.output_verify_threshold = 0.9f;
    cfg.adversarial_training = 1;
    cfg.attack_simulation_types = SNEPPX_ATTACK_FGSM | SNEPPX_ATTACK_PGD | SNEPPX_ATTACK_CW;
    cfg.attack_epsilon = 0.1f;
    return cfg;
}

/**
 * @brief Create Input Guard.
 *
 * @param input_dim [in] Input Dim value.
 * @param output_dim [in] Output Dim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXInputGuard* SNEPPX_input_guard_create(size_t input_dim, size_t output_dim, unsigned int seed) {
    SNEPPXInputGuard* guard = (SNEPPXInputGuard*)SNEPPX_malloc(sizeof(SNEPPXInputGuard), 64);
    if (!guard) return NULL;
    memset(guard, 0, sizeof(SNEPPXInputGuard));

    size_t shape_pp[] = {output_dim, input_dim};
    size_t shape_n1[] = {input_dim};
    guard->projection_matrix = SNEPPX_tensor_create(shape_pp, 2, SNEPPX_FLOAT32);
    guard->norm_stats_mean = SNEPPX_tensor_zeros(shape_n1, 1, SNEPPX_FLOAT32);
    guard->norm_stats_var = SNEPPX_tensor_zeros(shape_n1, 1, SNEPPX_FLOAT32);
    guard->anomaly_threshold = 2.5f;

    if (guard->projection_matrix) {
        unsigned long state = seed;
        float* pd = (float*)guard->projection_matrix->data;
        for (size_t i = 0; i < guard->projection_matrix->size; i += 2) {
            state = state * 1103515245UL + 12345UL;
            float u1 = (float)((state >> 16) & 0x7FFF) / 32767.0f;
            state = state * 1103515245UL + 12345UL;
            float u2 = (float)((state >> 16) & 0x7FFF) / 32767.0f;
            float r = sqrtf(-2.0f * logf(u1 + 1e-10f));
            float theta = 2.0f * 3.14159265f * u2;
            pd[i] = r * cosf(theta) * 0.01f;
            if (i + 1 < guard->projection_matrix->size)
                pd[i + 1] = r * sinf(theta) * 0.01f;
        }
    }
    if (guard->norm_stats_var) {
        float* vd = (float*)guard->norm_stats_var->data;
        for (size_t i = 0; i < input_dim; i++) vd[i] = 1.0f;
    }

    return guard;
}

/**
 * @brief Destroy Input Guard.
 */
void SNEPPX_input_guard_destroy(SNEPPXInputGuard* guard) {
    if (!guard) return;
    if (guard->projection_matrix) SNEPPX_tensor_destroy(guard->projection_matrix);
    if (guard->norm_stats_mean) SNEPPX_tensor_destroy(guard->norm_stats_mean);
    if (guard->norm_stats_var) SNEPPX_tensor_destroy(guard->norm_stats_var);
    SNEPPX_free(guard, sizeof(SNEPPXInputGuard));
}

/**
 * @brief Run the forward pass for Arc Input Guard.
 *
 * @param guard [out] Guard value.
 * @param input [in] Input value.
 * @param sanitized [out] Sanitized value.
 */
void SNEPPX_arc_input_guard_forward(SNEPPXInputGuard* guard, const SNEPPXTensor* input, SNEPPXTensor** sanitized, float* anomaly_score) {
    size_t batch = input->shape[0];
    size_t dim = input->shape[1];
    float* in_data = (float*)input->data;

    size_t shape_s[] = {batch, dim};
    *sanitized = SNEPPX_tensor_create(shape_s, 2, SNEPPX_FLOAT32);
    if (!*sanitized) { *anomaly_score = 0.0f; return; }
    float* out_data = (float*)(*sanitized)->data;

    float* mean_data = (float*)guard->norm_stats_mean->data;
    float* var_data = (float*)guard->norm_stats_var->data;
    float* proj_data = guard->projection_matrix ? (float*)guard->projection_matrix->data : NULL;

    float sum_norm = 0.0f;
    int flagged = 0;

    for (size_t b = 0; b < batch; b++) {
        float* in_b = in_data + b * dim;
        float* out_b = out_data + b * dim;

        float l2 = 0.0f;
        for (size_t i = 0; i < dim; i++) l2 += in_b[i] * in_b[i];
        l2 = sqrtf(l2 + 1e-10f);
        sum_norm += l2;

        float mean = 0.0f;
        for (size_t i = 0; i < dim; i++) mean += mean_data[i];
        mean /= (float)dim;
        float var = 0.0f;
        for (size_t i = 0; i < dim; i++) var += var_data[i];
        var /= (float)dim;
        float threshold = guard->anomaly_threshold;

        if (fabsf(l2 - mean) / sqrtf(var + 1e-6f) > threshold) {
            flagged++;
            if (proj_data) {
                for (size_t i = 0; i < dim; i++) {
                    float sum = 0.0f;
                    for (size_t j = 0; j < dim; j++)
                        sum += proj_data[i * dim + j] * in_b[j];
                    out_b[i] = sum;
                }
            } else {
                memcpy(out_b, in_b, dim * sizeof(float));
            }
            float clip_bound = 3.0f * sqrtf(var + 1e-6f);
            for (size_t i = 0; i < dim; i++) {
                if (out_b[i] > mean + clip_bound) out_b[i] = mean + clip_bound;
                if (out_b[i] < mean - clip_bound) out_b[i] = mean - clip_bound;
            }
        } else {
            memcpy(out_b, in_b, dim * sizeof(float));
        }

        for (size_t i = 0; i < dim; i++) {
            mean_data[i] = 0.99f * mean_data[i] + 0.01f * in_b[i];
            float diff = in_b[i] - mean_data[i];
            var_data[i] = 0.99f * var_data[i] + 0.01f * (diff * diff);
        }
    }

    *anomaly_score = (float)flagged / (float)batch;
}
