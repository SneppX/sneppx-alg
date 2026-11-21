#ifndef SNEPPX_AI_SHAP_PROVIDER_H
#define SNEPPX_AI_SHAP_PROVIDER_H

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_AI_SHAP_MAX_FEATURES 16384
#define SNEPPX_AI_SHAP_MAX_SAMPLES 100000
#define SNEPPX_AI_SHAP_BACKGROUND_SIZE 100

/*
 * SNEPPX - Shap Provider
 *
 * WHAT
 *   Shap Provider.
 *
 * CONCEPT
 *   Provides SHA hash family.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef struct {
    uint32_t feature_index;
    char* feature_name;
    size_t feature_name_len;
    double shap_value;
    double expected_value;
    double feature_value;
    double min_value;
    double max_value;
    double std_dev;
    double contribution_percent;
    uint8_t* interaction_matrix;
    size_t interaction_matrix_len;
    uint32_t* interacting_features;
    uint32_t num_interactions;
    uint8_t categorical : 1;
} SNEPPXAIExplainFeature;

typedef struct {
    uint64_t sample_id;
    double* feature_vector;
    uint32_t num_features;
    double prediction;
    double baseline_prediction;
    SNEPPXAIExplainFeature* features;
    uint32_t num_shap_features;
    double sum_shap_values;
    double prediction_error;
    double* shap_interaction_matrix;
    size_t interaction_matrix_dim;
    uint8_t explained : 1;
} SNEPPXAIExplainSample;

typedef struct {
    double* background_data;
    uint32_t background_samples;
    uint32_t num_features;
    uint8_t* model_weights;
    size_t model_weights_len;
    uint32_t feature_samples;
    uint32_t max_samples;
    uint64_t total_explain_time_ns;
    uint32_t total_explain_calls;
    uint8_t* feature_names;
    size_t feature_names_len;
    uint8_t kernel_shap : 1;
    uint8_t tree_shap : 1;
    uint8_t deep_shap : 1;
} SNEPPXAIExplainProvider;

int snepx_ai_shap_init(SNEPPXAIExplainProvider* provider, uint32_t num_features);
int snepx_ai_shap_load_background(SNEPPXAIExplainProvider* provider, const double* data, uint32_t num_samples, uint32_t num_features);
int snepx_ai_shap_explain(SNEPPXAIExplainProvider* provider, const double* sample_features, uint32_t num_features, SNEPPXAIExplainSample* result);
int snepx_ai_shap_explain_batch(SNEPPXAIExplainProvider* provider, const double* samples, uint32_t num_samples, uint32_t num_features, SNEPPXAIExplainSample* results);
int snepx_ai_shap_kernel_shap(SNEPPXAIExplainProvider* provider, const double* sample, SNEPPXAIExplainSample* result);
int snepx_ai_shap_compute_interactions(SNEPPXAIExplainProvider* provider, const SNEPPXAIExplainSample* sample, double* interaction_matrix, size_t* dim);
int snepx_ai_shap_global_importance(const SNEPPXAIExplainProvider* provider, SNEPPXAIExplainSample* samples, uint32_t num_samples, double* global_importance_out, uint32_t* num_features_out);
int snepx_ai_shap_summary_plot(const SNEPPXAIExplainProvider* provider, SNEPPXAIExplainSample* samples, uint32_t num_samples, uint8_t* plot_out, size_t* plot_len);
int snepx_ai_shap_dependence_plot(const SNEPPXAIExplainProvider* provider, const SNEPPXAIExplainSample* samples, uint32_t num_samples, uint32_t feature_index, uint8_t* plot_out, size_t* plot_len);
int snepx_ai_shap_provider_destroy(SNEPPXAIExplainProvider* provider);
int snepx_ai_shap_sample_destroy(SNEPPXAIExplainSample* sample);

#endif
