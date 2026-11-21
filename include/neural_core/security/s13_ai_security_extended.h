#ifndef SNEPPX_S13_AI_SECURITY_EXTENDED_H
#define SNEPPX_S13_AI_SECURITY_EXTENDED_H

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_FEDERATED_MAX_CLIENTS 1024
#define SNEPPX_MODEL_MAX_LAYERS 512
#define SNEPPX_ADVERSARIAL_MAX_EPS 64
#define SNEPPX_PRIVACY_MAX_BUDGET 100.0

/*
 * SNEPPX - S13 Ai Security Extended
 *
 * WHAT
 *   S13 Ai Security Extended.
 *
 * CONCEPT
 *   Provides the S13 Ai Security Extended.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_ATTACK_FGSM,
    SNEPPX_ATTACK_PGD,
    SNEPPX_ATTACK_CW,
    SNEPPX_ATTACK_DEEPFOOL,
    SNEPPX_ATTACK_BOUNDARY,
    SNEPPX_ATTACK_HOP_SKIP_JUMP,
    SNEPPX_ATTACK_AUTO_ATTACK,
    SNEPPX_ATTACK_SQUARE,
    SNEPPX_ATTACK_ZOO,
    SNEPPX_ATTACK_SPSA,
    SNEPPX_ATTACK_GEODA,
    SNEPPX_ATTACK_PIXEL
} SNEPPXAdversarialAttackType;

typedef struct {
    SNEPPXAdversarialAttackType attack_type;
    float epsilon;
    float epsilon_step;
    uint32_t num_iterations;
    uint32_t random_start;
    uint32_t targeted : 1;
    uint32_t linf_norm : 1;
    uint32_t l2_norm : 1;
    float confidence;
    uint32_t num_restarts;
    float* loss_history;
    size_t loss_history_len;
} SNEPPXAdversarialAttackConfig;

typedef struct {
    float* original_input;
    float* adversarial_input;
    size_t input_size;
    uint32_t original_label;
    uint32_t adversarial_label;
    float perturbation_norm_l2;
    float perturbation_norm_linf;
    float confidence_score;
    uint32_t attack_success : 1;
    uint32_t num_queries;
    double elapsed_ms;
} SNEPPXAdversarialSample;

typedef struct {
    uint32_t num_samples;
    SNEPPXAdversarialSample* samples;
    float mean_perturbation_l2;
    float mean_perturbation_linf;
    float attack_success_rate;
    float avg_queries;
    uint32_t total_attacks;
} SNEPPXAdversarialReport;

typedef struct {
    uint8_t* model_weights;
    size_t weights_size;
    uint32_t num_clients;
    uint32_t min_clients;
    float aggregation_rate;
    uint8_t secure_aggregation : 1;
    uint8_t differential_privacy : 1;
    float noise_multiplier;
    float clipping_threshold;
    uint32_t num_rounds;
    uint64_t round_timeout_ms;
} SNEPPXFederatedLearningConfig;

typedef struct {
    float* client_gradients;
    size_t gradient_size;
    uint32_t client_id;
    float* encrypted_gradients;
    size_t encrypted_size;
    uint8_t* masking_key[32];
    float noise_scale;
    uint64_t round_number;
} SNEPPXFederatedClientUpdate;

typedef struct {
    float* global_model;
    size_t model_size;
    uint32_t num_clients_aggregated;
    float total_noise_added;
    float privacy_spent;
    float remaining_budget;
} SNEPPXFederatedRound;

typedef struct {
    SNEPPXFederatedRound* rounds;
    uint32_t num_rounds;
    float total_epsilon_spent;
    float total_delta_spent;
    uint32_t max_clients_per_round;
} SNEPPXFederatedTrainer;

// Adversarial robustness
int snepx_adversarial_generate(SNEPPXAdversarialAttackConfig* config, const float* input, size_t input_size, uint32_t label, float* adversarial_out);
int snepx_adversarial_evaluate(SNEPPXAdversarialAttackConfig* config, const float* dataset, size_t dataset_size, const uint32_t* labels, size_t num_samples, SNEPPXAdversarialReport* report);
int snepx_adversarial_defend(SNEPPXAdversarialAttackConfig* config, float* model_weights, size_t weights_size, float* certified_radius_out);

// Federated learning
int snepx_federated_init(SNEPPXFederatedLearningConfig* config);
int snepx_federated_round_begin(SNEPPXFederatedLearningConfig* config, SNEPPXFederatedRound* round);
int snepx_federated_client_submit(SNEPPXFederatedRound* round, const SNEPPXFederatedClientUpdate* update);
int snepx_federated_aggregate(SNEPPXFederatedRound* round, float* global_model_out);
int snepx_federated_privacy_account(SNEPPXFederatedTrainer* trainer, float epsilon, float delta);
int snepx_federated_secure_aggregate(const SNEPPXFederatedClientUpdate* updates, uint32_t num_updates, float* aggregated, size_t aggregated_size, const uint8_t** masks);

// Model extraction defense
typedef struct {
    uint32_t max_queries_per_ip;
    uint32_t query_window_seconds;
    uint32_t perturbation_threshold;
    float confidence_threshold;
    uint32_t total_queries;
    uint32_t suspicious_ips_count;
    uint32_t blocked_ips_count;
} SNEPPXModelExtractionDefense;

int snepx_model_extraction_defense_init(SNEPPXModelExtractionDefense* defense);
int snepx_model_extraction_check(SNEPPXModelExtractionDefense* defense, const uint8_t* ip, size_t ip_len, const float* query, size_t query_size);
int snepx_model_extraction_block_ip(SNEPPXModelExtractionDefense* defense, const uint8_t* ip, size_t ip_len);

// Data poisoning defense
typedef struct {
    float outlier_threshold;
    uint32_t min_cluster_size;
    uint32_t max_outlier_fraction;
    uint8_t* training_data;
    size_t training_data_size;
    uint32_t* outlier_indices;
    size_t outlier_count;
    float* influence_scores;
    size_t influence_count;
} SNEPPXDataPoisoningDetector;

int snepx_data_poisoning_init(SNEPPXDataPoisoningDetector* detector, const float* training_data, size_t data_size, size_t num_samples);
int snepx_data_poisoning_detect(SNEPPXDataPoisoningDetector* detector, uint32_t* outlier_indices_out, size_t* outlier_count);
int snepx_data_poisoning_influence(SNEPPXDataPoisoningDetector* detector, const float* sample, size_t sample_size, float* influence_score);
int snepx_data_poisoning_remove(SNEPPXDataPoisoningDetector* detector, const uint32_t* indices, size_t count);

// Model watermarking
typedef struct {
    uint8_t watermark_key[32];
    uint8_t watermark_signal[256];
    size_t signal_len;
    float strength;
    uint32_t layer_index;
    uint8_t* embedded_weights;
    size_t embedded_weights_size;
} SNEPPXModelWatermark;

int snepx_watermark_embed(SNEPPXModelWatermark* watermark, float* weights, size_t weights_size);
int snepx_watermark_verify(SNEPPXModelWatermark* watermark, const float* weights, size_t weights_size, float* confidence);
int snepx_watermark_extract(SNEPPXModelWatermark* watermark, const float* weights, size_t weights_size, uint8_t* signal_out);

#endif
