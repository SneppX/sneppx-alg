#ifndef NEURAL_CORE_MODEL_ZOO_MODEL_CARD_H
#define NEURAL_CORE_MODEL_ZOO_MODEL_CARD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
/*
 * SNEPPX - Model Card
 *
 * WHAT
 *   Model Card.
 *
 * CONCEPT
 *   Provides the Model Card.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif

typedef struct ModelCard {
    // Identity
    char *name;
    char *version;
    char *description;
    char *author;
    char *license;
    char *repository;
    char *homepage;
    
    // Architecture
    char *architecture;
    char *model_type;       // e.g., "causal-lm", "seq2seq", "vision"
    char *framework;        // e.g., "sneppx", "pytorch", "tensorflow"
    
    // Model details
    int64_t num_parameters;
    int64_t num_layers;
    int64_t hidden_size;
    int64_t vocab_size;
    int64_t max_seq_len;
    
    // Training details
    char *training_data;
    char *training_procedure;
    char *optimizer;
    double learning_rate;
    int64_t batch_size;
    int64_t num_epochs;
    char *precision;        // e.g., "fp32", "fp16", "bf16", "int8"
    
    // Performance metrics
    double perplexity;
    double accuracy;
    double f1_score;
    double latency_ms;      // Inference latency
    double throughput;      // tokens/sec
    
    // Usage
    char *intended_use;
    char *limitations;
    char *ethical_considerations;
    
    // Citations
    char *citation;
    
    // Tags
    char **tags;
    int num_tags;
    
    // Files
    char *config_path;
    char *weights_path;
    char *tokenizer_path;
} ModelCard;

// Create/destroy
ModelCard *model_card_create(void);
void model_card_destroy(ModelCard *card);

// JSON serialization
char *model_card_to_json(const ModelCard *card, int pretty);
ModelCard *model_card_from_json(const char *json);

// File I/O
int model_card_save(const ModelCard *card, const char *path);
ModelCard *model_card_load(const char *path);

// Validation
int model_card_validate(const ModelCard *card, char **error_out);

// Setters
void model_card_set_name(ModelCard *card, const char *name);
void model_card_set_version(ModelCard *card, const char *version);
void model_card_set_description(ModelCard *card, const char *desc);
void model_card_set_author(ModelCard *card, const char *author);
void model_card_set_license(ModelCard *card, const char *license);
void model_card_set_repository(ModelCard *card, const char *repo);
void model_card_set_architecture(ModelCard *card, const char *arch);
void model_card_set_num_parameters(ModelCard *card, int64_t params);
void model_card_add_tag(ModelCard *card, const char *tag);

#ifdef __cplusplus
}
#endif

#endif // NEURAL_CORE_MODEL_ZOO_MODEL_CARD_H
