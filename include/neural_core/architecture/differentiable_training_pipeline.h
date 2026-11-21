#ifndef SNEPPX_TRAIN_H
#define SNEPPX_TRAIN_H

#ifdef __cplusplus
extern "C" {
#endif

#include "system_architecture_definitions.h"
#include "automatic_differentiation_framework.h"
#include "gradient_optimization_suite.h"
#include "multidimensional_tensor_engine.h"
#include <stddef.h>
/*
 * SNEPPX - Differentiable Training Pipeline
 *
 * WHAT
 *   Differentiable Training Pipeline.
 *
 * CONCEPT
 *   End-to-end differentiable training pipeline configuration and API.
 *
 * ROLE
 *   Declares the training pipeline API that wraps HSS, SER, and attention modules for end-to-end differentiable training.
 *
 * REFERENCES
 *   None (internal module).
 */



typedef struct {
    size_t num_epochs;
    size_t batch_size;
    float learning_rate;
    size_t log_interval;
    size_t save_interval;
    SNEPPXDevice device;
    int use_cuda_optimizer;
} SNEPPXTrainConfig;

typedef struct {
    SNEPPXModel* model;
    SNEPPXOptimizer* optimizer;
    SNEPPXTrainConfig config;
    SNEPPXTensor* loss_history;
    size_t step_count;
} SNEPPXTrainer;

/**
 * @brief Perform Train Config Default.
 *
 * @return The result value, or 0 on error.
 */
SNEPPXTrainConfig SNEPPX_train_config_default(void);
/**
 * @brief Create Trainer.
 *
 * @param model [out] Model value.
 * @param config [in] Config value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTrainer* SNEPPX_trainer_create(SNEPPXModel* model, const SNEPPXTrainConfig* config);
/**
 * @brief Destroy Trainer.
 *
 * @param trainer [out] Trainer value.
 */
void SNEPPX_trainer_destroy(SNEPPXTrainer* trainer);
/**
 * @brief Perform Trainer Train Step.
 *
 * @param trainer [out] Trainer value.
 * @param batch_input [in] Batch Input value.
 * @param batch_target [in] Batch Target value.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_trainer_train_step(SNEPPXTrainer* trainer, const SNEPPXTensor* batch_input, const SNEPPXTensor* batch_target);
/**
 * @brief Perform Trainer Evaluate.
 *
 * @param trainer [out] Trainer value.
 * @param val_input [in] Val Input value.
 * @param val_target [in] Val Target value.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_trainer_evaluate(SNEPPXTrainer* trainer, const SNEPPXTensor* val_input, const SNEPPXTensor* val_target);
/**
 * @brief Perform Trainer Save Checkpoint.
 *
 * @param trainer [out] Trainer value.
 * @param path [in] Path value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_trainer_save_checkpoint(SNEPPXTrainer* trainer, const char* path);
/**
 * @brief Perform Trainer Load Checkpoint.
 *
 * @param trainer [out] Trainer value.
 * @param path [in] Path value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_trainer_load_checkpoint(SNEPPXTrainer* trainer, const char* path);


#ifdef __cplusplus
}
#endif
#endif /* SNEPPX_TRAIN_H */
