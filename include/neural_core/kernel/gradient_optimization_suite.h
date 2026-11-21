#ifndef SNEPPX_OPTIMIZER_H
#define SNEPPX_OPTIMIZER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "multidimensional_tensor_engine.h"
#include <stddef.h>

/*
 * SNEPPX - Gradient Optimization Suite
 *
 * WHAT
 *   Gradient Optimization Suite.
 *
 * CONCEPT
 *   Provides the Gradient Optimization Suite.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_OPTIMIZER_SGD = 0,
    SNEPPX_OPTIMIZER_ADAM,
    SNEPPX_OPTIMIZER_ADAMW,
    SNEPPX_OPTIMIZER_ADAMAX,
    SNEPPX_OPTIMIZER_RMSPROP,
    SNEPPX_OPTIMIZER_ADAGRAD,
    SNEPPX_OPTIMIZER_ADADELTA,
    SNEPPX_OPTIMIZER_RADAM,
    SNEPPX_OPTIMIZER_SF_ADAMW,
} SNEPPXOptimizerType;

typedef struct {
    float learning_rate;
    float momentum;
    float weight_decay;
    float grad_clip;
    SNEPPXOptimizerType type;
    float beta1, beta2;
    float epsilon;
    float dampening;
    int nesterov;
    float rho;
} SNEPPXOptimizerConfig;

typedef struct {
    float learning_rate;
    float momentum;
    float weight_decay;
    float grad_clip;
    SNEPPXTensor** momentum_buffers;
    size_t num_params;
    size_t step_count;
    SNEPPXOptimizerType type;
    SNEPPXTensor** state_buf2;
    SNEPPXTensor** state_buf3;
    float beta1, beta2;
    float epsilon;
    float dampening;
    int nesterov;
    float rho;
} SNEPPXOptimizer;

/**
 * @brief Perform Optimizer Config Default.
 *
 * @return The result value, or 0 on error.
 */
SNEPPXOptimizerConfig SNEPPX_optimizer_config_default(void);
/**
 * @brief Create Optimizer.
 *
 * @param config [in] Config value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXOptimizer* SNEPPX_optimizer_create(const SNEPPXOptimizerConfig* config);
/**
 * @brief Destroy Optimizer.
 *
 * @param opt [out] Opt value.
 */
void SNEPPX_optimizer_destroy(SNEPPXOptimizer* opt);
/**
 * @brief Perform Optimizer Step.
 *
 * @param opt [out] Opt value.
 * @param params [out] Params value.
 * @param grads [out] Grads value.
 * @param num_params [in] Num Params value.
 */
void SNEPPX_optimizer_step(SNEPPXOptimizer* opt, SNEPPXTensor** params, SNEPPXTensor** grads, size_t num_params);

/**
 * @brief Create Sgd.
 *
 * @param lr [in] Lr value.
 * @param momentum [in] Momentum value.
 * @param weight_decay [in] Weight Decay value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXOptimizer* SNEPPX_sgd_create(float lr, float momentum, float weight_decay);
/**
 * @brief Create Adam.
 *
 * @param lr [in] Lr value.
 * @param beta1 [in] Beta1 value.
 * @param beta2 [in] Beta2 value.
 * @param eps [in] Eps value.
 * @param weight_decay [in] Weight Decay value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXOptimizer* SNEPPX_adam_create(float lr, float beta1, float beta2, float eps, float weight_decay);
/**
 * @brief Create Adamw.
 *
 * @param lr [in] Lr value.
 * @param beta1 [in] Beta1 value.
 * @param beta2 [in] Beta2 value.
 * @param eps [in] Eps value.
 * @param weight_decay [in] Weight Decay value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXOptimizer* SNEPPX_adamw_create(float lr, float beta1, float beta2, float eps, float weight_decay);
/**
 * @brief Create Rmsprop.
 *
 * @param lr [in] Lr value.
 * @param alpha [in] Alpha value.
 * @param eps [in] Eps value.
 * @param momentum [in] Momentum value.
 * @param weight_decay [in] Weight Decay value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXOptimizer* SNEPPX_rmsprop_create(float lr, float alpha, float eps, float momentum, float weight_decay);
/**
 * @brief Create Adagrad.
 *
 * @param lr [in] Lr value.
 * @param eps [in] Eps value.
 * @param weight_decay [in] Weight Decay value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXOptimizer* SNEPPX_adagrad_create(float lr, float eps, float weight_decay);

typedef enum {
    SNEPPX_LR_STEP,
    SNEPPX_LR_EXPONENTIAL,
    SNEPPX_LR_COSINE,
    SNEPPX_LR_REDUCE_ON_PLATEAU,
} SNEPPXLRSchedulerType;

typedef struct SNEPPXLRScheduler {
    SNEPPXLRSchedulerType type;
    float* lr_ptr;
    float gamma;
    float min_lr, max_lr;
    size_t step_size;
    float factor;
    size_t patience;
    size_t total_steps;
    size_t last_epoch;
    float best_loss;
    int wait_count;
    int mode_min;
} SNEPPXLRScheduler;

/**
 * @brief Perform Lr Scheduler Step Lr.
 *
 * @param lr_ptr [out] Lr Ptr value.
 * @param gamma [in] Gamma value.
 * @param step_size [in] Step Size value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXLRScheduler* SNEPPX_lr_scheduler_step_lr(float* lr_ptr, float gamma, size_t step_size);
/**
 * @brief Perform Lr Scheduler Exponential.
 *
 * @param lr_ptr [out] Lr Ptr value.
 * @param gamma [in] Gamma value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXLRScheduler* SNEPPX_lr_scheduler_exponential(float* lr_ptr, float gamma);
/**
 * @brief Perform Lr Scheduler Cosine.
 *
 * @param lr_ptr [out] Lr Ptr value.
 * @param min_lr [in] Min Lr value.
 * @param max_lr [in] Max Lr value.
 * @param total_steps [in] Total Steps value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXLRScheduler* SNEPPX_lr_scheduler_cosine(float* lr_ptr, float min_lr, float max_lr, size_t total_steps);
/**
 * @brief Perform Lr Scheduler Reduce On Plateau.
 *
 * @param lr_ptr [out] Lr Ptr value.
 * @param factor [in] Factor value.
 * @param patience [in] Patience value.
 * @param mode_min [in] Mode Min value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXLRScheduler* SNEPPX_lr_scheduler_reduce_on_plateau(float* lr_ptr, float factor, size_t patience, int mode_min);
/**
 * @brief Destroy Lr Scheduler.
 *
 * @param sched [out] Sched value.
 */
void SNEPPX_lr_scheduler_destroy(SNEPPXLRScheduler* sched);
/**
 * @brief Perform Lr Scheduler Step.
 *
 * @param sched [out] Sched value.
 * @param current_loss [in] Current Loss value.
 */
void SNEPPX_lr_scheduler_step(SNEPPXLRScheduler* sched, float current_loss);


#ifdef __cplusplus
}
#endif
#endif
