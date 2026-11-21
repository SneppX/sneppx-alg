#ifndef SNEPPX_OPTIMIZER_INTERNAL_H
#define SNEPPX_OPTIMIZER_INTERNAL_H
/*
 * SNEPPX - Optimizer Impl
 *
 * WHAT
 *   Optimizer Impl.
 *
 * CONCEPT
 *   Provides optimizer implementations.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/*
 * Optimizer Internal — v0.5
 *
 * PURPOSE: Per-parameter update kernels, gradient clipping, weight decay,
 * and learning-rate schedule internals.  The external API defines the
 * optimizer struct; this header holds the step implementations and state
 * tracking for SGD, Adam, AdamW, and LAMB.
 *
 * DEPENDENCIES: gradient_optimization_suite.h, multidimensional_tensor_engine.h
 * VERSION: v0.5
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void**     param_data;
    void**     grad_data;
    void**     momentum_buf;
    void**     velocity_buf;
    size_t     num_params;
    size_t     param_size;
    float      lr;
    float      beta1;
    float      beta2;
    float      eps;
    float      weight_decay;
    uint64_t   step;
} SNEPPXOptimizerState;

typedef enum {
    SNEPPX_SCHEDULE_CONSTANT,
    SNEPPX_SCHEDULE_COSINE,
    SNEPPX_SCHEDULE_LINEAR,
    SNEPPX_SCHEDULE_STEP_DECAY,
    SNEPPX_SCHEDULE_WARMUP_COSINE,
} SNEPPXLRSchedule;

typedef struct {
    SNEPPXLRSchedule type;
    float          base_lr;
    float          min_lr;
    size_t         warmup_steps;
    size_t         total_steps;
    size_t         step_count;
    float          current_lr;
} SNEPPXLRScheduler;

/**
 * @brief Initialize Optimizer State.
 *
 * @param state [out] State value.
 * @param num_params [in] Num Params value.
 * @param param_size [in] Param Size value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_optimizer_state_init(SNEPPXOptimizerState* state, size_t num_params, size_t param_size);
/**
 * @brief Destroy Optimizer State.
 *
 * @param state [out] State value.
 */
void SNEPPX_optimizer_state_destroy(SNEPPXOptimizerState* state);

/**
 * @brief Perform Optimizer Sgd Step.
 *
 * @param state [out] State value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_optimizer_sgd_step(SNEPPXOptimizerState* state);
/**
 * @brief Perform Optimizer Adam Step.
 *
 * @param state [out] State value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_optimizer_adam_step(SNEPPXOptimizerState* state);
/**
 * @brief Perform Optimizer Adamw Step.
 *
 * @param state [out] State value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_optimizer_adamw_step(SNEPPXOptimizerState* state);
/**
 * @brief Perform Optimizer Lamb Step.
 *
 * @param state [out] State value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_optimizer_lamb_step(SNEPPXOptimizerState* state);

/**
 * @brief Perform Gradient Clip Norm.
 *
 * @param grads [out] Grads value.
 * @param num_params [in] Num Params value.
 * @param param_size [in] Param Size value.
 * @param max_norm [in] Max Norm value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_gradient_clip_norm(void** grads, size_t num_params, size_t param_size, float max_norm);
/**
 * @brief Perform Gradient Clip Value.
 *
 * @param grads [out] Grads value.
 * @param num_params [in] Num Params value.
 * @param param_size [in] Param Size value.
 * @param min_val [in] Min Val value.
 * @param max_val [in] Max Val value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_gradient_clip_value(void** grads, size_t num_params, size_t param_size, float min_val, float max_val);

/**
 * @brief Initialize Lr Scheduler.
 *
 * @param sched [out] Sched value.
 * @param type [in] Type value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_lr_scheduler_init(SNEPPXLRScheduler* sched, SNEPPXLRSchedule type);
/**
 * @brief Perform Impl Lr Scheduler Step.
 *
 * @param sched [out] Sched value.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_impl_lr_scheduler_step(SNEPPXLRScheduler* sched);
/**
 * @brief Reset Lr Scheduler.
 *
 * @param sched [out] Sched value.
 */
void  SNEPPX_lr_scheduler_reset(SNEPPXLRScheduler* sched);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_OPTIMIZER_INTERNAL_H */
