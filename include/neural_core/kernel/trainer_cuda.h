#ifndef SNEPPX_TRAINER_CUDA_H
#define SNEPPX_TRAINER_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

#include "multidimensional_tensor_engine.h"
#include "gradient_optimization_suite.h"
#include <stddef.h>

/*
 * SNEPPX - Trainer Cuda
 *
 * WHAT
 *   Trainer Cuda.
 *
 * CONCEPT
 *   Provides the Trainer Cuda.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/**
 * @brief Perform Trainer Cuda Available.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_trainer_cuda_available(void);
/**
 * @brief Initialize Trainer Cuda.
 *
 * @param params [out] Params value.
 * @param n [in] N value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_trainer_cuda_init(SNEPPXTensor** params, size_t n);
/**
 * @brief Perform Trainer Cuda Shutdown.
 */
void SNEPPX_trainer_cuda_shutdown(void);
/**
 * @brief Perform Trainer Cuda Optimizer Step.
 *
 * @param opt [out] Opt value.
 * @param params [out] Params value.
 * @param grads [out] Grads value.
 * @param n [in] N value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_trainer_cuda_optimizer_step(SNEPPXOptimizer* opt, SNEPPXTensor** params, SNEPPXTensor** grads, size_t n);
/**
 * @brief Perform Trainer Cuda Transfer To Device.
 *
 * @param params [out] Params value.
 * @param n [in] N value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_trainer_cuda_transfer_to_device(SNEPPXTensor** params, size_t n);
/**
 * @brief Perform Trainer Cuda Transfer To Host.
 *
 * @param params [out] Params value.
 * @param n [in] N value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_trainer_cuda_transfer_to_host(SNEPPXTensor** params, size_t n);


#ifdef __cplusplus
}
#endif
#endif /* SNEPPX_TRAINER_CUDA_H */
