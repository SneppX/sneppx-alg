#ifndef SNEPPX_DIFFUSION_MODEL_H
#define SNEPPX_DIFFUSION_MODEL_H
#include <stddef.h>
#include <stdbool.h>
/*
 * SNEPPX - Diffusion Model
 *
 * WHAT
 *   Diffusion Model.
 *
 * CONCEPT
 *   Diffusion model configuration and forward API for the SNEPPX-Algo diffusion pipeline.
 *
 * ROLE
 *   Declares the diffusion model API for generative modeling within the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal module).
 */


#ifdef __cplusplus
extern "C" {
#endif

typedef struct SNEPPXDiffusion SNEPPXDiffusion;

/**
 * @brief Create Diffusion.
 *
 * @param img_channels [in] Img Channels value.
 * @param img_size [in] Img Size value.
 * @param hidden_dim [in] Hidden Dim value.
 * @param num_timesteps [in] Num Timesteps value.
 * @param noise_schedule_type [in] Noise Schedule Type value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXDiffusion* SNEPPX_diffusion_create(size_t img_channels, size_t img_size,
    size_t hidden_dim, int num_timesteps, int noise_schedule_type);
/**
 * @brief Destroy Diffusion.
 *
 * @param model [out] Model value.
 */
void SNEPPX_diffusion_destroy(void* model);
/**
 * @brief Perform Diffusion Sample.
 *
 * @param model [out] Model value.
 * @param output [out] Output value.
 * @param num_samples [in] Num Samples value.
 * @param cond [in] Cond value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_diffusion_sample(void* model, float* output, size_t num_samples, const float* cond);
/**
 * @brief Perform Diffusion Train Step.
 *
 * @param model [out] Model value.
 * @param images [in] Images value.
 * @param num_samples [in] Num Samples value.
 * @param loss [out] Loss value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_diffusion_train_step(void* model, const float* images, size_t num_samples, float* loss);

#ifdef __cplusplus
}
#endif
#endif
