#ifndef SNEPPX_GENERATIVE_ADVERSARIAL_NETWORK_H
#define SNEPPX_GENERATIVE_ADVERSARIAL_NETWORK_H
#include <stddef.h>
#include <stdbool.h>
/*
 * SNEPPX - Generative Adversarial Network
 *
 * WHAT
 *   Generative Adversarial Network.
 *
 * CONCEPT
 *   GAN configuration and training API for the SNEPPX-Algo system.
 *
 * ROLE
 *   Declares the GAN API for generative adversarial training within the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal module).
 */


#ifdef __cplusplus
extern "C" {
#endif

typedef struct SNEPPXGAN SNEPPXGAN;

/**
 * @brief Create Gan.
 *
 * @param latent_dim [in] Latent Dim value.
 * @param hidden_dim [in] Hidden Dim value.
 * @param output_dim [in] Output Dim value.
 * @param use_batch_norm [in] Use Batch Norm value.
 * @param use_spectral_norm [in] Use Spectral Norm value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXGAN* SNEPPX_gan_create(size_t latent_dim, size_t hidden_dim, size_t output_dim,
    int use_batch_norm, int use_spectral_norm);
/**
 * @brief Destroy Gan.
 *
 * @param gan [out] Gan value.
 */
void SNEPPX_gan_destroy(void* gan);
/**
 * @brief Perform Gan Generate.
 *
 * @param gan [out] Gan value.
 * @param noise [in] Noise value.
 * @param num_samples [in] Num Samples value.
 * @param output [out] Output value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_gan_generate(void* gan, const float* noise, size_t num_samples, float* output);
/**
 * @brief Perform Gan Train Step.
 *
 * @param gan [out] Gan value.
 * @param real_samples [in] Real Samples value.
 * @param num_samples [in] Num Samples value.
 * @param g_loss [out] G Loss value.
 * @param d_loss [out] D Loss value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_gan_train_step(void* gan, const float* real_samples, size_t num_samples,
    float* g_loss, float* d_loss);

#ifdef __cplusplus
}
#endif
#endif
