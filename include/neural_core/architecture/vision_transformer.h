#ifndef SNEPPX_VISION_TRANSFORMER_H
#define SNEPPX_VISION_TRANSFORMER_H
#include <stddef.h>
#include <stdbool.h>
/*
 * SNEPPX - Vision Transformer
 *
 * WHAT
 *   Vision Transformer.
 *
 * CONCEPT
 *   ViT configuration and forward API for image classification.
 *
 * ROLE
 *   Declares the Vision Transformer API for image understanding within the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal module).
 */


#ifdef __cplusplus
extern "C" {
#endif

typedef struct SNEPPXVisionTransformer SNEPPXVisionTransformer;

/**
 * @brief Create Vit.
 *
 * @param img_size [in] Img Size value.
 * @param patch_size [in] Patch Size value.
 * @param in_channels [in] In Channels value.
 * @param num_classes [in] Num Classes value.
 * @param hidden_dim [in] Hidden Dim value.
 * @param num_heads [in] Num Heads value.
 * @param num_layers [in] Num Layers value.
 * @param dropout [in] Dropout value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXVisionTransformer* SNEPPX_vit_create(size_t img_size, size_t patch_size,
    size_t in_channels, size_t num_classes, size_t hidden_dim, size_t num_heads,
    size_t num_layers, float dropout);
/**
 * @brief Destroy Vit.
 *
 * @param vit [out] Vit value.
 */
void SNEPPX_vit_destroy(void* vit);
/**
 * @brief Run the forward pass for Vit.
 *
 * @param vit [out] Vit value.
 * @param images [in] Images value.
 * @param batch_size [in] Batch Size value.
 * @param logits [out] Logits value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_vit_forward(void* vit, const float* images, size_t batch_size, float* logits);
/**
 * @brief Perform Vit Extract Features.
 *
 * @param vit [out] Vit value.
 * @param images [in] Images value.
 * @param batch_size [in] Batch Size value.
 * @param features [out] Features value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_vit_extract_features(void* vit, const float* images, size_t batch_size, float* features);

#ifdef __cplusplus
}
#endif
#endif
