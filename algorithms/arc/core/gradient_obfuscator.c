#include "adversarial_robustness_certification.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <math.h>
/*
 * SNEPPX - ARC Gradient Obfuscation
 *
 * WHAT
 *   ARC Gradient Obfuscation.
 *
 * CONCEPT
 *   Obfuscates gradients during adversarial training to improve robustness.
 *
 * ROLE
 *   Applies gradient obfuscation techniques (randomized smoothing of gradients, gradient clipping with noise) to make adversarial training more effective.
 *
 * REFERENCES
 *   None (internal algorithm).
 */



/**
 * @brief Create Gradient Obfuscator.
 *
 * @param max_params [in] Max Params value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXGradientObfuscator* SNEPPX_gradient_obfuscator_create(size_t max_params, unsigned int seed) {
    (void)seed;
    SNEPPXGradientObfuscator* obf = (SNEPPXGradientObfuscator*)SNEPPX_malloc(sizeof(SNEPPXGradientObfuscator), 64);
    if (!obf) return NULL;
    memset(obf, 0, sizeof(SNEPPXGradientObfuscator));

    size_t shape_m[] = {max_params};
    obf->noise_buffer = SNEPPX_tensor_zeros(shape_m, 1, SNEPPX_FLOAT32);
    obf->clamp_mask = SNEPPX_tensor_zeros(shape_m, 1, SNEPPX_FLOAT32);
    return obf;
}

/**
 * @brief Destroy Gradient Obfuscator.
 */
void SNEPPX_gradient_obfuscator_destroy(SNEPPXGradientObfuscator* obf) {
    if (!obf) return;
    if (obf->noise_buffer) SNEPPX_tensor_destroy(obf->noise_buffer);
    if (obf->clamp_mask) SNEPPX_tensor_destroy(obf->clamp_mask);
    SNEPPX_free(obf, sizeof(SNEPPXGradientObfuscator));
}

/**
 * @brief Perform Arc Obfuscate Gradients.
 *
 * @param obf [out] Obf value.
 * @param gradients [out] Gradients value.
 */
void SNEPPX_arc_obfuscate_gradients(SNEPPXGradientObfuscator* obf, SNEPPXTensor* gradients, int method) {
    if (!obf || !gradients) return;
    float* gd = (float*)gradients->data;
    size_t n = gradients->size;

    if (method & SNEPPX_OBF_NOISE) {
        float* nb = (float*)obf->noise_buffer->data;
        size_t max_n = obf->noise_buffer->size > n ? n : obf->noise_buffer->size;
        float scale = 0.01f;
        unsigned long state = 12345;
        for (size_t i = 0; i < max_n; i += 2) {
            state = state * 1103515245UL + 12345UL;
            float u1 = (float)((state >> 16) & 0x7FFF) / 32767.0f;
            state = state * 1103515245UL + 12345UL;
            float u2 = (float)((state >> 16) & 0x7FFF) / 32767.0f;
            float r = sqrtf(-2.0f * logf(u1 + 1e-10f));
            float theta = 2.0f * 3.14159265f * u2;
            nb[i] = r * cosf(theta) * scale;
            if (i + 1 < max_n)
                nb[i + 1] = r * sinf(theta) * scale;
        }
        for (size_t i = 0; i < n && i < obf->noise_buffer->size; i++) {
            gd[i] += nb[i];
        }
    }

    if (method & SNEPPX_OBF_CLAMP) {
        float* cm = (float*)obf->clamp_mask->data;
        float clip_max = 1.0f;
        for (size_t i = 0; i < n && i < obf->clamp_mask->size; i++) {
            if (gd[i] > clip_max) { gd[i] = clip_max; cm[i] = 1.0f; }
            else if (gd[i] < -clip_max) { gd[i] = -clip_max; cm[i] = 1.0f; }
            else { cm[i] = 0.0f; }
        }
    }
}
