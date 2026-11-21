#ifndef SNEPPX_ASLR_H
#define SNEPPX_ASLR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/*
 * SNEPPX - Address Space Randomization
 *
 * WHAT
 *   Address Space Randomization.
 *
 * CONCEPT
 *   Provides randomness generation.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/**
 * @brief Set Aslr Random Off.
 *
 * @param max_offset [in] Max Offset value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_aslr_random_offset(size_t max_offset);
/**
 * @brief Apply Aslr.
 *
 * @param base_ptr [out] Base Ptr value.
 * @param size [out] Size value.
 * @param max_random [in] Max Random value.
 */
void SNEPPX_aslr_apply(void** base_ptr, size_t* size, size_t max_random);


#ifdef __cplusplus
}
#endif
#endif
