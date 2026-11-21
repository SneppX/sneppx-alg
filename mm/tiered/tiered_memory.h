#ifndef SNEPPX_TIERED_MEMORY_H
#define SNEPPX_TIERED_MEMORY_H
#include <stddef.h>
#ifdef __cplusplus
/*
 * SNEPPX - Tiered Memory
 *
 * WHAT
 *   Tiered Memory.
 *
 * CONCEPT
 *   Provides memory management.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif
/**
 * @brief Create Tiered.
 *
 * @param tier_sizes [in] Tier Sizes value.
 * @param num_tiers [in] Num Tiers value.
 * @param tier_names [in] Tier Names value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_tiered_create(const size_t* tier_sizes, int num_tiers, const char** tier_names);
/**
 * @brief Destroy Tiered.
 *
 * @param tm [out] Tm value.
 */
void SNEPPX_tiered_destroy(void* tm);
/**
 * @brief Perform Tiered Alloc.
 *
 * @param tm [out] Tm value.
 * @param size [in] Size value.
 * @param preferred_tier [in] Preferred Tier value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_tiered_alloc(void* tm, size_t size, int preferred_tier);
/**
 * @brief Free Tiered.
 *
 * @param tm [out] Tm value.
 * @param ptr [out] Ptr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tiered_free(void* tm, void* ptr);
/**
 * @brief Perform Tiered Migrate.
 *
 * @param tm [out] Tm value.
 * @param ptr [out] Ptr value.
 * @param target_tier [in] Target Tier value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tiered_migrate(void* tm, void* ptr, int target_tier);
/**
 * @brief Perform Tiered Get Tier.
 *
 * @param tm [out] Tm value.
 * @param ptr [in] Ptr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tiered_get_tier(void* tm, const void* ptr);
/**
 * @brief Perform Tiered Used.
 *
 * @param tm [out] Tm value.
 * @param tier [in] Tier value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_tiered_used(void* tm, int tier);
/**
 * @brief Perform Tiered Capacity.
 *
 * @param tm [out] Tm value.
 * @param tier [in] Tier value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_tiered_capacity(void* tm, int tier);
#ifdef __cplusplus
}
#endif
#endif
