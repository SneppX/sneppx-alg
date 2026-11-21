#ifndef SNEPPX_SC_H
#define SNEPPX_SC_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/*
 * SNEPPX - Side Channel Resistant Primitives
 *
 * WHAT
 *   Side Channel Resistant Primitives.
 *
 * CONCEPT
 *   Provides the Side Channel Resistant Primitives.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/**
 * @brief Perform Sc Select U32.
 *
 * @param condition [in] Condition value.
 * @param a [in] A value.
 * @param b [in] B value.
 *
 * @return 0 on success, -1 on error.
 */
uint32_t SNEPPX_sc_select_u32(uint32_t condition, uint32_t a, uint32_t b);
/**
 * @brief Perform Sc Select U64.
 *
 * @param condition [in] Condition value.
 * @param a [in] A value.
 * @param b [in] B value.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_sc_select_u64(uint64_t condition, uint64_t a, uint64_t b);
/**
 * @brief Perform Sc Equal U32.
 *
 * @param a [in] A value.
 * @param b [in] B value.
 *
 * @return 0 on success, -1 on error.
 */
uint32_t SNEPPX_sc_equal_u32(uint32_t a, uint32_t b);
/**
 * @brief Perform Sc Lt U32.
 *
 * @param a [in] A value.
 * @param b [in] B value.
 *
 * @return 0 on success, -1 on error.
 */
uint32_t SNEPPX_sc_lt_u32(uint32_t a, uint32_t b);
/**
 * @brief Perform Sc Is Zero U32.
 *
 * @param a [in] A value.
 *
 * @return 0 on success, -1 on error.
 */
uint32_t SNEPPX_sc_is_zero_u32(uint32_t a);
/**
 * @brief Perform Sc Cond Copy.
 *
 * @param dst [out] Dst value.
 * @param src [in] Src value.
 * @param len [in] Len value.
 * @param condition [in] Condition value.
 */
void SNEPPX_sc_cond_copy(void* dst, const void* src, size_t len, uint32_t condition);


#ifdef __cplusplus
}
#endif
#endif
