#ifndef SNEPPX_CT_H
#define SNEPPX_CT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
/*
 * SNEPPX - Constant-Time Operations
 *
 * WHAT
 *   Constant-Time Operations.
 *
 * CONCEPT
 *   Primitives that execute in data-independent time to prevent timing side-channel leaks.
 *
 * ROLE
 *   Layer S5 side-channel resistance used by all crypto modules handling secret keys.
 *
 * REFERENCES
 *   None (internal utility).
 */



/**
 * @brief Perform Ct Equal.
 *
 * @param a [in] A value.
 * @param b [in] B value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ct_equal(const uint8_t* a, const uint8_t* b, size_t len);
/**
 * @brief Perform Ct Select.
 *
 * @param mask [in] Mask value.
 * @param a [in] A value.
 * @param b [in] B value.
 *
 * @return 0 on success, -1 on error.
 */
uint8_t SNEPPX_ct_select(uint8_t mask, uint8_t a, uint8_t b);
/**
 * @brief Perform Ct Copy.
 *
 * @param dst [out] Dst value.
 * @param src [in] Src value.
 * @param len [in] Len value.
 * @param condition [in] Condition value.
 */
void SNEPPX_ct_copy(uint8_t* dst, const uint8_t* src, size_t len, uint8_t condition);
/**
 * @brief Perform Ct Is Zero.
 *
 * @param data [in] Data value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ct_is_zero(const uint8_t* data, size_t len);
/**
 * @brief Perform Ct Compare 32.
 *
 * @param a [in] A value.
 * @param b [in] B value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
uint32_t SNEPPX_ct_compare_32(const uint32_t* a, const uint32_t* b, size_t len);


#ifdef __cplusplus
}
#endif
#endif
