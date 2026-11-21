#include "constant_time_operations.h"
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
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ct_equal(const uint8_t* a, const uint8_t* b, size_t len) {
    uint8_t diff = 0;
    for (size_t i = 0; i < len; i++) diff |= a[i] ^ b[i];
    return (int)((diff == 0) ? 1 : 0);
}

/**
 * @brief Perform Ct Select.
 *
 * @param mask [in] Mask value.
 * @param a [in] A value.
 *
 * @return 0 on success, -1 on error.
 */
uint8_t SNEPPX_ct_select(uint8_t mask, uint8_t a, uint8_t b) {
    return (uint8_t)(b ^ (mask & (a ^ b)));
}

/**
 * @brief Perform Ct Copy.
 *
 * @param dst [out] Dst value.
 * @param src [in] Src value.
 * @param len [in] Len value.
 */
void SNEPPX_ct_copy(uint8_t* dst, const uint8_t* src, size_t len, uint8_t condition) {
    uint16_t m = condition | ((uint16_t)condition << 8);
    m |= m >> 4; m |= m >> 2; m |= m >> 1;
    uint8_t mask = (uint8_t)(uint8_t)m;
    for (size_t i = 0; i < len; i++) dst[i] = (uint8_t)(dst[i] ^ (mask & (dst[i] ^ src[i])));
}

/**
 * @brief Perform Ct Is Zero.
 *
 * @param data [in] Data value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ct_is_zero(const uint8_t* data, size_t len) {
    uint8_t acc = 0;
    for (size_t i = 0; i < len; i++) acc |= data[i];
    return (int)((acc == 0) ? 1 : 0);
}

/**
 * @brief Perform Ct Compare 32.
 *
 * @param a [in] A value.
 * @param b [in] B value.
 *
 * @return 0 on success, -1 on error.
 */
uint32_t SNEPPX_ct_compare_32(const uint32_t* a, const uint32_t* b, size_t len) {
    uint32_t diff = 0;
    for (size_t i = 0; i < len; i++) diff |= a[i] ^ b[i];
    return diff;
}
