#include "side_channel_resistant_primitives.h"
/*
 * SNEPPX - Scalar Multiplication (Curve25519 / Ed25519)
 *
 * WHAT
 *   Scalar Multiplication (Curve25519 / Ed25519).
 *
 * CONCEPT
 *   Constant-time scalar multiplication using the Montgomery ladder on Curve25519 and Ed25519.
 *
 * ROLE
 *   Foundation for X25519 key exchange and Ed25519 signatures used by the key-vault.
 *
 * REFERENCES
 *   RFC 7748 (X25519), RFC 8032 (Ed25519).
 */



/**
 * @brief Perform Sc Select U32.
 *
 * @param condition [in] Condition value.
 * @param a [in] A value.
 *
 * @return 0 on success, -1 on error.
 */
uint32_t SNEPPX_sc_select_u32(uint32_t condition, uint32_t a, uint32_t b) {
    uint32_t mask = (uint32_t)((int32_t)condition >> 31);
    return (a & mask) | (b & ~mask);
}

/**
 * @brief Perform Sc Select U64.
 *
 * @param condition [in] Condition value.
 * @param a [in] A value.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_sc_select_u64(uint64_t condition, uint64_t a, uint64_t b) {
    uint64_t mask = (uint64_t)((int64_t)condition >> 63);
    return (a & mask) | (b & ~mask);
}

/**
 * @brief Perform Sc Equal U32.
 *
 * @param a [in] A value.
 *
 * @return 0 on success, -1 on error.
 */
uint32_t SNEPPX_sc_equal_u32(uint32_t a, uint32_t b) {
    uint32_t diff = a ^ b;
    diff |= diff >> 16;
    diff |= diff >> 8;
    diff |= diff >> 4;
    diff |= diff >> 2;
    diff |= diff >> 1;
    return (uint32_t)((diff & 1U) - 1U);
}

/**
 * @brief Perform Sc Lt U32.
 *
 * @param a [in] A value.
 *
 * @return 0 on success, -1 on error.
 */
uint32_t SNEPPX_sc_lt_u32(uint32_t a, uint32_t b) {
    uint64_t diff = (uint64_t)a - (uint64_t)b;
    return (uint32_t)(diff >> 32);
}

/**
 * @brief Perform Sc Is Zero U32.
 *
 * @return 0 on success, -1 on error.
 */
uint32_t SNEPPX_sc_is_zero_u32(uint32_t a) {
    uint32_t t = a;
    t |= t >> 16;
    t |= t >> 8;
    t |= t >> 4;
    t |= t >> 2;
    t |= t >> 1;
    return (uint32_t)((t & 1U) - 1U);
}

/**
 * @brief Perform Sc Cond Copy.
 *
 * @param dst [out] Dst value.
 * @param src [in] Src value.
 * @param len [in] Len value.
 */
void SNEPPX_sc_cond_copy(void* dst, const void* src, size_t len, uint32_t condition) {
    uint32_t mask = (uint32_t)((int32_t)condition >> 31);
    uint8_t* d = (uint8_t*)dst;
    const uint8_t* s = (const uint8_t*)src;
    for (size_t i = 0; i < len; i++) {
        d[i] = (uint8_t)((s[i] & mask) | (d[i] & ~mask));
    }
}
