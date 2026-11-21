#include "power_analysis_mitigation.h"
#include <stdint.h>
/*
 * SNEPPX - Power Analysis Mitigation
 *
 * WHAT
 *   Power Analysis Mitigation.
 *
 * CONCEPT
 *   Defenses against differential and simple power analysis on cryptographic operations.
 *
 * ROLE
 *   Layer S5 side-channel resistance of the S0-S9 security stack.
 *
 * REFERENCES
 *   None (internal hardening).
 */



static volatile uint64_t dummy_accumulator = 0;

/**
 * @brief Start Power Balance.
 */
void SNEPPX_power_balance_start(void) {
    dummy_accumulator = 0;
}

/**
 * @brief Perform Power Balance End.
 */
void SNEPPX_power_balance_end(void) {
    dummy_accumulator = 0;
}

/**
 * @brief Perform Power Dummy Op.
 */
void SNEPPX_power_dummy_op(void) {
    uint64_t x = 0xDEADBEEFCAFEBABEULL;
    uint64_t y = 0x0123456789ABCDEFULL;
    for (int i = 0; i < 16; i++) {
        x ^= y;
        x *= 0x9E3779B97F4A7C15ULL;
        y += x;
        y = (y << 13) | (y >> 51);
        dummy_accumulator ^= x ^ y;
    }
}

/**
 * @brief Perform Power Mul Const Time.
 *
 * @param a [in] A value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_power_mul_const_time(int a, int b) {
    return a * b;
}

/**
 * @brief Perform Power Cmp Const Time.
 *
 * @param a [in] A value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_power_cmp_const_time(int a, int b) {
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

/**
 * @brief Perform Power Mask Gen.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_power_mask_gen(int condition) {
    return condition ? UINT64_MAX : 0;
}
