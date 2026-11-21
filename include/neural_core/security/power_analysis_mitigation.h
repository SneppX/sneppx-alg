#ifndef SNEPPX_POWER_H
#define SNEPPX_POWER_H

#ifdef __cplusplus
extern "C" {
#endif

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



/**
 * @brief Start Power Balance.
 */
void SNEPPX_power_balance_start(void);
/**
 * @brief Perform Power Balance End.
 */
void SNEPPX_power_balance_end(void);
/**
 * @brief Perform Power Dummy Op.
 */
void SNEPPX_power_dummy_op(void);

/**
 * @brief Perform Power Mul Const Time.
 *
 * @param a [in] A value.
 * @param b [in] B value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_power_mul_const_time(int a, int b);
/**
 * @brief Perform Power Cmp Const Time.
 *
 * @param a [in] A value.
 * @param b [in] B value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_power_cmp_const_time(int a, int b);
/**
 * @brief Perform Power Mask Gen.
 *
 * @param condition [in] Condition value.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_power_mask_gen(int condition);


#ifdef __cplusplus
}
#endif
#endif
