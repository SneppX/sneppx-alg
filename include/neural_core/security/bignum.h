#ifndef SNEPPX_BIGNUM_H
#define SNEPPX_BIGNUM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_BN_MAX_WORDS 128
#define SNEPPX_BN_WORD uint64_t
#define SNEPPX_BN_HALF_WORD uint32_t
/*
 * SNEPPX - Big-Integer Arithmetic
 *
 * WHAT
 *   Big-Integer Arithmetic.
 *
 * CONCEPT
 *   Arbitrary-precision integer arithmetic for RSA, Dilithium, and other crypto modules.
 *
 * ROLE
 *   Foundation for RSA, Dilithium, and other number-theoretic crypto algorithms.
 *
 * REFERENCES
 *   None (internal utility).
 */



typedef struct {
    SNEPPX_BN_WORD words[SNEPPX_BN_MAX_WORDS];
    int used;
    int sign;
} SNEPPXBigNum;

/**
 * @brief Initialize Bn.
 *
 * @param bn [out] Bn value.
 */
void SNEPPX_bn_init(SNEPPXBigNum* bn);
/**
 * @brief Perform Bn Zero.
 *
 * @param bn [out] Bn value.
 */
void SNEPPX_bn_zero(SNEPPXBigNum* bn);
/**
 * @brief Perform Bn Set Word.
 *
 * @param bn [out] Bn value.
 * @param val [in] Val value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_bn_set_word(SNEPPXBigNum* bn, SNEPPX_BN_WORD val);
/**
 * @brief Perform Bn Set Array.
 *
 * @param bn [out] Bn value.
 * @param bytes [in] Bytes value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_bn_set_array(SNEPPXBigNum* bn, const uint8_t* bytes, size_t len);
/**
 * @brief Perform Bn From Hex.
 *
 * @param bn [out] Bn value.
 * @param hex [in] Hex value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_bn_from_hex(SNEPPXBigNum* bn, const char* hex);
/**
 * @brief Perform Bn To Array.
 *
 * @param bn [in] Bn value.
 * @param out [out] Out value.
 * @param out_len [out] Out Len value.
 */
void SNEPPX_bn_to_array(const SNEPPXBigNum* bn, uint8_t* out, size_t* out_len);

/**
 * @brief Perform Bn Copy.
 *
 * @param dst [out] Dst value.
 * @param src [in] Src value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_bn_copy(SNEPPXBigNum* dst, const SNEPPXBigNum* src);
/**
 * @brief Perform Bn Is Zero.
 *
 * @param bn [in] Bn value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_bn_is_zero(const SNEPPXBigNum* bn);
/**
 * @brief Perform Bn Is One.
 *
 * @param bn [in] Bn value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_bn_is_one(const SNEPPXBigNum* bn);
/**
 * @brief Perform Bn Cmp.
 *
 * @param a [in] A value.
 * @param b [in] B value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_bn_cmp(const SNEPPXBigNum* a, const SNEPPXBigNum* b);
/**
 * @brief Perform Bn Cmp Word.
 *
 * @param a [in] A value.
 * @param b [in] B value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_bn_cmp_word(const SNEPPXBigNum* a, SNEPPX_BN_WORD b);

/**
 * @brief Add Bn.
 *
 * @param r [out] R value.
 * @param a [in] A value.
 * @param b [in] B value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_bn_add(SNEPPXBigNum* r, const SNEPPXBigNum* a, const SNEPPXBigNum* b);
/**
 * @brief Perform Bn Sub.
 *
 * @param r [out] R value.
 * @param a [in] A value.
 * @param b [in] B value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_bn_sub(SNEPPXBigNum* r, const SNEPPXBigNum* a, const SNEPPXBigNum* b);
/**
 * @brief Perform Bn Mul.
 *
 * @param r [out] R value.
 * @param a [in] A value.
 * @param b [in] B value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_bn_mul(SNEPPXBigNum* r, const SNEPPXBigNum* a, const SNEPPXBigNum* b);
/**
 * @brief Perform Bn Div.
 *
 * @param q [out] Q value.
 * @param rem [out] Rem value.
 * @param a [in] A value.
 * @param b [in] B value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_bn_div(SNEPPXBigNum* q, SNEPPXBigNum* rem, const SNEPPXBigNum* a, const SNEPPXBigNum* b);
/**
 * @brief Perform Bn Mod.
 *
 * @param r [out] R value.
 * @param a [in] A value.
 * @param m [in] M value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_bn_mod(SNEPPXBigNum* r, const SNEPPXBigNum* a, const SNEPPXBigNum* m);
/**
 * @brief Perform Bn Exp Mod.
 *
 * @param r [out] R value.
 * @param base [in] Base value.
 * @param exp [in] Exp value.
 * @param mod [in] Mod value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_bn_exp_mod(SNEPPXBigNum* r, const SNEPPXBigNum* base, const SNEPPXBigNum* exp, const SNEPPXBigNum* mod);

/**
 * @brief Perform Bn Gcd.
 *
 * @param r [out] R value.
 * @param a [in] A value.
 * @param b [in] B value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_bn_gcd(SNEPPXBigNum* r, const SNEPPXBigNum* a, const SNEPPXBigNum* b);
/**
 * @brief Perform Bn Inv Mod.
 *
 * @param r [out] R value.
 * @param a [in] A value.
 * @param m [in] M value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_bn_inv_mod(SNEPPXBigNum* r, const SNEPPXBigNum* a, const SNEPPXBigNum* m);
/**
 * @brief Perform Bn Is Prime.
 *
 * @param bn [in] Bn value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_bn_is_prime(const SNEPPXBigNum* bn);
/**
 * @brief Perform Bn Print.
 *
 * @param bn [in] Bn value.
 */
void SNEPPX_bn_print(const SNEPPXBigNum* bn);

#ifdef __cplusplus
}
#endif
#endif
