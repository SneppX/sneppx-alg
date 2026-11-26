#ifndef SNEPPX_SPHINCS_PLUS_H
#define SNEPPX_SPHINCS_PLUS_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stddef.h>
#include <stdint.h>

#define SPHINCS_PUBLICKEYBYTES 32
#define SPHINCS_SECRETKEYBYTES 64
#define SPHINCS_SIGBYTES 14592
    /*
     * SNEPPX - Experimental round-3 SPHINCS+ (SLH-DSA lineage)
     *
     * WHAT
     *   Experimental round-3 SPHINCS+ stateless hash-based signatures.
     *
     * CONCEPT
     *   Hash-based signatures using round-3 parameter sets; not byte-compatible
     *   with finalized FIPS 205 SLH-DSA.
     *
     * ROLE
     *   Experimental Layer S0 post-quantum crypto for long-lived signatures.
     *
     * REFERENCES
     *   NIST PQC Round 3 SPHINCS+ finalist. No FIPS validation is claimed.
     */

    /**
     * @brief Generate Sphincs.
     *
     * @param pk [out] Pk value.
     * @param sk [out] Sk value.
     * @param variant [in] Variant value.
     *
     * @return 0 on success, -1 on error.
     */
    int SNEPPX_sphincs_keygen(uint8_t* pk, uint8_t* sk, int variant);
    /**
     * @brief Sign Sphincs.
     *
     * @param sig [out] Sig value.
     * @param siglen [out] Siglen value.
     * @param m [in] M value.
     * @param mlen [in] Mlen value.
     * @param sk [in] Sk value.
     * @param variant [in] Variant value.
     *
     * @return 0 on success, -1 on error.
     */
    int SNEPPX_sphincs_sign(uint8_t* sig, size_t* siglen, const uint8_t* m, size_t mlen, const uint8_t* sk,
                            int variant);
    /**
     * @brief Verify Sphincs.
     *
     * @param sig [in] Sig value.
     * @param siglen [in] Siglen value.
     * @param m [in] M value.
     * @param mlen [in] Mlen value.
     * @param pk [in] Pk value.
     * @param variant [in] Variant value.
     *
     * @return 0 on success, -1 on error.
     */
    int SNEPPX_sphincs_verify(const uint8_t* sig, size_t siglen, const uint8_t* m, size_t mlen, const uint8_t* pk,
                              int variant);

#ifdef __cplusplus
}
#endif
#endif
