#include "hkdf.h"
#include "hmac.h"
#include <string.h>
/*
 * SNEPPX - HKDF (HMAC-based Key Derivation Function)
 *
 * WHAT
 *   HKDF (HMAC-based Key Derivation Function).
 *
 * CONCEPT
 *   RFC 5869 HKDF: extract a pseudorandom key, then expand into derived keys.
 *
 * ROLE
 *   Used by the key-vault for sub-key derivation and secure-transport session key derivation.
 *
 * REFERENCES
 *   RFC 5869 (HKDF).
 */



/**
 * @brief Perform Hkdf Extract.
 *
 * @param salt [in] Salt value.
 * @param salt_len [in] Salt Len value.
 * @param ikm [in] Ikm value.
 * @param ikm_len [in] Ikm Len value.
 * @param prk [out] Prk value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hkdf_extract(const uint8_t* salt, size_t salt_len, const uint8_t* ikm, size_t ikm_len, uint8_t* prk, size_t prk_len) {
    if (!ikm||!prk||prk_len<32) return -1;
    const uint8_t* s=salt;
    size_t sl=salt_len;
    uint8_t zeros[32];
    if (!salt||salt_len==0) { memset(zeros,0,32); s=zeros; sl=32; }
    return SNEPPX_hmac_sha256(s,sl,ikm,ikm_len,prk);
}

/**
 * @brief Perform Hkdf Expand.
 *
 * @param prk [in] Prk value.
 * @param prk_len [in] Prk Len value.
 * @param info [in] Info value.
 * @param info_len [in] Info Len value.
 * @param okm [out] Okm value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hkdf_expand(const uint8_t* prk, size_t prk_len, const uint8_t* info, size_t info_len, uint8_t* okm, size_t okm_len) {
    if (!prk||!okm||okm_len>255*32) return -1;
    uint8_t t[32],block[32+256];
    size_t tl=0,offset=0;
    uint8_t n=1;
    while (offset<okm_len) {
        if (tl) memcpy(block,t,tl);
        if (info) memcpy(block+tl,info,info_len);
        block[tl+info_len]=n;
        int ret=SNEPPX_hmac_sha256(prk,prk_len,block,tl+info_len+1,t);
        if (ret!=0) return -1;
        tl=32;
        size_t copy=(okm_len-offset<32)?okm_len-offset:32;
        memcpy(okm+offset,t,copy);
        offset+=copy; n++;
    }
    return 0;
}

/**
 * @brief Perform Hkdf.
 *
 * @param salt [in] Salt value.
 * @param salt_len [in] Salt Len value.
 * @param ikm [in] Ikm value.
 * @param ikm_len [in] Ikm Len value.
 * @param info [in] Info value.
 * @param info_len [in] Info Len value.
 * @param okm [out] Okm value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hkdf(const uint8_t* salt, size_t salt_len, const uint8_t* ikm, size_t ikm_len, const uint8_t* info, size_t info_len, uint8_t* okm, size_t okm_len) {
    uint8_t prk[32];
    if (SNEPPX_hkdf_extract(salt,salt_len,ikm,ikm_len,prk,32)!=0) return -1;
    return SNEPPX_hkdf_expand(prk,32,info,info_len,okm,okm_len);
}
