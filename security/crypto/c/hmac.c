#include "hmac.h"
#include "sha512_hashing_implementation.h"
#include "constant_time_operations.h"
#include <string.h>
/*
 * SNEPPX - HMAC (Hash-based Message Authentication Code)
 *
 * WHAT
 *   HMAC (Hash-based Message Authentication Code).
 *
 * CONCEPT
 *   RFC 2104 HMAC: a keyed hash providing integrity and authenticity for arbitrary messages.
 *
 * ROLE
 *   Used by HKDF (extract phase), DRBG reseeding, and keyed integrity checks.
 *
 * REFERENCES
 *   RFC 2104 (HMAC), FIPS 198-1.
 */



static void hmac_compute(const uint8_t* key, size_t key_len, const uint8_t* data, size_t data_len, uint8_t* out, size_t out_len, int use_sha512) {
    uint8_t k[128],ipad[128],opad[128];
    size_t block_size=use_sha512?128:64;
    size_t hash_size=use_sha512?64:32;
    memset(k,0,block_size);
    if (key_len>block_size) {
        SNEPPXSHA512Context ctx;
        SNEPPX_sha512_init(&ctx);
        SNEPPX_sha512_update(&ctx,key,key_len);
        uint8_t h[64];
        SNEPPX_sha512_finish(&ctx,h);
        memcpy(k,h,use_sha512?64:32);
    } else memcpy(k,key,key_len);
    for (size_t i=0;i<block_size;i++) { ipad[i]=k[i]^0x36; opad[i]=k[i]^0x5C; }

    uint8_t inner_hash[64];
    {
        SNEPPXSHA512Context ctx;
        SNEPPX_sha512_init(&ctx);
        SNEPPX_sha512_update(&ctx,ipad,block_size);
        SNEPPX_sha512_update(&ctx,data,data_len);
        SNEPPX_sha512_finish(&ctx,inner_hash);
    }
    size_t final_hash_size = (out_len < hash_size) ? out_len : hash_size;
    {
        SNEPPXSHA512Context ctx;
        SNEPPX_sha512_init(&ctx);
        SNEPPX_sha512_update(&ctx,opad,block_size);
        SNEPPX_sha512_update(&ctx,inner_hash,final_hash_size);
        SNEPPX_sha512_finish(&ctx,out);
    }
    memset(k,0,sizeof(k)); memset(ipad,0,sizeof(ipad)); memset(opad,0,sizeof(opad));
}

/**
 * @brief Initialize Hmac.
 *
 * @param ctx [out] Ctx value.
 * @param key [in] Key value.
 * @param key_len [in] Key Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hmac_init(SNEPPXHMAC* ctx, const uint8_t* key, size_t key_len, int hash_type) {
    if (!ctx||!key) return -1;
    ctx->key_len=key_len<SNEPPX_HMAC_MAX_KEY?key_len:SNEPPX_HMAC_MAX_KEY;
    memcpy(ctx->key,key,ctx->key_len);
    ctx->hash_type=hash_type;
    return 0;
}

/**
 * @brief Compute Hmac.
 *
 * @param ctx [out] Ctx value.
 * @param data [in] Data value.
 * @param data_len [in] Data Len value.
 * @param out [out] Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hmac_compute(SNEPPXHMAC* ctx, const uint8_t* data, size_t data_len, uint8_t* out, size_t* out_len) {
    if (!ctx||!data||!out||!out_len) return -1;
    size_t hash_size=(ctx->hash_type==0)?32:64;
    if (*out_len<hash_size) return -1;
    hmac_compute(ctx->key,ctx->key_len,data,data_len,out,hash_size,ctx->hash_type!=0);
    *out_len=hash_size;
    return 0;
}

/**
 * @brief Perform Hmac Sha256.
 *
 * @param key [in] Key value.
 * @param key_len [in] Key Len value.
 * @param data [in] Data value.
 * @param data_len [in] Data Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hmac_sha256(const uint8_t* key, size_t key_len, const uint8_t* data, size_t data_len, uint8_t out[32]) {
    if (!key||!data||!out) return -1;
    hmac_compute(key,key_len,data,data_len,out,32,0);
    return 0;
}

/**
 * @brief Perform Hmac Sha512.
 *
 * @param key [in] Key value.
 * @param key_len [in] Key Len value.
 * @param data [in] Data value.
 * @param data_len [in] Data Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hmac_sha512(const uint8_t* key, size_t key_len, const uint8_t* data, size_t data_len, uint8_t out[64]) {
    if (!key||!data||!out) return -1;
    hmac_compute(key,key_len,data,data_len,out,64,1);
    return 0;
}
