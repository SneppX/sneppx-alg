#include "hmac.h"
#include "constant_time_operations.h"
#include "sha256.h"
#include "sha512_hashing_implementation.h"
#include <stdlib.h>
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

/*
 * HMAC-SHA256 uses SHA-256 for both hash passes. The SHA-512 context used by
 * the generic path writes a 64-byte digest, which overflowed the 32-byte
 * output buffer expected by SNEPPX_hmac_sha256 callers.
 */
static int hmac_sha256_compute(const uint8_t* key, size_t key_len, const uint8_t* data, size_t data_len,
                               uint8_t out[32])
{
    uint8_t k[64], ipad[64], opad[64], inner_hash[32], outer[96];
    if (key_len > sizeof(k))
    {
        SNEPPX_sha256(k, key, key_len);
        memset(k + 32, 0, sizeof(k) - 32);
    }
    else
    {
        memcpy(k, key, key_len);
        memset(k + key_len, 0, sizeof(k) - key_len);
    }
    for (size_t i = 0; i < sizeof(k); i++)
    {
        ipad[i] = (uint8_t) (k[i] ^ 0x36);
        opad[i] = (uint8_t) (k[i] ^ 0x5C);
    }
    memset(k, 0, sizeof(k));

    if (data_len > SIZE_MAX - sizeof(ipad))
        return -1;
    uint8_t* inner = (uint8_t*) malloc(sizeof(ipad) + data_len);
    if (!inner)
        return -1;
    memcpy(inner, ipad, sizeof(ipad));
    if (data_len > 0)
        memcpy(inner + sizeof(ipad), data, data_len);
    SNEPPX_sha256(inner_hash, inner, sizeof(ipad) + data_len);
    memset(inner, 0, sizeof(ipad) + data_len);
    free(inner);

    memcpy(outer, opad, sizeof(opad));
    memcpy(outer + sizeof(opad), inner_hash, sizeof(inner_hash));
    SNEPPX_sha256(out, outer, sizeof(outer));
    memset(outer, 0, sizeof(outer));
    memset(inner_hash, 0, sizeof(inner_hash));
    memset(ipad, 0, sizeof(ipad));
    memset(opad, 0, sizeof(opad));
    return 0;
}

static int hmac_compute(const uint8_t* key, size_t key_len, const uint8_t* data, size_t data_len, uint8_t* out,
                        size_t out_len, int use_sha512)
{
    if (!use_sha512)
    {
        uint8_t digest[32];
        if (hmac_sha256_compute(key, key_len, data, data_len, digest) != 0)
            return -1;
        size_t copy_len = out_len < sizeof(digest) ? out_len : sizeof(digest);
        memcpy(out, digest, copy_len);
        memset(digest, 0, sizeof(digest));
        return 0;
    }

    (void) out_len;
    uint8_t k[128], ipad[128], opad[128];
    size_t block_size = 128;
    memset(k, 0, block_size);
    if (key_len > block_size)
    {
        SNEPPXSHA512Context ctx;
        SNEPPX_sha512_init(&ctx);
        SNEPPX_sha512_update(&ctx, key, key_len);
        uint8_t h[64];
        SNEPPX_sha512_finish(&ctx, h);
        memcpy(k, h, 64);
    }
    else
        memcpy(k, key, key_len);
    for (size_t i = 0; i < block_size; i++)
    {
        ipad[i] = k[i] ^ 0x36;
        opad[i] = k[i] ^ 0x5C;
    }

    uint8_t inner_hash[64];
    {
        SNEPPXSHA512Context ctx;
        SNEPPX_sha512_init(&ctx);
        SNEPPX_sha512_update(&ctx, ipad, block_size);
        SNEPPX_sha512_update(&ctx, data, data_len);
        SNEPPX_sha512_finish(&ctx, inner_hash);
    }
    {
        SNEPPXSHA512Context ctx;
        SNEPPX_sha512_init(&ctx);
        SNEPPX_sha512_update(&ctx, opad, block_size);
        SNEPPX_sha512_update(&ctx, inner_hash, sizeof(inner_hash));
        SNEPPX_sha512_finish(&ctx, out);
    }
    memset(k, 0, sizeof(k));
    memset(ipad, 0, sizeof(ipad));
    memset(opad, 0, sizeof(opad));
    return 0;
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
int SNEPPX_hmac_init(SNEPPXHMAC* ctx, const uint8_t* key, size_t key_len, int hash_type)
{
    if (!ctx || !key)
        return -1;
    ctx->key_len = key_len < SNEPPX_HMAC_MAX_KEY ? key_len : SNEPPX_HMAC_MAX_KEY;
    memcpy(ctx->key, key, ctx->key_len);
    ctx->hash_type = hash_type;
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
int SNEPPX_hmac_compute(SNEPPXHMAC* ctx, const uint8_t* data, size_t data_len, uint8_t* out, size_t* out_len)
{
    if (!ctx || !data || !out || !out_len)
        return -1;
    size_t hash_size = (ctx->hash_type == 0) ? 32 : 64;
    if (*out_len < hash_size)
        return -1;
    if (hmac_compute(ctx->key, ctx->key_len, data, data_len, out, hash_size, ctx->hash_type != 0) != 0)
        return -1;
    *out_len = hash_size;
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
int SNEPPX_hmac_sha256(const uint8_t* key, size_t key_len, const uint8_t* data, size_t data_len, uint8_t out[32])
{
    if (!key || !data || !out)
        return -1;
    return hmac_compute(key, key_len, data, data_len, out, 32, 0);
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
int SNEPPX_hmac_sha512(const uint8_t* key, size_t key_len, const uint8_t* data, size_t data_len, uint8_t out[64])
{
    if (!key || !data || !out)
        return -1;
    return hmac_compute(key, key_len, data, data_len, out, 64, 1);
}
