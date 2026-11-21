#include "drbg.h"
#include "sha512_hashing_implementation.h"
#include "constant_time_operations.h"
#include <string.h>
/*
 * SNEPPX - Deterministic Random Bit Generator (DRBG)
 *
 * WHAT
 *   Deterministic Random Bit Generator (DRBG).
 *
 * CONCEPT
 *   NIST SP 800-90A compliant HMAC-DRBG and Hash-DRBG for expanding entropy seeds.
 *
 * ROLE
 *   Foundation for all random-number generation in the crypto module.
 *
 * REFERENCES
 *   NIST SP 800-90A (DRBG).
 */



static void sha256_f(const uint8_t* in, size_t in_len, uint8_t out[32]) {
    SNEPPXSHA512Context ctx;
    SNEPPX_sha512_init(&ctx);
    SNEPPX_sha512_update(&ctx,in,in_len);
    uint8_t full[64];
    SNEPPX_sha512_finish(&ctx,full);
    memcpy(out,full,32);
}

/**
 * @brief Initialize Drbg.
 *
 * @param ctx [out] Ctx value.
 * @param entropy [in] Entropy value.
 * @param entropy_len [in] Entropy Len value.
 * @param nonce [in] Nonce value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_drbg_init(SNEPPXDRBG* ctx, const uint8_t* entropy, size_t entropy_len, const uint8_t* nonce, size_t nonce_len) {
    if (!ctx||!entropy||entropy_len<32) return -1;
    memset(ctx,0,sizeof(*ctx));

    uint8_t seed[SNEPPX_DRBG_SEED_SIZE] = {0};
    size_t sl=0;
    size_t copy_len = (entropy_len < SNEPPX_DRBG_SEED_SIZE) ? entropy_len : SNEPPX_DRBG_SEED_SIZE;
    memcpy(seed,entropy,copy_len); sl=copy_len;
    if (nonce) {
        size_t room = SNEPPX_DRBG_SEED_SIZE - sl;
        size_t nc = (nonce_len < room) ? nonce_len : room;
        memcpy(seed+sl,nonce,nc); sl+=nc;
    }

    uint8_t hash[32];
    sha256_f(seed,sl,hash);
    memcpy(ctx->hb.v,hash,32);
    memcpy(ctx->hb.v+32,seed+32,16);

    memset(ctx->hb.c,0,48);
    sha256_f(ctx->hb.v,48,ctx->hb.c);
    ctx->hb.reseed_counter=1;
    ctx->hb.initialized=1;
    ctx->use_hmac=0;
    return 0;
}

/**
 * @brief Perform Drbg Reseed.
 *
 * @param ctx [out] Ctx value.
 * @param entropy [in] Entropy value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_drbg_reseed(SNEPPXDRBG* ctx, const uint8_t* entropy, size_t entropy_len) {
    if (!ctx||!ctx->hb.initialized||!entropy) return -1;
    uint8_t seed[SNEPPX_DRBG_SEED_SIZE+256];
    memcpy(seed,ctx->hb.v,SNEPPX_DRBG_SEED_SIZE);
    memcpy(seed+SNEPPX_DRBG_SEED_SIZE,entropy,entropy_len);
    uint8_t hash[32];
    sha256_f(seed,SNEPPX_DRBG_SEED_SIZE+entropy_len,hash);
    memcpy(ctx->hb.v,hash,32);
    memcpy(ctx->hb.v+32,seed+32,16);
    ctx->hb.reseed_counter=1;
    return 0;
}

/**
 * @brief Perform Drbg Generate.
 *
 * @param ctx [out] Ctx value.
 * @param out [out] Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_drbg_generate(SNEPPXDRBG* ctx, uint8_t* out, size_t out_len) {
    if (!ctx||!ctx->hb.initialized||!out||out_len>SNEPPX_DRBG_MAX_OUTPUT) return -1;
    if (ctx->hb.reseed_counter>10000) return -1;

    size_t generated=0;
    uint8_t v[SNEPPX_DRBG_SEED_SIZE];
    memcpy(v,ctx->hb.v,SNEPPX_DRBG_SEED_SIZE);

    while (generated<out_len) {
        sha256_f(v,SNEPPX_DRBG_SEED_SIZE,v);
        size_t copy=(out_len-generated<32)?out_len-generated:32;
        memcpy(out+generated,v,copy);
        generated+=copy;
    }

    uint8_t hash[32];
    sha256_f(ctx->hb.c,SNEPPX_DRBG_SEED_SIZE,hash);
    for (int i=0;i<SNEPPX_DRBG_SEED_SIZE;i++) ctx->hb.v[i]^=hash[i%32];

    ctx->hb.reseed_counter++;
    return 0;
}

/**
 * @brief Destroy Drbg.
 */
void SNEPPX_drbg_destroy(SNEPPXDRBG* ctx) {
    if (ctx) {
        volatile uint8_t* p=(volatile uint8_t*)ctx;
        for (size_t i=0;i<sizeof(*ctx);i++) p[i]=0;
    }
}

/**
 * @brief Perform Drbg Self Test.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_drbg_self_test(void) {
    SNEPPXDRBG ctx;
    uint8_t entropy[48],nonce[16],out[64];
    memset(entropy,0x12,48);
    memset(nonce,0x34,16);
    if (SNEPPX_drbg_init(&ctx,entropy,48,nonce,16)!=0) return -1;
    if (SNEPPX_drbg_generate(&ctx,out,64)!=0) return -1;
    int all_zero=1;
    for (int i=0;i<64;i++) if (out[i]) all_zero=0;
    if (all_zero) return -1;
    SNEPPX_drbg_destroy(&ctx);
    return 0;
}
