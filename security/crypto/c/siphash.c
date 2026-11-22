#include "siphash.h"
#include <string.h>
/*
 * SNEPPX - SipHash 2-4 Pseudorandom Function
 *
 * WHAT
 *   SipHash 2-4 Pseudorandom Function.
 *
 * CONCEPT
 *   Fast, cryptographically secure PRF designed by Aumasson and Bernstein for short inputs.
 *
 * ROLE
 *   Used for hash-table key hashing and as a building block in key derivation.
 *
 * REFERENCES
 *   Aumasson & Bernstein 2012 (SipHash: a fast short-input PRF).
 */

static uint64_t load64_le(const uint8_t* data)
{
    uint64_t value = 0;
    for (unsigned int i = 0; i < 8; ++i)
        value |= (uint64_t) data[i] << (8U * i);
    return value;
}

static uint64_t rotl64(uint64_t x, int b)
{
    return (x << b) | (x >> (64 - b));
}
static void sipround(SNEPPXSipHash* sh)
{
    sh->v0 += sh->v1;
    sh->v1 = rotl64(sh->v1, 13);
    sh->v1 ^= sh->v0;
    sh->v0 = rotl64(sh->v0, 32);
    sh->v2 += sh->v3;
    sh->v3 = rotl64(sh->v3, 16);
    sh->v3 ^= sh->v2;
    sh->v0 += sh->v3;
    sh->v3 = rotl64(sh->v3, 21);
    sh->v3 ^= sh->v0;
    sh->v2 += sh->v1;
    sh->v1 = rotl64(sh->v1, 17);
    sh->v1 ^= sh->v2;
    sh->v2 = rotl64(sh->v2, 32);
}

/**
 * @brief Initialize Siphash.
 *
 * @param sh [out] Sh value.
 */
void SNEPPX_siphash_init(SNEPPXSipHash* sh, const uint8_t key[16])
{
    if (!sh || !key)
        return;
    memset(sh, 0, sizeof(*sh));
    sh->k0 = load64_le(key);
    sh->k1 = load64_le(key + 8);
    sh->v0 = sh->k0 ^ 0x736F6D6570736575ULL;
    sh->v1 = sh->k1 ^ 0x646F72616E646F6DULL;
    sh->v2 = sh->k0 ^ 0x6C7967656E657261ULL;
    sh->v3 = sh->k1 ^ 0x7465646279746573ULL;
    sh->c_rounds = 2;
    sh->d_rounds = 4;
}

/**
 * @brief Update Siphash.
 *
 * @param sh [out] Sh value.
 * @param data [in] Data value.
 */
void SNEPPX_siphash_update(SNEPPXSipHash* sh, const uint8_t* data, size_t len)
{
    if (!sh || (!data && len != 0))
        return;
    uint64_t m;
    size_t off = 0;
    while (len - off >= 8)
    {
        m = load64_le(data + off);
        sh->v3 ^= m;
        for (int i = 0; i < sh->c_rounds; i++)
            sipround(sh);
        sh->v0 ^= m;
        off += 8;
    }
    m = (uint64_t) len << 56;
    for (size_t i = len; i > off; i--)
        m |= (uint64_t) data[i - 1] << ((i - 1 - off) * 8);
    sh->v3 ^= m;
    for (int i = 0; i < sh->c_rounds; i++)
        sipround(sh);
    sh->v0 ^= m;
}

/**
 * @brief Perform Siphash Finalize.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_siphash_finalize(SNEPPXSipHash* sh)
{
    if (!sh)
        return 0;
    sh->v2 ^= 0xFF;
    for (int i = 0; i < sh->d_rounds; i++)
        sipround(sh);
    return sh->v0 ^ sh->v1 ^ sh->v2 ^ sh->v3;
}

/**
 * @brief Hash Sip.
 *
 * @param data [in] Data value.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_siphash(const uint8_t key[16], const uint8_t* data, size_t len)
{
    SNEPPXSipHash sh;
    SNEPPX_siphash_init(&sh, key);
    SNEPPX_siphash_update(&sh, data, len);
    return SNEPPX_siphash_finalize(&sh);
}

/**
 * @brief Initialize Siphash 24.
 *
 * @param sh [out] Sh value.
 */
void SNEPPX_siphash_24_init(SNEPPXSipHash* sh, const uint8_t key[16])
{
    if (!sh || !key)
        return;
    SNEPPX_siphash_init(sh, key);
    sh->c_rounds = 2;
    sh->d_rounds = 4;
}

/**
 * @brief Perform Siphash 24.
 *
 * @param data [in] Data value.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_siphash_24(const uint8_t key[16], const uint8_t* data, size_t len)
{
    return SNEPPX_siphash(key, data, len);
}
