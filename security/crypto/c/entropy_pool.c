#include "entropy_pool.h"
#include "siphash.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#endif

/*
 * SNEPPX - Entropy Pool
 *
 * WHAT
 *   Legacy entropy pool retained for compatibility/testing.
 *
 * CONCEPT
 *   Uses a custom SipHash-based mixer; this is not a standardized DRBG or a
 *   production cryptographic random generator.
 *
 * ROLE
 *   Do not use this output for keys, nonces, tokens, or other security
 *   decisions. Production randomness must come from SNEPPX_random_bytes().
 *
 */

static uint64_t get_tsc(void)
{
#ifdef _WIN32
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (uint64_t) c.QuadPart;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t) tv.tv_sec * 1000000ULL + tv.tv_usec;
#endif
}

static void store64_le(uint8_t* out, uint64_t value)
{
    for (unsigned int i = 0; i < 8; ++i)
        out[i] = (uint8_t) (value >> (8U * i));
}

/*
 * One-way absorption: derive a SipHash output from the current pool and the
 * incoming bytes, then mix that output into the pool. Using the previous pool
 * as the key means an observer cannot recover earlier state from the new pool.
 */
static void entropy_absorb(SNEPPXEntropyPool* ep, const uint8_t* data, size_t len)
{
    uint8_t key[SNEPPX_SIPHASH_KEY_SIZE];
    uint8_t block[8 + 16];
    size_t off = 0;
    uint64_t counter = 0;

    while (off < len)
    {
        size_t take = len - off;
        if (take > 16)
            take = 16;
        memcpy(key, ep->pool, sizeof(key));
        store64_le(block, counter++);
        memcpy(block + 8, data + off, take);
        if (take < 16)
            memset(block + 8 + take, 0, 16 - take);
        uint64_t h = SNEPPX_siphash(key, block, sizeof(block));
        for (unsigned int i = 0; i < 8; ++i)
            ep->pool[(ep->pool_index + (int) i) % SNEPPX_ENTROPY_POOL_SIZE] ^= (uint8_t) (h >> (8U * i));
        ep->pool_index = (ep->pool_index + 8) % SNEPPX_ENTROPY_POOL_SIZE;
        off += take;
    }
}

/* Counter-mode SipHash stream. No timing source is used after seeding. */
static void entropy_extract(const SNEPPXEntropyPool* ep, uint8_t* out, size_t out_len)
{
    uint8_t key[SNEPPX_SIPHASH_KEY_SIZE];
    uint8_t msg[8 + SNEPPX_ENTROPY_POOL_SIZE];
    uint8_t block[8];
    uint64_t counter = 0;

    memcpy(key, ep->pool, sizeof(key));
    memcpy(msg + 8, ep->pool, SNEPPX_ENTROPY_POOL_SIZE);
    while (out_len > 0)
    {
        store64_le(msg, counter++);
        uint64_t h = SNEPPX_siphash(key, msg, sizeof(msg));
        store64_le(block, h);
        size_t take = out_len < 8 ? out_len : 8;
        memcpy(out, block, take);
        out += take;
        out_len -= take;
        for (unsigned int i = 0; i < 8; ++i)
            key[i] ^= block[i];
        for (size_t i = 8; i < sizeof(key); ++i)
            key[i] ^= key[i - 8];
    }
}

/**
 * @brief Initialize Entropy Pool.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_entropy_pool_init(SNEPPXEntropyPool* ep)
{
    if (!ep)
        return -1;
    memset(ep, 0, sizeof(*ep));
    for (int i = 0; i < SNEPPX_ENTROPY_SOURCES; i++)
        ep->source_available[i] = 1;
    return 0;
}

/**
 * @brief Add Entropy Pool.
 *
 * @param ep [out] Ep value.
 * @param src [in] Src value.
 * @param data [in] Data value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_entropy_pool_add(SNEPPXEntropyPool* ep, SNEPPXEntropySource src, const uint8_t* data, size_t len)
{
    if (!ep || !data || len == 0)
        return -1;
    if ((int) src < 0 || src >= SNEPPX_ENTROPY_SOURCES)
        return -1;
    entropy_absorb(ep, data, len);

    ep->source_available[src] = 1;
    ep->last_collection[src] = (uint64_t) time(NULL);
    ep->entropy_estimate += 2;
    if (ep->entropy_estimate > SNEPPX_ENTROPY_POOL_SIZE)
        ep->entropy_estimate = SNEPPX_ENTROPY_POOL_SIZE;
    return 0;
}

/**
 * @brief Perform Entropy Pool Add Rdtsc.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_entropy_pool_add_rdtsc(SNEPPXEntropyPool* ep)
{
    if (!ep)
        return -1;
    uint64_t tsc = get_tsc();
    uint64_t tsc2 = get_tsc();
    uint64_t jitter = tsc2 > tsc ? tsc2 - tsc : tsc - tsc2;
    return SNEPPX_entropy_pool_add(ep, SNEPPX_ENTROPY_SOURCE_RDTSC, (const uint8_t*) &jitter, sizeof(jitter));
}

/**
 * @brief Perform Entropy Pool Add Os.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_entropy_pool_add_os(SNEPPXEntropyPool* ep)
{
    if (!ep)
        return -1;
    uint8_t buf[32];
    int got = 0;
#ifdef _WIN32
    HCRYPTPROV hProv = 0;
    if (CryptAcquireContext(&hProv, NULL, NULL, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT))
    {
        if (CryptGenRandom(hProv, sizeof(buf), buf))
            got = 1;
        CryptReleaseContext(hProv, 0);
    }
#else
    FILE* f = fopen("/dev/urandom", "rb");
    if (f)
    {
        got = (fread(buf, 1, sizeof(buf), f) == sizeof(buf));
        fclose(f);
    }
#endif
    /*
     * SNEPPX - Entropy Pool
     *
     * WHAT
     *   Entropy Pool.
     *
     * CONCEPT
     *   Collects entropy from OS sources into a pool for the CSPRNG.
     *
     * ROLE
     *   Foundation for the random number subsystem feeding the DRBG.
     *
     * REFERENCES
     *   None (internal utility).
     */

    if (!got)
    {
        ep->source_available[SNEPPX_ENTROPY_SOURCE_OS] = 0;
        return -1;
    }
    return SNEPPX_entropy_pool_add(ep, SNEPPX_ENTROPY_SOURCE_OS, buf, sizeof(buf));
}

/**
 * @brief Perform Entropy Pool Collect.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_entropy_pool_collect(SNEPPXEntropyPool* ep)
{
    if (!ep)
        return -1;
    if (SNEPPX_entropy_pool_add_rdtsc(ep) != 0)
        return -1;
    return SNEPPX_entropy_pool_add_os(ep);
}

/**
 * @brief Get Entropy Pool.
 *
 * @param ep [out] Ep value.
 * @param out [out] Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_entropy_pool_get(SNEPPXEntropyPool* ep, uint8_t* out, size_t out_len)
{
    if (!ep || !out)
        return -1;
    if (out_len == 0)
        return 0;
    if (out_len > INT_MAX)
        return -1;
    if (ep->entropy_estimate < SNEPPX_ENTROPY_THRESHOLD && SNEPPX_entropy_pool_collect(ep) != 0)
        return -1;

    entropy_extract(ep, out, out_len);
    /* One-way rekey after generation: old state cannot be recovered. */
    entropy_absorb(ep, out, out_len);

    ep->entropy_estimate -= (int) out_len;
    if (ep->entropy_estimate < 0)
        ep->entropy_estimate = 0;
    return 0;
}

/**
 * @brief Perform Entropy Pool Estimate.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_entropy_pool_estimate(const SNEPPXEntropyPool* ep)
{
    return ep ? ep->entropy_estimate : 0;
}

/**
 * @brief Perform Entropy Pool Stir.
 */
void SNEPPX_entropy_pool_stir(SNEPPXEntropyPool* ep)
{
    if (!ep)
        return;
    for (int i = 0; i < 4; i++)
    {
        uint8_t hash[32];
        if (SNEPPX_entropy_pool_get(ep, hash, sizeof(hash)) != 0)
            return;
        for (int j = 0; j < 32 && (i * 32 + j) < SNEPPX_ENTROPY_POOL_SIZE; j++)
            ep->pool[i * 32 + j] = hash[j];
    }
}
