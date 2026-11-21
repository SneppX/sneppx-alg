#ifndef SNEPPX_ENTROPY_POOL_H
#define SNEPPX_ENTROPY_POOL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_ENTROPY_POOL_SIZE 256
#define SNEPPX_ENTROPY_THRESHOLD 128
#define SNEPPX_ENTROPY_SOURCES 8
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



typedef enum {
    SNEPPX_ENTROPY_SOURCE_RDTSC = 0,
    SNEPPX_ENTROPY_SOURCE_OS = 1,
    SNEPPX_ENTROPY_SOURCE_INTERRUPT_JITTER = 2,
    SNEPPX_ENTROPY_SOURCE_NETWORK = 3,
    SNEPPX_ENTROPY_SOURCE_DISK_TIMING = 4,
    SNEPPX_ENTROPY_SOURCE_MOUSE = 5,
    SNEPPX_ENTROPY_SOURCE_KEYBOARD = 6,
    SNEPPX_ENTROPY_SOURCE_MICROPHONE = 7,
} SNEPPXEntropySource;

typedef struct {
    uint8_t pool[SNEPPX_ENTROPY_POOL_SIZE];
    int pool_index;
    int entropy_estimate;
    int source_available[SNEPPX_ENTROPY_SOURCES];
    uint64_t last_collection[SNEPPX_ENTROPY_SOURCES];
} SNEPPXEntropyPool;

/**
 * @brief Initialize Entropy Pool.
 *
 * @param ep [out] Ep value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_entropy_pool_init(SNEPPXEntropyPool* ep);
/**
 * @brief Add Entropy Pool.
 *
 * @param ep [out] Ep value.
 * @param src [in] Src value.
 * @param data [in] Data value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_entropy_pool_add(SNEPPXEntropyPool* ep, SNEPPXEntropySource src, const uint8_t* data, size_t len);
/**
 * @brief Perform Entropy Pool Add Rdtsc.
 *
 * @param ep [out] Ep value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_entropy_pool_add_rdtsc(SNEPPXEntropyPool* ep);
/**
 * @brief Perform Entropy Pool Add Os.
 *
 * @param ep [out] Ep value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_entropy_pool_add_os(SNEPPXEntropyPool* ep);
/**
 * @brief Perform Entropy Pool Collect.
 *
 * @param ep [out] Ep value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_entropy_pool_collect(SNEPPXEntropyPool* ep);
/**
 * @brief Get Entropy Pool.
 *
 * @param ep [out] Ep value.
 * @param out [out] Out value.
 * @param out_len [in] Out Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_entropy_pool_get(SNEPPXEntropyPool* ep, uint8_t* out, size_t out_len);
/**
 * @brief Perform Entropy Pool Estimate.
 *
 * @param ep [in] Ep value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_entropy_pool_estimate(const SNEPPXEntropyPool* ep);
/**
 * @brief Perform Entropy Pool Stir.
 *
 * @param ep [out] Ep value.
 */
void SNEPPX_entropy_pool_stir(SNEPPXEntropyPool* ep);

#ifdef __cplusplus
}
#endif
#endif
