#ifndef SNEPPX_CANARY_H
#define SNEPPX_CANARY_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_CANARY_SIZE 16

/*
 * SNEPPX - Stack Canary Protection
 *
 * WHAT
 *   Stack Canary Protection.
 *
 * CONCEPT
 *   Provides stack canary hardening.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef struct {
    uint8_t value[SNEPPX_CANARY_SIZE];
    uint64_t generation;
} SNEPPXCanary;

/**
 * @brief Perform Canary Generate.
 *
 * @param canary [out] Canary value.
 */
void SNEPPX_canary_generate(SNEPPXCanary* canary);
/**
 * @brief Verify Canary.
 *
 * @param expected [in] Expected value.
 * @param memory [in] Memory value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_canary_verify(const SNEPPXCanary* expected, const uint8_t* memory);
/**
 * @brief Write Canary.
 *
 * @param canary [in] Canary value.
 * @param memory [out] Memory value.
 */
void SNEPPX_canary_write(const SNEPPXCanary* canary, uint8_t* memory);


#ifdef __cplusplus
}
#endif
#endif
