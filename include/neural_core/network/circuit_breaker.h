#ifndef SNEPPX_CIRCUIT_BREAKER_H
#define SNEPPX_CIRCUIT_BREAKER_H
#include <stddef.h>
#ifdef __cplusplus
/*
 * SNEPPX - Circuit Breaker
 *
 * WHAT
 *   Circuit Breaker.
 *
 * CONCEPT
 *   Provides the Circuit Breaker.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif
typedef struct { void* impl; } SNEPPXCircuitBreaker;
typedef enum { SNEPPX_CB_CLOSED, SNEPPX_CB_OPEN, SNEPPX_CB_HALF_OPEN } SNEPPXCBState;
/**
 * @brief Create Cb.
 *
 * @param name [in] Name value.
 * @param failure_threshold [in] Failure Threshold value.
 * @param recovery_timeout_ms [in] Recovery Timeout Ms value.
 * @param half_open_max_requests [in] Half Open Max Requests value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXCircuitBreaker* SNEPPX_cb_create(const char* name, int failure_threshold, int recovery_timeout_ms, int half_open_max_requests);
/**
 * @brief Destroy Cb.
 *
 * @param cb [out] Cb value.
 */
void SNEPPX_cb_destroy(SNEPPXCircuitBreaker* cb);
int SNEPPX_cb_call(SNEPPXCircuitBreaker* cb, int (*fn)(void*), void* arg);
/**
 * @brief Perform Cb State.
 *
 * @param cb [out] Cb value.
 *
 * @return The result value, or 0 on error.
 */
SNEPPXCBState SNEPPX_cb_state(SNEPPXCircuitBreaker* cb);
/**
 * @brief Reset Cb.
 *
 * @param cb [out] Cb value.
 */
void SNEPPX_cb_reset(SNEPPXCircuitBreaker* cb);
/**
 * @brief Perform Cb Failure Count.
 *
 * @param cb [out] Cb value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_cb_failure_count(SNEPPXCircuitBreaker* cb);
#ifdef __cplusplus
}
#endif
#endif
