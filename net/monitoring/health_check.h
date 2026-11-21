#ifndef SNEPPX_HEALTH_CHECK_H
#define SNEPPX_HEALTH_CHECK_H
#include <stddef.h>
#ifdef __cplusplus
/*
 * SNEPPX - Health Check
 *
 * WHAT
 *   Health Check.
 *
 * CONCEPT
 *   Provides the Health Check.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif
/**
 * @brief Create Health Check.
 *
 * @param interval_ms [in] Interval Ms value.
 * @param timeout_ms [in] Timeout Ms value.
 * @param unhealthy_threshold [in] Unhealthy Threshold value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_health_check_create(int interval_ms, int timeout_ms, int unhealthy_threshold);
/**
 * @brief Destroy Health Check.
 *
 * @param hc [out] Hc value.
 */
void SNEPPX_health_check_destroy(void* hc);
int SNEPPX_health_check_add_endpoint(void* hc, const char* name, const char* url, int (*check_fn)(void* ctx), void* ctx);
/**
 * @brief Start Health Check.
 *
 * @param hc [out] Hc value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_health_check_start(void* hc);
/**
 * @brief Stop Health Check.
 *
 * @param hc [out] Hc value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_health_check_stop(void* hc);
/**
 * @brief Perform Health Check Get Status.
 *
 * @param hc [out] Hc value.
 * @param name [in] Name value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_health_check_get_status(void* hc, const char* name);
/**
 * @brief Perform Health Check Is Healthy.
 *
 * @param hc [out] Hc value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_health_check_is_healthy(void* hc);
/**
 * @brief Perform Health Check Get Stats.
 *
 * @param hc [out] Hc value.
 * @param name [in] Name value.
 * @param total_checks [out] Total Checks value.
 * @param passed_checks [out] Passed Checks value.
 * @param failed_checks [out] Failed Checks value.
 * @param avg_latency [out] Avg Latency value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_health_check_get_stats(void* hc, const char* name, int* total_checks, int* passed_checks, int* failed_checks, double* avg_latency);
int SNEPPX_health_check_set_on_unhealthy(void* hc, void (*cb)(void* hc, const char* name, int status));
#ifdef __cplusplus
}
#endif
#endif
