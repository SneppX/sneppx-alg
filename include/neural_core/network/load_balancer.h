#ifndef SNEPPX_LOAD_BALANCER_H
#define SNEPPX_LOAD_BALANCER_H
#include <stddef.h>
#ifdef __cplusplus
/*
 * SNEPPX - Load Balancer
 *
 * WHAT
 *   Load Balancer.
 *
 * CONCEPT
 *   Provides the Load Balancer.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif
typedef struct { void* impl; } SNEPPXLoadBalancer;
typedef enum { SNEPPX_LB_ROUND_ROBIN, SNEPPX_LB_LEAST_CONN, SNEPPX_LB_RANDOM, SNEPPX_LB_WEIGHTED } SNEPPXLBStrategy;
/**
 * @brief Create Lb.
 *
 * @param strategy [in] Strategy value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXLoadBalancer* SNEPPX_lb_create(SNEPPXLBStrategy strategy);
/**
 * @brief Destroy Lb.
 *
 * @param lb [out] Lb value.
 */
void SNEPPX_lb_destroy(SNEPPXLoadBalancer* lb);
/**
 * @brief Perform Lb Add Backend.
 *
 * @param lb [out] Lb value.
 * @param host [in] Host value.
 * @param port [in] Port value.
 * @param weight [in] Weight value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_lb_add_backend(SNEPPXLoadBalancer* lb, const char* host, int port, int weight);
/**
 * @brief Perform Lb Remove Backend.
 *
 * @param lb [out] Lb value.
 * @param host [in] Host value.
 * @param port [in] Port value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_lb_remove_backend(SNEPPXLoadBalancer* lb, const char* host, int port);
/**
 * @brief Perform Lb Next Backend.
 *
 * @param lb [out] Lb value.
 * @param host [out] Host value.
 * @param host_max [in] Host Max value.
 * @param port [out] Port value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_lb_next_backend(SNEPPXLoadBalancer* lb, char* host, size_t host_max, int* port);
/**
 * @brief Perform Lb Mark Unhealthy.
 *
 * @param lb [out] Lb value.
 * @param host [in] Host value.
 * @param port [in] Port value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_lb_mark_unhealthy(SNEPPXLoadBalancer* lb, const char* host, int port);
/**
 * @brief Perform Lb Mark Healthy.
 *
 * @param lb [out] Lb value.
 * @param host [in] Host value.
 * @param port [in] Port value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_lb_mark_healthy(SNEPPXLoadBalancer* lb, const char* host, int port);
#ifdef __cplusplus
}
#endif
#endif
