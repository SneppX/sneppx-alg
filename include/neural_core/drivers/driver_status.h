#ifndef SNEPPX_DRIVER_STATUS_H
#define SNEPPX_DRIVER_STATUS_H

#include <stddef.h>

#ifdef __cplusplus
/*
 * SNEPPX - Driver Status
 *
 * WHAT
 *   Driver Status.
 *
 * CONCEPT
 *   Provides the Driver Status.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif

/* Unified status codes returned by accelerator driver entry points.
 * Stub backends (Vulkan, Metal, NPU, OneAPI, Qualcomm, Intel, AMD, SGX,
 * shim, TPU) return SNEPPX_DRIVER_UNSUPPORTED so callers can fall back
 * to CUDA/ROCm or the CPU reference path instead of silently doing nothing. */
typedef enum {
    SNEPPX_DRIVER_OK = 0,
    SNEPPX_DRIVER_UNSUPPORTED = -1,
    SNEPPX_DRIVER_ERROR = -2
} sneppx_driver_status_t;

/* Returns a short, human-readable reason a backend is unavailable. */
const char* sneppx_driver_status_string(sneppx_driver_status_t status);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_DRIVER_STATUS_H */
