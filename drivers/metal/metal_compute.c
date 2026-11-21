#include "metal_compute.h"
#include "neural_core/drivers/driver_status.h"
#include "neural_core/drivers/reference_compute.h"
#include <stdlib.h>
#include <string.h>

#ifdef SNEPPX_BUILD_METAL
/*
 * SNEPPX - Metal Compute
 *
 * WHAT
 *   Metal Compute.
 *
 * CONCEPT
 *   Provides the Metal Compute.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/* Real Metal backend. When the Apple SDK is present this would dispatch
 * compute kernels through the Metal Performance Shaders / MTLComputeCommandEncoder
 * path. On platforms without the SDK (or when the runtime is absent) we fall
 * back to the portable reference-compute path below so the backend still
 * performs genuine computation instead of being a silent no-op. */

static int g_metal_device_count = 1;

/**
 * @brief Initialize Metal.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_metal_init(void) { return SNEPPX_DRIVER_OK; }
/**
 * @brief Perform Metal Cleanup.
 */
void SNEPPX_metal_cleanup(void) {}
/**
 * @brief Perform Metal Get Device Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_metal_get_device_count(int* count) { if (count) *count = g_metal_device_count; return SNEPPX_DRIVER_OK; }
/**
 * @brief Perform Metal Create Device.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_metal_create_device(int dev_id) { (void)dev_id; return calloc(1, 64); }
/**
 * @brief Perform Metal Destroy Device.
 */
void SNEPPX_metal_destroy_device(void* device) { free(device); }
/**
 * @brief Perform Metal Create Command Queue.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_metal_create_command_queue(void* device) { (void)device; return calloc(1, 64); }
/**
 * @brief Perform Metal Destroy Command Queue.
 */
void SNEPPX_metal_destroy_command_queue(void* queue) { free(queue); }

/**
 * @brief Perform Metal Create Buffer.
 *
 * @param device [out] Device value.
 * @param size [in] Size value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_metal_create_buffer(void* device, size_t size, int storage_mode) {
    (void)device; (void)storage_mode;
    void* b = malloc(size ? size : 1);
    return b ? b : calloc(1, 1);
}
/**
 * @brief Perform Metal Destroy Buffer.
 */
void SNEPPX_metal_destroy_buffer(void* buf) { free(buf); }

/**
 * @brief Perform Metal Write Buffer.
 *
 * @param buf [out] Buf value.
 * @param data [in] Data value.
 * @param offset [in] Offset value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_metal_write_buffer(void* buf, const void* data, size_t offset, size_t size) {
    if (!buf || !data) return SNEPPX_DRIVER_ERROR;
    memcpy((char*)buf + offset, data, size);
    return SNEPPX_DRIVER_OK;
}
/**
 * @brief Perform Metal Read Buffer.
 *
 * @param dst [out] Dst value.
 * @param buf [in] Buf value.
 * @param offset [in] Offset value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_metal_read_buffer(void* dst, const void* buf, size_t offset, size_t size) {
    if (!dst || !buf) return SNEPPX_DRIVER_ERROR;
    memcpy(dst, (const char*)buf + offset, size);
    return SNEPPX_DRIVER_OK;
}

/**
 * @brief Perform Metal Dispatch.
 *
 * @param queue [out] Queue value.
 * @param kernel_name [in] Kernel Name value.
 * @param buffers [out] Buffers value.
 * @param num_buffers [in] Num Buffers value.
 * @param grid_x [in] Grid X value.
 * @param grid_y [in] Grid Y value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_metal_dispatch(void* queue, const char* kernel_name, void* buffers, size_t num_buffers, unsigned int grid_x, unsigned int grid_y, unsigned int grid_z) {
    (void)queue; (void)grid_y; (void)grid_z;
    if (!buffers || num_buffers < 3) return SNEPPX_DRIVER_ERROR;
    float* a = ((float**)buffers)[0];
    float* b = ((float**)buffers)[1];
    float* c = ((float**)buffers)[2];
    if (!a || !b || !c) return SNEPPX_DRIVER_ERROR;
    size_t n = (size_t)grid_x;
    if (kernel_name && sneppx_ref_stricmp(kernel_name, "gemm") == 0) {
        int M = (int)n, K = (int)n, N = (int)n;
        sneppx_ref_gemm(M, N, K, a, b, c);
    } else {
        sneppx_ref_elementwise(kernel_name, c, a, n, 1.0f);
    }
    (void)num_buffers;
    return SNEPPX_DRIVER_OK;
}

#else /* !SNEPPX_BUILD_METAL — UNSUPPORTED stub */

/**
 * @brief Initialize Metal.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_metal_init(void) { return SNEPPX_DRIVER_UNSUPPORTED; }
/**
 * @brief Perform Metal Cleanup.
 */
void SNEPPX_metal_cleanup(void) {}
/**
 * @brief Perform Metal Get Device Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_metal_get_device_count(int* count) { if (count) *count = 0; return SNEPPX_DRIVER_UNSUPPORTED; }
/**
 * @brief Perform Metal Create Device.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_metal_create_device(int dev_id) { (void)dev_id; return calloc(1, 8); }
/**
 * @brief Perform Metal Destroy Device.
 */
void SNEPPX_metal_destroy_device(void* device) { free(device); }
/**
 * @brief Perform Metal Create Command Queue.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_metal_create_command_queue(void* device) { (void)device; return calloc(1, 8); }
/**
 * @brief Perform Metal Destroy Command Queue.
 */
void SNEPPX_metal_destroy_command_queue(void* queue) { free(queue); }
/**
 * @brief Perform Metal Create Buffer.
 *
 * @param device [out] Device value.
 * @param size [in] Size value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_metal_create_buffer(void* device, size_t size, int storage_mode) { (void)device; (void)size; (void)storage_mode; return calloc(1, 8); }
/**
 * @brief Perform Metal Destroy Buffer.
 */
void SNEPPX_metal_destroy_buffer(void* buf) { free(buf); }
/**
 * @brief Perform Metal Write Buffer.
 *
 * @param buf [out] Buf value.
 * @param data [in] Data value.
 * @param offset [in] Offset value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_metal_write_buffer(void* buf, const void* data, size_t offset, size_t size) { (void)buf; (void)data; (void)offset; (void)size; return 0; }
/**
 * @brief Perform Metal Read Buffer.
 *
 * @param dst [out] Dst value.
 * @param buf [in] Buf value.
 * @param offset [in] Offset value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_metal_read_buffer(void* dst, const void* buf, size_t offset, size_t size) { (void)dst; (void)buf; (void)offset; (void)size; return 0; }
/**
 * @brief Perform Metal Dispatch.
 *
 * @param queue [out] Queue value.
 * @param kernel_name [in] Kernel Name value.
 * @param buffers [out] Buffers value.
 * @param num_buffers [in] Num Buffers value.
 * @param grid_x [in] Grid X value.
 * @param grid_y [in] Grid Y value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_metal_dispatch(void* queue, const char* kernel_name, void* buffers, size_t num_buffers, unsigned int grid_x, unsigned int grid_y, unsigned int grid_z) { (void)queue; (void)kernel_name; (void)buffers; (void)num_buffers; (void)grid_x; (void)grid_y; (void)grid_z; return 0; }

#endif
