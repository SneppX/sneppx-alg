#ifndef SNEPPX_VULKAN_COMPUTE_H
#define SNEPPX_VULKAN_COMPUTE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
/*
 * SNEPPX - Vulkan Compute
 *
 * WHAT
 *   Vulkan Compute.
 *
 * CONCEPT
 *   Provides the Vulkan Compute.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif

/* Vulkan compute backend. When SNEPPX_BUILD_VULKAN is defined the backend
 * performs genuine computation through the portable reference-compute kernels
 * (see neural_core/drivers/reference_compute.h) so it is not a silent no-op.
 * Without the flag the entry points report SNEPPX_DRIVER_UNSUPPORTED. */

/**
 * @brief Initialize Vulkan.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_vulkan_init(void);
/**
 * @brief Perform Vulkan Cleanup.
 */
void SNEPPX_vulkan_cleanup(void);
/**
 * @brief Perform Vulkan Get Device Count.
 *
 * @param count [out] Count value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_vulkan_get_device_count(int* count);
/**
 * @brief Perform Vulkan Get Device Props.
 *
 * @param dev_id [in] Dev Id value.
 * @param name [out] Name value.
 * @param name_max [in] Name Max value.
 * @param global_mem [out] Global Mem value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_vulkan_get_device_props(int dev_id, char* name, size_t name_max, size_t* global_mem);

/**
 * @brief Perform Vulkan Create Compute Pipeline.
 *
 * @param shader_path [in] Shader Path value.
 * @param entry_point [in] Entry Point value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_vulkan_create_compute_pipeline(const char* shader_path, const char* entry_point);
/**
 * @brief Perform Vulkan Destroy Compute Pipeline.
 *
 * @param pipeline [out] Pipeline value.
 */
void  SNEPPX_vulkan_destroy_compute_pipeline(void* pipeline);

/**
 * @brief Perform Vulkan Create Buffer.
 *
 * @param buf [out] Buf value.
 * @param size [in] Size value.
 * @param usage [in] Usage value.
 * @param pipeline [out] Pipeline value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_vulkan_create_buffer(void** buf, size_t size, int usage, void* pipeline);
/**
 * @brief Perform Vulkan Destroy Buffer.
 *
 * @param buf [out] Buf value.
 * @param pipeline [out] Pipeline value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_vulkan_destroy_buffer(void* buf, void* pipeline);
/**
 * @brief Perform Vulkan Write Buffer.
 *
 * @param buf [out] Buf value.
 * @param data [in] Data value.
 * @param offset [in] Offset value.
 * @param size [in] Size value.
 * @param pipeline [out] Pipeline value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_vulkan_write_buffer(void* buf, const void* data, size_t offset, size_t size, void* pipeline);
/**
 * @brief Perform Vulkan Read Buffer.
 *
 * @param dst [out] Dst value.
 * @param buf [in] Buf value.
 * @param offset [in] Offset value.
 * @param size [in] Size value.
 * @param pipeline [out] Pipeline value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_vulkan_read_buffer(void* dst, const void* buf, size_t offset, size_t size, void* pipeline);

/* Dispatch a compute kernel over `buffers` (float* array of `num_buffers`).
 * `entry_point` selects the operation ("gemm", "relu", "sigmoid", "tanh", ...).
 * `group_x` is the problem dimension (e.g. matrix size for gemm). */
/**
 * @brief Perform Vulkan Dispatch.
 *
 * @param pipeline [out] Pipeline value.
 * @param buffers [out] Buffers value.
 * @param num_buffers [in] Num Buffers value.
 * @param group_x [in] Group X value.
 * @param group_y [in] Group Y value.
 * @param group_z [in] Group Z value.
 * @param entry_point [in] Entry Point value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_vulkan_dispatch(void* pipeline, void** buffers, size_t num_buffers,
                           unsigned int group_x, unsigned int group_y, unsigned int group_z,
                           const char* entry_point);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_VULKAN_COMPUTE_H */
