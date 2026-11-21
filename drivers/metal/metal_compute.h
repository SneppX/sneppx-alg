#ifndef SNEPPX_METAL_COMPUTE_H
#define SNEPPX_METAL_COMPUTE_H
#include <stddef.h>
#ifdef __cplusplus
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


extern "C" {
#endif
/**
 * @brief Initialize Metal.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_metal_init(void);
/**
 * @brief Perform Metal Cleanup.
 */
void SNEPPX_metal_cleanup(void);
/**
 * @brief Perform Metal Get Device Count.
 *
 * @param count [out] Count value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_metal_get_device_count(int* count);
/**
 * @brief Perform Metal Create Device.
 *
 * @param dev_id [in] Dev Id value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_metal_create_device(int dev_id);
/**
 * @brief Perform Metal Destroy Device.
 *
 * @param device [out] Device value.
 */
void  SNEPPX_metal_destroy_device(void* device);
/**
 * @brief Perform Metal Create Command Queue.
 *
 * @param device [out] Device value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_metal_create_command_queue(void* device);
/**
 * @brief Perform Metal Destroy Command Queue.
 *
 * @param queue [out] Queue value.
 */
void  SNEPPX_metal_destroy_command_queue(void* queue);
/**
 * @brief Perform Metal Create Buffer.
 *
 * @param device [out] Device value.
 * @param size [in] Size value.
 * @param storage_mode [in] Storage Mode value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_metal_create_buffer(void* device, size_t size, int storage_mode);
/**
 * @brief Perform Metal Destroy Buffer.
 *
 * @param buf [out] Buf value.
 */
void  SNEPPX_metal_destroy_buffer(void* buf);
/**
 * @brief Perform Metal Write Buffer.
 *
 * @param buf [out] Buf value.
 * @param data [in] Data value.
 * @param offset [in] Offset value.
 * @param size [in] Size value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_metal_write_buffer(void* buf, const void* data, size_t offset, size_t size);
/**
 * @brief Perform Metal Read Buffer.
 *
 * @param dst [out] Dst value.
 * @param buf [in] Buf value.
 * @param offset [in] Offset value.
 * @param size [in] Size value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_metal_read_buffer(void* dst, const void* buf, size_t offset, size_t size);
/**
 * @brief Perform Metal Dispatch.
 *
 * @param queue [out] Queue value.
 * @param kernel_name [in] Kernel Name value.
 * @param buffers [out] Buffers value.
 * @param num_buffers [in] Num Buffers value.
 * @param grid_x [in] Grid X value.
 * @param grid_y [in] Grid Y value.
 * @param grid_z [in] Grid Z value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_metal_dispatch(void* queue, const char* kernel_name, void* buffers, size_t num_buffers, unsigned int grid_x, unsigned int grid_y, unsigned int grid_z);
#ifdef __cplusplus
}
#endif
#endif
