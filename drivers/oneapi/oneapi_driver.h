#ifndef SNEPPX_ONEAPI_DRIVER_H
#define SNEPPX_ONEAPI_DRIVER_H
#include <stddef.h>
#ifdef __cplusplus
/*
 * SNEPPX - Oneapi Driver
 *
 * WHAT
 *   Oneapi Driver.
 *
 * CONCEPT
 *   Provides the Oneapi Driver.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif
/**
 * @brief Perform Oneapi Register.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_oneapi_register(void);
/**
 * @brief Perform Oneapi Get Device Count.
 *
 * @param count [out] Count value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_oneapi_get_device_count(int* count);
/**
 * @brief Perform Oneapi Get Device Props.
 *
 * @param dev_id [in] Dev Id value.
 * @param name [out] Name value.
 * @param name_max [in] Name Max value.
 * @param global_mem [out] Global Mem value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_oneapi_get_device_props(int dev_id, char* name, size_t name_max, size_t* global_mem);
/**
 * @brief Perform Oneapi Create Queue.
 *
 * @param dev_id [in] Dev Id value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_oneapi_create_queue(int dev_id);
/**
 * @brief Perform Oneapi Destroy Queue.
 *
 * @param queue [out] Queue value.
 */
void  SNEPPX_oneapi_destroy_queue(void* queue);
/**
 * @brief Perform Oneapi Mem Alloc.
 *
 * @param dev_ptr [out] Dev Ptr value.
 * @param bytes [in] Bytes value.
 * @param queue [out] Queue value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_oneapi_mem_alloc(void** dev_ptr, size_t bytes, void* queue);
/**
 * @brief Free Oneapi Mem.
 *
 * @param dev_ptr [out] Dev Ptr value.
 * @param queue [out] Queue value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_oneapi_mem_free(void* dev_ptr, void* queue);
/**
 * @brief Perform Oneapi Memcpy Htod.
 *
 * @param dev_dst [out] Dev Dst value.
 * @param host_src [in] Host Src value.
 * @param bytes [in] Bytes value.
 * @param queue [out] Queue value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_oneapi_memcpy_htod(void* dev_dst, const void* host_src, size_t bytes, void* queue);
/**
 * @brief Perform Oneapi Memcpy Dtoh.
 *
 * @param host_dst [out] Host Dst value.
 * @param dev_src [in] Dev Src value.
 * @param bytes [in] Bytes value.
 * @param queue [out] Queue value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_oneapi_memcpy_dtoh(void* host_dst, const void* dev_src, size_t bytes, void* queue);
/**
 * @brief Perform Oneapi Launch Kernel.
 *
 * @param kernel_name [in] Kernel Name value.
 * @param queue [out] Queue value.
 * @param args [out] Args value.
 * @param args_size [in] Args Size value.
 * @param global_size [in] Global Size value.
 * @param local_size [in] Local Size value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_oneapi_launch_kernel(const char* kernel_name, void* queue, void* args, size_t args_size, size_t global_size, size_t local_size);
#ifdef __cplusplus
}
#endif
#endif
