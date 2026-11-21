#ifndef SNEPPX_INTEL_DRIVER_H
#define SNEPPX_INTEL_DRIVER_H
#include <stddef.h>
#ifdef __cplusplus
/*
 * SNEPPX - Intel Driver
 *
 * WHAT
 *   Intel Driver.
 *
 * CONCEPT
 *   Provides the Intel Driver.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif
/**
 * @brief Perform Intel Register.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_intel_register(void);
/**
 * @brief Perform Intel Get Device Count.
 *
 * @param count [out] Count value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_intel_get_device_count(int* count);
/**
 * @brief Perform Intel Get Device Props.
 *
 * @param dev_id [in] Dev Id value.
 * @param name [out] Name value.
 * @param name_max [in] Name Max value.
 * @param global_mem [out] Global Mem value.
 * @param eu_count [out] Eu Count value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_intel_get_device_props(int dev_id, char* name, size_t name_max, size_t* global_mem, int* eu_count);
/**
 * @brief Perform Intel Create Context.
 *
 * @param device_id [in] Device Id value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_intel_create_context(int device_id);
/**
 * @brief Perform Intel Destroy Context.
 *
 * @param ctx [out] Ctx value.
 */
void  SNEPPX_intel_destroy_context(void* ctx);
/**
 * @brief Perform Intel Mem Alloc.
 *
 * @param dev_ptr [out] Dev Ptr value.
 * @param bytes [in] Bytes value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_intel_mem_alloc(void** dev_ptr, size_t bytes);
/**
 * @brief Free Intel Mem.
 *
 * @param dev_ptr [out] Dev Ptr value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_intel_mem_free(void* dev_ptr);
/**
 * @brief Perform Intel Memcpy Htod.
 *
 * @param dev_dst [out] Dev Dst value.
 * @param host_src [in] Host Src value.
 * @param bytes [in] Bytes value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_intel_memcpy_htod(void* dev_dst, const void* host_src, size_t bytes);
/**
 * @brief Perform Intel Memcpy Dtoh.
 *
 * @param host_dst [out] Host Dst value.
 * @param dev_src [in] Dev Src value.
 * @param bytes [in] Bytes value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_intel_memcpy_dtoh(void* host_dst, const void* dev_src, size_t bytes);
/**
 * @brief Perform Intel Launch Kernel.
 *
 * @param kernel_name [in] Kernel Name value.
 * @param queue [out] Queue value.
 * @param global_size [in] Global Size value.
 * @param local_size [in] Local Size value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_intel_launch_kernel(const char* kernel_name, void* queue, size_t global_size, size_t local_size);
#ifdef __cplusplus
}
#endif
#endif
