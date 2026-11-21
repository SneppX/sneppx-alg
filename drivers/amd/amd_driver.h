#ifndef SNEPPX_AMD_DRIVER_H
#define SNEPPX_AMD_DRIVER_H
#include <stddef.h>
#ifdef __cplusplus
/*
 * SNEPPX - Amd Driver
 *
 * WHAT
 *   Amd Driver.
 *
 * CONCEPT
 *   Provides the Amd Driver.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif
/**
 * @brief Perform Amd Register.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_amd_register(void);
/**
 * @brief Perform Amd Get Device Count.
 *
 * @param count [out] Count value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_amd_get_device_count(int* count);
/**
 * @brief Perform Amd Get Device Props.
 *
 * @param dev_id [in] Dev Id value.
 * @param name [out] Name value.
 * @param name_max [in] Name Max value.
 * @param global_mem [out] Global Mem value.
 * @param compute_units [out] Compute Units value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_amd_get_device_props(int dev_id, char* name, size_t name_max, size_t* global_mem, int* compute_units);
/**
 * @brief Perform Amd Create Context.
 *
 * @param device_id [in] Device Id value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_amd_create_context(int device_id);
/**
 * @brief Perform Amd Destroy Context.
 *
 * @param ctx [out] Ctx value.
 */
void  SNEPPX_amd_destroy_context(void* ctx);
/**
 * @brief Perform Amd Mem Alloc.
 *
 * @param dev_ptr [out] Dev Ptr value.
 * @param bytes [in] Bytes value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_amd_mem_alloc(void** dev_ptr, size_t bytes);
/**
 * @brief Free Amd Mem.
 *
 * @param dev_ptr [out] Dev Ptr value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_amd_mem_free(void* dev_ptr);
/**
 * @brief Perform Amd Memcpy Htod.
 *
 * @param dev_dst [out] Dev Dst value.
 * @param host_src [in] Host Src value.
 * @param bytes [in] Bytes value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_amd_memcpy_htod(void* dev_dst, const void* host_src, size_t bytes);
/**
 * @brief Perform Amd Memcpy Dtoh.
 *
 * @param host_dst [out] Host Dst value.
 * @param dev_src [in] Dev Src value.
 * @param bytes [in] Bytes value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_amd_memcpy_dtoh(void* host_dst, const void* dev_src, size_t bytes);
/**
 * @brief Perform Amd Launch Kernel.
 *
 * @param kernel_name [in] Kernel Name value.
 * @param queue [out] Queue value.
 * @param args [out] Args value.
 * @param num_args [in] Num Args value.
 * @param global_size [in] Global Size value.
 * @param local_size [in] Local Size value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_amd_launch_kernel(const char* kernel_name, void* queue, void** args, size_t num_args, size_t global_size, size_t local_size);
#ifdef __cplusplus
}
#endif
#endif
