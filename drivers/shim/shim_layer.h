#ifndef SNEPPX_SHIM_LAYER_H
#define SNEPPX_SHIM_LAYER_H
#include <stddef.h>
#ifdef __cplusplus
/*
 * SNEPPX - Shim Layer
 *
 * WHAT
 *   Shim Layer.
 *
 * CONCEPT
 *   Provides the Shim Layer.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif
/**
 * @brief Initialize Shim.
 *
 * @param backend_type [in] Backend Type value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shim_init(const char* backend_type);
/**
 * @brief Perform Shim Cleanup.
 */
void SNEPPX_shim_cleanup(void);
/**
 * @brief Perform Shim Get Device Count.
 *
 * @param count [out] Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shim_get_device_count(int* count);
/**
 * @brief Perform Shim Get Device Props.
 *
 * @param dev_id [in] Dev Id value.
 * @param name [out] Name value.
 * @param name_max [in] Name Max value.
 * @param dev_type [out] Dev Type value.
 * @param global_mem [out] Global Mem value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shim_get_device_props(int dev_id, char* name, size_t name_max, int* dev_type, size_t* global_mem);
/**
 * @brief Perform Shim Create Context.
 *
 * @param device_id [in] Device Id value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_shim_create_context(int device_id);
/**
 * @brief Perform Shim Destroy Context.
 *
 * @param ctx [out] Ctx value.
 */
void SNEPPX_shim_destroy_context(void* ctx);
/**
 * @brief Perform Shim Mem Alloc.
 *
 * @param dev_ptr [out] Dev Ptr value.
 * @param bytes [in] Bytes value.
 * @param ctx [out] Ctx value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shim_mem_alloc(void** dev_ptr, size_t bytes, void* ctx);
/**
 * @brief Free Shim Mem.
 *
 * @param dev_ptr [out] Dev Ptr value.
 * @param ctx [out] Ctx value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shim_mem_free(void* dev_ptr, void* ctx);
/**
 * @brief Perform Shim Memcpy Htod.
 *
 * @param dev_dst [out] Dev Dst value.
 * @param host_src [in] Host Src value.
 * @param bytes [in] Bytes value.
 * @param ctx [out] Ctx value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shim_memcpy_htod(void* dev_dst, const void* host_src, size_t bytes, void* ctx);
/**
 * @brief Perform Shim Memcpy Dtoh.
 *
 * @param host_dst [out] Host Dst value.
 * @param dev_src [in] Dev Src value.
 * @param bytes [in] Bytes value.
 * @param ctx [out] Ctx value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shim_memcpy_dtoh(void* host_dst, const void* dev_src, size_t bytes, void* ctx);
/**
 * @brief Perform Shim Launch Kernel.
 *
 * @param kernel_name [in] Kernel Name value.
 * @param ctx [out] Ctx value.
 * @param args [out] Args value.
 * @param num_args [in] Num Args value.
 * @param global_size [in] Global Size value.
 * @param local_size [in] Local Size value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shim_launch_kernel(const char* kernel_name, void* ctx, void** args, size_t num_args, size_t global_size, size_t local_size);
/**
 * @brief Perform Shim Synchronize.
 *
 * @param ctx [out] Ctx value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shim_synchronize(void* ctx);
#ifdef __cplusplus
}
#endif
#endif
