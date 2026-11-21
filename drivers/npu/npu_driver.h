#ifndef SNEPPX_NPU_DRIVER_H
#define SNEPPX_NPU_DRIVER_H
#include <stddef.h>
#ifdef __cplusplus
/*
 * SNEPPX - Npu Driver
 *
 * WHAT
 *   Npu Driver.
 *
 * CONCEPT
 *   Provides the Npu Driver.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif
/**
 * @brief Perform Npu Register Driver.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_npu_register_driver(void);
/**
 * @brief Perform Npu Get Device Count.
 *
 * @param count [out] Count value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_npu_get_device_count(int* count);
/**
 * @brief Perform Npu Get Device Props.
 *
 * @param dev_id [in] Dev Id value.
 * @param props [out] Props value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_npu_get_device_props(int dev_id, void* props);
/**
 * @brief Perform Npu Create Context.
 *
 * @param device_id [in] Device Id value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_npu_create_context(int device_id);
/**
 * @brief Perform Npu Destroy Context.
 *
 * @param ctx [out] Ctx value.
 */
void  SNEPPX_npu_destroy_context(void* ctx);
/**
 * @brief Perform Npu Mem Alloc.
 *
 * @param dev_ptr [out] Dev Ptr value.
 * @param bytes [in] Bytes value.
 * @param ctx [out] Ctx value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_npu_mem_alloc(void** dev_ptr, size_t bytes, void* ctx);
/**
 * @brief Free Npu Mem.
 *
 * @param dev_ptr [out] Dev Ptr value.
 * @param ctx [out] Ctx value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_npu_mem_free(void* dev_ptr, void* ctx);
/**
 * @brief Perform Npu Mem Htod.
 *
 * @param dev_dst [out] Dev Dst value.
 * @param host_src [in] Host Src value.
 * @param bytes [in] Bytes value.
 * @param ctx [out] Ctx value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_npu_mem_htod(void* dev_dst, const void* host_src, size_t bytes, void* ctx);
/**
 * @brief Perform Npu Mem Dtoh.
 *
 * @param host_dst [out] Host Dst value.
 * @param dev_src [in] Dev Src value.
 * @param bytes [in] Bytes value.
 * @param ctx [out] Ctx value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_npu_mem_dtoh(void* host_dst, const void* dev_src, size_t bytes, void* ctx);
/**
 * @brief Perform Npu Execute.
 *
 * @param exec [out] Exec value.
 * @param inputs [out] Inputs value.
 * @param num_inputs [in] Num Inputs value.
 * @param outputs [out] Outputs value.
 * @param num_outputs [in] Num Outputs value.
 * @param ctx [out] Ctx value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_npu_execute(void* exec, void** inputs, size_t num_inputs, void** outputs, size_t num_outputs, void* ctx);
#ifdef __cplusplus
}
#endif
#endif
