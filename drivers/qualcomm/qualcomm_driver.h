#ifndef SNEPPX_QUALCOMM_DRIVER_H
#define SNEPPX_QUALCOMM_DRIVER_H
#include <stddef.h>
#ifdef __cplusplus
/*
 * SNEPPX - Qualcomm Driver
 *
 * WHAT
 *   Qualcomm Driver.
 *
 * CONCEPT
 *   Provides the Qualcomm Driver.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif
/**
 * @brief Perform Qualcomm Register.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_qualcomm_register(void);
/**
 * @brief Perform Qualcomm Get Device Count.
 *
 * @param count [out] Count value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_qualcomm_get_device_count(int* count);
/**
 * @brief Perform Qualcomm Get Device Props.
 *
 * @param dev_id [in] Dev Id value.
 * @param name [out] Name value.
 * @param name_max [in] Name Max value.
 * @param total_mem [out] Total Mem value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_qualcomm_get_device_props(int dev_id, char* name, size_t name_max, unsigned long long* total_mem);
/**
 * @brief Perform Qualcomm Create Context.
 *
 * @param model_path [in] Model Path value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_qualcomm_create_context(const char* model_path);
/**
 * @brief Perform Qualcomm Destroy Context.
 *
 * @param ctx [out] Ctx value.
 */
void  SNEPPX_qualcomm_destroy_context(void* ctx);
/**
 * @brief Perform Qualcomm Set Input.
 *
 * @param ctx [out] Ctx value.
 * @param name [in] Name value.
 * @param data [in] Data value.
 * @param size [in] Size value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_qualcomm_set_input(void* ctx, const char* name, const float* data, size_t size);
/**
 * @brief Perform Qualcomm Run Inference.
 *
 * @param ctx [out] Ctx value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_qualcomm_run_inference(void* ctx);
/**
 * @brief Perform Qualcomm Get Output.
 *
 * @param ctx [out] Ctx value.
 * @param name [in] Name value.
 * @param data [out] Data value.
 * @param size [in] Size value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_qualcomm_get_output(void* ctx, const char* name, float* data, size_t size);
#ifdef __cplusplus
}
#endif
#endif
