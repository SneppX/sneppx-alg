#ifndef SNEPPX_PTH_FORMAT_H
#define SNEPPX_PTH_FORMAT_H
#include <stddef.h>
#ifdef __cplusplus
/*
 * SNEPPX - Pth Format
 *
 * WHAT
 *   Pth Format.
 *
 * CONCEPT
 *   Provides the Pth Format.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif
/**
 * @brief Load Pth.
 *
 * @param path [in] Path value.
 * @param device [in] Device value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_pth_load(const char* path, const char* device);
/**
 * @brief Destroy Pth.
 *
 * @param state [out] State value.
 */
void SNEPPX_pth_destroy(void* state);
/**
 * @brief Perform Pth Get Tensor.
 *
 * @param state [out] State value.
 * @param key [in] Key value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_pth_get_tensor(void* state, const char* key);
/**
 * @brief Perform Pth Get Keys.
 *
 * @param state [out] State value.
 * @param keys [out] Keys value.
 * @param count [out] Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_pth_get_keys(void* state, char*** keys, size_t* count);
/**
 * @brief Perform Pth Set Tensor.
 *
 * @param state [out] State value.
 * @param key [in] Key value.
 * @param data [in] Data value.
 * @param shape [in] Shape value.
 * @param ndim [in] Ndim value.
 * @param dtype [in] Dtype value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_pth_set_tensor(void* state, const char* key, const void* data, const size_t* shape, size_t ndim, int dtype);
/**
 * @brief Save Pth.
 *
 * @param state [out] State value.
 * @param path [in] Path value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_pth_save(void* state, const char* path);
#ifdef __cplusplus
}
#endif
#endif
