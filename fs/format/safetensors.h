#ifndef SNEPPX_SAFETENSORS_H
#define SNEPPX_SAFETENSORS_H
#include <stddef.h>
#ifdef __cplusplus
/*
 * SNEPPX - Safetensors
 *
 * WHAT
 *   Safetensors.
 *
 * CONCEPT
 *   Provides tensor operations.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif

/* Tensor dtype codes (mirror the safetensors dtype strings). */
typedef enum {
    SNEPPX_ST_DTYPE_F32 = 0,
    SNEPPX_ST_DTYPE_F16,
    SNEPPX_ST_DTYPE_F64,
    SNEPPX_ST_DTYPE_I64,
    SNEPPX_ST_DTYPE_I32,
    SNEPPX_ST_DTYPE_I16,
    SNEPPX_ST_DTYPE_I8,
    SNEPPX_ST_DTYPE_U8,
    SNEPPX_ST_DTYPE_BOOL,
    SNEPPX_ST_DTYPE_BF16,
    SNEPPX_ST_DTYPE_F8_E5M2,
    SNEPPX_ST_DTYPE_F8_E4M3FN
} SNEPPXSafetensorsDType;

/**
 * @brief Open Safetensors.
 *
 * @param path [in] Path value.
 * @param mode [in] Mode value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_safetensors_open(const char* path, const char* mode);
/**
 * @brief Close Safetensors.
 *
 * @param st [out] St value.
 */
void SNEPPX_safetensors_close(void* st);
/**
 * @brief Perform Safetensors Get Tensor Count.
 *
 * @param st [out] St value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_safetensors_get_tensor_count(void* st);
/**
 * @brief Perform Safetensors Get Tensor Names.
 *
 * @param st [out] St value.
 * @param names [out] Names value.
 * @param count [out] Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_safetensors_get_tensor_names(void* st, char*** names, size_t* count);
/**
 * @brief Perform Safetensors Read Tensor.
 *
 * @param st [out] St value.
 * @param name [in] Name value.
 * @param ndim [out] Ndim value.
 * @param shape [out] Shape value.
 * @param dtype [out] Dtype value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_safetensors_read_tensor(void* st, const char* name, size_t* ndim, size_t** shape, int* dtype);
/**
 * @brief Perform Safetensors Write Tensor.
 *
 * @param st [out] St value.
 * @param name [in] Name value.
 * @param data [in] Data value.
 * @param shape [in] Shape value.
 * @param ndim [in] Ndim value.
 * @param dtype [in] Dtype value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_safetensors_write_tensor(void* st, const char* name, const void* data, const size_t* shape, size_t ndim, int dtype);
/**
 * @brief Save Safetensors.
 *
 * @param st [out] St value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_safetensors_save(void* st);
/**
 * @brief Perform Safetensors Get Metadata.
 *
 * @param st [out] St value.
 * @param key [in] Key value.
 * @param value [out] Value value.
 * @param value_max [in] Value Max value.
 *
 * @return The result value, or 0 on error.
 */
unsigned long long SNEPPX_safetensors_get_metadata(void* st, const char* key, char* value, size_t value_max);

/* Element size in bytes for a dtype code (0 if unknown). */
/**
 * @brief Perform Safetensors Dtype Size.
 *
 * @param dtype [in] Dtype value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_safetensors_dtype_size(int dtype);

#ifdef __cplusplus
}
#endif
#endif
