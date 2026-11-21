#ifndef SNEPPX_SGX_ENCLAVE_H
#define SNEPPX_SGX_ENCLAVE_H
#include <stddef.h>
#ifdef __cplusplus
/*
 * SNEPPX - Sgx Enclave
 *
 * WHAT
 *   Sgx Enclave.
 *
 * CONCEPT
 *   Provides the Sgx Enclave.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif
/**
 * @brief Initialize Sgx.
 *
 * @param enclave_path [in] Enclave Path value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_sgx_init(const char* enclave_path);
/**
 * @brief Destroy Sgx.
 */
void SNEPPX_sgx_destroy(void);
/**
 * @brief Perform Sgx Create Enclave.
 *
 * @param name [in] Name value.
 * @param heap_size [in] Heap Size value.
 * @param stack_size [in] Stack Size value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_sgx_create_enclave(const char* name, size_t heap_size, size_t stack_size);
/**
 * @brief Perform Sgx Destroy Enclave.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_sgx_destroy_enclave(void);
/**
 * @brief Perform Sgx Call.
 *
 * @param func_name [in] Func Name value.
 * @param input [out] Input value.
 * @param input_len [in] Input Len value.
 * @param output [out] Output value.
 * @param output_len [in] Output Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_sgx_call(const char* func_name, void* input, size_t input_len, void* output, size_t output_len);
/**
 * @brief Perform Sgx Seal Data.
 *
 * @param data [in] Data value.
 * @param data_len [in] Data Len value.
 * @param sealed [out] Sealed value.
 * @param sealed_len [out] Sealed Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_sgx_seal_data(const unsigned char* data, size_t data_len, unsigned char* sealed, size_t* sealed_len);
/**
 * @brief Perform Sgx Unseal Data.
 *
 * @param sealed [in] Sealed value.
 * @param sealed_len [in] Sealed Len value.
 * @param data [out] Data value.
 * @param data_len [out] Data Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_sgx_unseal_data(const unsigned char* sealed, size_t sealed_len, unsigned char* data, size_t* data_len);
#ifdef __cplusplus
}
#endif
#endif
