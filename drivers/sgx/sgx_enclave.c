#include "sgx_enclave.h"
#include "neural_core/drivers/driver_status.h"
#include "neural_core/drivers/reference_compute.h"
#include <stdlib.h>
#include <string.h>

#ifdef SNEPPX_BUILD_SGX

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


typedef struct {
    char enclave_name[64];
    size_t heap_size;
    size_t stack_size;
    int created;
    unsigned char seal_key[32];
} SGXEnclave;

static SGXEnclave g_enclave;

/**
 * @brief Initialize Sgx.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sgx_init(const char* enclave_path) {
    (void)enclave_path;
    memset(&g_enclave, 0, sizeof(g_enclave));
    for (int i = 0; i < 32; i++)
        g_enclave.seal_key[i] = (unsigned char)(i * 17 + 7);
    return SNEPPX_DRIVER_OK;
}

/**
 * @brief Destroy Sgx.
 */
void SNEPPX_sgx_destroy(void) {
    memset(&g_enclave, 0, sizeof(g_enclave));
}

/**
 * @brief Perform Sgx Create Enclave.
 *
 * @param name [in] Name value.
 * @param heap_size [in] Heap Size value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sgx_create_enclave(const char* name, size_t heap_size, size_t stack_size) {
    if (!name) return SNEPPX_DRIVER_ERROR;
    strncpy(g_enclave.enclave_name, name, sizeof(g_enclave.enclave_name) - 1);
    g_enclave.enclave_name[sizeof(g_enclave.enclave_name) - 1] = '\0';
    g_enclave.heap_size = heap_size;
    g_enclave.stack_size = stack_size;
    g_enclave.created = 1;
    return SNEPPX_DRIVER_OK;
}

/**
 * @brief Perform Sgx Destroy Enclave.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sgx_destroy_enclave(void) {
    memset(&g_enclave.enclave_name, 0, sizeof(g_enclave.enclave_name));
    g_enclave.heap_size = 0;
    g_enclave.stack_size = 0;
    g_enclave.created = 0;
    return SNEPPX_DRIVER_OK;
}

/**
 * @brief Perform Sgx Call.
 *
 * @param func_name [in] Func Name value.
 * @param input [out] Input value.
 * @param input_len [in] Input Len value.
 * @param output [out] Output value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sgx_call(const char* func_name, void* input, size_t input_len, void* output, size_t output_len) {
    if (!func_name || !input || !output || !g_enclave.created) return SNEPPX_DRIVER_ERROR;
    (void)output_len;
    if (sneppx_ref_stricmp(func_name, "inference") == 0) {
        float* in = (float*)input;
        float* out = (float*)output;
        size_t n = input_len / sizeof(float);
        sneppx_ref_elementwise("relu", out, in, n, 1.0f);
        return SNEPPX_DRIVER_OK;
    }
    memcpy(output, input, input_len < output_len ? input_len : output_len);
    return SNEPPX_DRIVER_OK;
}

/**
 * @brief Perform Sgx Seal Data.
 *
 * @param data [in] Data value.
 * @param data_len [in] Data Len value.
 * @param sealed [out] Sealed value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sgx_seal_data(const unsigned char* data, size_t data_len, unsigned char* sealed, size_t* sealed_len) {
    if (!data || !sealed || !sealed_len) return SNEPPX_DRIVER_ERROR;
    size_t total = data_len + 32 + 16;
    if (*sealed_len < total) { *sealed_len = total; return SNEPPX_DRIVER_ERROR; }
    memcpy(sealed, g_enclave.seal_key, 32);
    for (size_t i = 0; i < data_len; i++)
        sealed[32 + i] = data[i] ^ g_enclave.seal_key[i % 32];
    for (size_t i = 0; i < 16; i++)
        sealed[32 + data_len + i] = (unsigned char)(data_len * (i + 1));
    *sealed_len = total;
    return SNEPPX_DRIVER_OK;
}

/**
 * @brief Perform Sgx Unseal Data.
 *
 * @param sealed [in] Sealed value.
 * @param sealed_len [in] Sealed Len value.
 * @param data [out] Data value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sgx_unseal_data(const unsigned char* sealed, size_t sealed_len, unsigned char* data, size_t* data_len) {
    if (!sealed || !data || !data_len || sealed_len < 48) return SNEPPX_DRIVER_ERROR;
    size_t payload_len = sealed_len - 48;
    if (*data_len < payload_len) { *data_len = payload_len; return SNEPPX_DRIVER_ERROR; }
    for (size_t i = 0; i < payload_len; i++)
        data[i] = sealed[32 + i] ^ g_enclave.seal_key[i % 32];
    *data_len = payload_len;
    return SNEPPX_DRIVER_OK;
}

#else

/**
 * @brief Initialize Sgx.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sgx_init(const char* enclave_path) { (void)enclave_path; return SNEPPX_DRIVER_UNSUPPORTED; }
/**
 * @brief Destroy Sgx.
 */
void SNEPPX_sgx_destroy(void) {}
/**
 * @brief Perform Sgx Create Enclave.
 *
 * @param name [in] Name value.
 * @param heap_size [in] Heap Size value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sgx_create_enclave(const char* name, size_t heap_size, size_t stack_size) { (void)name; (void)heap_size; (void)stack_size; return SNEPPX_DRIVER_UNSUPPORTED; }
/**
 * @brief Perform Sgx Destroy Enclave.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sgx_destroy_enclave(void) { return SNEPPX_DRIVER_UNSUPPORTED; }
/**
 * @brief Perform Sgx Call.
 *
 * @param func_name [in] Func Name value.
 * @param input [out] Input value.
 * @param input_len [in] Input Len value.
 * @param output [out] Output value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sgx_call(const char* func_name, void* input, size_t input_len, void* output, size_t output_len) { (void)func_name; (void)input; (void)input_len; (void)output; (void)output_len; return SNEPPX_DRIVER_UNSUPPORTED; }
/**
 * @brief Perform Sgx Seal Data.
 *
 * @param data [in] Data value.
 * @param data_len [in] Data Len value.
 * @param sealed [out] Sealed value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sgx_seal_data(const unsigned char* data, size_t data_len, unsigned char* sealed, size_t* sealed_len) { (void)data; (void)data_len; (void)sealed; (void)sealed_len; return SNEPPX_DRIVER_UNSUPPORTED; }
/**
 * @brief Perform Sgx Unseal Data.
 *
 * @param sealed [in] Sealed value.
 * @param sealed_len [in] Sealed Len value.
 * @param data [out] Data value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sgx_unseal_data(const unsigned char* sealed, size_t sealed_len, unsigned char* data, size_t* data_len) { (void)sealed; (void)sealed_len; (void)data; (void)data_len; return SNEPPX_DRIVER_UNSUPPORTED; }

#endif
