#include "neural_core/drivers/driver_status.h"
#include "test_gtest.h"
#include "sgx_enclave.h"
#include <stdio.h>
#include <string.h>

/*
 * SNEPPX - Test Backend Sgx
 *
 * WHAT
 *   Test Backend Sgx.
 *
 * CONCEPT
 *   Provides the Test Backend Sgx.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static int pass = 0, fail = 0;

TEST(test_backend_sgx, suite) {
    int r = SNEPPX_sgx_init("enclave.signed.so");
    if (r == SNEPPX_DRIVER_OK) {
        SX_CHECK(SNEPPX_sgx_create_enclave("test", 65536, 8192) == SNEPPX_DRIVER_OK, "sgx create enclave");
        float input[4] = { -1.0f, 2.0f, -3.0f, 4.0f };
        float output[4] = {0};
        SX_CHECK(SNEPPX_sgx_call("inference", input, sizeof(input), output, sizeof(output)) == SNEPPX_DRIVER_OK, "sgx call inference");
        SX_CHECK(output[0] == 0.0f && output[1] == 2.0f, "sgx inference result");
        unsigned char data[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
        unsigned char sealed[64] = {0}; size_t sealed_len = sizeof(sealed);
        SX_CHECK(SNEPPX_sgx_seal_data(data, 16, sealed, &sealed_len) == SNEPPX_DRIVER_OK, "sgx seal data");
        unsigned char unsealed[32] = {0}; size_t unsealed_len = sizeof(unsealed);
        SX_CHECK(SNEPPX_sgx_unseal_data(sealed, sealed_len, unsealed, &unsealed_len) == SNEPPX_DRIVER_OK, "sgx unseal data");
        SX_CHECK(unsealed_len == 16 && memcmp(unsealed, data, 16) == 0, "sgx seal roundtrip");
        SX_CHECK(SNEPPX_sgx_destroy_enclave() == SNEPPX_DRIVER_OK, "sgx destroy enclave");
    } else {
        SX_CHECK(r == SNEPPX_DRIVER_UNSUPPORTED, "sgx reports unsupported");
    }
    SNEPPX_sgx_destroy();
}
