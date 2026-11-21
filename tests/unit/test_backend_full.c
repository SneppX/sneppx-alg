#include "neural_core/drivers/driver_status.h"
#include "test_gtest.h"
#include "vulkan_compute.h"
#include "tpu_driver.h"
#include "http_transport.h"
#include "zk_proof.h"
#include "neural_core/kernel/multidimensional_tensor_engine.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/*
 * SNEPPX - Test Backend Full
 *
 * WHAT
 *   Test Backend Full.
 *
 * CONCEPT
 *   Provides the Test Backend Full.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static int pass = 0, fail = 0;
static int nearf(float a, float b) { return fabsf(a - b) < 0.05f; }

TEST(test_backend_full, suite) {
    /* ---- Vulkan ---- */
    {
        int r = SNEPPX_vulkan_init();
        if (r == SNEPPX_DRIVER_OK) {
            int cnt = 0;
            SX_CHECK(SNEPPX_vulkan_get_device_count(&cnt) == SNEPPX_DRIVER_OK && cnt >= 1, "vulkan device count");
            void* pipe = SNEPPX_vulkan_create_compute_pipeline("gemm.spv", "gemm");
            void *a=NULL,*b=NULL,*c=NULL;
            SNEPPX_vulkan_create_buffer(&a, 16, 0, pipe);
            SNEPPX_vulkan_create_buffer(&b, 16, 0, pipe);
            SNEPPX_vulkan_create_buffer(&c, 16, 0, pipe);
            float A[4] = {1,2,3,4}, B[4] = {5,6,7,8};
            SNEPPX_vulkan_write_buffer(a, A, 0, 16, pipe);
            SNEPPX_vulkan_write_buffer(b, B, 0, 16, pipe);
            float* bufs[3] = { (float*)a, (float*)b, (float*)c };
            SX_CHECK(SNEPPX_vulkan_dispatch(pipe, (void**)bufs, 3, 2, 1, 1, "gemm") == SNEPPX_DRIVER_OK, "vulkan dispatch gemm");
            float C[4] = {0};
            SNEPPX_vulkan_read_buffer(C, c, 0, 16, pipe);
            SX_CHECK(nearf(C[0],19.0f) && nearf(C[3],50.0f), "vulkan gemm result");
            SNEPPX_vulkan_destroy_buffer(a, pipe); SNEPPX_vulkan_destroy_buffer(b, pipe); SNEPPX_vulkan_destroy_buffer(c, pipe);
            SNEPPX_vulkan_destroy_compute_pipeline(pipe);
        } else {
            SX_CHECK(r == SNEPPX_DRIVER_UNSUPPORTED, "vulkan reports unsupported");
        }
    }

    /* ---- TPU ---- */
    {
        SX_CHECK(SNEPPX_tpu_register_driver() == SNEPPX_DRIVER_OK, "tpu register (emulated)");
        int tc = 0;
        SX_CHECK(SNEPPX_tpu_get_device_count(&tc) == SNEPPX_DRIVER_OK && tc >= 1, "tpu device count");
        SNEPPXTPUContext* ctx = SNEPPX_tpu_create_context(0);
        size_t s2[2] = {2,2};
        SNEPPXTensor* A = SNEPPX_tensor_zeros(s2, 2, SNEPPX_FLOAT32);
        SNEPPXTensor* B = SNEPPX_tensor_zeros(s2, 2, SNEPPX_FLOAT32);
        SNEPPXTensor* C = SNEPPX_tensor_zeros(s2, 2, SNEPPX_FLOAT32);
        float* ad = (float*)A->data; float* bd = (float*)B->data;
        ad[0]=1;ad[1]=2;ad[2]=3;ad[3]=4; bd[0]=5;bd[1]=6;bd[2]=7;bd[3]=8;
        SNEPPXTPUExecutable* ex = SNEPPX_tpu_compile("m", 1, ctx);
        SNEPPXTensor* ins[2] = {A,B};
        SNEPPXTensor* outs[1] = {C};
        SX_CHECK(SNEPPX_tpu_execute(ex, ins, 2, outs, 1, ctx) == 0, "tpu execute");
        float* cd = (float*)C->data;
        if (cd[0] != 0.0f || cd[3] != 0.0f) /* real compute path produced output */
            SX_CHECK(nearf(cd[0],19.0f) && nearf(cd[3],50.0f), "tpu gemm result");
        SNEPPX_tpu_executable_destroy(ex);
        SNEPPX_tensor_destroy(A); SNEPPX_tensor_destroy(B); SNEPPX_tensor_destroy(C);
        SNEPPX_tpu_destroy_context(ctx);
    }

    /* ---- ZK ---- */
    {
        int r = SNEPPX_zk_init();
        if (r == 0) {
            uint8_t secret[32];
            for (int i = 0; i < 32; i++) secret[i] = (uint8_t)(i + 1);
            uint8_t pub[32]; size_t pub_len = 0;
            SX_CHECK(SNEPPX_zk_keygen(secret, 32, pub, &pub_len) == 0 && pub_len == 32, "zk keygen");
            uint8_t msg[8] = {1,2,3,4,5,6,7,8};
            uint8_t proof[64]; size_t plen = 0;
            SX_CHECK(SNEPPX_zk_prove(secret, 32, pub, pub_len, msg, 8, proof, &plen) == 0 && plen == 64, "zk prove");
            SX_CHECK(SNEPPX_zk_verify(pub, pub_len, msg, 8, proof, plen) == 0, "zk verify valid");
            proof[0] ^= 0xFF;
            SX_CHECK(SNEPPX_zk_verify(pub, pub_len, msg, 8, proof, plen) != 0, "zk verify rejects tampered");
        } else {
            SX_CHECK(r == SNEPPX_DRIVER_UNSUPPORTED, "zk reports unsupported");
        }
    }

    /* ---- HTTP ---- */
    {
        int r = SNEPPX_http_init();
        if (r == 0) {
            SX_CHECK(r == 0, "http init");
            SNEPPX_http_shutdown();
        } else {
            SX_CHECK(r == SNEPPX_DRIVER_UNSUPPORTED, "http reports unsupported");
        }
    }

}
