#include "neural_core/drivers/driver_status.h"
#include "test_gtest.h"
#include "shim_layer.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

/*
 * SNEPPX - Test Backend Shim
 *
 * WHAT
 *   Test Backend Shim.
 *
 * CONCEPT
 *   Provides the Test Backend Shim.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static int pass = 0, fail = 0;

TEST(test_backend_shim, suite) {
    int r = SNEPPX_shim_init("amd");
    if (r == SNEPPX_DRIVER_OK) {
        int cnt = 0;
        SX_CHECK(SNEPPX_shim_get_device_count(&cnt) == SNEPPX_DRIVER_OK && cnt >= 1, "shim device count");
        void* ctx = SNEPPX_shim_create_context(0);
        SX_CHECK(ctx != NULL, "shim create context");
        void *a = NULL, *b = NULL, *cptr = NULL;
        SX_CHECK(SNEPPX_shim_mem_alloc(&a, 16, ctx) == SNEPPX_DRIVER_OK && a != NULL, "shim mem alloc");
        SX_CHECK(SNEPPX_shim_mem_alloc(&b, 16, ctx) == SNEPPX_DRIVER_OK, "shim mem alloc b");
        SX_CHECK(SNEPPX_shim_mem_alloc(&cptr, 16, ctx) == SNEPPX_DRIVER_OK, "shim mem alloc c");
        float A[4] = {1,2,3,4}, B[4] = {5,6,7,8};
        SX_CHECK(SNEPPX_shim_memcpy_htod(a, A, 16, ctx) == SNEPPX_DRIVER_OK, "shim htod");
        SX_CHECK(SNEPPX_shim_memcpy_htod(b, B, 16, ctx) == SNEPPX_DRIVER_OK, "shim htod b");
        void* args[3] = {a, b, cptr};
        SX_CHECK(SNEPPX_shim_launch_kernel("gemm", ctx, args, 3, 2, 1) == SNEPPX_DRIVER_OK, "shim launch gemm");
        float C[4] = {0};
        SX_CHECK(SNEPPX_shim_memcpy_dtoh(C, cptr, 16, ctx) == SNEPPX_DRIVER_OK, "shim dtoh");
        SX_CHECK(fabsf(C[0] - 19.0f) < 0.05f && fabsf(C[3] - 50.0f) < 0.05f, "shim gemm result");
        SX_CHECK(SNEPPX_shim_synchronize(ctx) == SNEPPX_DRIVER_OK, "shim synchronize");
        SNEPPX_shim_mem_free(a, ctx); SNEPPX_shim_mem_free(b, ctx); SNEPPX_shim_mem_free(cptr, ctx);
        SNEPPX_shim_destroy_context(ctx);
    } else {
        SX_CHECK(r == SNEPPX_DRIVER_UNSUPPORTED, "shim reports unsupported on default build (use -DSNEPPX_BUILD_AMD=ON -DSNEPPX_BUILD_SHIM=ON)");
    }
    SNEPPX_shim_cleanup();
}
