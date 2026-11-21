#include "neural_core/drivers/driver_status.h"
#include "test_gtest.h"
#include "intel_driver.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

/*
 * SNEPPX - Test Backend Intel
 *
 * WHAT
 *   Test Backend Intel.
 *
 * CONCEPT
 *   Provides the Test Backend Intel.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static int pass = 0, fail = 0;

TEST(test_backend_intel, suite) {
    int r = SNEPPX_intel_register();
    if (r == SNEPPX_DRIVER_OK) {
        int cnt = 0;
        SX_CHECK(SNEPPX_intel_get_device_count(&cnt) == SNEPPX_DRIVER_OK && cnt >= 1, "intel device count");
        char name[64]; size_t mem = 0; int eu = 0;
        SX_CHECK(SNEPPX_intel_get_device_props(0, name, sizeof(name), &mem, &eu) == SNEPPX_DRIVER_OK, "intel device props");
        void* ctx = SNEPPX_intel_create_context(0);
        SX_CHECK(ctx != NULL, "intel create context");
        void *a = NULL, *b = NULL;
        SX_CHECK(SNEPPX_intel_mem_alloc(&a, 16) == SNEPPX_DRIVER_OK && a != NULL, "intel mem alloc");
        SX_CHECK(SNEPPX_intel_mem_alloc(&b, 16) == SNEPPX_DRIVER_OK, "intel mem alloc b");
        float A[4] = {1,2,3,4};
        SX_CHECK(SNEPPX_intel_memcpy_htod(a, A, 16) == SNEPPX_DRIVER_OK, "intel htod");
        float B[4] = {0};
        SX_CHECK(SNEPPX_intel_memcpy_dtoh(B, a, 16) == SNEPPX_DRIVER_OK, "intel dtoh");
        SX_CHECK(B[0] == 1.0f && B[3] == 4.0f, "intel memcpy roundtrip");
        SX_CHECK(SNEPPX_intel_launch_kernel("relu", ctx, 4, 1) == SNEPPX_DRIVER_OK, "intel launch kernel");
        SNEPPX_intel_mem_free(a); SNEPPX_intel_mem_free(b);
        SNEPPX_intel_destroy_context(ctx);
    } else {
        SX_CHECK(r == SNEPPX_DRIVER_UNSUPPORTED, "intel reports unsupported");
    }
}
