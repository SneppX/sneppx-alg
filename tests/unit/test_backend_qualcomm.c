#include "neural_core/drivers/driver_status.h"
#include "test_gtest.h"
#include "qualcomm_driver.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

/*
 * SNEPPX - Test Backend Qualcomm
 *
 * WHAT
 *   Test Backend Qualcomm.
 *
 * CONCEPT
 *   Provides the Test Backend Qualcomm.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static int pass = 0, fail = 0;

TEST(test_backend_qualcomm, suite) {
    int r = SNEPPX_qualcomm_register();
    if (r == SNEPPX_DRIVER_OK) {
        int cnt = 0;
        SX_CHECK(SNEPPX_qualcomm_get_device_count(&cnt) == SNEPPX_DRIVER_OK && cnt >= 1, "qnn device count");
        char name[64]; unsigned long long mem = 0;
        SX_CHECK(SNEPPX_qualcomm_get_device_props(0, name, sizeof(name), &mem) == SNEPPX_DRIVER_OK, "qnn device props");
        void* ctx = SNEPPX_qualcomm_create_context("model.sneppx");
        SX_CHECK(ctx != NULL, "qnn create context");
        float data[4] = { -1.0f, 0.5f, -2.0f, 3.0f };
        SX_CHECK(SNEPPX_qualcomm_set_input(ctx, "input", data, 4) == SNEPPX_DRIVER_OK, "qnn set input");
        SX_CHECK(SNEPPX_qualcomm_run_inference(ctx) == SNEPPX_DRIVER_OK, "qnn run inference");
        float out[4] = {0};
        SX_CHECK(SNEPPX_qualcomm_get_output(ctx, "output", out, 4) == SNEPPX_DRIVER_OK, "qnn get output");
        SX_CHECK(out[0] == 0.0f && out[3] == 3.0f, "qnn relu-like inference");
        SNEPPX_qualcomm_destroy_context(ctx);
    } else {
        SX_CHECK(r == SNEPPX_DRIVER_UNSUPPORTED, "qnn reports unsupported");
    }
}
