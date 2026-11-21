#include "cuda_driver.h"
#include "test_gtest.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/*
 * SNEPPX - Test Cuda Driver
 *
 * WHAT
 *   Test Cuda Driver.
 *
 * CONCEPT
 *   Provides the Test Cuda Driver.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */





static void test_register_driver(void) {
    int ret = SNEPPX_cuda_register_driver();
    SX_ASSERT(ret == 0, "driver registration");
}

static void test_get_device_count(void) {
    int count = -1;
    int ret = SNEPPX_cuda_get_device_count(&count);
    SX_ASSERT(ret == 0, "get device count returns 0");
    SX_ASSERT(count >= 0, "device count is non-negative");
}

static void test_get_device_props(void) {
    SNEPPXCUDADeviceProps props;
    memset(&props, 0, sizeof(props));
    int ret = SNEPPX_cuda_get_device_props(0, &props);
    SX_ASSERT(ret == 0, "get device props returns 0");
}

static void test_set_get_device(void) {
    int ret = SNEPPX_cuda_set_device(0);
    SX_ASSERT(ret == 0, "set device 0");
    int dev = -1;
    ret = SNEPPX_cuda_get_device(&dev);
    SX_ASSERT(ret == 0, "get device");
}

static void test_context_create_destroy(void) {
    SNEPPXCUDAContext* ctx = SNEPPX_cuda_create_context(0);
    /* May return NULL if CUDA not available - that's OK */
    if (ctx) {
        SX_ASSERT(ctx->device_id == 0, "context device id");
        int err = SNEPPX_cuda_context_error(ctx);
        SX_ASSERT(err == 0, "no context error");
        SNEPPX_cuda_destroy_context(ctx);
    }
}

static void test_stream_create_destroy(void) {
    SNEPPXCUDAStream* stream = NULL;
    int ret = SNEPPX_cuda_stream_create(&stream, 0);
    SX_ASSERT(ret == 0, "stream create");
    if (stream) {
        SNEPPX_cuda_stream_destroy(stream);
    }
}

static void test_mem_alloc_free(void) {
    void* ptr = NULL;
    int ret = SNEPPX_cuda_mem_alloc(&ptr, 1024);
    SX_ASSERT(ret == 0, "mem alloc");
    if (ptr) {
        ret = SNEPPX_cuda_mem_free(ptr);
        SX_ASSERT(ret == 0, "mem free");
    }
}

static void test_mem_htod_dtoh(void) {
    float host_src[] = {1.0f, 2.0f, 3.0f};
    void* dev = NULL;
    SNEPPX_cuda_mem_alloc(&dev, sizeof(host_src));
    if (dev) {
        int ret = SNEPPX_cuda_mem_htod(dev, host_src, sizeof(host_src));
        SX_ASSERT(ret == 0, "htod");
        float host_dst[3] = {0};
        ret = SNEPPX_cuda_mem_dtoh(host_dst, dev, sizeof(host_src));
        SX_ASSERT(ret == 0, "dtoh");
        SNEPPX_cuda_mem_free(dev);
    }
}

static void test_event_create_destroy(void) {
    SNEPPXCUDAEvent* event = NULL;
    int ret = SNEPPX_cuda_event_create(&event);
    SX_ASSERT(ret == 0, "event create");
    if (event) {
        SNEPPX_cuda_event_destroy(event);
    }
}

static void test_error_string(void) {
    const char* err = SNEPPX_cuda_error_string(0);
    SX_ASSERT(err != NULL, "error string non-null");
}


TEST(test_cuda_driver, register_driver) { test_register_driver(); }
TEST(test_cuda_driver, get_device_count) { test_get_device_count(); }
TEST(test_cuda_driver, get_device_props) { test_get_device_props(); }
TEST(test_cuda_driver, set_get_device) { test_set_get_device(); }
TEST(test_cuda_driver, context_create_destroy) { test_context_create_destroy(); }
TEST(test_cuda_driver, stream_create_destroy) { test_stream_create_destroy(); }
TEST(test_cuda_driver, mem_alloc_free) { test_mem_alloc_free(); }
TEST(test_cuda_driver, mem_htod_dtoh) { test_mem_htod_dtoh(); }
TEST(test_cuda_driver, event_create_destroy) { test_event_create_destroy(); }
TEST(test_cuda_driver, error_string) { test_error_string(); }
