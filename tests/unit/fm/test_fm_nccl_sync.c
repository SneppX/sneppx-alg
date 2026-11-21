#include "fractal_memory_orchestrator.h"
#include "test_gtest.h"
#include "polymorphic_memory_allocator.h"
#include <stdio.h>
#include <string.h>

/*
 * SNEPPX - Test Fm Nccl Sync
 *
 * WHAT
 *   Test Fm Nccl Sync.
 *
 * CONCEPT
 *   Provides NCCL communication layer.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */




static int dummy_allreduce(void* data, size_t count, void* ctx) {
    (void)ctx;
    return 0;
}

static void test_fm_nccl_sync_null_callback(void) {
    SNEPPXFMConfig cfg = SNEPPX_fm_config_default();
    cfg.num_nodes = 2;
    cfg.memory_dim = 4;
    cfg.memory_capacity = 8;
    SNEPPXFMController* ctrl = SNEPPX_fm_controller_create(&cfg);
    SX_ASSERT(ctrl != NULL, "controller created");
    int rc = SNEPPX_fm_sync_nccl(ctrl, NULL, NULL);
    SX_ASSERT(rc == 0, "sync with null callback ok");
    SNEPPX_fm_controller_destroy(ctrl);
}

static void test_fm_nccl_sync_with_callback(void) {
    SNEPPXFMConfig cfg = SNEPPX_fm_config_default();
    cfg.num_nodes = 2;
    cfg.memory_dim = 4;
    cfg.memory_capacity = 8;
    SNEPPXFMController* ctrl = SNEPPX_fm_controller_create(&cfg);
    SX_ASSERT(ctrl != NULL, "controller created");

    size_t shape_k[] = {1, cfg.memory_dim};
    SNEPPXTensor* key = SNEPPX_tensor_ones(shape_k, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* val = SNEPPX_tensor_ones(shape_k, 2, SNEPPX_FLOAT32);
    for (size_t n = 0; n < cfg.num_nodes; n++) {
        SNEPPX_fm_memory_bank_write(ctrl->nodes[n]->memory_bank, key, val);
    }

    int rc = SNEPPX_fm_sync_nccl(ctrl, dummy_allreduce, &ctrl);
    SX_ASSERT(rc == 0, "sync with dummy callback ok");
    SX_ASSERT(ctrl->sync_state.sync_round > 0, "sync round incremented");

    SNEPPX_tensor_destroy(key);
    SNEPPX_tensor_destroy(val);
    SNEPPX_fm_controller_destroy(ctrl);
}


TEST(test_fm_nccl_sync, FM_NCCL_sync_null_callback) { test_fm_nccl_sync_null_callback(); }
TEST(test_fm_nccl_sync, FM_NCCL_sync_with_dummy_callback) { test_fm_nccl_sync_with_callback(); }
