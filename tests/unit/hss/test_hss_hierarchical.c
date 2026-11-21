#include "hierarchical_state_space.h"
#include "test_gtest.h"
#include <stdio.h>
#include <math.h>

/*
 * SNEPPX - Test Hss Hierarchical
 *
 * WHAT
 *   Test Hss Hierarchical.
 *
 * CONCEPT
 *   Provides neural architecture.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */






static void test_hierarchical_config(void) {
    SNEPPXHSSConfig cfg = SNEPPX_hss_config_default();
    SX_ASSERT(cfg.state_dim == 64, "default state_dim 64");
    SX_ASSERT(cfg.input_dim == 512, "default input_dim 512");
    SX_ASSERT(cfg.num_layers == 2, "default num_layers 2");
    cfg.use_hierarchical = 1;
    SX_ASSERT(cfg.use_hierarchical == 1, "hierarchical enabled");
}

static void test_hierarchical_level_count(void) {
    int levels = SNEPPX_hss_hierarchical_levels(64, 4);
    SX_ASSERT(levels == 4, "log2(64/4) levels = 4");
    levels = SNEPPX_hss_hierarchical_levels(16, 2);
    SX_ASSERT(levels == 3, "log2(16/2) levels = 3");
}

static void test_hierarchical_create(void) {
    SNEPPXHSSConfig cfg = SNEPPX_hss_config_default();
    cfg.state_dim = 16;
    cfg.input_dim = 16;
    cfg.use_hierarchical = 1;
    SNEPPXHSSModel* model = SNEPPX_hss_model_create(&cfg, 42);
    SX_ASSERT(model != NULL, "hierarchical model created");
    SNEPPX_hss_model_destroy(model);
}


TEST(test_hss_hierarchical, hierarchical_config) { test_hierarchical_config(); }
TEST(test_hss_hierarchical, hierarchical_level_count) { test_hierarchical_level_count(); }
TEST(test_hss_hierarchical, hierarchical_create) { test_hierarchical_create(); }
