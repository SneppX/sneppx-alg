#include "hierarchical_state_space.h"
#include "test_gtest.h"
#include <stdio.h>
#include <math.h>

/*
 * SNEPPX - Test Hss Discretize
 *
 * WHAT
 *   Test Hss Discretize.
 *
 * CONCEPT
 *   Provides the Test Hss Discretize.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */






static void test_discretize_zoh(void) {
    float A[] = {-0.5f, 0.0f, 0.0f, -0.5f};
    float B[] = {1.0f, 0.0f, 0.0f, 1.0f};
    float dt = 0.1f;
    float Ad[4], Bd[4];
    SNEPPX_hss_discretize_zoh(A, B, 2, dt, Ad, Bd);
    SX_ASSERT(Ad[0] > 0.9f, "Ad[0] positive");
    SX_ASSERT(Ad[3] > 0.9f, "Ad[3] positive");
    SX_ASSERT_NEAR(Ad[1], 0.0f, 1e-6f, "Ad[1] zero");
    SX_ASSERT_NEAR(Ad[2], 0.0f, 1e-6f, "Ad[2] zero");
}

static void test_discretize_bilinear(void) {
    float A[] = {-1.0f};
    float B[] = {1.0f};
    float dt = 0.1f;
    float Ad[1], Bd[1];
    SNEPPX_hss_discretize_bilinear(A, B, 1, dt, Ad, Bd);
    SX_ASSERT(Ad[0] > 0.8f, "Ad bilinear positive");
    SX_ASSERT(Bd[0] > 0.0f, "Bd bilinear positive");
}

static void test_discretize_identity(void) {
    float A[] = {0.0f};
    float B[] = {1.0f};
    float dt = 1.0f;
    float Ad[1], Bd[1];
    SNEPPX_hss_discretize_zoh(A, B, 1, dt, Ad, Bd);
    SX_ASSERT_NEAR(Ad[0], 1.0f, 1e-4f, "Ad identity");
    SX_ASSERT_NEAR(Bd[0], 1.0f, 1e-4f, "Bd identity");
}


TEST(test_hss_discretize, discretize_zoh) { test_discretize_zoh(); }
TEST(test_hss_discretize, discretize_bilinear) { test_discretize_bilinear(); }
TEST(test_hss_discretize, discretize_identity) { test_discretize_identity(); }
