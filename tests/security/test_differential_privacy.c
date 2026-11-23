#include "differential_privacy.h"
#include "test_gtest.h"
#include <math.h>
#include <stdio.h>

/*
 * SNEPPX - Test Differential Privacy
 *
 * WHAT
 *   Test Differential Privacy.
 *
 * CONCEPT
 *   Provides the Test Differential Privacy.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */

TEST(test_differential_privacy, suite)
{
    dp_stats_t stats;
    double val = SNEPPX_dp_laplace_mech(10.0, 1.0, 1.0);
    printf("Laplace(10.0, 1.0, 1.0) = %.4f\n", val);
    EXPECT_TRUE(std::isfinite(val)) << "M-10: Laplace noise must be finite";
    double val2 = SNEPPX_dp_laplace_mech(10.0, 1.0, 1.0);
    EXPECT_TRUE(val != val2) << "M-10: DP noise must be freshly randomized";
    val = SNEPPX_dp_gaussian_mech(10.0, 1.0, 1.0, 1e-5);
    printf("Gaussian(10.0, 1.0, 1.0, 1e-5) = %.4f\n", val);
    EXPECT_TRUE(std::isfinite(val)) << "M-10: Gaussian noise must be finite";
    SNEPPX_dp_set_budget(1.0, 1e-5);
    double params[4] = {1.0, 2.0, 3.0, 4.0};
    double grads[4] = {0.1, 0.2, 0.3, 0.4};
    EXPECT_EQ(SNEPPX_dp_dpsgd_step(params, grads, 4, 0.01, 1.0, 1e-5, 1.0), 0);
    EXPECT_TRUE(std::isfinite(params[0])) << "M-10: DP-SGD noise must be finite";
    printf("DP-SGD step OK\n");
    if (SNEPPX_dp_get_stats(&stats) == 0)
    {
        printf("Budget remaining: eps=%.4f del=%.4e\n", stats.budget_remaining_epsilon, stats.budget_remaining_delta);
    }
    printf("PASS: Differential privacy OK\n");
    return;
}
