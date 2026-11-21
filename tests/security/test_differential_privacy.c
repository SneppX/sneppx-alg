#include "differential_privacy.h"
#include "test_gtest.h"
#include <stdio.h>
#include <math.h>

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


TEST(test_differential_privacy, suite) {
    dp_stats_t stats;
    double val = SNEPPX_dp_laplace_mech(10.0, 1.0, 1.0);
    printf("Laplace(10.0, 1.0, 1.0) = %.4f\n", val);
    val = SNEPPX_dp_gaussian_mech(10.0, 1.0, 1.0, 1e-5);
    printf("Gaussian(10.0, 1.0, 1.0, 1e-5) = %.4f\n", val);
    SNEPPX_dp_set_budget(1.0, 1e-5);
    double params[4] = {1.0, 2.0, 3.0, 4.0};
    double grads[4] = {0.1, 0.2, 0.3, 0.4};
    SNEPPX_dp_dpsgd_step(params, grads, 4, 0.01, 1.0, 1e-5, 1.0);
    printf("DP-SGD step OK\n");
    if (SNEPPX_dp_get_stats(&stats) == 0) {
        printf("Budget remaining: eps=%.4f del=%.4e\n", stats.budget_remaining_epsilon, stats.budget_remaining_delta);
    }
    printf("PASS: Differential privacy OK\n");
    return;
}
