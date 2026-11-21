/*
 * SNEPPX - Test GTest shim
 *
 * WHAT
 *   Compatibility layer between the legacy SNEPPX C test harness and
 *   GoogleTest.
 *
 * CONCEPT
 *   The legacy test suite used hand-rolled harnesses (per-file ASSERT*
 *   macros, run_test() registration, pass/fail counters). Migrated test
 *   sources are now compiled as C++ and link against GoogleTest, while the
 *   test bodies keep their original assertion calls. Each SX_* macro maps a
 *   legacy assertion to a GoogleTest ASSERT_* / EXPECT_* so a failing check
 *   aborts only the current test function (mirroring the legacy `return`
 *   behaviour) without cascading NULL dereferences.
 *
 *   Call-site renames performed by scripts/migrate_tests_to_gtest.py:
 *
 *     ASSERT(cond, msg)           -> SX_ASSERT(cond, msg)
 *     ASSERT_EQ(a, b, msg)        -> SX_ASSERT_EQ(a, b, msg)
 *     ASSERT_NEAR(a,b,eps,msg)    -> SX_ASSERT_NEAR(a,b,eps,msg)
 *     ASSERT_NULL(p, msg)         -> SX_ASSERT_NULL(p, msg)
 *     ASSERT_NOT_NULL(p, msg)     -> SX_ASSERT_NOT_NULL(p, msg)
 *     ASSERT_STREQ/STR_EQ(a,b,msg)-> SX_ASSERT_STREQ(a,b,msg)
 *     ASSERT_NEAR_ARR(a,b,n,e,msg)-> SX_ASSERT_NEAR_ARR(a,b,n,e,msg)
 *     TEST(name, expr)            -> SX_TEST(name, expr)   (per-file TEST macro)
 *     CHECK(c, m)                 -> SX_CHECK(c, m)        (program tests)
 *     FLOAT_CLOSE(a,b)            -> SX_FLOAT_CLOSE(a,b)
 *
 * ROLE
 *   SNEPPX-Algo test infrastructure. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 */

#ifndef SNEPPX_TEST_GTEST_H
#define SNEPPX_TEST_GTEST_H

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* Fatal assertion (stops the current test body), matching legacy `return`. */
#define SX_ASSERT(cond, msg) \
    ASSERT_TRUE(cond) << (msg) << " [" #cond "]"

/* Exact-equality compare, matching the legacy `(a) != (b)` semantics. */
#define SX_ASSERT_EQ(a, b, msg) \
    ASSERT_EQ((a), (b)) << (msg)

/* Tolerated difference, matching the legacy `fabsf((a)-(b)) > (eps)` test. */
#define SX_ASSERT_NEAR(a, b, eps, msg) \
    ASSERT_NEAR((a), (b), (eps)) << (msg)

#define SX_ASSERT_STREQ(a, b, msg) \
    ASSERT_STREQ((a), (b)) << (msg)

#define SX_ASSERT_STR_EQ(a, b, msg) \
    ASSERT_STREQ((a), (b)) << (msg)

#define SX_ASSERT_NULL(p, msg) \
    ASSERT_EQ((p), nullptr) << (msg)

#define SX_ASSERT_NOT_NULL(p, msg) \
    ASSERT_NE((p), nullptr) << (msg)

/* Element-wise near-compare; aborts on the first mismatching element. */
#define SX_ASSERT_NEAR_ARR(a, b, n, eps, msg) do { \
    for (size_t _sx_gi = 0; _sx_gi < (size_t)(n); _sx_gi++) { \
        ASSERT_NEAR((a)[_sx_gi], (b)[_sx_gi], (eps)) \
            << (msg) << " [element " << _sx_gi << "]"; \
    } \
} while (0)

/* Per-file TEST(name, expr) macro (Pattern B files): non-fatal boolean check. */
#define SX_TEST(name, expr) \
    EXPECT_TRUE(expr) << (name)

/* Program-style CHECK(c, m) macro (Pattern C files): non-fatal boolean check. */
#define SX_CHECK(c, m) \
    EXPECT_TRUE(c) << (m)

/* Legacy FLOAT_CLOSE(a, b) helper used by autodiff gradient checks. */
#define SX_FLOAT_CLOSE(a, b) \
    (::std::fabs((a) - (b)) < 5e-2f)

/*
 * C99 compound literal replacement. `(size_t[]){1, 4}` (invalid C++)
 * becomes `SX_ARR_C(size_t, 2, 1, 4)`: a temporary std::array whose
 * .data() pointer is valid for the duration of the enclosing full
 * expression, i.e. while the callee consumes the argument.
 */
#define SX_ARR_C(TYPE, COUNT, ...) \
    (::std::array<TYPE, COUNT>{{__VA_ARGS__}}.data())

#endif /* SNEPPX_TEST_GTEST_H */
