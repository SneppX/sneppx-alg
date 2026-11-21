#ifndef SNEPPX_TEST_COMMON_H
#define SNEPPX_TEST_COMMON_H

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifdef __cplusplus
#include <cfloat>
#define ASSERT_NEAR_CXX(a, b, eps, msg) do { \
    if (fabs((double)(a) - (double)(b)) > (double)(eps)) { \
        printf("FAIL: %s (got %f, expected %f, eps %f)\n", msg, (double)(a), (double)(b), (double)(eps)); \
        tests_failed++; return; \
    } \
} while(0)
#endif

/*
 * SNEPPX - Test Common
 *
 * WHAT
 *   Test Common.
 *
 * CONCEPT
 *   Provides the Test Common.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("FAIL: %s (%s)\n", msg, #cond); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, eps, msg) do { \
    if (fabsf((float)(a) - (float)(b)) > (float)(eps)) { \
        printf("FAIL: %s (got %f, expected %f, eps %f)\n", msg, (float)(a), (float)(b), (float)(eps)); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_STREQ(a, b, msg) do { \
    if (strcmp((a), (b)) != 0) { \
        printf("FAIL: %s (got \"%s\", expected \"%s\")\n", msg, (a), (b)); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_NULL(ptr, msg) do { \
    if ((ptr) != NULL) { \
        printf("FAIL: %s (expected NULL)\n", msg); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_NOT_NULL(ptr, msg) do { \
    if ((ptr) == NULL) { \
        printf("FAIL: %s (expected non-NULL)\n", msg); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_EQ(a, b, msg) do { \
    if ((a) != (b)) { \
        printf("FAIL: %s (got %lld, expected %lld)\n", msg, (long long)(a), (long long)(b)); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_NEAR_ARR(a, b, n, eps, msg) do { \
    for (size_t _i = 0; _i < (n); _i++) { \
        if (fabsf((float)(a)[_i] - (float)(b)[_i]) > (float)(eps)) { \
            printf("FAIL: %s [%zu] (got %f, expected %f)\n", msg, _i, (float)(a)[_i], (float)(b)[_i]); \
            tests_failed++; return; \
        } \
    } \
} while(0)

static void run_test(const char* name, void (*fn)(void)) {
    printf("Running %s... ", name);
    fflush(stdout);
    fn();
    printf("PASS\n");
    tests_passed++;
}

#define RUN_ALL_TESTS() do { \
    printf("\nResults: %d passed, %d failed out of %d\n", \
           tests_passed, tests_failed, tests_passed + tests_failed); \
    return tests_failed > 0 ? 1 : 0; \
} while(0)

#endif
