#include <stdio.h>
#include "test_gtest.h"
#include <string.h>
#include <stdlib.h>
#include "cryptographic_random_generator.h"

/*
 * SNEPPX - Test Random
 *
 * WHAT
 *   Test Random.
 *
 * CONCEPT
 *   Provides randomness generation.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */




void test_uniformity(void) {
    printf("\n--- test_uniformity ---\n");
    size_t len = 1024 * 1024;
    uint8_t* buf = (uint8_t*)malloc(len);
    if (!buf) { printf("SKIP: uniformity (malloc failed)\n"); return; }
    int ret = SNEPPX_random_bytes(buf, len);
    SX_TEST("generate 1MB random", ret == 0);
    size_t zeros = 0, ones = 0;
    for (size_t i = 0; i < len; i++) {
        for (int b = 0; b < 8; b++) {
            if ((buf[i] >> b) & 1) ones++; else zeros++;
        }
    }
    double ratio = (double)ones / (double)(ones + zeros);
    SX_TEST("bit ratio near 0.5", ratio > 0.49 && ratio < 0.51);
    free(buf);
}

void test_no_duplicates(void) {
    printf("\n--- test_no_duplicates ---\n");
    uint32_t vals[1000];
    for (int i = 0; i < 1000; i++) vals[i] = SNEPPX_random_uint32();
    int dup = 0;
    for (int i = 0; i < 1000 && !dup; i++)
        for (int j = i + 1; j < 1000 && !dup; j++)
            if (vals[i] == vals[j]) dup = 1;
    SX_TEST("no duplicates in 1000 values", !dup);
}

void test_uniform_bounded(void) {
    printf("\n--- test_uniform_bounded ---\n");
    uint32_t bound = 100;
    uint32_t value = 123;
    SX_TEST("zero bound rejected", SNEPPX_random_uniform_checked(0, &value) == -1);
    SX_TEST("invalid bound leaves output unchanged", value == 123);
    SX_TEST("null output rejected", SNEPPX_random_uniform_checked(bound, NULL) == -1);
    int ok = 1;
    for (int i = 0; i < 10000; i++) {
        if (SNEPPX_random_uniform_checked(bound, &value) != 0 || value >= bound) {
            ok = 0;
            break;
        }
    }
    SX_TEST("checked values in [0, bound)", ok == 1);
}


TEST(test_random, test_uniformity) { test_uniformity(); }
TEST(test_random, test_no_duplicates) { test_no_duplicates(); }
TEST(test_random, test_uniform_bounded) { test_uniform_bounded(); }
