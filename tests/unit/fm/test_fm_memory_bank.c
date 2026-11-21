#include "fractal_memory_orchestrator.h"
#include "test_gtest.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

/*
 * SNEPPX - Test Fm Memory Bank
 *
 * WHAT
 *   Test Fm Memory Bank.
 *
 * CONCEPT
 *   Provides memory management.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static void test_bank_create(void) {
    SNEPPXFMMemoryBank* bank = SNEPPX_fm_memory_bank_create(32, 64);
    SX_ASSERT(bank != NULL, "bank not null");
    SX_ASSERT(bank->keys != NULL, "keys not null");
    SX_ASSERT(bank->values != NULL, "values not null");
    SX_ASSERT(bank->keys->shape[0] == 64, "keys rows 64");
    SX_ASSERT(bank->keys->shape[1] == 32, "keys cols 32");
    SX_ASSERT(bank->num_entries == 0, "empty");
    SX_ASSERT(bank->max_entries == 64, "max 64");
    SNEPPX_fm_memory_bank_destroy(bank);
}

static void test_bank_write_read(void) {
    SNEPPXFMMemoryBank* bank = SNEPPX_fm_memory_bank_create(8, 16);
    size_t key_shape[] = {8};
    size_t val_shape[] = {8};
    for (int k = 0; k < 5; k++) {
        SNEPPXTensor* key = SNEPPX_tensor_zeros(key_shape, 1, SNEPPX_FLOAT32);
        SNEPPXTensor* val = SNEPPX_tensor_zeros(val_shape, 1, SNEPPX_FLOAT32);
        ((float*)key->data)[0] = (float)k;
        ((float*)val->data)[0] = (float)(k * 10);
        int r = SNEPPX_fm_memory_bank_write(bank, key, val);
        SX_ASSERT(r != 0, "write ok");
        SNEPPX_tensor_destroy(key);
        SNEPPX_tensor_destroy(val);
    }
    SX_ASSERT(bank->num_entries == 5, "5 entries");

    SNEPPXTensor* q = SNEPPX_tensor_zeros(key_shape, 1, SNEPPX_FLOAT32);
    ((float*)q->data)[0] = 2.0f;
    SNEPPXTensor* result = SNEPPX_fm_memory_bank_read(bank, q);
    SX_ASSERT(result != NULL, "read hit");
    SX_ASSERT(fabsf(((float*)result->data)[0] - 20.0f) < 1e-5f, "value 20");
    SNEPPX_tensor_destroy(q);
    SNEPPX_tensor_destroy(result);
    SNEPPX_fm_memory_bank_destroy(bank);
}

static void test_bank_overwrite(void) {
    SNEPPXFMMemoryBank* bank = SNEPPX_fm_memory_bank_create(4, 8);
    size_t sh[] = {4};
    SNEPPXTensor* k = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* v1 = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    ((float*)v1->data)[0] = 100.0f;
    SNEPPX_fm_memory_bank_write(bank, k, v1);
    SNEPPXTensor* v2 = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    ((float*)v2->data)[0] = 200.0f;
    SNEPPX_fm_memory_bank_write(bank, k, v2);
    SNEPPXTensor* r = SNEPPX_fm_memory_bank_read(bank, k);
    SX_ASSERT(r != NULL, "read hit");
    SX_ASSERT(fabsf(((float*)r->data)[0] - 200.0f) < 1e-5f, "overwritten to 200");
    SX_ASSERT(bank->num_entries == 1, "still 1 entry");
    SNEPPX_tensor_destroy(k);
    SNEPPX_tensor_destroy(v1);
    SNEPPX_tensor_destroy(v2);
    SNEPPX_tensor_destroy(r);
    SNEPPX_fm_memory_bank_destroy(bank);
}

static void test_bank_eviction(void) {
    SNEPPXFMMemoryBank* bank = SNEPPX_fm_memory_bank_create(2, 3);
    size_t sh[] = {2};
    for (int i = 0; i < 5; i++) {
        SNEPPXTensor* k = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
        SNEPPXTensor* v = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
        ((float*)k->data)[0] = (float)i;
        ((float*)v->data)[0] = (float)i;
        SNEPPX_fm_memory_bank_write(bank, k, v);
        SNEPPX_tensor_destroy(k);
        SNEPPX_tensor_destroy(v);
    }
    SX_ASSERT(bank->num_entries == 3, "capped at 3");
    SNEPPX_fm_memory_bank_destroy(bank);
}

static void test_bank_forget(void) {
    SNEPPXFMMemoryBank* bank = SNEPPX_fm_memory_bank_create(4, 6);
    size_t sh[] = {4};
    for (int i = 0; i < 6; i++) {
        SNEPPXTensor* k = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
        SNEPPXTensor* v = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
        ((float*)k->data)[0] = (float)i;
        ((float*)v->data)[0] = (float)i;
        SNEPPX_fm_memory_bank_write(bank, k, v);
        SNEPPX_tensor_destroy(k);
        SNEPPX_tensor_destroy(v);
    }

    SNEPPXTensor* q = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    ((float*)q->data)[0] = 3.0f;
    for (int j = 0; j < 5; j++) {
        SNEPPXTensor* r = SNEPPX_fm_memory_bank_read(bank, q);
        SNEPPX_tensor_destroy(r);
    }
    SNEPPX_tensor_destroy(q);

    SNEPPX_fm_memory_bank_forget(bank, 0.5f);
    int retained = 0;
    for (size_t i = 0; i < bank->num_entries; i++) {
        float* vd = (float*)bank->values->data + i * 4;
        if (fabsf(vd[0]) > 1e-6f) retained++;
    }
    SX_ASSERT(retained >= 2, "some retained after forget");
    SNEPPX_fm_memory_bank_destroy(bank);
}


TEST(test_fm_memory_bank, test_bank_create) { test_bank_create(); }
TEST(test_fm_memory_bank, test_bank_write_read) { test_bank_write_read(); }
TEST(test_fm_memory_bank, test_bank_overwrite) { test_bank_overwrite(); }
TEST(test_fm_memory_bank, test_bank_eviction) { test_bank_eviction(); }
TEST(test_fm_memory_bank, test_bank_forget) { test_bank_forget(); }
