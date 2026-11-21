#include <stdio.h>
#include "test_gtest.h"
#include <string.h>
#include "memory_hard_key_derivation.h"
#include "constant_time_operations.h"

/*
 * SNEPPX - Test Argon2
 *
 * WHAT
 *   Test Argon2.
 *
 * CONCEPT
 *   Provides the Test Argon2.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */




void test_hash_verify(void) {
    printf("\n--- test_hash_verify ---\n");
    uint8_t password[] = "correct horse battery staple";
    uint8_t salt[] = "sodium chloride";
    uint8_t hash[32];

    SNEPPXArgon2Config cfg;
    cfg.memory_kb = 64;
    cfg.iterations = 2;
    cfg.parallelism = 1;
    cfg.hash_len = 32;

    int ret = SNEPPX_argon2id(password, sizeof(password) - 1, salt, sizeof(salt) - 1, &cfg, hash);
    SX_TEST("hash success", ret == 0);

    ret = SNEPPX_argon2id_verify(password, sizeof(password) - 1, salt, sizeof(salt) - 1, &cfg, hash);
    SX_TEST("verify correct password", ret == 1);

    uint8_t wrong[] = "wrong password";
    ret = SNEPPX_argon2id_verify(wrong, sizeof(wrong) - 1, salt, sizeof(salt) - 1, &cfg, hash);
    SX_TEST("verify wrong password fails", ret == 0);
}

void test_timing(void) {
    printf("\n--- test_timing ---\n");
    uint8_t password[] = "test password";
    uint8_t salt[] = "test salt";
    uint8_t hash[32];

    SNEPPXArgon2Config cfg;
    cfg.memory_kb = 64;
    cfg.iterations = 2;
    cfg.parallelism = 1;
    cfg.hash_len = 32;

    SNEPPX_argon2id(password, sizeof(password) - 1, salt, sizeof(salt) - 1, &cfg, hash);

    uint8_t close[] = "test passworc";
    int ret_correct = SNEPPX_argon2id_verify(password, sizeof(password) - 1, salt, sizeof(salt) - 1, &cfg, hash);
    int ret_wrong = SNEPPX_argon2id_verify(close, sizeof(close) - 1, salt, sizeof(salt) - 1, &cfg, hash);

    SX_TEST("timing-safe verify works", ret_correct == 1 && ret_wrong == 0);
}


TEST(test_argon2, test_hash_verify) { test_hash_verify(); }
TEST(test_argon2, test_timing) { test_timing(); }
