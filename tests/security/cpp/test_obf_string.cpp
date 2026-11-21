#include "string_obfuscation_technique.h"
#include <cstdio>
#include <cstring>
#include <cassert>
#include <string>
#include <iostream>

/*
 * SNEPPX - Test Obf String
 *
 * WHAT
 *   Test Obf String.
 *
 * CONCEPT
 *   Provides the Test Obf String.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
/** @brief Securely zero a memory buffer. */
void SNEPPX_secure_zero(void* ptr, size_t len);
}

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { printf("  TEST: %s ... ", name); } while(0)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { const std::string fail_message = (msg); printf("FAIL: %s\n", fail_message.c_str()); tests_failed++; } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)

void test_encrypt_decrypt() {
    TEST("encrypt_decrypt");
    SNEPPX::SNEPPXObfStringPool pool;

    std::string plain = "SNEPPX-Algo";
    SNEPPX::SNEPPXObfString crypt = pool.encrypt(plain, 0xA71A75A7);

    ASSERT(crypt.length == plain.size(), "length mismatch");

    char output[64] = {0};
    pool.decrypt(crypt, output, sizeof(output));

    ASSERT(std::string(output) == plain, "decrypted text mismatch");
    PASS();
}

void test_pool() {
    TEST("pool_100_strings");
    SNEPPX::SNEPPXObfStringPool pool;

    std::string strings[100];
    for (int i = 0; i < 100; i++) {
        strings[i] = "test_string_" + std::to_string(i);
        pool.pool_register(strings[i]);
    }

    ASSERT(pool.count() == 100, "pool count mismatch");

    pool.pool_decrypt_all();

    for (int i = 0; i < 100; i++) {
        const char* dec = pool.get_string(i);
        ASSERT(dec != nullptr, "null decrypted string");
        ASSERT(std::string(dec) == strings[i], "decrypted string mismatch at index " + std::to_string(i));
    }

    PASS();
}

void test_wipe() {
    TEST("wipe_after_decrypt");
    SNEPPX::SNEPPXObfStringPool pool;

    std::string plain = "sensitive_data";
    SNEPPX::SNEPPXObfString crypt = pool.encrypt(plain, 0x12345678);

    char output[64] = {0};
    pool.decrypt(crypt, output, sizeof(output));

    ASSERT(std::string(output) == plain, "decrypt before wipe");

    SNEPPX_secure_zero(output, sizeof(output));

    bool all_zero = true;
    for (size_t i = 0; i < plain.size() && i < sizeof(output); i++) {
        if (output[i] != 0) { all_zero = false; break; }
    }

    ASSERT(all_zero, "stack was not zeroed");
    PASS();
}

void test_compile_time_encrypt() {
    TEST("compile_time_encrypt");
    constexpr uint32_t key = 0x42424242;
    constexpr auto encrypted = SNEPPX::SNEPPX_obf_encrypt_literal("TEST", key);

    ASSERT(encrypted[0] == ('T' ^ 0x42), "byte 0 mismatch");
    ASSERT(encrypted[1] == ('E' ^ 0x42), "byte 1 mismatch");
    ASSERT(encrypted[2] == ('S' ^ 0x42), "byte 2 mismatch");
    ASSERT(encrypted[3] == ('T' ^ 0x42), "byte 3 mismatch");
    ASSERT(encrypted[4] == ('\0' ^ 0x42), "null terminator mismatch");

    PASS();
}

int main() {
    printf("\n=== S2 Obfuscation String Tests ===\n\n");

    test_encrypt_decrypt();
    test_pool();
    test_wipe();
    test_compile_time_encrypt();

    printf("\nResults: %d passed, %d failed out of %d\n",
           tests_passed, tests_failed, tests_passed + tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
