#include "cryptographic_random_generator.h"
#include "sha512_hashing_implementation.h"

#ifdef _WIN32
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
/*
 * SNEPPX - Random
 *
 * WHAT
 *   Random.
 *
 * CONCEPT
 *   Provides randomness generation.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


#elif defined(__linux__) || defined(__unix__)
#include <sys/syscall.h>
#include <unistd.h>
#include <fcntl.h>
#elif defined(__APPLE__)
#include <stdlib.h>
#endif

/**
 * @brief Perform Random Bytes.
 *
 * @param buffer [out] Buffer value.
 *
 * @return 0 on success, -1 on error.
 */
/* Test-only deterministic RNG (SHA-512 hash-chain). Disabled by default;
 * enabled via SNEPPX_random_bytes_set_seed for reproducible known-answer tests. */
static int g_test_rng = 0;
static uint8_t g_test_state[64];
static size_t g_test_state_used = 64;

void SNEPPX_random_bytes_set_seed(const uint8_t seed[32]) {
    SNEPPXSHA512Context ctx;
    SNEPPX_sha512_init(&ctx);
    SNEPPX_sha512_update(&ctx, seed, 32);
    SNEPPX_sha512_finish(&ctx, g_test_state);
    g_test_state_used = 64;
    g_test_rng = 1;
}

void SNEPPX_random_bytes_clear_seed(void) {
    g_test_rng = 0;
    g_test_state_used = 64;
}

static int test_rng_generate(uint8_t* out, size_t len) {
    size_t produced = 0;
    while (produced < len) {
        if (g_test_state_used >= 64) {
            SNEPPXSHA512Context ctx;
            SNEPPX_sha512_init(&ctx);
            SNEPPX_sha512_update(&ctx, g_test_state, 64);
            SNEPPX_sha512_finish(&ctx, g_test_state);
            g_test_state_used = 0;
        }
        size_t avail = 64 - g_test_state_used;
        size_t take = (len - produced < avail) ? (len - produced) : avail;
        memcpy(out + produced, g_test_state + g_test_state_used, take);
        g_test_state_used += take;
        produced += take;
    }
    return 0;
}

int SNEPPX_random_bytes(uint8_t* buffer, size_t len) {
    if (!buffer || !len) return -1;
    if (g_test_rng) return test_rng_generate(buffer, len);
#ifdef _WIN32
    NTSTATUS status = BCryptGenRandom(NULL, buffer, (ULONG)len, BCRYPT_USE_SYSTEM_PREFERRED_RNG);
    if (status < 0) return -1;
    return 0;
#elif defined(__linux__)
    long ret = syscall(SYS_getrandom, buffer, len, 0);
    if (ret == (long)len) return 0;
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd < 0) return -1;
    size_t total = 0;
    while (total < len) {
        ssize_t n = read(fd, buffer + total, len - total);
        if (n <= 0) { close(fd); return -1; }
        total += (size_t)n;
    }
    close(fd);
    return 0;
#elif defined(__APPLE__)
    arc4random_buf(buffer, len);
    return 0;
#else
    (void)buffer; (void)len;
    return -1;
#endif
/*
 * SNEPPX - Cryptographic Random Number Wrapper
 *
 * WHAT
 *   Cryptographic Random Number Wrapper.
 *
 * CONCEPT
 *   Public API wrapper routing all random requests through the CSPRNG with auto reseeding.
 *
 * ROLE
 *   Public entry point for all cryptographic randomness used by every crypto module.
 *
 * REFERENCES
 *   None (internal utility).
 */


}

/**
 * @brief Perform Random Uint32.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_random_uint32_checked(uint32_t* value) {
    if (!value) return -1;
    return SNEPPX_random_bytes((uint8_t*)value, sizeof(*value));
}

uint32_t SNEPPX_random_uint32(void) {
    uint32_t value = 0;
    (void)SNEPPX_random_uint32_checked(&value);
    return value;
}

/**
 * @brief Generate a uniformly distributed integer below upper_bound.
 *
 * @param upper_bound Exclusive upper bound; must be non-zero.
 * @param value [out] Receives the generated value on success.
 *
 * @return 0 on success, -1 on invalid arguments or entropy failure.
 */
int SNEPPX_random_uniform_checked(uint32_t upper_bound, uint32_t* value) {
    if (upper_bound == 0 || !value) return -1;
    const uint32_t threshold = (uint32_t)(-upper_bound % upper_bound);
    for (;;) {
        uint32_t random_value = 0;
        if (SNEPPX_random_uint32_checked(&random_value) != 0) return -1;
        if (random_value >= threshold) {
            *value = random_value % upper_bound;
            return 0;
        }
    }
}

/**
 * @brief Compatibility wrapper for bounded random values.
 *
 * Returns 0 if the bound is invalid or the system entropy source fails.
 * Call SNEPPX_random_uniform_checked when the caller must distinguish errors.
 */
uint32_t SNEPPX_random_uniform(uint32_t upper_bound) {
    uint32_t value = 0;
    (void)SNEPPX_random_uniform_checked(upper_bound, &value);
    return value;
}
