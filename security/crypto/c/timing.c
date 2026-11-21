#include "timing_attack_countermeasure.h"
#include "side_channel_resistant_primitives.h"
#include "constant_time_operations.h"
#include "cryptographic_random_generator.h"

#if defined(_MSC_VER)
#include <intrin.h>
/*
 * SNEPPX - Timing
 *
 * WHAT
 *   Timing.
 *
 * CONCEPT
 *   Provides timing attack countermeasures.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


#elif defined(_WIN32)
#include <windows.h>
#elif defined(__x86_64__) || defined(__amd64__)
#include <x86intrin.h>
#elif defined(__aarch64__)
#include <arm_acle.h>
#endif

/**
 * @brief Start Timing.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_timing_start(void) {
#if defined(_MSC_VER)
    __cpuidex((int[]){0,0,0,0}, 0, 0);
    return __rdtsc();
#elif defined(__x86_64__) || defined(__amd64__)
    uint32_t lo, hi;
    __asm__ volatile("cpuid\n\t"
                     "rdtsc\n\t"
                     "mov %%edx, %0\n\t"
                     "mov %%eax, %1\n\t"
                     : "=r"(hi), "=r"(lo)
                     :
                     : "%rax", "%rbx", "%rcx", "%rdx");
    return ((uint64_t)hi << 32) | (uint64_t)lo;
#elif defined(_WIN32)
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (uint64_t)t.QuadPart;
#elif defined(__aarch64__)
    __asm__ volatile("isb" : : : "memory");
    uint64_t t;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(t));
    return t;
#else
    return 0;
#endif
}

/**
 * @brief Perform Timing End.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_timing_end(void) {
#if defined(_MSC_VER)
    unsigned int aux;
    uint64_t t = __rdtscp(&aux);
    __cpuidex((int[]){0,0,0,0}, 0, 0);
    return t;
#elif defined(__x86_64__) || defined(__amd64__)
    uint32_t lo, hi;
    __asm__ volatile("rdtscp\n\t"
                     "mov %%edx, %0\n\t"
                     "mov %%eax, %1\n\t"
                     "cpuid\n\t"
                     : "=r"(hi), "=r"(lo)
                     :
                     : "%rax", "%rbx", "%rcx", "%rdx");
    return ((uint64_t)hi << 32) | (uint64_t)lo;
#elif defined(_WIN32)
    LARGE_INTEGER freq, t;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    return (uint64_t)(t.QuadPart * 1000000000ULL / freq.QuadPart);
#elif defined(__aarch64__)
    uint64_t t;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(t));
    __asm__ volatile("isb" : : : "memory");
    return t;
#else
    return 0;
#endif
}

/**
 * @brief Perform Timing Random Delay.
 *
 * @param min_ns [in] Min Ns value.
 */
void SNEPPX_timing_random_delay(uint32_t min_ns, uint32_t max_ns) {
    if (max_ns <= min_ns) return;
    uint32_t range = max_ns - min_ns;
    uint32_t r;
    SNEPPX_random_bytes((uint8_t*)&r, sizeof(r));
    uint32_t delay = min_ns + (r % range);
    uint64_t start = SNEPPX_timing_start();
    while (1) {
        uint64_t elapsed = SNEPPX_timing_end() - start;
        if (elapsed >= delay) break;
    }
}

/**
 * @brief Perform Timing Safe Equal.
 *
 * @param a [in] A value.
 * @param b [in] B value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_timing_safe_equal(const uint8_t* a, const uint8_t* b, size_t len, uint64_t* timing_ns) {
    uint64_t t0 = SNEPPX_timing_start();
    int result = SNEPPX_ct_equal(a, b, len);
    uint64_t t1 = SNEPPX_timing_end();
    if (timing_ns) {
#if defined(__x86_64__) || defined(__amd64__) || defined(_MSC_VER)
        *timing_ns = (t1 - t0) / 2400;
#else
        *timing_ns = t1 - t0;
#endif
/*
 * SNEPPX - Timing Attack Countermeasures
 *
 * WHAT
 *   Timing Attack Countermeasures.
 *
 * CONCEPT
 *   Defenses against timing attacks exploiting data-dependent execution time.
 *
 * ROLE
 *   Layer S5 side-channel resistance used by all crypto modules handling secret keys.
 *
 * REFERENCES
 *   None (internal utility).
 */


    }
    return result;
}
