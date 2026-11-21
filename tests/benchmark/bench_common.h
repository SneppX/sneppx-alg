#ifndef SNEPPX_BENCH_COMMON_H
#define SNEPPX_BENCH_COMMON_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

/*
 * SNEPPX - Bench Common
 *
 * WHAT
 *   Bench Common.
 *
 * CONCEPT
 *   Provides the Bench Common.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef struct {
    double min_time;
    double max_time;
    double total_time;
    double mean_time;
    long long iterations;
} BenchStats;

#ifdef _WIN32
static double get_time_sec(void) {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart;
}
#else
static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
#endif

#define BENCH_INIT(name) BenchStats name = {1e30, 0, 0, 0, 0}

#define BENCH_START(bench, runs, warmup, code_block) do { \
    for (int _w = 0; _w < (warmup); _w++) { code_block } \
    bench.total_time = 0; \
    bench.min_time = 1e30; \
    bench.max_time = 0; \
    bench.iterations = (runs); \
    for (long long _r = 0; _r < (runs); _r++) { \
        double _t0 = get_time_sec(); \
        { code_block } \
        double _t1 = get_time_sec(); \
        double _dt = _t1 - _t0; \
        bench.total_time += _dt; \
        if (_dt < bench.min_time) bench.min_time = _dt; \
        if (_dt > bench.max_time) bench.max_time = _dt; \
    } \
    bench.mean_time = bench.total_time / bench.iterations; \
} while(0)

static void bench_print(const char* name, BenchStats* s) {
    printf("  %-35s %8.3f ms  (min %8.3f  max %8.3f  runs %lld)\n",
           name, s->mean_time * 1000.0,
           s->min_time * 1000.0, s->max_time * 1000.0,
           (long long)s->iterations);
}

typedef struct {
    const char* name;
    void (*fn)(void);
} BenchEntry;

static int g_bench_passed = 0, g_bench_failed = 0, g_bench_skipped = 0;

#define BENCH_RUN(name, fn) do { \
    printf("Bench %s...\n", name); \
    fn(); \
    g_bench_passed++; \
} while(0)

#define BENCH_MAIN() do { \
    printf("\nResults: %d passed, %d failed, %d skipped\n", \
           g_bench_passed, g_bench_failed, g_bench_skipped); \
    return g_bench_failed > 0 ? 1 : 0; \
} while(0)

#endif
