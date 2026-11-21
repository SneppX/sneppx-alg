#include "concurrent_workload_dispatch.h"
#include "test_gtest.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <intrin.h>
#pragma intrinsic(_InterlockedIncrement)
#endif

/*
 * SNEPPX - Test Thread
 *
 * WHAT
 *   Test Thread.
 *
 * CONCEPT
 *   Provides the Test Thread.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */





/* ── Shared data for counter tests ────────────────────────── */

typedef struct {
    int*   counter;
    int    iterations;
    size_t tid;
} IncArg;

static void inc_func(void* arg) {
    IncArg* a = (IncArg*)arg;
    for (int i = 0; i < a->iterations; i++) {
#ifdef _WIN32
        _InterlockedIncrement((volatile long*)a->counter);
#else
        __sync_add_and_fetch(a->counter, 1);
#endif
    }
}

/* ── Pool create / destroy ────────────────────────────────── */

static void test_pool_create_destroy(void) {
    /*
     * Destroy immediately after creation to exercise the worker startup /
     * condition-variable wait race. Repeat to make lost wakeups reproducible.
     */
    for (int i = 0; i < 32; i++) {
        SNEPPXThreadPool* pool = SNEPPX_threadpool_create(2);
        SX_ASSERT(pool != NULL, "pool created");
        SNEPPX_threadpool_destroy(pool);
    }
}

/* ── Default count at least 2 ─────────────────────────────── */

static void test_default_count(void) {
    size_t n = SNEPPX_threadpool_default_count();
    SX_ASSERT(n >= 2, "default count >= 2");
}

/* ── Submit and execute one task ──────────────────────────── */

static void test_submit_one(void) {
    SNEPPXThreadPool* pool = SNEPPX_threadpool_create(2);
    SX_ASSERT(pool != NULL, "pool created");

    int counter = 0;
    IncArg arg = { &counter, 1000, 0 };
    SNEPPXTask task;
    task.func = inc_func;
    task.arg  = &arg;

    SX_ASSERT(SNEPPX_threadpool_submit(pool, task) == 0, "submit ok");
    SNEPPX_threadpool_wait(pool);
    SX_ASSERT(counter == 1000, "counter == 1000");

    SNEPPX_threadpool_destroy(pool);
}

/* ── Multiple tasks ───────────────────────────────────────── */

static void test_multiple_tasks(void) {
    SNEPPXThreadPool* pool = SNEPPX_threadpool_create(4);
    SX_ASSERT(pool != NULL, "pool created");

    int counter = 0;
    int N = 100;
    for (int i = 0; i < N; i++) {
        IncArg* arg = (IncArg*)malloc(sizeof(IncArg));
        SX_ASSERT(arg != NULL, "arg malloc");
        arg->counter    = &counter;
        arg->iterations = 1000;
        arg->tid        = (size_t)i;
        SNEPPXTask task;
        task.func = inc_func;
        task.arg  = arg;
        SNEPPX_threadpool_submit(pool, task);
    }

    SNEPPX_threadpool_wait(pool);
    SX_ASSERT(counter == N * 1000, "counter == N*1000");

    SNEPPX_threadpool_destroy(pool);
}

/* ── Future ───────────────────────────────────────────────── */

typedef struct {
    int value;
} FutArg;

static void fut_func(void* arg) {
    FutArg* a = (FutArg*)arg;
    a->value = 42;
}

static void test_future(void) {
    SNEPPXThreadPool* pool = SNEPPX_threadpool_create(2);
    SX_ASSERT(pool != NULL, "pool created");

    SNEPPXFuture* fut = SNEPPX_future_create();
    SX_ASSERT(fut != NULL, "future created");

    FutArg arg;
    arg.value = 0;

    SNEPPXTask task;
    task.func = fut_func;
    task.arg  = &arg;

    SX_ASSERT(!SNEPPX_future_is_ready(fut), "future not ready yet");
    SNEPPX_threadpool_submit_future(pool, task, fut);

    SNEPPX_future_wait(fut);
    SX_ASSERT(SNEPPX_future_is_ready(fut), "future ready after wait");
    SX_ASSERT(arg.value == 42, "future task executed");

    SNEPPX_future_destroy(fut);
    SNEPPX_threadpool_destroy(pool);
}

/* ── Future without pool ──────────────────────────────────── */

static void test_future_solo(void) {
    SNEPPXFuture* fut = SNEPPX_future_create();
    SX_ASSERT(fut != NULL, "future created");
    SX_ASSERT(!SNEPPX_future_is_ready(fut), "not ready");

    SNEPPX_future_set_result(fut, (void*)(intptr_t)99);
    SX_ASSERT(SNEPPX_future_is_ready(fut), "ready");
    SX_ASSERT(SNEPPX_future_get_result(fut) == (void*)(intptr_t)99, "result == 99");

    SNEPPX_future_destroy(fut);
}

/* ── Parallel For ─────────────────────────────────────────── */

typedef struct {
    int*  array;
    int   value;
} ForArg;

static void for_func(size_t start, size_t end, void* arg) {
    ForArg* fa = (ForArg*)arg;
    for (size_t i = start; i < end; i++) {
        fa->array[i] = fa->value;
    }
}

static void test_parallel_for(void) {
    SNEPPXThreadPool* pool = SNEPPX_threadpool_create(4);
    SX_ASSERT(pool != NULL, "pool created");

    enum { N = 10000 };
    int* arr = (int*)calloc(N, sizeof(int));
    SX_ASSERT(arr != NULL, "array malloc");

    ForArg arg;
    arg.array = arr;
    arg.value = 7;

    SNEPPX_parallel_for(pool, 0, N, for_func, &arg);
    for (int i = 0; i < N; i++) {
        SX_ASSERT(arr[i] == 7, "array[i] == 7");
    }

    free(arr);
    SNEPPX_threadpool_destroy(pool);
}

/* ── Parallel Reduce (sum) ────────────────────────────────── */

typedef struct {
    int*  array;
    size_t len;
} ReduceArg;

static void sum_func(size_t start, size_t end, void* arg, void* result) {
    ReduceArg* ra = (ReduceArg*)arg;
    int* sum = (int*)result;
    for (size_t i = start; i < end; i++) {
        *sum += ra->array[i];
    }
}

static void combine_int(void* dst, const void* src) {
    *(int*)dst += *(const int*)src;
}

static void test_parallel_reduce(void) {
    SNEPPXThreadPool* pool = SNEPPX_threadpool_create(4);
    SX_ASSERT(pool != NULL, "pool created");

    enum { N = 10000 };
    int* arr = (int*)malloc(N * sizeof(int));
    SX_ASSERT(arr != NULL, "array malloc");

    int expected = 0;
    for (int i = 0; i < N; i++) {
        arr[i] = i + 1;
        expected += arr[i];
    }

    ReduceArg rarg;
    rarg.array = arr;
    rarg.len   = N;

    int result = 0;
    int init   = 0;
    SNEPPX_parallel_reduce(pool, 0, N, &init, sizeof(int),
                         sum_func, combine_int, &result, &rarg);
    SX_ASSERT(result == expected, "reduce sum matches expected");

    free(arr);
    SNEPPX_threadpool_destroy(pool);
}

/* ── Single-threaded pool (1 thread) ──────────────────────── */

static void test_single_thread_pool(void) {
    SNEPPXThreadPool* pool = SNEPPX_threadpool_create(1);
    SX_ASSERT(pool != NULL, "pool created");

    int counter = 0;
    IncArg arg = { &counter, 500, 0 };
    SNEPPXTask task;
    task.func = inc_func;
    task.arg  = &arg;

    SNEPPX_threadpool_submit(pool, task);
    SNEPPX_threadpool_wait(pool);
    SX_ASSERT(counter == 500, "counter == 500");

    SNEPPX_threadpool_destroy(pool);
}

/* ── Many iterations (stress) ─────────────────────────────── */

static void test_stress(void) {
    SNEPPXThreadPool* pool = SNEPPX_threadpool_create(4);
    SX_ASSERT(pool != NULL, "pool created");

    enum { N = 500, ITERS = 500 };
    int* counters = (int*)calloc(N, sizeof(int));
    SX_ASSERT(counters != NULL, "counters malloc");

    for (int i = 0; i < N; i++) {
        IncArg* arg = (IncArg*)malloc(sizeof(IncArg));
        SX_ASSERT(arg != NULL, "arg malloc");
        arg->counter    = &counters[i];
        arg->iterations = ITERS;
        arg->tid        = (size_t)i;
        SNEPPXTask task;
        task.func = inc_func;
        task.arg  = arg;
        SNEPPX_threadpool_submit(pool, task);
    }

    SNEPPX_threadpool_wait(pool);
    for (int i = 0; i < N; i++) {
        SX_ASSERT(counters[i] == ITERS, "counter == ITERS");
    }

    free(counters);
    SNEPPX_threadpool_destroy(pool);
}

/* ── Null-safe destroy ────────────────────────────────────── */

static void test_null_destroy(void) {
    SNEPPX_threadpool_destroy(NULL);
}

/* ─────────────────────────────────────────────────────────── */


TEST(test_thread, pool_create_destroy) { test_pool_create_destroy(); }
TEST(test_thread, default_count) { test_default_count(); }
TEST(test_thread, submit_one) { test_submit_one(); }
TEST(test_thread, multiple_tasks) { test_multiple_tasks(); }
TEST(test_thread, future) { test_future(); }
TEST(test_thread, future_solo) { test_future_solo(); }
TEST(test_thread, parallel_for) { test_parallel_for(); }
TEST(test_thread, parallel_reduce) { test_parallel_reduce(); }
TEST(test_thread, single_thread_pool) { test_single_thread_pool(); }
TEST(test_thread, stress) { test_stress(); }
TEST(test_thread, null_destroy) { test_null_destroy(); }
