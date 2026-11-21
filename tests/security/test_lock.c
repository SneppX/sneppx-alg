#include "synchronization_lock_interface.h"
#include "test_gtest.h"
#include <stdio.h>

/*
 * SNEPPX - Test Lock
 *
 * WHAT
 *   Test Lock.
 *
 * CONCEPT
 *   Provides the Test Lock.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */





static void test_lock_init_destroy(void) {
    SNEPPXLock* lock = SNEPPX_lock_init();
    SX_ASSERT(lock != NULL, "lock initialized");
    SNEPPX_lock_destroy(lock);
}

static void test_lock_acquire_release(void) {
    SNEPPXLock* lock = SNEPPX_lock_init();
    SNEPPX_lock_acquire(lock);
    SX_ASSERT(SNEPPX_lock_is_held(lock) == 1, "lock acquired");
    SNEPPX_lock_release(lock);
    SNEPPX_lock_destroy(lock);
}

static void test_lock_try_acquire(void) {
    SNEPPXLock* lock = SNEPPX_lock_init();
    int ok = SNEPPX_lock_try_acquire(lock);
    SX_ASSERT(ok == 0 || ok == 1, "try_acquire returns bool");
    SNEPPX_lock_release(lock);
    SNEPPX_lock_destroy(lock);
}


TEST(test_lock, lock_init_destroy) { test_lock_init_destroy(); }
TEST(test_lock, lock_acquire_release) { test_lock_acquire_release(); }
TEST(test_lock, lock_try_acquire) { test_lock_try_acquire(); }
