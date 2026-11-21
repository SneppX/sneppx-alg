#include "integrity_monitor.h"
#include "test_gtest.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/*
 * SNEPPX - Test Integrity Monitor
 *
 * WHAT
 *   Test Integrity Monitor.
 *
 * CONCEPT
 *   Provides the Test Integrity Monitor.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static int g_event_count = 0;



static void event_callback(const SNEPPXMonitorEvent* event) {
    (void)event;
    g_event_count++;
}

static void test_init_shutdown(void) {
    int ret = SNEPPX_monitor_init();
    SX_ASSERT(ret == 0, "monitor init");
    SNEPPX_monitor_shutdown();
}

static void test_start_stop(void) {
    SNEPPX_monitor_init();
    int ret = SNEPPX_monitor_start(100);
    SX_ASSERT(ret == 0, "monitor start");
    ret = SNEPPX_monitor_stop();
    SX_ASSERT(ret == 0, "monitor stop");
    SNEPPX_monitor_shutdown();
}

static void test_region_registration(void) {
    SNEPPX_monitor_init();
    int data[] = {1, 2, 3, 4, 5};
    int ret = SNEPPX_monitor_register_region("test_data", data, sizeof(data));
    SX_ASSERT(ret == 0, "region registered");

    ret = SNEPPX_monitor_verify_region("test_data");
    SX_ASSERT(ret == 0, "region verified unchanged");

    ret = SNEPPX_monitor_unregister_region("test_data");
    SX_ASSERT(ret == 0, "region unregistered");

    ret = SNEPPX_monitor_verify_region("test_data");
    SX_ASSERT(ret == -1, "unregistered region returns -1");
    SNEPPX_monitor_shutdown();
}

static void test_verify_all(void) {
    SNEPPX_monitor_init();
    int data[] = {10, 20, 30};
    SNEPPX_monitor_register_region("verify_data", data, sizeof(data));
    int violations = SNEPPX_monitor_verify_all();
    SX_ASSERT(violations == 0, "verify all returns 0 violations");
    SNEPPX_monitor_unregister_region("verify_data");
    SNEPPX_monitor_shutdown();
}

static void test_canary(void) {
    SNEPPX_monitor_init();
    int ok = SNEPPX_monitor_check_canary();
    SX_ASSERT(ok != 0, "canary check passes initially");
    SNEPPX_monitor_refresh_canary();
    ok = SNEPPX_monitor_check_canary();
    SX_ASSERT(ok != 0, "canary check passes after refresh");
    SNEPPX_monitor_shutdown();
}

static void test_callback(void) {
    SNEPPX_monitor_init();
    SNEPPX_monitor_set_callback(event_callback);
    g_event_count = 0;

    /* Register a small region and modify it to trigger an event */
    char data[] = "original data";
    SNEPPX_monitor_register_region("callback_data", data, sizeof(data));

    /* Modify the data */
    data[0] = 'X';

    int violations = SNEPPX_monitor_verify_all();
    SX_ASSERT(violations > 0, "violations detected after modification");
    SX_ASSERT(g_event_count > 0, "callback fired");
    SNEPPX_monitor_unregister_region("callback_data");
    SNEPPX_monitor_shutdown();
}


TEST(test_integrity_monitor, init_shutdown) { test_init_shutdown(); }
TEST(test_integrity_monitor, start_stop) { test_start_stop(); }
TEST(test_integrity_monitor, region_registration) { test_region_registration(); }
TEST(test_integrity_monitor, verify_all) { test_verify_all(); }
TEST(test_integrity_monitor, canary) { test_canary(); }
TEST(test_integrity_monitor, callback) { test_callback(); }
