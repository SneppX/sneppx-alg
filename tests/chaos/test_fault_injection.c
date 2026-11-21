#include "security/chaos/fault_injection.h"
#include <assert.h>
#include <string.h>

/*
 * SNEPPX - Test Fault Injection
 *
 * WHAT
 *   Test Fault Injection.
 *
 * CONCEPT
 *   Provides the Test Fault Injection.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static void test_experiment_create(void) {
    SNEPPXChaosExperiment exp;
    memset(&exp, 0, sizeof(exp));
    int ret = snepx_chaos_experiment_create(&exp, "network_test", "Network should survive packet loss", 37);
    assert(ret == 0);
    assert(exp.status == SNEPPX_CHAOS_EXPERIMENT_RUNNING);
    snepx_chaos_experiment_destroy(&exp);
}

static void test_add_fault(void) {
    SNEPPXChaosExperiment exp;
    snepx_chaos_experiment_create(&exp, "fault_test", "Fault injection test", 20);
    SNEPPXChaosFault fault;
    memset(&fault, 0, sizeof(fault));
    fault.fault_type = SNEPPX_CHAOS_FAULT_NETWORK_DELAY;
    fault.duration_seconds = 30;
    fault.delay_ms = 200;
    int ret = snepx_chaos_experiment_add_fault(&exp, &fault);
    assert(ret == 0);
    assert(exp.num_faults == 1);
    snepx_chaos_experiment_destroy(&exp);
}

static void test_circuit_breaker(void) {
    SNEPPXChaosCircuitBreaker cb;
    snepx_chaos_circuit_breaker_init(&cb, 0.5f, 0.8f, 60000000000ULL);
    assert(cb.hystrix_status == SNEPPX_CHAOS_HYSTRIX_CLOSED);
    for (int i = 0; i < 10; i++) {
        snepx_chaos_circuit_breaker_record_failure(&cb);
    }
    assert(snepx_chaos_circuit_breaker_is_open(&cb) == 1);
}

int main(void) {
    test_experiment_create();
    test_add_fault();
    test_circuit_breaker();
    return 0;
}
