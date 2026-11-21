#ifndef SNEPPX_CHAOS_FAULT_INJECTION_H
#define SNEPPX_CHAOS_FAULT_INJECTION_H

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_CHAOS_MAX_EXPERIMENTS 256
#define SNEPPX_CHAOS_MAX_TARGETS 1024
#define SNEPPX_CHAOS_MAX_PROBES 64
#define SNEPPX_CHAOS_HYSTRIX_CIRCUIT_BREAKER_THRESHOLD 0.5f

/*
 * SNEPPX - Fault Injection
 *
 * WHAT
 *   Fault Injection.
 *
 * CONCEPT
 *   Provides the Fault Injection.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_CHAOS_FAULT_POD_KILL,
    SNEPPX_CHAOS_FAULT_CONTAINER_KILL,
    SNEPPX_CHAOS_FAULT_NETWORK_DELAY,
    SNEPPX_CHAOS_FAULT_NETWORK_PACKET_LOSS,
    SNEPPX_CHAOS_FAULT_NETWORK_PARTITION,
    SNEPPX_CHAOS_FAULT_CPU_STRESS,
    SNEPPX_CHAOS_FAULT_MEMORY_STRESS,
    SNEPPX_CHAOS_FAULT_DISK_IO_STRESS,
    SNEPPX_CHAOS_FAULT_DISK_FILL,
    SNEPPX_CHAOS_FAULT_PROCESS_KILL,
    SNEPPX_CHAOS_FAULT_SERVICE_DISRUPTION,
    SNEPPX_CHAOS_FAULT_DNS_FAILURE,
    SNEPPX_CHAOS_FAULT_HTTP_ERROR,
    SNEPPX_CHAOS_FAULT_CERTIFICATE_EXPIRY,
    SNEPPX_CHAOS_FAULT_DATABASE_FAILOVER,
    SNEPPX_CHAOS_FAULT_NODE_DRAIN,
    SNEPPX_CHAOS_FAULT_CLOCK_SKEW
} SNEPPXChaosFaultType;

typedef enum {
    SNEPPX_CHAOS_EXPERIMENT_RUNNING,
    SNEPPX_CHAOS_EXPERIMENT_PAUSED,
    SNEPPX_CHAOS_EXPERIMENT_STOPPED,
    SNEPPX_CHAOS_EXPERIMENT_COMPLETED,
    SNEPPX_CHAOS_EXPERIMENT_FAILED,
    SNEPPX_CHAOS_EXPERIMENT_ABORTED
} SNEPPXChaosExperimentStatus;

typedef enum {
    SNEPPX_CHAOS_HYSTRIX_CLOSED,
    SNEPPX_CHAOS_HYSTRIX_OPEN,
    SNEPPX_CHAOS_HYSTRIX_HALF_OPEN
} SNEPPXChaosCircuitBreakerStatus;

typedef struct {
    SNEPPXChaosFaultType fault_type;
    uint8_t* target_selector;
    size_t target_selector_len;
    uint32_t duration_seconds;
    uint32_t intensity_percent;
    uint32_t delay_ms;
    uint32_t packet_loss_percent;
    uint8_t* additional_params;
    size_t additional_params_len;
    uint8_t* namespace;
    size_t namespace_len;
    uint8_t* labels;
    size_t labels_len;
} SNEPPXChaosFault;

typedef struct {
    char* name;
    size_t name_len;
    uint8_t* experiment_id;
    size_t experiment_id_len;
    SNEPPXChaosFault* faults;
    uint32_t num_faults;
    SNEPPXChaosExperimentStatus status;
    uint64_t started_at_ns;
    uint64_t completed_at_ns;
    uint32_t steady_state_hits;
    uint32_t steady_state_tolerance;
    uint8_t* rollback_plan;
    size_t rollback_plan_len;
    uint8_t* hypothesis;
    size_t hypothesis_len;
    uint8_t auto_rollback : 1;
    uint8_t dry_run : 1;
    uint8_t stop_on_failure : 1;
} SNEPPXChaosExperiment;

typedef struct {
    char* probe_name;
    size_t probe_name_len;
    uint8_t* endpoint;
    size_t endpoint_len;
    uint32_t interval_ms;
    uint32_t timeout_ms;
    uint32_t success_threshold;
    uint32_t failure_threshold;
    uint8_t* expected_response;
    size_t expected_response_len;
    uint8_t* expected_status_codes;
    size_t status_codes_len;
    uint64_t last_success_ns;
    uint64_t last_failure_ns;
    uint32_t consecutive_failures;
    uint8_t active : 1;
    uint8_t passing : 1;
} SNEPPXChaosProbe;

typedef struct {
    SNEPPXChaosCircuitBreakerStatus hystrix_status;
    uint64_t last_open_ns;
    uint64_t last_half_open_ns;
    uint32_t failure_count;
    uint32_t success_count;
    uint64_t open_duration_ns;
    uint64_t half_open_max_requests;
    uint64_t half_open_current_requests;
    float failure_threshold;
    float success_threshold;
} SNEPPXChaosCircuitBreaker;

int snepx_chaos_experiment_create(SNEPPXChaosExperiment* exp, const char* name, const uint8_t* hypothesis, size_t hypothesis_len);
int snepx_chaos_experiment_add_fault(SNEPPXChaosExperiment* exp, const SNEPPXChaosFault* fault);
int snepx_chaos_experiment_add_probe(SNEPPXChaosExperiment* exp, const SNEPPXChaosProbe* probe);
int snepx_chaos_experiment_run(SNEPPXChaosExperiment* exp);
int snepx_chaos_experiment_pause(SNEPPXChaosExperiment* exp);
int snepx_chaos_experiment_stop(SNEPPXChaosExperiment* exp);
int snepx_chaos_experiment_rollback(SNEPPXChaosExperiment* exp);
int snepx_chaos_experiment_validate_hypothesis(SNEPPXChaosExperiment* exp, uint8_t* hypothesis_valid);
int snepx_chaos_experiment_report(SNEPPXChaosExperiment* exp, uint8_t* report_out, size_t* report_len);

int snepx_chaos_inject_fault(SNEPPXChaosFault* fault);
int snepx_chaos_remove_fault(SNEPPXChaosFault* fault);
int snepx_chaos_network_delay(const char* target, uint32_t delay_ms);
int snepx_chaos_network_packet_loss(const char* target, uint32_t loss_percent);
int snepx_chaos_network_partition(const char** targets, uint32_t num_targets, uint32_t duration_seconds);
int snepx_chaos_cpu_stress(uint32_t cpu_cores, uint32_t duration_seconds);
int snepx_chaos_memory_stress(uint64_t memory_mb, uint32_t duration_seconds);
int snepx_chaos_disk_fill(uint64_t size_mb, const char* path);
int snepx_chaos_process_kill(const char* process_name);
int snepx_chaos_dns_failure(const char* domain);

int snepx_chaos_circuit_breaker_init(SNEPPXChaosCircuitBreaker* cb, float failure_threshold, float success_threshold, uint64_t open_duration_ns);
int snepx_chaos_circuit_breaker_record_success(SNEPPXChaosCircuitBreaker* cb);
int snepx_chaos_circuit_breaker_record_failure(SNEPPXChaosCircuitBreaker* cb);
uint8_t snepx_chaos_circuit_breaker_is_open(SNEPPXChaosCircuitBreaker* cb);

int snepx_chaos_probe_execute(SNEPPXChaosProbe* probe);
int snepx_chaos_probe_http_get(SNEPPXChaosProbe* probe, const char* url);
int snepx_chaos_probe_tcp_connect(SNEPPXChaosProbe* probe, const char* host, uint16_t port);

int snepx_chaos_experiment_destroy(SNEPPXChaosExperiment* exp);

#endif
