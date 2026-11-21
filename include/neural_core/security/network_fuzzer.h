#ifndef SNEPPX_NETWORK_FUZZER_H
#define SNEPPX_NETWORK_FUZZER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

/*
 * SNEPPX - Network Fuzzer
 *
 * WHAT
 *   Network Fuzzer.
 *
 * CONCEPT
 *   Provides the Network Fuzzer.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    FUZZ_PROTO_RAW = 0,
    FUZZ_PROTO_TLS,
    FUZZ_PROTO_HTTP,
    FUZZ_PROTO_QUIC,
    FUZZ_PROTO_DNS,
    FUZZ_PROTO_SSH
} fuzz_protocol_t;

typedef enum {
    FUZZ_CRASH_NONE = 0,
    FUZZ_CRASH_SEGFAULT,
    FUZZ_CRASH_ABORT,
    FUZZ_CRASH_TIMEOUT,
    FUZZ_CRASH_OOM,
    FUZZ_CRASH_ASSERTION
} fuzz_crash_type_t;

typedef struct {
    size_t max_input_size;
    int max_mutations;
    int timeout_ms;
    int enable_coverage;
    int enable_crash_detection;
    double mutate_prob_flip;
    double mutate_prob_erase;
    double mutate_prob_insert;
    double mutate_prob_splice;
    double mutate_prob_havoc;
} fuzz_config_t;

typedef struct {
    uint32_t input_crc;
    uint64_t execution_id;
    int crashed;
    fuzz_crash_type_t crash_type;
    int new_coverage;
    uint64_t coverage_edges;
} fuzz_result_t;

typedef struct {
    uint64_t total_executions;
    uint64_t total_crashes;
    uint64_t total_mutations;
    int corpus_size;
    int coverage_edges;
    int coverage_total;
    double coverage_percent;
    double crashes_per_k;
} fuzz_stats_t;

/**
 * @brief Initialize Fuzzer.
 *
 * @param protocol [in] Protocol value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fuzzer_init(fuzz_protocol_t protocol);
/**
 * @brief Perform Fuzzer Add Seed.
 *
 * @param data [in] Data value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fuzzer_add_seed(const uint8_t *data, size_t len);
/**
 * @brief Perform Fuzzer Generate.
 *
 * @param buf [out] Buf value.
 * @param len [out] Len value.
 * @param max_len [in] Max Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fuzzer_generate(uint8_t *buf, size_t *len, size_t max_len);
/**
 * @brief Perform Fuzzer Execute.
 *
 * @param input [in] Input value.
 * @param len [in] Len value.
 * @param result [out] Result value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fuzzer_execute(const uint8_t *input, size_t len, fuzz_result_t *result);
/**
 * @brief Perform Fuzzer Get Stats.
 *
 * @param stats [out] Stats value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fuzzer_get_stats(fuzz_stats_t *stats);
/**
 * @brief Perform Fuzzer Set Config.
 *
 * @param config [in] Config value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fuzzer_set_config(const fuzz_config_t *config);
/**
 * @brief Reset Fuzzer.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fuzzer_reset(void);
/**
 * @brief Get Fuzzer Fuzz Tar.
 *
 * @param target_data [in] Target Data value.
 * @param target_len [in] Target Len value.
 * @param iterations [in] Iterations value.
 * @param results [out] Results value.
 * @param max_results [in] Max Results value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fuzzer_fuzz_target(const uint8_t *target_data, size_t target_len, int iterations, fuzz_result_t *results, int max_results);


#ifdef __cplusplus
}
#endif
#endif
