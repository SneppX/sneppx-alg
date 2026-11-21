#ifndef SNEPPX_INTEGRITY_MONITOR_H
#define SNEPPX_INTEGRITY_MONITOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
/*
 * SNEPPX - Integrity Monitor
 *
 * WHAT
 *   Integrity Monitor.
 *
 * CONCEPT
 *   Provides the Integrity Monitor.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif

typedef enum {
    SNEPPX_MONITOR_EVENT_TEXT_MODIFIED,
    SNEPPX_MONITOR_EVENT_CANARY_TRIGGERED,
    SNEPPX_MONITOR_EVENT_FUNC_PTR_MODIFIED,
    SNEPPX_MONITOR_EVENT_HEAP_CORRUPTION,
    SNEPPX_MONITOR_EVENT_HEARTBEAT_MISS,
    SNEPPX_MONITOR_EVENT_SELF_TAMPER,
    SNEPPX_MONITOR_EVENT_PATTERN_FOUND,
    SNEPPX_MONITOR_EVENT_FREQ_ANOMALY,
} SNEPPXMonitorEventType;

typedef struct {
    SNEPPXMonitorEventType type;
    const char*          description;
    uint64_t             address;
    size_t               size;
    uint64_t             timestamp;
} SNEPPXMonitorEvent;

typedef void (*SNEPPXMonitorCallback)(const SNEPPXMonitorEvent* event);

/**
 * @brief Initialize Monitor.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_monitor_init(void);
/**
 * @brief Perform Monitor Shutdown.
 */
void SNEPPX_monitor_shutdown(void);
/**
 * @brief Start Monitor.
 *
 * @param interval_ms [in] Interval Ms value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_monitor_start(uint64_t interval_ms);
/**
 * @brief Stop Monitor.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_monitor_stop(void);

/**
 * @brief Perform Monitor Register Region.
 *
 * @param name [in] Name value.
 * @param addr [in] Addr value.
 * @param size [in] Size value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_monitor_register_region(const char* name, const void* addr, size_t size);
/**
 * @brief Perform Monitor Unregister Region.
 *
 * @param name [in] Name value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_monitor_unregister_region(const char* name);

/**
 * @brief Perform Monitor Verify All.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_monitor_verify_all(void);
/**
 * @brief Perform Monitor Verify Region.
 *
 * @param name [in] Name value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_monitor_verify_region(const char* name);

/**
 * @brief Perform Monitor Check Canary.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_monitor_check_canary(void);
/**
 * @brief Perform Monitor Refresh Canary.
 */
void SNEPPX_monitor_refresh_canary(void);

/**
 * @brief Perform Monitor Set Callback.
 *
 * @param cb [in] Cb value.
 */
void SNEPPX_monitor_set_callback(SNEPPXMonitorCallback cb);

/**
 * @brief Perform Monitor Freq Analyze.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_monitor_freq_analyze(void);
/**
 * @brief Reset Monitor Freq.
 */
void SNEPPX_monitor_freq_reset(void);

/**
 * @brief Perform Monitor Timing Set Baseline.
 *
 * @param mean [in] Mean value.
 * @param stddev [in] Stddev value.
 */
void SNEPPX_monitor_timing_set_baseline(double mean, double stddev);
/**
 * @brief Perform Monitor Timing Check.
 *
 * @param elapsed_us [in] Elapsed Us value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_monitor_timing_check(uint64_t elapsed_us);

/**
 * @brief Perform Monitor Api Hook Check.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_monitor_api_hook_check(void);
/**
 * @brief Perform Monitor Api Hook Enable.
 *
 * @param base [in] Base value.
 * @param size [in] Size value.
 */
void SNEPPX_monitor_api_hook_enable(const void* base, size_t size);

/**
 * @brief Perform Monitor Syscall Track.
 *
 * @param syscall_num [in] Syscall Num value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_monitor_syscall_track(int syscall_num);
/**
 * @brief Perform Monitor Syscall Analyze.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_monitor_syscall_analyze(void);
/**
 * @brief Perform Monitor Syscall Learn Baseline.
 */
void SNEPPX_monitor_syscall_learn_baseline(void);
/**
 * @brief Perform Monitor Syscall Enable.
 */
void SNEPPX_monitor_syscall_enable(void);

/**
 * @brief Perform Monitor Verify Single Region.
 *
 * @param name [in] Name value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_monitor_verify_single_region(const char* name);
/**
 * @brief Perform Monitor Set Canary.
 *
 * @param depth [in] Depth value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_monitor_set_canary(int depth);
/**
 * @brief Perform Monitor Check Canary At.
 *
 * @param depth [in] Depth value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_monitor_check_canary_at(int depth);
/**
 * @brief Perform Monitor Get Events.
 *
 * @param buffer [out] Buffer value.
 * @param max [in] Max value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_monitor_get_events(SNEPPXMonitorEvent* buffer, int max);
/**
 * @brief Perform Monitor Scan Memory For Pattern.
 *
 * @param pattern [in] Pattern value.
 * @param pattern_len [in] Pattern Len value.
 * @param start [in] Start value.
 * @param end [in] End value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_monitor_scan_memory_for_pattern(const unsigned char* pattern, size_t pattern_len, const void* start, const void* end);
/**
 * @brief Perform Monitor Set Anomaly Threshold.
 *
 * @param threshold [in] Threshold value.
 */
void SNEPPX_monitor_set_anomaly_threshold(int threshold);
/**
 * @brief Perform Monitor Check Self.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_monitor_check_self(void);
/**
 * @brief Perform Monitor Set Heartbeat.
 *
 * @param interval_ms [in] Interval Ms value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_monitor_set_heartbeat(uint64_t interval_ms);
/**
 * @brief Perform Monitor Add Callback.
 *
 * @param cb [in] Cb value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_monitor_add_callback(SNEPPXMonitorCallback cb);
/**
 * @brief Perform Monitor Remove Callback.
 *
 * @param cb [in] Cb value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_monitor_remove_callback(SNEPPXMonitorCallback cb);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_INTEGRITY_MONITOR_H */
