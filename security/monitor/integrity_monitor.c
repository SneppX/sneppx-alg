#include "integrity_monitor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#define SNEPPX_MONITOR_MAX_REGIONS 64
#define SNEPPX_MONITOR_CANARY_SIZE 8
#define SNEPPX_MONITOR_MAX_FREQ_SAMPLES 256
#define SNEPPX_MONITOR_MAX_SYSCALL_TABLE 512
#define SNEPPX_MONITOR_MAX_CANARIES 16
#define SNEPPX_MONITOR_EVENT_LOG_SIZE 256
#define SNEPPX_MONITOR_MAX_CALLBACKS 4
#define SNEPPX_MONITOR_SELF_PTR_COUNT 16

#define SNEPPX_MONITOR_ALERT_LOW 0
#define SNEPPX_MONITOR_ALERT_MEDIUM 1
#define SNEPPX_MONITOR_ALERT_HIGH 2
#define SNEPPX_MONITOR_ALERT_CRITICAL 3

#define SNEPPX_MONITOR_SLIDING_WINDOW_SIZE 32
#define SNEPPX_MONITOR_EVENT_TYPE_COUNT 16

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


typedef struct {
    char name[128];
    const void* addr;
    size_t size;
    uint32_t baseline_crc;
    int active;
} MonitoredRegion;

typedef struct {
    SNEPPXMonitorEventType type;
    char description[256];
    uint64_t address;
    size_t size;
    uint64_t timestamp;
    int severity;
} EventLogEntry;

typedef struct {
    void* func_ptrs[SNEPPX_MONITOR_SELF_PTR_COUNT];
    uint32_t magic;
    char version[8];
} SelfCheckBlock;

typedef struct {
    int check_count;
    uint64_t last_check_time;
    uint64_t min_interval_us;
    uint64_t max_interval_us;
} RegionFreqTrack;

typedef struct {
    uint64_t timestamp;
    int event_type;
    int count;
} FreqSample;

static SNEPPXMonitorCallback g_callbacks[SNEPPX_MONITOR_MAX_CALLBACKS];
static int g_callback_count = 0;

static int g_monitor_running = 0;
static uint64_t g_monitor_interval_ms = 0;

static MonitoredRegion g_regions[SNEPPX_MONITOR_MAX_REGIONS];
static int g_region_count = 0;

static unsigned char g_canaries[SNEPPX_MONITOR_MAX_CANARIES][SNEPPX_MONITOR_CANARY_SIZE];
static unsigned char g_canary_checks[SNEPPX_MONITOR_MAX_CANARIES][SNEPPX_MONITOR_CANARY_SIZE];
static int g_canary_count = 1;
static unsigned long g_prng_state = 0xDEADBEEF;

static EventLogEntry g_event_log[SNEPPX_MONITOR_EVENT_LOG_SIZE];
static int g_event_log_head = 0;
static int g_event_log_count = 0;

static SelfCheckBlock g_self_block;
static uint32_t g_self_crc_baseline = 0;

static uint64_t g_heartbeat_interval_ms = 0;
static uint64_t g_last_heartbeat_time = 0;

static RegionFreqTrack g_region_freq[SNEPPX_MONITOR_MAX_REGIONS];

static FreqSample g_freq_samples[SNEPPX_MONITOR_MAX_FREQ_SAMPLES];
static int g_freq_sample_count = 0;
static int g_freq_baseline[4];
static int g_anomaly_threshold = 5;

static uint64_t g_last_verify_time = 0;
static double g_timing_baseline = 0.0;
static double g_timing_stddev = 1.0;
static int g_timing_samples = 0;

static const void* g_module_base = NULL;
static size_t g_module_size = 0;
static int g_api_hook_check_enabled = 0;

static int g_syscall_counts[SNEPPX_MONITOR_MAX_SYSCALL_TABLE];
static int g_syscall_baseline[SNEPPX_MONITOR_MAX_SYSCALL_TABLE];
static int g_syscall_enabled = 0;

static int g_auto_learn_enabled = 0;
static uint64_t g_auto_learn_start = 0;
static uint64_t g_auto_learn_duration = 0;
static int g_sensitivity_level = 5;
static int g_anomaly_total_count = 0;
static uint64_t g_verify_interval_ms = 0;
static uint64_t g_last_verify_called = 0;
static int g_adaptive_threshold_enabled = 0;

static double g_sliding_window[SNEPPX_MONITOR_SLIDING_WINDOW_SIZE];
static int g_sliding_window_index = 0;
static int g_sliding_window_count = 0;

static int g_event_type_counts[SNEPPX_MONITOR_EVENT_TYPE_COUNT];
static int g_event_type_baselines[SNEPPX_MONITOR_EVENT_TYPE_COUNT];

static int g_alert_severity_levels[SNEPPX_MONITOR_EVENT_TYPE_COUNT];

static uint64_t g_monitor_start_time = 0;

static char g_monitor_instance_name[64] = {0};

static int g_event_filter[SNEPPX_MONITOR_EVENT_TYPE_COUNT];

static uint64_t g_bulk_verify_index = 0;

/**
 * @brief Perform Prng Next.
 *
 * @return The result value, or 0 on error.
 */
unsigned long SNEPPX_prng_next(void);

/**
 * @brief Initialize Monitor.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_init(void);
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
int SNEPPX_monitor_start(uint64_t interval_ms);
/**
 * @brief Stop Monitor.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_stop(void);
/**
 * @brief Perform Monitor Register Region.
 *
 * @param name [in] Name value.
 * @param addr [in] Addr value.
 * @param size [in] Size value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_register_region(const char* name, const void* addr, size_t size);
/**
 * @brief Perform Monitor Unregister Region.
 *
 * @param name [in] Name value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_unregister_region(const char* name);
/**
 * @brief Perform Monitor Verify All.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_verify_all(void);
/**
 * @brief Perform Monitor Verify Region.
 *
 * @param name [in] Name value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_verify_region(const char* name);
/**
 * @brief Perform Monitor Check Canary.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_check_canary(void);
/**
 * @brief Perform Monitor Refresh Canary.
 */
void SNEPPX_monitor_refresh_canary(void);
/**
 * @brief Perform Monitor Freq Analyze.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_freq_analyze(void);
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
int SNEPPX_monitor_timing_check(uint64_t elapsed_us);
/**
 * @brief Perform Monitor Api Hook Check.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_api_hook_check(void);
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
int SNEPPX_monitor_syscall_track(int syscall_num);
/**
 * @brief Perform Monitor Syscall Analyze.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_syscall_analyze(void);
/**
 * @brief Perform Monitor Syscall Learn Baseline.
 */
void SNEPPX_monitor_syscall_learn_baseline(void);
/**
 * @brief Perform Monitor Syscall Enable.
 */
void SNEPPX_monitor_syscall_enable(void);
static unsigned long prng_next(void) {
    g_prng_state = g_prng_state * 1103515245UL + 12345UL;
    return g_prng_state;
}

/**
 * @brief Perform Prng Next.
 *
 * @return The result value, or 0 on error.
 */
unsigned long SNEPPX_prng_next(void) {
    return prng_next();
}

static uint32_t crc32_c(const void* data, size_t len) {
    const unsigned char* buf = (const unsigned char*)data;
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; i++) {
        crc ^= buf[i];
        for (int j = 0; j < 8; j++)
            crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
    }
    return ~crc;
}

static void push_event_log(SNEPPXMonitorEventType type, const char* desc, uint64_t addr, size_t size) {
    EventLogEntry* e = &g_event_log[g_event_log_head];
    e->type = type;
    strncpy(e->description, desc, sizeof(e->description) - 1);
    e->description[sizeof(e->description) - 1] = '\0';
    e->address = addr;
    e->size = size;
    e->timestamp = (uint64_t)time(NULL);
    e->severity = g_alert_severity_levels[(int)type % SNEPPX_MONITOR_EVENT_TYPE_COUNT];
    g_event_log_head = (g_event_log_head + 1) % SNEPPX_MONITOR_EVENT_LOG_SIZE;
    if (g_event_log_count < SNEPPX_MONITOR_EVENT_LOG_SIZE)
        g_event_log_count++;
    g_anomaly_total_count++;
}

static void fire_event(SNEPPXMonitorEventType type, const char* desc, uint64_t addr, size_t size) {
    int etype_idx = (int)type % SNEPPX_MONITOR_EVENT_TYPE_COUNT;
    if (g_event_filter[etype_idx] == 0) return;
    push_event_log(type, desc, addr, size);
    g_event_type_counts[etype_idx]++;
    if (g_callback_count == 0) return;
    SNEPPXMonitorEvent ev;
    ev.type = type;
    ev.description = desc;
    ev.address = addr;
    ev.size = size;
    ev.timestamp = (uint64_t)time(NULL);
    for (int i = 0; i < g_callback_count; i++) {
        if (g_callbacks[i])
            g_callbacks[i](&ev);
    }
}

static void populate_self_block(void) {
    g_self_block.func_ptrs[0] = (void*)SNEPPX_monitor_init;
    g_self_block.func_ptrs[1] = (void*)SNEPPX_monitor_shutdown;
    g_self_block.func_ptrs[2] = (void*)SNEPPX_monitor_start;
    g_self_block.func_ptrs[3] = (void*)SNEPPX_monitor_stop;
    g_self_block.func_ptrs[4] = (void*)SNEPPX_monitor_register_region;
    g_self_block.func_ptrs[5] = (void*)SNEPPX_monitor_unregister_region;
    g_self_block.func_ptrs[6] = (void*)SNEPPX_monitor_verify_all;
    g_self_block.func_ptrs[7] = (void*)SNEPPX_monitor_verify_region;
    g_self_block.func_ptrs[8] = (void*)SNEPPX_monitor_check_canary;
    g_self_block.func_ptrs[9] = (void*)SNEPPX_monitor_refresh_canary;
    g_self_block.func_ptrs[10] = (void*)SNEPPX_monitor_freq_analyze;
    g_self_block.func_ptrs[11] = (void*)SNEPPX_monitor_freq_reset;
    g_self_block.func_ptrs[12] = (void*)SNEPPX_monitor_timing_set_baseline;
    g_self_block.func_ptrs[13] = (void*)SNEPPX_monitor_timing_check;
    g_self_block.func_ptrs[14] = (void*)SNEPPX_monitor_api_hook_check;
    g_self_block.func_ptrs[15] = (void*)SNEPPX_monitor_syscall_enable;
    g_self_block.magic = 0xA5A5A5A5;
    strncpy(g_self_block.version, "v3.2", sizeof(g_self_block.version) - 1);
    g_self_block.version[sizeof(g_self_block.version) - 1] = '\0';
}

static void freq_record_sample(int event_type) {
    if (g_freq_sample_count >= SNEPPX_MONITOR_MAX_FREQ_SAMPLES) {
        memmove(g_freq_samples, g_freq_samples + 1,
                (SNEPPX_MONITOR_MAX_FREQ_SAMPLES - 1) * sizeof(FreqSample));
        g_freq_sample_count--;
    }
    FreqSample* s = &g_freq_samples[g_freq_sample_count++];
    s->timestamp = (uint64_t)time(NULL);
    s->event_type = event_type;
    s->count = ++g_freq_baseline[event_type % 4];
}

/**
 * @brief Perform Monitor Freq Analyze.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_freq_analyze(void) {
    if (g_freq_sample_count < 10) return 0;
    int recent[4] = {0};
    int start = (g_freq_sample_count > 20) ? g_freq_sample_count - 20 : 0;
    for (int i = start; i < g_freq_sample_count; i++) {
        recent[g_freq_samples[i].event_type % 4]++;
    }
    int anomalies = 0;
    int effective_threshold = g_adaptive_threshold_enabled ? (g_anomaly_threshold > 2 ? g_anomaly_threshold - 1 : 1) : g_anomaly_threshold;
    for (int i = 0; i < 4; i++) {
        if (g_freq_baseline[i] > 0) {
            double ratio = (double)recent[i] / (double)g_freq_baseline[i];
            if (ratio > (double)effective_threshold || ratio < 1.0 / (double)effective_threshold) {
                char desc[256];
                snprintf(desc, sizeof(desc), "Frequency anomaly event type %d: ratio %.2f", i, ratio);
                fire_event(SNEPPX_MONITOR_EVENT_TEXT_MODIFIED, desc, 0, 0);
                anomalies++;
            }
        }
    }
    return anomalies;
}

/**
 * @brief Reset Monitor Freq.
 */
void SNEPPX_monitor_freq_reset(void) {
    memset(g_freq_baseline, 0, sizeof(g_freq_baseline));
    g_freq_sample_count = 0;
}

/**
 * @brief Perform Monitor Timing Set Baseline.
 *
 * @param mean [in] Mean value.
 */
void SNEPPX_monitor_timing_set_baseline(double mean, double stddev) {
    g_timing_baseline = mean;
    g_timing_stddev = (stddev > 0.001) ? stddev : 1.0;
}

/**
 * @brief Perform Monitor Timing Check.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_timing_check(uint64_t elapsed_us) {
    if (g_timing_samples < 5) {
        g_timing_samples++;
        return 0;
    }
    if (g_sliding_window_count < SNEPPX_MONITOR_SLIDING_WINDOW_SIZE) {
        g_sliding_window[g_sliding_window_count++] = (double)elapsed_us;
    } else {
        g_sliding_window[g_sliding_window_index] = (double)elapsed_us;
        g_sliding_window_index = (g_sliding_window_index + 1) % SNEPPX_MONITOR_SLIDING_WINDOW_SIZE;
    }
    double sum = 0;
    int n = g_sliding_window_count > 0 ? g_sliding_window_count : 1;
    for (int i = 0; i < g_sliding_window_count; i++) sum += g_sliding_window[i];
    double avg = sum / n;
    double variance = 0;
    for (int i = 0; i < g_sliding_window_count; i++) {
        double diff = g_sliding_window[i] - avg;
        variance += diff * diff;
    }
    double effective_stddev = (n > 1) ? sqrt(variance / (n - 1)) : g_timing_stddev;
    if (effective_stddev < 0.001) effective_stddev = g_timing_stddev;
    double z = (fabs((double)elapsed_us - g_timing_baseline)) / effective_stddev;
    if (z > 3.0) {
        char desc[256];
        snprintf(desc, sizeof(desc), "Timing anomaly: z-score %.2f (elapsed %llu us, baseline %.0f us)",
                 z, (unsigned long long)elapsed_us, g_timing_baseline);
        fire_event(SNEPPX_MONITOR_EVENT_CANARY_TRIGGERED, desc, 0, 0);
        return 1;
    }
    return 0;
}

/**
 * @brief Perform Monitor Api Hook Check.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_api_hook_check(void) {
    if (!g_api_hook_check_enabled) return 0;
    if (!g_module_base || g_module_size == 0) return -1;
    uint32_t crc = crc32_c(g_module_base, g_module_size);
    if (crc != g_regions[0].baseline_crc) {
        fire_event(SNEPPX_MONITOR_EVENT_FUNC_PTR_MODIFIED,
                   "API hooking detected: module CRC mismatch",
                   (uint64_t)(uintptr_t)g_module_base, g_module_size);
        return 1;
    }
    return 0;
}

/**
 * @brief Perform Monitor Api Hook Enable.
 *
 * @param base [in] Base value.
 */
void SNEPPX_monitor_api_hook_enable(const void* base, size_t size) {
    g_module_base = base;
    g_module_size = size;
    g_api_hook_check_enabled = 1;
}

/**
 * @brief Perform Monitor Syscall Track.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_syscall_track(int syscall_num) {
    if (!g_syscall_enabled) return 0;
    if (syscall_num >= 0 && syscall_num < SNEPPX_MONITOR_MAX_SYSCALL_TABLE) {
        g_syscall_counts[syscall_num]++;
    }
    return 0;
}

/**
 * @brief Perform Monitor Syscall Analyze.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_syscall_analyze(void) {
    if (!g_syscall_enabled) return 0;
    int anomalies = 0;
    for (int i = 0; i < 200; i++) {
        if (g_syscall_baseline[i] > 0) {
            double ratio = (double)g_syscall_counts[i] / (double)g_syscall_baseline[i];
            if (ratio > 10.0 || (ratio < 0.1 && g_syscall_baseline[i] > 5)) {
                char desc[256];
                snprintf(desc, sizeof(desc), "Syscall anomaly #%d: ratio %.2f", i, ratio);
                fire_event(SNEPPX_MONITOR_EVENT_HEAP_CORRUPTION, desc, (uint64_t)i, 0);
                anomalies++;
            }
        }
    }
    return anomalies;
}

/**
 * @brief Perform Monitor Syscall Learn Baseline.
 */
void SNEPPX_monitor_syscall_learn_baseline(void) {
    for (int i = 0; i < SNEPPX_MONITOR_MAX_SYSCALL_TABLE; i++)
        g_syscall_baseline[i] = g_syscall_counts[i];
}

/**
 * @brief Perform Monitor Syscall Enable.
 */
void SNEPPX_monitor_syscall_enable(void) {
    g_syscall_enabled = 1;
}
/**
 * @brief Initialize Monitor.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_init(void) {
    g_prng_state = (unsigned long)time(NULL) ^ 0x7F3C5A1B;
    g_region_count = 0;
    g_callback_count = 0;
    memset(g_callbacks, 0, sizeof(g_callbacks));
    memset(g_regions, 0, sizeof(g_regions));
    memset(g_canaries, 0, sizeof(g_canaries));
    memset(g_canary_checks, 0, sizeof(g_canary_checks));
    memset(g_freq_baseline, 0, sizeof(g_freq_baseline));
    memset(g_syscall_counts, 0, sizeof(g_syscall_counts));
    memset(g_syscall_baseline, 0, sizeof(g_syscall_baseline));
    memset(g_region_freq, 0, sizeof(g_region_freq));
    memset(g_event_log, 0, sizeof(g_event_log));
    g_freq_sample_count = 0;
    g_timing_samples = 0;
    g_api_hook_check_enabled = 0;
    g_syscall_enabled = 0;
    g_last_verify_time = 0;
    g_anomaly_threshold = 5;
    g_event_log_head = 0;
    g_event_log_count = 0;
    g_heartbeat_interval_ms = 0;
    g_last_heartbeat_time = 0;
    g_canary_count = 1;
    g_self_crc_baseline = 0;
    g_auto_learn_enabled = 0;
    g_auto_learn_start = 0;
    g_auto_learn_duration = 0;
    g_sensitivity_level = 5;
    g_anomaly_total_count = 0;
    g_verify_interval_ms = 0;
    g_last_verify_called = 0;
    g_adaptive_threshold_enabled = 0;
    g_sliding_window_index = 0;
    g_sliding_window_count = 0;
    memset(g_sliding_window, 0, sizeof(g_sliding_window));
    memset(g_event_type_counts, 0, sizeof(g_event_type_counts));
    memset(g_event_type_baselines, 0, sizeof(g_event_type_baselines));
    memset(g_monitor_instance_name, 0, sizeof(g_monitor_instance_name));
    for (int i = 0; i < SNEPPX_MONITOR_EVENT_TYPE_COUNT; i++) {
        g_alert_severity_levels[i] = SNEPPX_MONITOR_ALERT_MEDIUM;
        g_event_filter[i] = 1;
    }
    g_monitor_start_time = (uint64_t)time(NULL);
    g_bulk_verify_index = 0;
    populate_self_block();
    g_self_crc_baseline = crc32_c(&g_self_block, sizeof(g_self_block));
    SNEPPX_monitor_refresh_canary();
    return 0;
}
/**
 * @brief Perform Monitor Shutdown.
 */
void SNEPPX_monitor_shutdown(void) {
    SNEPPX_monitor_stop();
    memset(g_callbacks, 0, sizeof(g_callbacks));
    g_callback_count = 0;
    g_region_count = 0;
    g_api_hook_check_enabled = 0;
    g_syscall_enabled = 0;
    g_heartbeat_interval_ms = 0;
    g_last_heartbeat_time = 0;
    g_event_log_head = 0;
    g_event_log_count = 0;
    g_canary_count = 0;
    g_self_crc_baseline = 0;
    memset(g_regions, 0, sizeof(g_regions));
    memset(g_region_freq, 0, sizeof(g_region_freq));
    memset(g_event_log, 0, sizeof(g_event_log));
}

/**
 * @brief Start Monitor.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_start(uint64_t interval_ms) {
    if (g_monitor_running) return 0;
    g_monitor_interval_ms = interval_ms;
    g_monitor_running = 1;
    return 0;
}

/**
 * @brief Stop Monitor.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_stop(void) {
    g_monitor_running = 0;
    return 0;
}

/**
 * @brief Perform Monitor Register Region.
 *
 * @param name [in] Name value.
 * @param addr [in] Addr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_register_region(const char* name, const void* addr, size_t size) {
    if (!name || !addr || size == 0 || g_region_count >= SNEPPX_MONITOR_MAX_REGIONS) return -1;
    MonitoredRegion* reg = &g_regions[g_region_count];
    strncpy(reg->name, name, sizeof(reg->name) - 1);
    reg->name[sizeof(reg->name) - 1] = '\0';
    reg->addr = addr;
    reg->size = size;
    reg->baseline_crc = crc32_c(addr, size);
    reg->active = 1;
    g_region_freq[g_region_count].check_count = 0;
    g_region_freq[g_region_count].last_check_time = 0;
    g_region_freq[g_region_count].min_interval_us = 0;
    g_region_freq[g_region_count].max_interval_us = 0;
    g_region_count++;
    return 0;
}

/**
 * @brief Perform Monitor Unregister Region.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_unregister_region(const char* name) {
    if (!name) return -1;
    for (int i = 0; i < g_region_count; i++) {
        if (g_regions[i].active && strcmp(g_regions[i].name, name) == 0) {
            g_regions[i].active = 0;
            g_region_freq[i].check_count = 0;
            return 0;
        }
    }
    return -1;
}

/**
 * @brief Perform Monitor Verify All.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_verify_all(void) {
    int violations = 0;
    uint64_t now_us = (uint64_t)time(NULL) * 1000000;

    if (g_verify_interval_ms > 0) {
        uint64_t now_ms = (uint64_t)time(NULL) * 1000;
        if (now_ms - g_last_verify_called < g_verify_interval_ms)
            return 0;
        g_last_verify_called = now_ms;
    }

    if (g_heartbeat_interval_ms > 0) {
        uint64_t now_ms = (uint64_t)time(NULL) * 1000;
        if (g_last_heartbeat_time > 0) {
            uint64_t elapsed_heartbeat = now_ms - g_last_heartbeat_time;
            if (elapsed_heartbeat > g_heartbeat_interval_ms) {
                char desc[256];
                snprintf(desc, sizeof(desc), "Heartbeat miss: %llu ms since last verify (interval %llu ms)",
                         (unsigned long long)elapsed_heartbeat, (unsigned long long)g_heartbeat_interval_ms);
                fire_event(SNEPPX_MONITOR_EVENT_HEARTBEAT_MISS, desc, 0, 0);
                violations++;
            }
        }
        g_last_heartbeat_time = now_ms;
    }

    if (g_last_verify_time > 0) {
        uint64_t elapsed = now_us - g_last_verify_time;
        SNEPPX_monitor_timing_check(elapsed);
        freq_record_sample(0);
    }
    g_last_verify_time = now_us;

    for (int i = 0; i < g_region_count; i++) {
        if (!g_regions[i].active) continue;
        g_region_freq[i].check_count++;
        uint64_t interval = (g_region_freq[i].last_check_time > 0) ?
            (now_us - g_region_freq[i].last_check_time) : 0;
        if (interval > 0) {
            if (g_region_freq[i].min_interval_us == 0 || interval < g_region_freq[i].min_interval_us)
                g_region_freq[i].min_interval_us = interval;
            if (interval > g_region_freq[i].max_interval_us)
                g_region_freq[i].max_interval_us = interval;
        }
        g_region_freq[i].last_check_time = now_us;
        if (interval > 0 && interval < 100 && g_region_freq[i].check_count > 5) {
            char desc[256];
            snprintf(desc, sizeof(desc), "Anti-fuzzing: region '%s' checked too often (interval %llu us)",
                     g_regions[i].name, (unsigned long long)interval);
            fire_event(SNEPPX_MONITOR_EVENT_FREQ_ANOMALY, desc, 0, 0);
            violations++;
        }
        if (g_region_freq[i].check_count > 50 && g_last_verify_time > 0) {
            uint64_t total_span = now_us - g_last_verify_time;
            if (total_span > 0) {
                double checks_per_sec = (double)g_region_freq[i].check_count / ((double)total_span / 1000000.0);
                if (checks_per_sec < 0.01 && g_region_freq[i].check_count > 0) {
                    char desc[256];
                    snprintf(desc, sizeof(desc), "Evasion suspect: region '%s' checked only %.2f/sec (%d checks)",
                             g_regions[i].name, checks_per_sec, g_region_freq[i].check_count);
                    fire_event(SNEPPX_MONITOR_EVENT_FREQ_ANOMALY, desc, 0, 0);
                    violations++;
                }
            }
        }
        uint32_t current = crc32_c(g_regions[i].addr, g_regions[i].size);
        if (current != g_regions[i].baseline_crc) {
            char desc[256];
            snprintf(desc, sizeof(desc), "Region '%s' modified: CRC mismatch (expected 0x%08X, got 0x%08X)",
                     g_regions[i].name, g_regions[i].baseline_crc, current);
            fire_event(SNEPPX_MONITOR_EVENT_TEXT_MODIFIED, desc,
                       (uint64_t)(uintptr_t)g_regions[i].addr, g_regions[i].size);
            violations++;
        }
    }

    if (!violations) {
        int canary_ok = SNEPPX_monitor_check_canary();
        if (!canary_ok) {
            fire_event(SNEPPX_MONITOR_EVENT_CANARY_TRIGGERED, "Stack canary corrupted", 0, 0);
            violations++;
        }
    }

    violations += SNEPPX_monitor_freq_analyze();
    violations += SNEPPX_monitor_api_hook_check();

    if (g_auto_learn_enabled) {
        uint64_t now = (uint64_t)time(NULL);
        if (now - g_auto_learn_start < g_auto_learn_duration) {
            for (int i = 0; i < g_region_count; i++) {
                if (g_regions[i].active) {
                    uint32_t current = crc32_c(g_regions[i].addr, g_regions[i].size);
                    if (current != g_regions[i].baseline_crc) {
                        g_regions[i].baseline_crc = current;
                    }
                }
            }
            memcpy(g_event_type_baselines, g_event_type_counts, sizeof(g_event_type_baselines));
        } else {
            g_auto_learn_enabled = 0;
        }
    }

    return violations;
}

/**
 * @brief Perform Monitor Verify Region.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_verify_region(const char* name) {
    if (!name) return -1;
    for (int i = 0; i < g_region_count; i++) {
        if (!g_regions[i].active) continue;
        if (strcmp(g_regions[i].name, name) == 0) {
            uint32_t current = crc32_c(g_regions[i].addr, g_regions[i].size);
            if (current != g_regions[i].baseline_crc) {
                char desc[256];
                snprintf(desc, sizeof(desc), "Region '%s' modified", name);
                fire_event(SNEPPX_MONITOR_EVENT_TEXT_MODIFIED, desc,
                           (uint64_t)(uintptr_t)g_regions[i].addr, g_regions[i].size);
                return 1;
            }
            return 0;
        }
    }
    return -1;
}

/**
 * @brief Perform Monitor Check Canary.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_check_canary(void) {
    return memcmp(g_canaries[0], g_canary_checks[0], SNEPPX_MONITOR_CANARY_SIZE) == 0;
}

/**
 * @brief Perform Monitor Refresh Canary.
 */
void SNEPPX_monitor_refresh_canary(void) {
    for (size_t i = 0; i < SNEPPX_MONITOR_CANARY_SIZE; i++) {
        g_canaries[0][i] = (unsigned char)(prng_next() & 0xFF);
        g_canary_checks[0][i] = g_canaries[0][i];
    }
    for (int d = 1; d < g_canary_count; d++) {
        for (size_t i = 0; i < SNEPPX_MONITOR_CANARY_SIZE; i++) {
            g_canaries[d][i] = (unsigned char)(prng_next() & 0xFF);
            g_canary_checks[d][i] = g_canaries[d][i];
        }
    }
}

/**
 * @brief Perform Monitor Set Callback.
 */
void SNEPPX_monitor_set_callback(SNEPPXMonitorCallback cb) {
    g_callbacks[0] = cb;
    g_callback_count = (cb != NULL) ? 1 : 0;
}

/**
 * @brief Perform Monitor Verify Single Region.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_verify_single_region(const char* name) {
    if (!name) return -1;
    for (int i = 0; i < g_region_count; i++) {
        if (!g_regions[i].active) continue;
        if (strcmp(g_regions[i].name, name) == 0) {
            uint32_t current = crc32_c(g_regions[i].addr, g_regions[i].size);
            uint32_t expected = g_regions[i].baseline_crc;
            char desc[256];
            if (current != expected) {
                snprintf(desc, sizeof(desc), "Region '%s' VERIFY FAIL: CRC 0x%08X != baseline 0x%08X (addr=%p, size=%zu)",
                         name, current, expected, g_regions[i].addr, g_regions[i].size);
                fire_event(SNEPPX_MONITOR_EVENT_TEXT_MODIFIED, desc,
                           (uint64_t)(uintptr_t)g_regions[i].addr, g_regions[i].size);
                return 1;
            }
            snprintf(desc, sizeof(desc), "Region '%s' VERIFY OK: CRC 0x%08X (addr=%p, size=%zu)",
                     name, current, g_regions[i].addr, g_regions[i].size);
            push_event_log(SNEPPX_MONITOR_EVENT_TEXT_MODIFIED, desc,
                           (uint64_t)(uintptr_t)g_regions[i].addr, g_regions[i].size);
            return 0;
        }
    }
    return -1;
}

/**
 * @brief Perform Monitor Set Canary.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_set_canary(int depth) {
    if (depth < 0 || depth >= SNEPPX_MONITOR_MAX_CANARIES) return -1;
    if (depth >= g_canary_count)
        g_canary_count = depth + 1;
    for (size_t i = 0; i < SNEPPX_MONITOR_CANARY_SIZE; i++) {
        g_canaries[depth][i] = (unsigned char)(prng_next() & 0xFF);
        g_canary_checks[depth][i] = g_canaries[depth][i];
    }
    return 0;
}

/**
 * @brief Perform Monitor Check Canary At.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_check_canary_at(int depth) {
    if (depth < 0 || depth >= g_canary_count) return -1;
    return memcmp(g_canaries[depth], g_canary_checks[depth], SNEPPX_MONITOR_CANARY_SIZE) == 0;
}

/**
 * @brief Perform Monitor Get Events.
 *
 * @param buffer [out] Buffer value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_events(SNEPPXMonitorEvent* buffer, int max) {
    if (!buffer || max <= 0) return 0;
    int to_copy = (max < g_event_log_count) ? max : g_event_log_count;
    int start = (g_event_log_head - g_event_log_count + SNEPPX_MONITOR_EVENT_LOG_SIZE) % SNEPPX_MONITOR_EVENT_LOG_SIZE;
    for (int i = 0; i < to_copy; i++) {
        int idx = (start + i) % SNEPPX_MONITOR_EVENT_LOG_SIZE;
        buffer[i].type = g_event_log[idx].type;
        buffer[i].description = g_event_log[idx].description;
        buffer[i].address = g_event_log[idx].address;
        buffer[i].size = g_event_log[idx].size;
        buffer[i].timestamp = g_event_log[idx].timestamp;
    }
    return to_copy;
}

/**
 * @brief Perform Monitor Scan Memory For Pattern.
 *
 * @param pattern [in] Pattern value.
 * @param pattern_len [in] Pattern Len value.
 * @param start [in] Start value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_scan_memory_for_pattern(const unsigned char* pattern, size_t pattern_len, const void* start, const void* end) {
    if (!pattern || pattern_len == 0 || !start || !end || start >= end) return -1;
    const unsigned char* p = (const unsigned char*)start;
    const unsigned char* last = (const unsigned char*)end - pattern_len;
    int found = 0;
    while (p <= last) {
        if (memcmp(p, pattern, pattern_len) == 0) {
            char desc[256];
            snprintf(desc, sizeof(desc), "Pattern found at %p (len %zu)", (const void*)p, pattern_len);
            fire_event(SNEPPX_MONITOR_EVENT_PATTERN_FOUND, desc, (uint64_t)(uintptr_t)p, pattern_len);
            found++;
            p += pattern_len;
        } else {
            p++;
        }
    }
    return found;
}

/**
 * @brief Perform Monitor Set Anomaly Threshold.
 */
void SNEPPX_monitor_set_anomaly_threshold(int threshold) {
    if (threshold < 1) threshold = 1;
    g_anomaly_threshold = threshold;
}

/**
 * @brief Perform Monitor Check Self.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_check_self(void) {
    SelfCheckBlock check_block;
    memcpy(&check_block, &g_self_block, sizeof(SelfCheckBlock));
    uint32_t current_crc = crc32_c(&check_block, sizeof(check_block));
    if (g_self_crc_baseline == 0) return -1;
    if (current_crc != g_self_crc_baseline) {
        char desc[256];
        snprintf(desc, sizeof(desc), "Self-integrity FAIL: CRC 0x%08X != baseline 0x%08X",
                 current_crc, g_self_crc_baseline);
        fire_event(SNEPPX_MONITOR_EVENT_SELF_TAMPER, desc, 0, 0);
        return 1;
    }
    return 0;
}

/**
 * @brief Perform Monitor Set Heartbeat.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_set_heartbeat(uint64_t interval_ms) {
    g_heartbeat_interval_ms = interval_ms;
    g_last_heartbeat_time = (uint64_t)time(NULL) * 1000;
    return 0;
}

/**
 * @brief Perform Monitor Add Callback.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_add_callback(SNEPPXMonitorCallback cb) {
    if (!cb) return -1;
    if (g_callback_count >= SNEPPX_MONITOR_MAX_CALLBACKS) return -1;
    for (int i = 0; i < g_callback_count; i++) {
        if (g_callbacks[i] == cb) return 0;
    }
    g_callbacks[g_callback_count++] = cb;
    return 0;
}

/**
 * @brief Perform Monitor Remove Callback.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_remove_callback(SNEPPXMonitorCallback cb) {
    if (!cb) return -1;
    for (int i = 0; i < g_callback_count; i++) {
        if (g_callbacks[i] == cb) {
            g_callbacks[i] = g_callbacks[g_callback_count - 1];
            g_callbacks[g_callback_count - 1] = NULL;
            g_callback_count--;
            return 0;
        }
    }
    return -1;
}

/**
 * @brief Perform Monitor Auto Learn.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_auto_learn(uint64_t seconds) {
    if (seconds == 0) {
        g_auto_learn_enabled = 0;
        return 0;
    }
    g_auto_learn_enabled = 1;
    g_auto_learn_start = (uint64_t)time(NULL);
    g_auto_learn_duration = seconds;
    return 0;
}

/**
 * @brief Perform Monitor Set Sensitivity.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_set_sensitivity(int level) {
    if (level < 0) level = 0;
    if (level > 10) level = 10;
    g_sensitivity_level = level;
    g_anomaly_threshold = 11 - level;
    if (g_anomaly_threshold < 1) g_anomaly_threshold = 1;
    return 0;
}

/**
 * @brief Perform Monitor Get Anomaly Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_anomaly_count(void) {
    return g_anomaly_total_count;
}

/**
 * @brief Perform Monitor Export Log.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_export_log(const char* path) {
    if (!path) return -1;
    FILE* f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "{\"event_log\":[\n");
    int start = (g_event_log_head - g_event_log_count + SNEPPX_MONITOR_EVENT_LOG_SIZE) % SNEPPX_MONITOR_EVENT_LOG_SIZE;
    for (int i = 0; i < g_event_log_count; i++) {
        int idx = (start + i) % SNEPPX_MONITOR_EVENT_LOG_SIZE;
        fprintf(f, "{\"type\":%d,\"desc\":\"%s\",\"addr\":%llu,\"size\":%zu,\"ts\":%llu,\"sev\":%d}%s\n",
                (int)g_event_log[idx].type, g_event_log[idx].description,
                (unsigned long long)g_event_log[idx].address, g_event_log[idx].size,
                (unsigned long long)g_event_log[idx].timestamp, g_event_log[idx].severity,
                (i < g_event_log_count - 1) ? "," : "");
    }
    fprintf(f, "]}\n");
    fclose(f);
    return 0;
}

/**
 * @brief Perform Monitor Set Verify Interval.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_set_verify_interval(uint64_t ms) {
    g_verify_interval_ms = ms;
    g_last_verify_called = (uint64_t)time(NULL) * 1000;
    return 0;
}

/**
 * @brief Perform Monitor Set Adaptive Threshold.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_set_adaptive_threshold(int enabled) {
    g_adaptive_threshold_enabled = (enabled != 0);
    return 0;
}

/**
 * @brief Perform Monitor Check Module Integrity.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_check_module_integrity(const char* module_name) {
    if (!module_name) return -1;
    for (int i = 0; i < g_region_count; i++) {
        if (g_regions[i].active && strcmp(g_regions[i].name, module_name) == 0) {
            uint32_t current = crc32_c(g_regions[i].addr, g_regions[i].size);
            if (current != g_regions[i].baseline_crc) {
                char desc[256];
                snprintf(desc, sizeof(desc), "Module '%s' integrity fail", module_name);
                fire_event(SNEPPX_MONITOR_EVENT_TEXT_MODIFIED, desc,
                           (uint64_t)(uintptr_t)g_regions[i].addr, g_regions[i].size);
                return 1;
            }
            return 0;
        }
    }
    return -1;
}
static void sliding_window_reset(void) {
    g_sliding_window_index = 0;
    g_sliding_window_count = 0;
    memset(g_sliding_window, 0, sizeof(g_sliding_window));
}

static int event_type_get_count(int type) {
    if (type < 0 || type >= SNEPPX_MONITOR_EVENT_TYPE_COUNT) return 0;
    return g_event_type_counts[type];
}

static int event_type_get_baseline(int type) {
    if (type < 0 || type >= SNEPPX_MONITOR_EVENT_TYPE_COUNT) return 0;
    return g_event_type_baselines[type];
}

static void reset_event_type_counts(void) {
    memset(g_event_type_counts, 0, sizeof(g_event_type_counts));
}

static void reset_event_type_baselines(void) {
    memset(g_event_type_baselines, 0, sizeof(g_event_type_baselines));
}

/**
 * @brief Perform Monitor Get Region Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_region_count(void) {
    return g_region_count;
}

/**
 * @brief Perform Monitor Get Active Region Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_active_region_count(void) {
    int count = 0;
    for (int i = 0; i < g_region_count; i++) {
        if (g_regions[i].active) count++;
    }
    return count;
}

/**
 * @brief Perform Monitor Get Event Log Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_event_log_count(void) {
    return g_event_log_count;
}

/**
 * @brief Perform Monitor Clear Event Log.
 */
void SNEPPX_monitor_clear_event_log(void) {
    memset(g_event_log, 0, sizeof(g_event_log));
    g_event_log_head = 0;
    g_event_log_count = 0;
}

/**
 * @brief Perform Monitor Get Last Verify Time.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_monitor_get_last_verify_time(void) {
    return g_last_verify_time;
}

/**
 * @brief Perform Monitor Get Anomaly Threshold.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_anomaly_threshold(void) {
    return g_anomaly_threshold;
}

/**
 * @brief Perform Monitor Get Sensitivity.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_sensitivity(void) {
    return g_sensitivity_level;
}

/**
 * @brief Perform Monitor Is Auto Learning.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_is_auto_learning(void) {
    return g_auto_learn_enabled;
}

/**
 * @brief Perform Monitor Get Verify Interval.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_monitor_get_verify_interval(void) {
    return g_verify_interval_ms;
}

/**
 * @brief Perform Monitor Is Adaptive Threshold.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_is_adaptive_threshold(void) {
    return g_adaptive_threshold_enabled ? 1 : 0;
}

/**
 * @brief Perform Monitor Event Type String.
 *
 * @return Pointer on success, NULL on error.
 */
const char* SNEPPX_monitor_event_type_string(SNEPPXMonitorEventType type) {
    switch (type) {
        case SNEPPX_MONITOR_EVENT_TEXT_MODIFIED: return "TEXT_MODIFIED";
        case SNEPPX_MONITOR_EVENT_CANARY_TRIGGERED: return "CANARY_TRIGGERED";
        case SNEPPX_MONITOR_EVENT_FUNC_PTR_MODIFIED: return "FUNC_PTR_MODIFIED";
        case SNEPPX_MONITOR_EVENT_HEAP_CORRUPTION: return "HEAP_CORRUPTION";
        case SNEPPX_MONITOR_EVENT_HEARTBEAT_MISS: return "HEARTBEAT_MISS";
        case SNEPPX_MONITOR_EVENT_SELF_TAMPER: return "SELF_TAMPER";
        case SNEPPX_MONITOR_EVENT_PATTERN_FOUND: return "PATTERN_FOUND";
        case SNEPPX_MONITOR_EVENT_FREQ_ANOMALY: return "FREQ_ANOMALY";
        default: return "UNKNOWN";
    }
}

/**
 * @brief Perform Monitor Set Severity Level.
 *
 * @param type [in] Type value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_set_severity_level(SNEPPXMonitorEventType type, int severity) {
    int idx = (int)type % SNEPPX_MONITOR_EVENT_TYPE_COUNT;
    if (severity < SNEPPX_MONITOR_ALERT_LOW) severity = SNEPPX_MONITOR_ALERT_LOW;
    if (severity > SNEPPX_MONITOR_ALERT_CRITICAL) severity = SNEPPX_MONITOR_ALERT_CRITICAL;
    g_alert_severity_levels[idx] = severity;
    return 0;
}

/**
 * @brief Perform Monitor Get Severity Level.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_severity_level(SNEPPXMonitorEventType type) {
    int idx = (int)type % SNEPPX_MONITOR_EVENT_TYPE_COUNT;
    return g_alert_severity_levels[idx];
}
static void auto_learn_adjust_threshold(void) {
    if (!g_auto_learn_enabled) return;
    int total_events = g_anomaly_total_count;
    if (total_events < 10) return;
    double rate = (double)total_events / (double)((uint64_t)time(NULL) - g_auto_learn_start + 1);
    if (rate > 10.0) {
        g_anomaly_threshold = g_anomaly_threshold < 10 ? g_anomaly_threshold + 1 : 10;
    } else if (rate < 1.0) {
        g_anomaly_threshold = g_anomaly_threshold > 2 ? g_anomaly_threshold - 1 : 1;
    }
}

static void auto_learn_update_baselines(void) {
    if (!g_auto_learn_enabled) return;
    for (int i = 0; i < g_region_count; i++) {
        if (g_regions[i].active) {
            uint32_t current = crc32_c(g_regions[i].addr, g_regions[i].size);
            if (current != g_regions[i].baseline_crc) {
                g_regions[i].baseline_crc = current;
            }
        }
    }
    memcpy(g_event_type_baselines, g_event_type_counts, sizeof(g_event_type_baselines));
    memcpy(g_syscall_baseline, g_syscall_counts, sizeof(g_syscall_baseline));
}

static void auto_learn_finalize(void) {
    g_auto_learn_enabled = 0;
    auto_learn_update_baselines();
}

/**
 * @brief Perform Monitor Get Region Crc.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_region_crc(const char* name) {
    if (!name) return -1;
    for (int i = 0; i < g_region_count; i++) {
        if (g_regions[i].active && strcmp(g_regions[i].name, name) == 0)
            return (int)g_regions[i].baseline_crc;
    }
    return -1;
}

/**
 * @brief Perform Monitor Region Is Active.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_region_is_active(const char* name) {
    if (!name) return 0;
    for (int i = 0; i < g_region_count; i++) {
        if (strcmp(g_regions[i].name, name) == 0)
            return g_regions[i].active ? 1 : 0;
    }
    return 0;
}

/**
 * @brief Perform Monitor Get Region Check Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_region_check_count(const char* name) {
    if (!name) return -1;
    for (int i = 0; i < g_region_count; i++) {
        if (strcmp(g_regions[i].name, name) == 0)
            return g_region_freq[i].check_count;
    }
    return -1;
}

/**
 * @brief Perform Monitor Get Heartbeat Interval.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_monitor_get_heartbeat_interval(void) {
    return g_heartbeat_interval_ms;
}

/**
 * @brief Perform Monitor Is Running.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_is_running(void) {
    return g_monitor_running ? 1 : 0;
}

/**
 * @brief Perform Monitor Get Interval Ms.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_monitor_get_interval_ms(void) {
    return g_monitor_interval_ms;
}

/**
 * @brief Perform Monitor Get Uptime Seconds.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_monitor_get_uptime_seconds(void) {
    if (g_monitor_start_time == 0) return 0;
    return (uint64_t)time(NULL) - g_monitor_start_time;
}

/**
 * @brief Perform Monitor Reset Region.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_reset_region(const char* name) {
    if (!name) return -1;
    for (int i = 0; i < g_region_count; i++) {
        if (strcmp(g_regions[i].name, name) == 0) {
            g_regions[i].baseline_crc = crc32_c(g_regions[i].addr, g_regions[i].size);
            g_region_freq[i].check_count = 0;
            g_region_freq[i].last_check_time = 0;
            return 0;
        }
    }
    return -1;
}

/**
 * @brief Perform Monitor Set Start Time.
 */
void SNEPPX_monitor_set_start_time(void) {
    g_monitor_start_time = (uint64_t)time(NULL);
}
/**
 * @brief Perform Monitor Reset Anomaly Count.
 */
void SNEPPX_monitor_reset_anomaly_count(void) { g_anomaly_total_count = 0; }
/**
 * @brief Perform Monitor Get Syscall Enabled.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_syscall_enabled(void) { return g_syscall_enabled; }
/**
 * @brief Perform Monitor Set Prng State.
 */
void SNEPPX_monitor_set_prng_state(unsigned long s) { g_prng_state = s; }
/**
 * @brief Perform Monitor Get Prng State.
 *
 * @return The result value, or 0 on error.
 */
unsigned long SNEPPX_monitor_get_prng_state(void) { return g_prng_state; }

/**
 * @brief Perform Monitor Export Json.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_export_json(const char* path) {
    if (!path) return -1;
    FILE* f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "{\"monitor_instance\":\"%s\",\"uptime\":%llu,\"event_count\":%d,\"anomaly_count\":%d,\"region_count\":%d,\"event_log\":[\n",
            g_monitor_instance_name,
            (unsigned long long)SNEPPX_monitor_get_uptime_seconds(),
            g_anomaly_total_count, g_anomaly_total_count, g_region_count);
    int start = (g_event_log_head - g_event_log_count + SNEPPX_MONITOR_EVENT_LOG_SIZE) % SNEPPX_MONITOR_EVENT_LOG_SIZE;
    for (int i = 0; i < g_event_log_count; i++) {
        int idx = (start + i) % SNEPPX_MONITOR_EVENT_LOG_SIZE;
        fprintf(f, "{\"type\":%d,\"desc\":\"%s\",\"addr\":%llu,\"size\":%zu,\"ts\":%llu,\"sev\":%d}%s\n",
                (int)g_event_log[idx].type, g_event_log[idx].description,
                (unsigned long long)g_event_log[idx].address, g_event_log[idx].size,
                (unsigned long long)g_event_log[idx].timestamp, g_event_log[idx].severity,
                (i < g_event_log_count - 1) ? "," : "");
    }
    fprintf(f, "],\"regions\":[");
    for (int i = 0; i < g_region_count; i++) {
        if (!g_regions[i].active) continue;
        fprintf(f, "{\"name\":\"%s\",\"addr\":%p,\"size\":%zu,\"crc\":0x%08X}%s",
                g_regions[i].name, g_regions[i].addr, g_regions[i].size, g_regions[i].baseline_crc,
                (i < g_region_count - 1) ? "," : "");
    }
    fprintf(f, "]}\n");
    fclose(f);
    return 0;
}

/**
 * @brief Perform Monitor Set Name.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_set_name(const char* name) {
    if (!name) return -1;
    strncpy(g_monitor_instance_name, name, sizeof(g_monitor_instance_name) - 1);
    g_monitor_instance_name[sizeof(g_monitor_instance_name) - 1] = '\0';
    return 0;
}

/**
 * @brief Perform Monitor Get Uptime.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_monitor_get_uptime(void) {
    return SNEPPX_monitor_get_uptime_seconds();
}

/**
 * @brief Perform Monitor Get Event Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_event_count(void) {
    return g_anomaly_total_count;
}

/**
 * @brief Perform Monitor Set Event Filter.
 *
 * @param type [in] Type value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_set_event_filter(SNEPPXMonitorEventType type, int enabled) {
    int idx = (int)type % SNEPPX_MONITOR_EVENT_TYPE_COUNT;
    g_event_filter[idx] = (enabled != 0) ? 1 : 0;
    return 0;
}

/**
 * @brief Perform Monitor Timing Set Window.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_timing_set_window(size_t size) {
    if (size < 2) size = 2;
    if (size > SNEPPX_MONITOR_SLIDING_WINDOW_SIZE) size = SNEPPX_MONITOR_SLIDING_WINDOW_SIZE;
    sliding_window_reset();
    return 0;
}

/**
 * @brief Perform Monitor Timing Get Baseline.
 *
 * @param mean [out] Mean value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_timing_get_baseline(double* mean, double* stddev) {
    if (!mean||!stddev) return -1;
    *mean = g_timing_baseline;
    *stddev = g_timing_stddev;
    return 0;
}

/**
 * @brief Perform Monitor Freq Get Threshold.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_freq_get_threshold(void) {
    return g_anomaly_threshold;
}

/**
 * @brief Perform Monitor Freq Set Threshold.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_freq_set_threshold(int t) {
    if (t < 1) t = 1;
    g_anomaly_threshold = t;
    return 0;
}

/**
 * @brief Perform Monitor Syscall Set Enabled.
 *
 * @param num [in] Num value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_syscall_set_enabled(int num, int enabled) {
    if (num < 0 || num >= SNEPPX_MONITOR_MAX_SYSCALL_TABLE) return -1;
    if (enabled) {
        g_syscall_enabled = 1;
    }
    (void)num;
    return 0;
}

/**
 * @brief Perform Monitor Syscall Get Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_syscall_get_count(int num) {
    if (num < 0 || num >= SNEPPX_MONITOR_MAX_SYSCALL_TABLE) return -1;
    return g_syscall_counts[num];
}

/**
 * @brief Reset Monitor Syscall.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_syscall_reset(void) {
    memset(g_syscall_counts, 0, sizeof(g_syscall_counts));
    memset(g_syscall_baseline, 0, sizeof(g_syscall_baseline));
    return 0;
}

/**
 * @brief Perform Monitor Region Get Baseline.
 *
 * @param name [in] Name value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_region_get_baseline(const char* name, uint32_t* crc_out) {
    if (!name||!crc_out) return -1;
    for (int i = 0; i < g_region_count; i++) {
        if (g_regions[i].active && strcmp(g_regions[i].name, name) == 0) {
            *crc_out = g_regions[i].baseline_crc;
            return 0;
        }
    }
    return -1;
}

/**
 * @brief Perform Monitor Region Update Baseline.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_region_update_baseline(const char* name) {
    if (!name) return -1;
    for (int i = 0; i < g_region_count; i++) {
        if (strcmp(g_regions[i].name, name) == 0) {
            g_regions[i].baseline_crc = crc32_c(g_regions[i].addr, g_regions[i].size);
            return 0;
        }
    }
    return -1;
}

/**
 * @brief Perform Monitor Region Get Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_region_get_count(void) {
    return g_region_count;
}

/**
 * @brief Perform Monitor Region List.
 *
 * @param buffer [out] Buffer value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_region_list(char* buffer, int max) {
    if (!buffer||max<1) return -1;
    int pos = 0;
    for (int i = 0; i < g_region_count && pos < max - 2; i++) {
        if (!g_regions[i].active) continue;
        int n = snprintf(buffer+pos, max-pos, "%s\n", g_regions[i].name);
        if (n > 0) pos += n;
    }
    buffer[pos] = '\0';
    return 0;
}

/**
 * @brief Perform Monitor Heartbeat Check.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_heartbeat_check(void) {
    if (g_heartbeat_interval_ms == 0) return 0;
    uint64_t now_ms = (uint64_t)time(NULL) * 1000;
    if (g_last_heartbeat_time == 0) {
        g_last_heartbeat_time = now_ms;
        return 0;
    }
    uint64_t elapsed = now_ms - g_last_heartbeat_time;
    if (elapsed > g_heartbeat_interval_ms) {
        char desc[256];
        snprintf(desc, sizeof(desc), "Heartbeat miss: %llu ms elapsed (interval %llu ms)",
                 (unsigned long long)elapsed, (unsigned long long)g_heartbeat_interval_ms);
        fire_event(SNEPPX_MONITOR_EVENT_HEARTBEAT_MISS, desc, 0, 0);
        return 1;
    }
    return 0;
}

/**
 * @brief Reset Monitor Heartbeat.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_heartbeat_reset(void) {
    g_last_heartbeat_time = (uint64_t)time(NULL) * 1000;
    return 0;
}

/**
 * @brief Perform Monitor Register Region Ex.
 *
 * @param name [in] Name value.
 * @param addr [in] Addr value.
 * @param size [in] Size value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_register_region_ex(const char* name, const void* addr, size_t size, int flags) {
    int ret = SNEPPX_monitor_register_region(name, addr, size);
    if (ret == 0) {
        if (flags & 1) {
            if (g_region_count > 0) {
                g_regions[g_region_count-1].active = 1;
            }
        }
    }
    return ret;
}

/**
 * @brief Verify Monitor Bulk.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_bulk_verify(void) {
    int violations = 0;
    int checked = 0;
    int start = (int)g_bulk_verify_index;
    for (int i = 0; i < g_region_count && checked < 10; i++) {
        int idx = (start + i) % g_region_count;
        if (!g_regions[idx].active) continue;
        uint32_t current = crc32_c(g_regions[idx].addr, g_regions[idx].size);
        if (current != g_regions[idx].baseline_crc) {
            char desc[256];
            snprintf(desc, sizeof(desc), "Bulk verify: region '%s' modified", g_regions[idx].name);
            fire_event(SNEPPX_MONITOR_EVENT_TEXT_MODIFIED, desc,
                       (uint64_t)(uintptr_t)g_regions[idx].addr, g_regions[idx].size);
            violations++;
        }
        checked++;
    }
    g_bulk_verify_index = (g_bulk_verify_index + checked) % (g_region_count > 0 ? g_region_count : 1);
    return violations;
}

static int g_region_flags[SNEPPX_MONITOR_MAX_REGIONS];

static void verify_region_now(int idx) {
    if (idx<0||idx>=g_region_count) return;
    if (!g_regions[idx].active) return;
    uint32_t current = crc32_c(g_regions[idx].addr, g_regions[idx].size);
    if (current != g_regions[idx].baseline_crc) {
        char desc[256];
        snprintf(desc, sizeof(desc), "Region '%s' modified", g_regions[idx].name);
        fire_event(SNEPPX_MONITOR_EVENT_TEXT_MODIFIED, desc,
                   (uint64_t)(uintptr_t)g_regions[idx].addr, g_regions[idx].size);
    }
}

/**
 * @brief Perform Monitor Freq Get Baseline Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_freq_get_baseline_count(int index) {
    if (index<0||index>=4) return -1;
    return g_freq_baseline[index];
}

/**
 * @brief Perform Monitor Freq Get Sample Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_freq_get_sample_count(void) {
    return g_freq_sample_count;
}

/**
 * @brief Perform Monitor Timing Get Current Mean.
 *
 * @return The result value, or 0 on error.
 */
double SNEPPX_monitor_timing_get_current_mean(void) {
    if (g_sliding_window_count==0) return 0.0;
    double sum=0.0;
    for (int i=0;i<g_sliding_window_count;i++) sum+=g_sliding_window[i];
    return sum/g_sliding_window_count;
}

/**
 * @brief Perform Monitor Timing Get Current Stddev.
 *
 * @return The result value, or 0 on error.
 */
double SNEPPX_monitor_timing_get_current_stddev(void) {
    if (g_sliding_window_count<2) return 0.0;
    double mean = SNEPPX_monitor_timing_get_current_mean();
    double var=0.0;
    for (int i=0;i<g_sliding_window_count;i++) {
        double d=g_sliding_window[i]-mean;
        var+=d*d;
    }
    return sqrt(var/(g_sliding_window_count-1));
}

/**
 * @brief Perform Monitor Get Self Crc.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_self_crc(void) {
    return (int)g_self_crc_baseline;
}

/**
 * @brief Perform Monitor Get Callback Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_callback_count(void) {
    return g_callback_count;
}

/**
 * @brief Perform Monitor Set Verify Interval Now.
 */
void SNEPPX_monitor_set_verify_interval_now(uint64_t ms) {
    g_verify_interval_ms = ms;
    g_last_verify_called = (uint64_t)time(NULL)*1000;
}

/**
 * @brief Perform Monitor Set Heartbeat Now.
 */
void SNEPPX_monitor_set_heartbeat_now(uint64_t interval_ms) {
    g_heartbeat_interval_ms = interval_ms;
    g_last_heartbeat_time = (uint64_t)time(NULL)*1000;
}

/**
 * @brief Perform Monitor Set Bulk Index.
 */
void SNEPPX_monitor_set_bulk_index(uint64_t idx) {
    g_bulk_verify_index = idx;
}

/**
 * @brief Perform Monitor Get Bulk Index.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_monitor_get_bulk_index(void) {
    return g_bulk_verify_index;
}

static int g_max_events_stored = SNEPPX_MONITOR_EVENT_LOG_SIZE;

/**
 * @brief Perform Monitor Set Max Events.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_set_max_events(int max) {
    if (max<16) max=16;
    if (max>4096) max=4096;
    g_max_events_stored = max;
    return 0;
}

/**
 * @brief Perform Monitor Get Max Events.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_max_events(void) {
    return g_max_events_stored;
}

/**
 * @brief Perform Monitor Get Event At.
 *
 * @param index [in] Index value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_event_at(int index, SNEPPXMonitorEvent* ev) {
    if (!ev||index<0||index>=g_event_log_count) return -1;
    int start = (g_event_log_head - g_event_log_count + SNEPPX_MONITOR_EVENT_LOG_SIZE) % SNEPPX_MONITOR_EVENT_LOG_SIZE;
    int idx = (start + index) % SNEPPX_MONITOR_EVENT_LOG_SIZE;
    ev->type = g_event_log[idx].type;
    ev->description = g_event_log[idx].description;
    ev->address = g_event_log[idx].address;
    ev->size = g_event_log[idx].size;
    ev->timestamp = g_event_log[idx].timestamp;
    return 0;
}

/**
 * @brief Perform Monitor Set All Filters.
 */
void SNEPPX_monitor_set_all_filters(int enabled) {
    int val = enabled?1:0;
    for (int i=0;i<SNEPPX_MONITOR_EVENT_TYPE_COUNT;i++) {
        g_event_filter[i]=val;
    }
}

/**
 * @brief Perform Monitor Get Filter.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_filter(SNEPPXMonitorEventType type) {
    int idx = (int)type % SNEPPX_MONITOR_EVENT_TYPE_COUNT;
    return g_event_filter[idx];
}

/**
 * @brief Perform Monitor Timing Reset Samples.
 */
void SNEPPX_monitor_timing_reset_samples(void) {
    g_timing_samples = 0;
    sliding_window_reset();
}

/**
 * @brief Perform Monitor Get Timing Samples.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_timing_samples(void) {
    return g_timing_samples;
}

/**
 * @brief Perform Monitor Get Sliding Window Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_sliding_window_count(void) {
    return g_sliding_window_count;
}

/**
 * @brief Perform Monitor Get Event Rate.
 *
 * @return The result value, or 0 on error.
 */
double SNEPPX_monitor_get_event_rate(void) {
    uint64_t uptime = SNEPPX_monitor_get_uptime_seconds();
    if (uptime==0) return 0.0;
    return (double)g_anomaly_total_count/(double)uptime;
}

/**
 * @brief Perform Monitor Get Event Type Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_event_type_count(int type_idx) {
    if (type_idx<0||type_idx>=SNEPPX_MONITOR_EVENT_TYPE_COUNT) return 0;
    return g_event_type_counts[type_idx];
}

/**
 * @brief Perform Monitor Reset Event Type Counts.
 */
void SNEPPX_monitor_reset_event_type_counts(void) {
    memset(g_event_type_counts,0,sizeof(g_event_type_counts));
}

/**
 * @brief Perform Monitor Get Event Type Baseline.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_event_type_baseline(int type) {
    return event_type_get_baseline(type);
}

/**
 * @brief Perform Monitor Get Start Time.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_monitor_get_start_time(void) {
    return g_monitor_start_time;
}

/**
 * @brief Perform Monitor Set Region Active.
 *
 * @param name [in] Name value.
 */
void SNEPPX_monitor_set_region_active(const char* name, int active) {
    if (!name) return;
    for (int i=0;i<g_region_count;i++) {
        if (strcmp(g_regions[i].name,name)==0) {
            g_regions[i].active = active?1:0;
            return;
        }
    }
}

/**
 * @brief Perform Monitor Is Region Active.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_is_region_active(const char* name) {
    return SNEPPX_monitor_region_is_active(name);
}

/**
 * @brief Perform Monitor Get Event Type Baseline Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_event_type_baseline_count(int type) {
    if (type<0||type>=SNEPPX_MONITOR_EVENT_TYPE_COUNT) return 0;
    return g_event_type_baselines[type];
}

/**
 * @brief Perform Monitor Get Region Index.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_region_index(const char* name) {
    if (!name) return -1;
    for (int i=0;i<g_region_count;i++) {
        if (strcmp(g_regions[i].name,name)==0) return i;
    }
    return -1;
}

/**
 * @brief Perform Monitor Get Region Crc By Index.
 *
 * @return 0 on success, -1 on error.
 */
uint32_t SNEPPX_monitor_get_region_crc_by_index(int index) {
    if (index<0||index>=g_region_count) return 0;
    return g_regions[index].baseline_crc;
}

/**
 * @brief Perform Monitor Get Region Name By Index.
 *
 * @return Pointer on success, NULL on error.
 */
const char* SNEPPX_monitor_get_region_name_by_index(int index) {
    if (index<0||index>=g_region_count) return NULL;
    return g_regions[index].name;
}

/**
 * @brief Perform Monitor Verify Range.
 *
 * @param start [in] Start value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_verify_range(int start, int end) {
    if (start<0) start=0;
    if (end>=g_region_count) end=g_region_count-1;
    int violations=0;
    for (int i=start;i<=end;i++) {
        if (!g_regions[i].active) continue;
        uint32_t current=crc32_c(g_regions[i].addr,g_regions[i].size);
        if (current!=g_regions[i].baseline_crc) {
            char desc[256];
            snprintf(desc,sizeof(desc),"Range verify: region '%s' modified",g_regions[i].name);
            fire_event(SNEPPX_MONITOR_EVENT_TEXT_MODIFIED,desc,
                       (uint64_t)(uintptr_t)g_regions[i].addr,g_regions[i].size);
            violations++;
        }
    }
    return violations;
}

/**
 * @brief Perform Monitor Set Heartbeat Interval Now.
 */
void SNEPPX_monitor_set_heartbeat_interval_now(void) {
    g_last_heartbeat_time=(uint64_t)time(NULL)*1000;
}

/**
 * @brief Perform Monitor Get Heartbeat Elapsed.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_monitor_get_heartbeat_elapsed(void) {
    if (g_last_heartbeat_time==0) return 0;
    uint64_t now_ms=(uint64_t)time(NULL)*1000;
    if (now_ms<g_last_heartbeat_time) return 0;
    return now_ms-g_last_heartbeat_time;
}

/**
 * @brief Perform Monitor Get Region Addr.
 *
 * @param name [in] Name value.
 * @param addr_out [in] Addr Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_region_addr(const char* name, const void** addr_out, size_t* size_out) {
    if (!name||!addr_out||!size_out) return -1;
    for (int i=0;i<g_region_count;i++) {
        if (g_regions[i].active&&strcmp(g_regions[i].name,name)==0) {
            *addr_out=g_regions[i].addr;
            *size_out=g_regions[i].size;
            return 0;
        }
    }
    return -1;
}

/**
 * @brief Perform Monitor Get Region By Index.
 *
 * @param index [in] Index value.
 * @param name_out [out] Name Out value.
 * @param name_max [in] Name Max value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_region_by_index(int index, char* name_out, int name_max, uint32_t* crc_out) {
    if (index<0||index>=g_region_count||!name_out||name_max<1) return -1;
    strncpy(name_out,g_regions[index].name,name_max-1);
    name_out[name_max-1]=0;
    if (crc_out) *crc_out=g_regions[index].baseline_crc;
    return 0;
}

/**
 * @brief Perform Monitor Get Timing Mean Std.
 *
 * @param mean [out] Mean value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_timing_mean_std(double* mean, double* stddev) {
    if (!mean||!stddev) return -1;
    if (g_sliding_window_count<2) return -1;
    double s=0.0;
    for (int i=0;i<g_sliding_window_count;i++) s+=g_sliding_window[i];
    *mean=s/g_sliding_window_count;
    double v=0.0;
    for (int i=0;i<g_sliding_window_count;i++) { double d=g_sliding_window[i]-*mean; v+=d*d; }
    *stddev=sqrt(v/(g_sliding_window_count-1));
    return 0;
}

/**
 * @brief Perform Monitor Get Timing Mean.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_timing_mean(double* mean) {
    if (!mean) return -1;
    if (g_sliding_window_count==0) return -1;
    double s=0.0;
    for (int i=0;i<g_sliding_window_count;i++) s+=g_sliding_window[i];
    *mean=s/g_sliding_window_count;
    return 0;
}

/**
 * @brief Perform Monitor Get Timing Stddev.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_timing_stddev(double* stddev) {
    if (!stddev) return -1;
    if (g_sliding_window_count<2) return -1;
    double mean;
    int ret=SNEPPX_monitor_get_timing_mean(&mean);
    if (ret!=0) return -1;
    double v=0.0;
    for (int i=0;i<g_sliding_window_count;i++) { double d=g_sliding_window[i]-mean; v+=d*d; }
    *stddev=sqrt(v/(g_sliding_window_count-1));
    return 0;
}

/**
 * @brief Perform Monitor Get Freq Baseline Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_freq_baseline_count(void) {
    int c=0;
    for (int i=0;i<4;i++) c+=g_freq_baseline[i];
    return c;
}

/**
 * @brief Perform Monitor Set Heartbeat Timer.
 */
void SNEPPX_monitor_set_heartbeat_timer(uint64_t ms) {
    g_heartbeat_interval_ms=ms;
    g_last_heartbeat_time=(uint64_t)time(NULL)*1000;
}

/**
 * @brief Perform Monitor Is Heartbeat Expired.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_is_heartbeat_expired(void) {
    if (g_heartbeat_interval_ms==0) return 0;
    uint64_t now_ms=(uint64_t)time(NULL)*1000;
    if (g_last_heartbeat_time==0) { g_last_heartbeat_time=now_ms; return 0; }
    return (now_ms-g_last_heartbeat_time>g_heartbeat_interval_ms)?1:0;
}

/**
 * @brief Perform Monitor Region Exists.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_region_exists(const char* name) {
    if (!name) return 0;
    for (int i=0;i<g_region_count;i++) {
        if (strcmp(g_regions[i].name,name)==0) return 1;
    }
    return 0;
}

/**
 * @brief Perform Monitor Set Start Time Now.
 */
void SNEPPX_monitor_set_start_time_now(void) {
    g_monitor_start_time=(uint64_t)time(NULL);
}

/**
 * @brief Perform Monitor Get Region Size.
 *
 * @param name [in] Name value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_region_size(const char* name, size_t* size_out) {
    if (!name||!size_out) return -1;
    for (int i=0;i<g_region_count;i++) {
        if (g_regions[i].active&&strcmp(g_regions[i].name,name)==0) {
            *size_out=g_regions[i].size;
            return 0;
        }
    }
    return -1;
}

/**
 * @brief Perform Monitor Get Syscall Baseline.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_syscall_baseline(int num) {
    if (num<0||num>=SNEPPX_MONITOR_MAX_SYSCALL_TABLE) return -1;
    return g_syscall_baseline[num];
}

/**
 * @brief Perform Monitor Get Syscall Ratio.
 *
 * @param num [in] Num value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_syscall_ratio(int num, double* ratio_out) {
    if (num<0||num>=SNEPPX_MONITOR_MAX_SYSCALL_TABLE||!ratio_out) return -1;
    if (g_syscall_baseline[num]==0) { *ratio_out=0.0; return 0; }
    *ratio_out=(double)g_syscall_counts[num]/(double)g_syscall_baseline[num];
    return 0;
}

/**
 * @brief Perform Monitor Get Anomaly Rate.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_anomaly_rate(double* rate_out) {
    if (!rate_out) return -1;
    uint64_t uptime=SNEPPX_monitor_get_uptime_seconds();
    if (uptime==0) { *rate_out=0.0; return 0; }
    *rate_out=(double)g_anomaly_total_count/(double)uptime;
    return 0;
}

/**
 * @brief Perform Monitor Get Event Rate For Type.
 *
 * @param type [in] Type value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_event_rate_for_type(int type, double* rate_out) {
    if (type<0||type>=SNEPPX_MONITOR_EVENT_TYPE_COUNT||!rate_out) return -1;
    uint64_t uptime=SNEPPX_monitor_get_uptime_seconds();
    if (uptime==0) { *rate_out=0.0; return 0; }
    *rate_out=(double)g_event_type_counts[type]/(double)uptime;
    return 0;
}

/**
 * @brief Perform Monitor Set Event Type Baseline.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_set_event_type_baseline(int type) {
    if (type<0||type>=SNEPPX_MONITOR_EVENT_TYPE_COUNT) return -1;
    g_event_type_baselines[type]=g_event_type_counts[type];
    return 0;
}

/**
 * @brief Perform Monitor Compare Event Type.
 *
 * @param type [in] Type value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_compare_event_type(int type, int* diff_out) {
    if (type<0||type>=SNEPPX_MONITOR_EVENT_TYPE_COUNT||!diff_out) return -1;
    *diff_out=g_event_type_counts[type]-g_event_type_baselines[type];
    return 0;
}

/**
 * @brief Perform Monitor Set All Event Baselines.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_set_all_event_baselines(void) {
    memcpy(g_event_type_baselines,g_event_type_counts,sizeof(g_event_type_baselines));
    return 0;
}

/**
 * @brief Perform Monitor Get Total Event Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_total_event_count(void) {
    int total=0;
    for (int i=0;i<SNEPPX_MONITOR_EVENT_TYPE_COUNT;i++) total+=g_event_type_counts[i];
    return total;
}

/**
 * @brief Perform Monitor Get Region Check Count By Index.
 *
 * @param index [in] Index value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_region_check_count_by_index(int index, int* count_out) {
    if (index<0||index>=g_region_count||!count_out) return -1;
    *count_out=g_region_freq[index].check_count;
    return 0;
}

/**
 * @brief Perform Monitor Get Region Check Interval.
 *
 * @param index [in] Index value.
 * @param min_out [out] Min Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_region_check_interval(int index, uint64_t* min_out, uint64_t* max_out) {
    if (index<0||index>=g_region_count||!min_out||!max_out) return -1;
    *min_out=g_region_freq[index].min_interval_us;
    *max_out=g_region_freq[index].max_interval_us;
    return 0;
}

/**
 * @brief Perform Monitor Get Region Crc String.
 *
 * @param name [in] Name value.
 * @param buf [out] Buf value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_region_crc_string(const char* name, char* buf, size_t size) {
    if (!name||!buf||size<16) return -1;
    for (int i=0;i<g_region_count;i++) {
        if (g_regions[i].active&&strcmp(g_regions[i].name,name)==0) {
            snprintf(buf,size,"0x%08X",g_regions[i].baseline_crc);
            return 0;
        }
    }
    return -1;
}

/**
 * @brief Perform Monitor Is Region Modified.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_is_region_modified(const char* name) {
    if (!name) return -1;
    for (int i=0;i<g_region_count;i++) {
        if (g_regions[i].active&&strcmp(g_regions[i].name,name)==0) {
            uint32_t current=crc32_c(g_regions[i].addr,g_regions[i].size);
            return (current!=g_regions[i].baseline_crc)?1:0;
        }
    }
    return -1;
}

/**
 * @brief Perform Monitor Disable All Filters.
 */
void SNEPPX_monitor_disable_all_filters(void) {
    memset(g_event_filter,0,sizeof(g_event_filter));
}

/**
 * @brief Perform Monitor Enable All Filters.
 */
void SNEPPX_monitor_enable_all_filters(void) {
    memset(g_event_filter,1,sizeof(g_event_filter));
}

/**
 * @brief Perform Monitor Get Region Count Active.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_region_count_active(void) {
    return SNEPPX_monitor_get_active_region_count();
}

/**
 * @brief Perform Monitor Get Syscall Count Total.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_syscall_count_total(void) {
    int total=0;
    for (int i=0;i<SNEPPX_MONITOR_MAX_SYSCALL_TABLE;i++) total+=g_syscall_counts[i];
    return total;
}

/**
 * @brief Perform Monitor Get Syscall Baseline Total.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_syscall_baseline_total(void) {
    int total=0;
    for (int i=0;i<SNEPPX_MONITOR_MAX_SYSCALL_TABLE;i++) total+=g_syscall_baseline[i];
    return total;
}

/**
 * @brief Perform Monitor Reset Freq Baselines.
 */
void SNEPPX_monitor_reset_freq_baselines(void) {
    memset(g_freq_baseline,0,sizeof(g_freq_baseline));
}

/**
 * @brief Perform Monitor Get Freq Baseline For Type.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_freq_baseline_for_type(int type) {
    if (type<0||type>=4) return 0;
    return g_freq_baseline[type];
}

/**
 * @brief Perform Monitor Get Self Check Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_self_check_count(void) {
    (void)0;
    return 0;
}

/**
 * @brief Perform Monitor Check Regions Batch.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_check_regions_batch(int indices[], int count) {
    if (!indices||count<1) return -1;
    int violations=0;
    for (int n=0;n<count;n++) {
        int i=indices[n];
        if (i<0||i>=g_region_count) continue;
        if (!g_regions[i].active) continue;
        uint32_t current=crc32_c(g_regions[i].addr,g_regions[i].size);
        if (current!=g_regions[i].baseline_crc) {
            char desc[256];
            snprintf(desc,sizeof(desc),"Batch verify: region '%s' modified",g_regions[i].name);
            fire_event(SNEPPX_MONITOR_EVENT_TEXT_MODIFIED,desc,
                       (uint64_t)(uintptr_t)g_regions[i].addr,g_regions[i].size);
            violations++;
        }
    }
    return violations;
}

/**
 * @brief Perform Monitor Get Canary Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_canary_count(void) {
    return g_canary_count;
}

/**
 * @brief Perform Monitor Get Canary Size.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_canary_size(void) {
    return SNEPPX_MONITOR_CANARY_SIZE;
}

/**
 * @brief Perform Monitor Has Callback.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_has_callback(SNEPPXMonitorCallback cb) {
    if (!cb) return 0;
    for (int i=0;i<g_callback_count;i++) {
        if (g_callbacks[i]==cb) return 1;
    }
    return 0;
}

/**
 * @brief Perform Monitor Get Event Type Count Safe.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_event_type_count_safe(int type) {
    return SNEPPX_monitor_get_event_type_count(type);
}

/**
 * @brief Perform Monitor Set Interval Now.
 */
void SNEPPX_monitor_set_interval_now(uint64_t ms) {
    g_monitor_interval_ms=ms;
}

/**
 * @brief Perform Monitor Get Interval.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_monitor_get_interval(void) {
    return g_monitor_interval_ms;
}

/**
 * @brief Perform Monitor Get Region Flags.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_region_flags(const char* name) {
    if (!name) return -1;
    for (int i=0;i<g_region_count;i++) {
        if (strcmp(g_regions[i].name,name)==0) return g_region_flags[i];
    }
    return -1;
}

/**
 * @brief Perform Monitor Set Region Flags.
 *
 * @param name [in] Name value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_set_region_flags(const char* name, int flags) {
    if (!name) return -1;
    for (int i=0;i<g_region_count;i++) {
        if (strcmp(g_regions[i].name,name)==0) {
            g_region_flags[i]=flags;
            return 0;
        }
    }
    return -1;
}

/**
 * @brief Perform Monitor Get Region Count Total.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_region_count_total(void) {
    return g_region_count;
}

/**
 * @brief Perform Monitor Get Heartbeat Miss Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_heartbeat_miss_count(void) {
    static int miss_count=0;
    if (SNEPPX_monitor_heartbeat_check()) miss_count++;
    return miss_count;
}

/**
 * @brief Perform Monitor Reset Heartbeat Miss Count.
 */
void SNEPPX_monitor_reset_heartbeat_miss_count(void) {
    (void)0;
}

/**
 * @brief Perform Monitor Get Verify Interval Remaining.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_monitor_get_verify_interval_remaining(void) {
    if (g_verify_interval_ms==0) return 0;
    uint64_t now_ms=(uint64_t)time(NULL)*1000;
    uint64_t elapsed=now_ms-g_last_verify_called;
    if (elapsed>=g_verify_interval_ms) return 0;
    return (int)(g_verify_interval_ms-elapsed);
}
