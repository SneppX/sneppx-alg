#include "container_breakout.h"
#include <string.h>
#include <time.h>

#define BREAKOUT_MAX_EVENTS 4096
#define BREAKOUT_WATCH_FILES 64
#define BREAKOUT_SENSITIVITY 5

/*
 * SNEPPX - Container Breakout
 *
 * WHAT
 *   Container Breakout.
 *
 * CONCEPT
 *   Provides the Container Breakout.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef struct {
    int pid;
    char comm[64];
    uint64_t syscall_count;
    uint64_t ns_change_count;
    uint64_t mount_count;
    uint64_t cap_usage_count;
    uint8_t suspicious;
    double anomaly_score;
    time_t first_seen;
    time_t last_seen;
} breakout_process_t;

typedef struct {
    char path[256];
    uint64_t access_count;
    uint64_t write_count;
    uint8_t suspicious;
} breakout_file_t;

static breakout_process_t processes[BREAKOUT_MAX_EVENTS];
static int process_count = 0;
static breakout_file_t watch_files[BREAKOUT_WATCH_FILES];
static int watch_file_count = 0;
static uint64_t total_breakout_attempts = 0;
static uint64_t total_blocks = 0;

static const char *suspicious_paths[] = {
    "/etc/passwd", "/etc/shadow", "/proc/1/environ",
    "/proc/1/cmdline", "/proc/1/root", "/var/run/docker.sock",
    "/proc/sysrq-trigger", "/sys/kernel/security",
    "/dev/mem", "/dev/kmem", "/dev/port",
    "/proc/kcore", "/proc/kallsyms",
    "/proc/modules", "/sys/firmware",
    NULL
};

static const char *suspicious_syscalls[] = {
    "mount", "unshare", "clone", "pivot_root",
    "chroot", "ptrace", "bpf", "open_by_handle_at",
    "keyctl", "add_key", "request_key",
    "process_vm_readv", "process_vm_writev",
    "perf_event_open", "kexec_load",
    "reboot", "sethostname", "setdomainname",
    NULL
};

static int find_process(int pid) {
    for (int i = 0; i < process_count; i++)
        if (processes[i].pid == pid) return i;
    return -1;
}

static int add_process(int pid, const char *comm) {
    if (process_count >= BREAKOUT_MAX_EVENTS) return -1;
    processes[process_count].pid = pid;
    strncpy(processes[process_count].comm, comm ? comm : "unknown", 63);
    processes[process_count].syscall_count = 0;
    processes[process_count].ns_change_count = 0;
    processes[process_count].mount_count = 0;
    processes[process_count].cap_usage_count = 0;
    processes[process_count].suspicious = 0;
    processes[process_count].anomaly_score = 0.0;
    processes[process_count].first_seen = time(NULL);
    processes[process_count].last_seen = time(NULL);
    return process_count++;
}

/**
 * @brief Perform Breakout Detect Ns Change.
 *
 * @param pid [in] Pid value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_breakout_detect_ns_change(int pid, const char *comm) {
    int idx = find_process(pid);
    if (idx < 0) idx = add_process(pid, comm);
    if (idx < 0) return 0;
    processes[idx].ns_change_count++;
    processes[idx].last_seen = time(NULL);
    if (processes[idx].ns_change_count > BREAKOUT_SENSITIVITY) {
        processes[idx].suspicious = 1;
        processes[idx].anomaly_score += 0.3;
        total_breakout_attempts++;
        return 1;
    }
    return 0;
}

/**
 * @brief Perform Breakout Detect Mount.
 *
 * @param pid [in] Pid value.
 * @param target [in] Target value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_breakout_detect_mount(int pid, const char *target, const char *comm) {
    int idx = find_process(pid);
    if (idx < 0) idx = add_process(pid, comm);
    if (idx < 0) return 0;
    processes[idx].mount_count++;
    processes[idx].last_seen = time(NULL);
    if (target) {
        for (int i = 0; suspicious_paths[i]; i++) {
            if (strcmp(target, suspicious_paths[i]) == 0) {
                processes[idx].suspicious = 1;
                processes[idx].anomaly_score += 0.5;
                total_breakout_attempts++;
                total_blocks++;
                return 2;
            }
        }
    }
    if (processes[idx].mount_count > BREAKOUT_SENSITIVITY) {
        processes[idx].suspicious = 1;
        processes[idx].anomaly_score += 0.2;
        return 1;
    }
    return 0;
}

/**
 * @brief Perform Breakout Detect Capability.
 *
 * @param pid [in] Pid value.
 * @param cap_effective [in] Cap Effective value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_breakout_detect_capability(int pid, uint64_t cap_effective, const char *comm) {
    int idx = find_process(pid);
    if (idx < 0) idx = add_process(pid, comm);
    if (idx < 0) return 0;
    uint64_t dangerous_caps = 0;
    dangerous_caps |= (1ULL << 0);
    dangerous_caps |= (1ULL << 5);
    dangerous_caps |= (1ULL << 6);
    dangerous_caps |= (1ULL << 12);
    dangerous_caps |= (1ULL << 17);
    dangerous_caps |= (1ULL << 18);
    dangerous_caps |= (1ULL << 21);
    dangerous_caps |= (1ULL << 22);
    dangerous_caps |= (1ULL << 24);
    if (cap_effective & dangerous_caps) {
        processes[idx].cap_usage_count++;
        processes[idx].anomaly_score += 0.4;
        total_breakout_attempts++;
        if (processes[idx].cap_usage_count > 2) {
            processes[idx].suspicious = 1;
            total_blocks++;
            return 2;
        }
        return 1;
    }
    return 0;
}

/**
 * @brief Perform Breakout Check File Access.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_breakout_check_file_access(const char *path) {
    if (!path) return 0;
    for (int i = 0; suspicious_paths[i]; i++) {
        if (strstr(path, suspicious_paths[i]) == path ||
            strstr(path, suspicious_paths[i]) == path + 1) {
            total_breakout_attempts++;
            total_blocks++;
            return 1;
        }
    }
    return 0;
}

/**
 * @brief Perform Breakout Detect Syscall.
 *
 * @param pid [in] Pid value.
 * @param syscall_name [in] Syscall Name value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_breakout_detect_syscall(int pid, const char *syscall_name, const char *comm) {
    if (!syscall_name) return 0;
    for (int i = 0; suspicious_syscalls[i]; i++) {
        if (strcmp(syscall_name, suspicious_syscalls[i]) == 0) {
            int idx = find_process(pid);
            if (idx < 0) idx = add_process(pid, comm);
            if (idx >= 0) {
                processes[idx].syscall_count++;
                processes[idx].anomaly_score += 0.25;
                processes[idx].suspicious = processes[idx].anomaly_score > 1.0;
                if (processes[idx].suspicious) total_breakout_attempts++;
            }
            return 1;
        }
    }
    return 0;
}

/**
 * @brief Perform Breakout Add Watch File.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_breakout_add_watch_file(const char *path) {
    if (!path || watch_file_count >= BREAKOUT_WATCH_FILES) return -1;
    strncpy(watch_files[watch_file_count].path, path, 255);
    watch_files[watch_file_count].access_count = 0;
    watch_files[watch_file_count].write_count = 0;
    watch_files[watch_file_count].suspicious = 0;
    watch_file_count++;
    return 0;
}

/**
 * @brief Perform Breakout Get Stats.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_breakout_get_stats(breakout_stats_t *stats) {
    if (!stats) return -1;
    stats->total_breakout_attempts = total_breakout_attempts;
    stats->total_blocks = total_blocks;
    stats->active_processes = process_count;
    stats->suspicious_processes = 0;
    stats->watch_files = watch_file_count;
    for (int i = 0; i < process_count; i++)
        if (processes[i].suspicious) stats->suspicious_processes++;
    return 0;
}

/**
 * @brief Reset Breakout.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_breakout_reset(void) {
    process_count = 0;
    watch_file_count = 0;
    total_breakout_attempts = 0;
    total_blocks = 0;
    memset(processes, 0, sizeof(processes));
    memset(watch_files, 0, sizeof(watch_files));
    return 0;
}
