#ifndef SNEPPX_CONTAINER_BREAKOUT_H
#define SNEPPX_CONTAINER_BREAKOUT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

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
    uint64_t total_breakout_attempts;
    uint64_t total_blocks;
    int active_processes;
    int suspicious_processes;
    int watch_files;
} breakout_stats_t;

/**
 * @brief Perform Breakout Detect Ns Change.
 *
 * @param pid [in] Pid value.
 * @param comm [in] Comm value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_breakout_detect_ns_change(int pid, const char *comm);
/**
 * @brief Perform Breakout Detect Mount.
 *
 * @param pid [in] Pid value.
 * @param target [in] Target value.
 * @param comm [in] Comm value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_breakout_detect_mount(int pid, const char *target, const char *comm);
/**
 * @brief Perform Breakout Detect Capability.
 *
 * @param pid [in] Pid value.
 * @param cap_effective [in] Cap Effective value.
 * @param comm [in] Comm value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_breakout_detect_capability(int pid, uint64_t cap_effective, const char *comm);
/**
 * @brief Perform Breakout Check File Access.
 *
 * @param path [in] Path value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_breakout_check_file_access(const char *path);
/**
 * @brief Perform Breakout Detect Syscall.
 *
 * @param pid [in] Pid value.
 * @param syscall_name [in] Syscall Name value.
 * @param comm [in] Comm value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_breakout_detect_syscall(int pid, const char *syscall_name, const char *comm);
/**
 * @brief Perform Breakout Add Watch File.
 *
 * @param path [in] Path value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_breakout_add_watch_file(const char *path);
/**
 * @brief Perform Breakout Get Stats.
 *
 * @param stats [out] Stats value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_breakout_get_stats(breakout_stats_t *stats);
/**
 * @brief Reset Breakout.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_breakout_reset(void);


#ifdef __cplusplus
}
#endif
#endif
