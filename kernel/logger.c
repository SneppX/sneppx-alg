#include "logger.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

/*
 * SNEPPX - Logger
 *
 * WHAT
 *   Logger.
 *
 * CONCEPT
 *   Provides structured logging.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/* ANSI color codes */
#define COLOR_RESET   "\033[0m"
#define COLOR_RED     "\033[31m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_BLUE    "\033[34m"
#define COLOR_MAGENTA "\033[35m"
#define COLOR_CYAN    "\033[36m"
#define COLOR_GRAY    "\033[37m"

static const char* level_colors[] = {
    [SNEPPX_LOG_TRACE] = COLOR_GRAY,
    [SNEPPX_LOG_DEBUG] = COLOR_CYAN,
    [SNEPPX_LOG_INFO]  = COLOR_GREEN,
    [SNEPPX_LOG_WARN]  = COLOR_YELLOW,
    [SNEPPX_LOG_ERROR] = COLOR_RED,
    [SNEPPX_LOG_FATAL] = COLOR_MAGENTA
};

static const char* level_strings[] = {
    [SNEPPX_LOG_TRACE] = "TRACE",
    [SNEPPX_LOG_DEBUG] = "DEBUG",
    [SNEPPX_LOG_INFO]  = "INFO",
    [SNEPPX_LOG_WARN]  = "WARN",
    [SNEPPX_LOG_ERROR] = "ERROR",
    [SNEPPX_LOG_FATAL] = "FATAL"
};

/**
 * @brief Perform Log Level String.
 *
 * @return Pointer on success, NULL on error.
 */
const char* SNEPPX_log_level_string(SNEPPX_LogLevel level) {
    if (level > SNEPPX_LOG_FATAL) return "UNKNOWN";
    return level_strings[level];
}

struct SNEPPX_Logger {
    SNEPPX_LogConfig config;
    FILE* json_fp;
};

static int64_t snepx_log_ns(void) {
#ifdef _WIN32
    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);
    return ((int64_t)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
#else
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
#endif
/*
 * SNEPPX - Kernel Module
 *
 * WHAT
 *   Kernel Module.
 *
 * CONCEPT
 *   Kernel Module implementation.
 *
 * ROLE
 *   Core kernel module used throughout the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal kernel module).
 */


}

/**
 * @brief Initialize Logger.
 *
 * @param logger [out] Logger value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_logger_init(SNEPPX_Logger** logger, const SNEPPX_LogConfig* config) {
    if (!logger) return -1;
    SNEPPX_Logger* log = (SNEPPX_Logger*)calloc(1, sizeof(SNEPPX_Logger));
    if (!log) return -1;
    if (config) {
        log->config = *config;
    } else {
        log->config.min_level = SNEPPX_LOG_INFO;
        log->config.enable_stdout = 1;
        log->config.enable_json = 0;
        log->config.json_path = NULL;
        log->config.rank = 0;
        log->config.enable_timestamp = 1;
        log->config.enable_color = 1;
    }
    log->json_fp = NULL;
    if (log->config.enable_json && log->config.json_path) {
        log->json_fp = fopen(log->config.json_path, "a");
        if (log->json_fp) {
            fprintf(log->json_fp, "[\n");
        }
    }
    *logger = log;
    return 0;
}

/**
 * @brief Destroy Logger.
 */
void SNEPPX_logger_destroy(SNEPPX_Logger* logger) {
    if (!logger) return;
    if (logger->json_fp) {
        fprintf(logger->json_fp, "\n]\n");
        fclose(logger->json_fp);
    }
    free(logger);
}

/**
 * @brief Perform Logger Set Level.
 *
 * @param logger [out] Logger value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_logger_set_level(SNEPPX_Logger* logger, SNEPPX_LogLevel level) {
    if (!logger) return -1;
    logger->config.min_level = level;
    return 0;
}

/**
 * @brief Perform Logger Set Output.
 *
 * @param logger [out] Logger value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_logger_set_output(SNEPPX_Logger* logger, const char* json_path) {
    if (!logger) return -1;
    if (logger->json_fp) {
        fprintf(logger->json_fp, "\n]\n");
        fclose(logger->json_fp);
    }
    logger->config.json_path = json_path;
    logger->config.enable_json = (json_path != NULL);
    if (json_path) {
        logger->json_fp = fopen(json_path, "a");
        if (logger->json_fp) fprintf(logger->json_fp, "[\n");
    } else {
        logger->json_fp = NULL;
    }
    return 0;
}

/**
 * @brief Write Log.
 *
 * @param logger [out] Logger value.
 * @param level [in] Level value.
 * @param file [in] File value.
 * @param line [in] Line value.
 * @param fmt [in] Fmt value.
 */
void SNEPPX_log_write(SNEPPX_Logger* logger, SNEPPX_LogLevel level,
                       const char* file, int line,
                       const char* fmt, ...) {
    if (!logger || level < logger->config.min_level) return;

    int64_t ts = snepx_log_ns();
    const char* level_str = SNEPPX_log_level_string(level);

    /* Format message */
    char msg_buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(msg_buf, sizeof(msg_buf), fmt, args);
    va_end(args);

    /* Stdout output */
    if (logger->config.enable_stdout) {
        if (logger->config.enable_color) {
            const char* color = level_colors[level <= SNEPPX_LOG_FATAL ? level : 0];
            if (logger->config.enable_timestamp) {
                printf("%s[%lld] [%s] [rank %d] %s:%d %s%s\n",
                       color, (long long)ts, level_str, logger->config.rank,
                       file, line, msg_buf, COLOR_RESET);
            } else {
                printf("%s[%s] [rank %d] %s:%d %s%s\n",
                       color, level_str, logger->config.rank,
                       file, line, msg_buf, COLOR_RESET);
            }
        } else {
            if (logger->config.enable_timestamp) {
                printf("[%lld] [%s] [rank %d] %s:%d %s\n",
                       (long long)ts, level_str, logger->config.rank,
                       file, line, msg_buf);
            } else {
                printf("[%s] [rank %d] %s:%d %s\n",
                       level_str, logger->config.rank,
                       file, line, msg_buf);
            }
        }
    }

    /* JSON output */
    if (logger->config.enable_json && logger->json_fp) {
        fprintf(logger->json_fp,
                "{\"timestamp\":%lld,\"level\":\"%s\",\"rank\":%d,"
                "\"file\":\"%s\",\"line\":%d,\"message\":\"%s\"},\n",
                (long long)ts, level_str, logger->config.rank,
                file, line, msg_buf);
        fflush(logger->json_fp);
    }
}
