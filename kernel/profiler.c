#include "profiler.h"
#include "security/obfuscator.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>

#ifdef SNEPPX_HAS_CUDA
#include <cuda_runtime.h>
#endif

/*
 * SNEPPX - Profiler
 *
 * WHAT
 *   Profiler.
 *
 * CONCEPT
 *   Provides performance profiling.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/* =========================================================================
 * Range marker stack (lightweight NVTX substitute)
 * ========================================================================= */

#define SNEPPX_RANGE_MAX_DEPTH 64

static struct {
    char names[SNEPPX_RANGE_MAX_DEPTH][64];
    int depth;
} _range_stack = {0};

/**
 * @brief Perform Range Push.
 */
void SNEPPX_range_push(const char* name) {
    if (_range_stack.depth < SNEPPX_RANGE_MAX_DEPTH) {
        strncpy(_range_stack.names[_range_stack.depth], name ? name : "unknown", 63);
        _range_stack.names[_range_stack.depth][63] = '\0';
        _range_stack.depth++;
    }
}

/**
 * @brief Perform Range Pop.
 */
void SNEPPX_range_pop(void) {
    if (_range_stack.depth > 0) {
        _range_stack.depth--;
    }
}

/**
 * @brief Perform Range Get Depth.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_range_get_depth(void) {
    return _range_stack.depth;
}

/* =========================================================================
 * Kernel Timer
 * ========================================================================= */

#ifdef SNEPPX_HAS_CUDA

/**
 * @brief Initialize Kernel Timer.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_kernel_timer_init(SNEPPX_KernelTimer* kt) {
    if (!kt) return -1;
    cudaEventCreate(&kt->start);
    cudaEventCreate(&kt->end);
    return 0;
}

/**
 * @brief Start Kernel Timer.
 *
 * @param kt [out] Kt value.
 */
void SNEPPX_kernel_timer_start(SNEPPX_KernelTimer* kt, cudaStream_t stream) {
    if (kt) cudaEventRecord(kt->start, stream);
}

/**
 * @brief Stop Kernel Timer.
 *
 * @param kt [out] Kt value.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_kernel_timer_stop(SNEPPX_KernelTimer* kt, cudaStream_t stream) {
    if (!kt) return 0.0f;
    cudaEventRecord(kt->end, stream);
    cudaEventSynchronize(kt->end);
    float ms = 0;
    cudaEventElapsedTime(&ms, kt->start, kt->end);
    return ms;
}

/**
 * @brief Destroy Kernel Timer.
 */
void SNEPPX_kernel_timer_destroy(SNEPPX_KernelTimer* kt) {
    if (!kt) return;
    cudaEventDestroy(kt->start);
    cudaEventDestroy(kt->end);
}

#else
/* Stubs come from profiler.h macros when SNEPPX_HAS_CUDA is not defined */
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



/* =========================================================================
 * Profiler
 * ========================================================================= */

/**
 * @brief Initialize Profiler.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_profiler_init(SNEPPX_Profiler* prof) {
    if (!prof) return -1;
    memset(prof, 0, sizeof(SNEPPX_Profiler));
    for (int i = 0; i < SNEPPX_PROFILER_MAX_ENTRIES; i++) {
        prof->entries[i].min_time_ms = FLT_MAX;
    }
    prof->enabled = 1;
    return 0;
}

/**
 * @brief Destroy Profiler.
 */
void SNEPPX_profiler_destroy(SNEPPX_Profiler* prof) {
    (void)prof;
}

/**
 * @brief Perform Profiler Enable.
 *
 * @param prof [out] Prof value.
 */
void SNEPPX_profiler_enable(SNEPPX_Profiler* prof, int enabled) {
    if (prof) prof->enabled = enabled;
}

/**
 * @brief Perform Profiler Record.
 *
 * @param prof [out] Prof value.
 * @param name [in] Name value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_profiler_record(SNEPPX_Profiler* prof, const char* name, float elapsed_ms) {
    if (!prof || !name || !prof->enabled) return -1;
    for (int i = 0; i < prof->num_entries; i++) {
        if (strcmp(prof->entries[i].name, name) == 0) {
            SNEPPX_ProfilerEntry* e = &prof->entries[i];
            e->num_calls++;
            e->total_time_ms += elapsed_ms;
            if (elapsed_ms < e->min_time_ms) e->min_time_ms = elapsed_ms;
            if (elapsed_ms > e->max_time_ms) e->max_time_ms = elapsed_ms;
            e->avg_time_ms = e->total_time_ms / e->num_calls;
            return 0;
        }
    }
    if (prof->num_entries >= SNEPPX_PROFILER_MAX_ENTRIES) return -1;
    SNEPPX_ProfilerEntry* e = &prof->entries[prof->num_entries++];
    strncpy(e->name, name, SNEPPX_PROFILER_NAME_MAX - 1);
    e->name[SNEPPX_PROFILER_NAME_MAX - 1] = '\0';
    e->num_calls = 1;
    e->total_time_ms = elapsed_ms;
    e->min_time_ms = elapsed_ms;
    e->max_time_ms = elapsed_ms;
    e->avg_time_ms = elapsed_ms;
    return 0;
}

/**
 * @brief Get Profiler.
 *
 * @param prof [out] Prof value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPX_ProfilerEntry* SNEPPX_profiler_get(SNEPPX_Profiler* prof, const char* name) {
    if (!prof || !name) return NULL;
    for (int i = 0; i < prof->num_entries; i++) {
        if (strcmp(prof->entries[i].name, name) == 0) {
            return &prof->entries[i];
        }
    }
    return NULL;
}

/**
 * @brief Reset Profiler.
 */
void SNEPPX_profiler_reset(SNEPPX_Profiler* prof) {
    if (!prof) return;
    for (int i = 0; i < prof->num_entries; i++) {
        prof->entries[i].num_calls = 0;
        prof->entries[i].total_time_ms = 0.0f;
        prof->entries[i].min_time_ms = FLT_MAX;
        prof->entries[i].max_time_ms = 0.0f;
        prof->entries[i].avg_time_ms = 0.0f;
    }
    prof->num_entries = 0;
}

/**
 * @brief Perform Profiler Print.
 */
void SNEPPX_profiler_print(const SNEPPX_Profiler* prof) {
    if (!prof) return;
    char obf_summary[64];
    sneppx_obfuscate_string("SNEPPX Profiler Summary", obf_summary, sizeof(obf_summary));
    char deobf_summary[64];
    sneppx_deobfuscate_string(obf_summary, deobf_summary, sizeof(deobf_summary));
    printf("\n=== %s ===\n", deobf_summary);
    printf("%-32s %8s %12s %10s %10s %10s\n",
           "Operation", "Calls", "Total(ms)", "Avg(ms)", "Min(ms)", "Max(ms)");
    printf("------------------------------------------------------------\n");
    for (int i = 0; i < prof->num_entries; i++) {
        const SNEPPX_ProfilerEntry* e = &prof->entries[i];
        printf("%-32s %8d %12.3f %10.3f %10.3f %10.3f\n",
               e->name, e->num_calls, e->total_time_ms,
               e->avg_time_ms, e->min_time_ms, e->max_time_ms);
    }
    printf("============================================================\n");
}

/**
 * @brief Perform Profiler To Json.
 *
 * @return Pointer on success, NULL on error.
 */
char* SNEPPX_profiler_to_json(const SNEPPX_Profiler* prof) {
    if (!prof) return NULL;
    size_t cap = 256;
    char* buf = (char*)malloc(cap);
    if (!buf) return NULL;
    size_t pos = 0;
    pos += snprintf(buf + pos, cap - pos, "{\n  \"profiler\": [\n");
    for (int i = 0; i < prof->num_entries; i++) {
        const SNEPPX_ProfilerEntry* e = &prof->entries[i];
        if (pos + 512 >= cap) {
            cap = cap * 2 + 512;
            char* nb = (char*)realloc(buf, cap);
            if (!nb) { free(buf); return NULL; }
            buf = nb;
        }
        int written = snprintf(buf + pos, cap - pos,
            "    {\"name\":\"%s\",\"calls\":%d,\"total_ms\":%.3f,"
            "\"avg_ms\":%.3f,\"min_ms\":%.3f,\"max_ms\":%.3f}%s\n",
            e->name, e->num_calls, e->total_time_ms,
            e->avg_time_ms, e->min_time_ms, e->max_time_ms,
            (i < prof->num_entries - 1) ? "," : "");
        pos += written;
        if (pos + 256 >= cap) {
            cap *= 2;
            char* nb = (char*)realloc(buf, cap);
            if (!nb) { free(buf); return NULL; }
            buf = nb;
        }
    }
    pos += snprintf(buf + pos, cap - pos, "  ]\n}\n");
    return buf;
}
