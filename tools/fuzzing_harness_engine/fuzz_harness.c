/*
 * SNEPPX - Fuzz Harness
 *
 * WHAT
 *   Fuzz Harness.
 *
 * CONCEPT
 *   Provides the Fuzz Harness.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/*
 * SNEPPX Fuzzing Harness — SKELETON
 * VERSION: v0.5
 *
 * PURPOSE: Coverage-guided fuzzing entry points for structured tensor
 * inputs, binary format parsers (checkpoint files), and network
 * messages.  Designed to be linked with libFuzzer or AFL.
 *
 * Each fuzz target consumes a byte array and returns 0.
 * Register new targets via SNEPPX_fuzz_register_target().
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct {
    const char* name;
    int (*target)(const uint8_t* data, size_t size);
} SNEPPXFuzzTarget;

#define SNEPPX_MAX_FUZZ_TARGETS 64
static SNEPPXFuzzTarget g_targets[SNEPPX_MAX_FUZZ_TARGETS];
static int g_num_targets = 0;

int SNEPPX_fuzz_register_target(const char* name, int (*target)(const uint8_t*, size_t)) {
    if (g_num_targets >= SNEPPX_MAX_FUZZ_TARGETS) return -1;
    g_targets[g_num_targets].name = name;
    g_targets[g_num_targets].target = target;
    g_num_targets++;
    return 0;
}

static int fuzz_tensor_create(const uint8_t* data, size_t size) {
    (void)data; (void)size;
    /* TODO(v0.5): feed bytes as shape+dtype, attempt SNEPPX_tensor_create */
    return 0;
}

static int fuzz_checkpoint_load(const uint8_t* data, size_t size) {
    (void)data; (void)size;
    /* TODO(v0.5): feed bytes as checkpoint file, attempt SNEPPX_ckpt_read_open */
    return 0;
}

/* Register fuzz targets at static initialization time so they are available
 * even when libFuzzer provides its own main() (in which case our main()
 * is never linked). */
/**
 * @brief Perform Fuzz Register Defaults.
 */
static void SNEPPX_fuzz_register_defaults(void) {
    static int registered = 0;
    if (registered) return;
    registered = 1;
    SNEPPX_fuzz_register_target("tensor_create", fuzz_tensor_create);
    SNEPPX_fuzz_register_target("checkpoint_load", fuzz_checkpoint_load);
}

/* GCC/Clang constructor — runs before main() or before libFuzzer's main() */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((constructor))
/**
 * @brief Perform Fuzz Auto Register.
 */
static void SNEPPX_fuzz_auto_register(void) {
    SNEPPX_fuzz_register_defaults();
}
#elif defined(_MSC_VER)
/* MSVC: use a static initializer via a dummy variable */
static int SNEPPX_fuzz_init = (SNEPPX_fuzz_register_defaults(), 0);
#endif

/* LibFuzzer entry point */
int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    /* Run all registered targets */
    for (int i = 0; i < g_num_targets; i++) {
        if (g_targets[i].target)
            g_targets[i].target(data, size);
    }
    return 0;
}

int main(void) {
    printf("SNEPPX Fuzz Harness (skeleton)\n");
    printf("Registered targets: %d\n", g_num_targets);
    return 0;
}
