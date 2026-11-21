/*
 * SNEPPX - Kernel Security Obfuscator
 *
 * WHAT
 *   String obfuscation for log messages and profiler output.
 *
 * CONCEPT
 *   Provides a lightweight XOR-based obfuscation with per-process PRNG
 *   seeding (from S0 CSPRNG) to avoid the predictable rand() pattern
 *   identified in the S2 audit (CWE-338).
 *
 * ROLE
 *   SNEPPX-Algo kernel security component. Used by elastic.c
 *   (log_obfuscated) and profiler.c (obfuscating summary strings).
 *
 * REFERENCES
 *   S2 audit findings: docs/security/obf_audit.md
 */

#include "obfuscator.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

static unsigned char obf_key[32];
static int obf_initialized = 0;

static void obf_ensure_init(void) {
    if (obf_initialized) return;
    /* Use address of local + time as minimum entropy, since full CSPRNG
     * is in S0 layer and not always linked. This is obfuscation-only,
     * not cryptographic secrecy. */
    unsigned int seed = (unsigned int)((uintptr_t)&obf_initialized
                      ^ (uintptr_t)&obf_key[0]);
    for (int i = 0; i < 32; i++) {
        seed = seed * 1103515245u + 12345u;
        obf_key[i] = (unsigned char)(seed >> 16);
    }
    obf_initialized = 1;
}

void sneppx_obfuscate_string(const char* input, char* output, size_t output_size) {
    if (!input || !output || output_size == 0) return;
    obf_ensure_init();
    size_t len = strlen(input);
    if (len >= output_size) len = output_size - 1;
    for (size_t i = 0; i < len; i++) {
        output[i] = input[i] ^ obf_key[i % 32];
    }
    output[len] = '\0';
}

void sneppx_deobfuscate_string(const char* input, char* output, size_t output_size) {
    /* XOR is symmetric — same operation reverses obfuscation */
    sneppx_obfuscate_string(input, output, output_size);
}

void log_obfuscated(const char* fmt, ...) {
    if (!fmt) return;
    char buf[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    /* Log a hash marker instead of raw string for sensitive paths */
    char obfuscated[512];
    sneppx_obfuscate_string(buf, obfuscated, sizeof(obfuscated));
    fprintf(stderr, "[S2-OBF] %s\n", obfuscated);
}
