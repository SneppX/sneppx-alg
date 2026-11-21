/*
 * SNEPPX - Kernel Security Obfuscator Header
 *
 * WHAT
 *   String obfuscation API for log messages and profiler output.
 *
 * CONCEPT
 *   Lightweight XOR-based obfuscation with per-process PRNG keying
 *   (avoids predictable rand() usage flagged in S2 audit).
 *
 * ROLE
 *   SNEPPX-Algo kernel security component.
 *
 * REFERENCES
 *   S2 audit findings: docs/security/obf_audit.md
 */

#ifndef SNEPPX_KERNEL_OBFUSCATOR_H
#define SNEPPX_KERNEL_OBFUSCATOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdarg.h>

/**
 * @brief Obfuscate a string via XOR with a per-process key.
 *
 * @param input [in] Plaintext string (null-terminated).
 * @param output [out] Output buffer for obfuscated bytes.
 * @param output_size [in] Size of output buffer.
 */
void sneppx_obfuscate_string(const char* input, char* output, size_t output_size);

/**
 * @brief Deobfuscate a previously obfuscated string (XOR is symmetric).
 *
 * @param input [in] Obfuscated string.
 * @param output [out] Output buffer for plaintext.
 * @param output_size [in] Size of output buffer.
 */
void sneppx_deobfuscate_string(const char* input, char* output, size_t output_size);

/**
 * @brief Log a formatted message with obfuscated content.
 *
 * @param fmt [in] printf-style format string.
 */
void log_obfuscated(const char* fmt, ...);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_KERNEL_OBFUSCATOR_H */
