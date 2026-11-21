#include "stack_canary_protection.h"
#include "cryptographic_random_generator.h"
#include "constant_time_operations.h"
#include <string.h>

/*
 * SNEPPX - Stack Canary Protection
 *
 * WHAT
 *   Generates, writes, and verifies stack canary values for buffer-overflow
 *   detection.
 *
 * CONCEPT
 *   A random canary value is generated from the CSPRNG and placed at the
 *   boundary of a stack buffer. Before the buffer is freed or returns, the
 *   canary is verified in constant time; a mismatch indicates a stack smash.
 *
 * ROLE
 *   Internal hardening module (stack canary). Used by secure_mem and the
 *   stack protector components.
 *
 * REFERENCES
 *   Internal hardening module.
 */

static uint64_t generation_counter = 0;

/**
 * @brief Generate a random stack canary value.
 * @param canary[out] Canary structure to fill.
 */
void SNEPPX_canary_generate(SNEPPXCanary* canary) {
    if (!canary) return;
    SNEPPX_random_bytes(canary->value, SNEPPX_CANARY_SIZE);
    canary->generation = generation_counter++;
}

/**
 * @brief Verify a canary value against stored memory in constant time.
 * @param expected[in] Expected canary value.
 * @param memory[in]   Memory region to compare against.
 * @return 1 if the canary matches, 0 otherwise.
 */
int SNEPPX_canary_verify(const SNEPPXCanary* expected, const uint8_t* memory) {
    if (!expected || !memory) return 0;
    return SNEPPX_ct_equal(expected->value, memory, SNEPPX_CANARY_SIZE);
}

/**
 * @brief Write a canary value to memory.
 * @param canary[in] Canary structure containing the value to write.
 * @param memory[out] Destination memory buffer.
 */
void SNEPPX_canary_write(const SNEPPXCanary* canary, uint8_t* memory) {
    if (!canary || !memory) return;
    memcpy(memory, canary->value, SNEPPX_CANARY_SIZE);
}
