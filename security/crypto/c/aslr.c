#include "address_space_randomization.h"
#include "cryptographic_random_generator.h"

/*
 * SNEPPX - Address Space Layout Randomization
 *
 * WHAT
 *   Generates random offsets and applies ASLR to base pointers for
 *   address-space layout randomization hardening.
 *
 * CONCEPT
 *   Produces a random offset aligned to the page boundary (4096 bytes)
 *   using the CSPRNG, then shifts a base pointer and shrinks the size
 *   to create a randomized view of an allocated region.
 *
 * ROLE
 *   Internal hardening module (S3 memory layout randomization). Used by
 *   the secure memory allocator and stack protector components.
 *
 * REFERENCES
 *   Internal hardening module.
 */

/**
 * @brief Set Aslr Random Off.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_aslr_random_offset(size_t max_offset) {
    if (max_offset == 0) return 0;
    uint64_t r;
    SNEPPX_random_bytes((uint8_t*)&r, sizeof(r));
    size_t offset = (size_t)(r % max_offset);
    size_t page = 4096;
    offset &= ~(page - 1);
    return offset;
}

/**
 * @brief Apply Aslr.
 *
 * @param base_ptr [out] Base Ptr value.
 * @param size [out] Size value.
 */
void SNEPPX_aslr_apply(void** base_ptr, size_t* size, size_t max_random) {
    if (!base_ptr || !*base_ptr || !size) return;
    size_t off = SNEPPX_aslr_random_offset(max_random);
    *base_ptr = (uint8_t*)*base_ptr + off;
    *size -= off;
}
