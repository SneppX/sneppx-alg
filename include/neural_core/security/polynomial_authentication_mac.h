#ifndef SNEPPX_POLY1305_H
#define SNEPPX_POLY1305_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/*
 * SNEPPX - Polynomial Authentication Mac
 *
 * WHAT
 *   Polynomial Authentication Mac.
 *
 * CONCEPT
 *   Provides the Polynomial Authentication Mac.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef struct {
    uint64_t r[5];
    uint64_t h[5];
    uint64_t s[2];
    uint8_t buf[16];
    size_t buflen;
} SNEPPXPoly1305State;

/**
 * @brief Initialize Poly1305.
 *
 * @param state [out] State value.
 */
void SNEPPX_poly1305_init(SNEPPXPoly1305State* state, const uint8_t key[32]);
/**
 * @brief Update Poly1305.
 *
 * @param state [out] State value.
 * @param data [in] Data value.
 * @param len [in] Len value.
 */
void SNEPPX_poly1305_update(SNEPPXPoly1305State* state, const uint8_t* data, size_t len);
/**
 * @brief Perform Poly1305 Finish.
 *
 * @param state [out] State value.
 */
void SNEPPX_poly1305_finish(SNEPPXPoly1305State* state, uint8_t mac[16]);


#ifdef __cplusplus
}
#endif
#endif
