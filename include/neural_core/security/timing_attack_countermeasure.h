#ifndef SNEPPX_TIMING_H
#define SNEPPX_TIMING_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
/*
 * SNEPPX - Timing Attack Countermeasures
 *
 * WHAT
 *   Timing Attack Countermeasures.
 *
 * CONCEPT
 *   Defenses against timing attacks exploiting data-dependent execution time.
 *
 * ROLE
 *   Layer S5 side-channel resistance used by all crypto modules handling secret keys.
 *
 * REFERENCES
 *   None (internal utility).
 */



/**
 * @brief Start Timing.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_timing_start(void);
/**
 * @brief Perform Timing End.
 *
 * @return 0 on success, -1 on error.
 */
uint64_t SNEPPX_timing_end(void);
/**
 * @brief Perform Timing Random Delay.
 *
 * @param min_ns [in] Min Ns value.
 * @param max_ns [in] Max Ns value.
 */
void SNEPPX_timing_random_delay(uint32_t min_ns, uint32_t max_ns);
/**
 * @brief Perform Timing Safe Equal.
 *
 * @param a [in] A value.
 * @param b [in] B value.
 * @param len [in] Len value.
 * @param timing_ns [out] Timing Ns value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_timing_safe_equal(const uint8_t* a, const uint8_t* b, size_t len, uint64_t* timing_ns);


#ifdef __cplusplus
}
#endif
#endif
