#ifndef SNEPPX_PROMPT_FILTER_H
#define SNEPPX_PROMPT_FILTER_H
/*
 * SNEPPX - Prompt Filter
 *
 * WHAT
 *   Prompt Filter.
 *
 * CONCEPT
 *   Provides the Prompt Filter.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/*
 * S5 AI Sanitizer — Prompt Injection & Jailbreak Detection
 * Filters incoming prompts for known injection patterns, jailbreak attempts,
 * and adversarial instructions.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_MAX_PATTERNS 256
#define SNEPPX_PATTERN_MAX_LEN 128

typedef enum {
    SNEPPX_FILTER_CLEAN = 0,
    SNEPPX_FILTER_INJECTION = 1,
    SNEPPX_FILTER_JAILBREAK = 2,
    SNEPPX_FILTER_ADVERSARIAL = 3,
    SNEPPX_FILTER_SUSPICIOUS = 4,
} SNEPPXFilterResult;

typedef struct {
    char pattern[SNEPPX_PATTERN_MAX_LEN];
    SNEPPXFilterResult classification;
    int is_active;
} SNEPPXFilterPattern;

typedef struct {
    SNEPPXFilterPattern patterns[SNEPPX_MAX_PATTERNS];
    int pattern_count;
    int enabled;
    int max_token_length;
    double anomaly_threshold;
} SNEPPXPromptFilter;

/**
 * @brief Initialize Prompt Filter.
 *
 * @param pf [out] Pf value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_prompt_filter_init(SNEPPXPromptFilter* pf);
/**
 * @brief Destroy Prompt Filter.
 *
 * @param pf [out] Pf value.
 */
void SNEPPX_prompt_filter_destroy(SNEPPXPromptFilter* pf);
/**
 * @brief Perform Prompt Filter Add Pattern.
 *
 * @param pf [out] Pf value.
 * @param pattern [in] Pattern value.
 * @param classification [in] Classification value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_prompt_filter_add_pattern(SNEPPXPromptFilter* pf, const char* pattern,
                                     SNEPPXFilterResult classification);
/**
 * @brief Perform Prompt Filter Scan.
 *
 * @param pf [out] Pf value.
 * @param prompt [in] Prompt value.
 * @param len [in] Len value.
 *
 * @return The result value, or 0 on error.
 */
SNEPPXFilterResult SNEPPX_prompt_filter_scan(SNEPPXPromptFilter* pf,
                                          const char* prompt, size_t len);
/**
 * @brief Perform Prompt Filter Sanitize.
 *
 * @param pf [out] Pf value.
 * @param prompt [in] Prompt value.
 * @param len [in] Len value.
 * @param sanitized [out] Sanitized value.
 * @param sanitized_len [out] Sanitized Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_prompt_filter_sanitize(SNEPPXPromptFilter* pf,
                                  const char* prompt, size_t len,
                                  char* sanitized, size_t* sanitized_len);
/**
 * @brief Perform Prompt Filter Load Defaults.
 *
 * @param pf [out] Pf value.
 */
void SNEPPX_prompt_filter_load_defaults(SNEPPXPromptFilter* pf);

#ifdef __cplusplus
}
#endif
#endif
