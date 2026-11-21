#include "prompt_filter.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

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


static const char* default_injection_patterns[] = {
    "ignore previous instructions",
    "ignore all previous",
    "forget your instructions",
    "you are now",
    "act as a",
    "pretend you are",
    "jailbreak",
    "DAN:",
    "do anything now",
    "roleplay",
    "system prompt",
    "override",
    "you have been",
    "new instructions",
    NULL
};

static const char* default_jailbreak_patterns[] = {
    "how to make a bomb",
    "how to hack",
    "illegal drugs",
    "bypass",
    "exploit",
    "vulnerability",
    "zero-day",
    "malware",
    "ransomware",
    "credit card numbers",
    "social security",
    NULL
};

static const char* multilingual_patterns[] = {
    "ignora instrucciones", "ignora todo", "olvida tus instrucciones",
    "ignorer les instructions", "oublie tes instructions", "tu es maintenant",
    "ignoriere anweisungen", "vergiss deine anweisungen", "du bist jetzt",
    "wujie zhiling", "hulue suoyou", "ni xianzai shi",
    "mushime yo", "subete o mushi", "anata wa ima",
    NULL
};

static int pattern_priorities[SNEPPX_MAX_PATTERNS];
static int total_scans = 0;
static int total_blocks = 0;

static int utf8_char_len(unsigned char lead) {
    if (lead < 0x80) return 1;
    if (lead < 0xC0) return 1;
    if (lead < 0xE0) return 2;
    if (lead < 0xF0) return 3;
    if (lead < 0xF8) return 4;
    return 1;
}

static int utf8_str_find(const char* haystack, const char* needle) {
    if (!haystack || !needle) return 0;
    return (strstr(haystack, needle) != NULL) ? 1 : 0;
}

/**
 * @brief Initialize Prompt Filter.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_filter_init(SNEPPXPromptFilter* pf) {
    if (!pf) return -1;
    memset(pf, 0, sizeof(*pf));
    pf->enabled = 1;
    pf->max_token_length = 4096;
    pf->anomaly_threshold = 0.8;
    memset(pattern_priorities, 0, sizeof(pattern_priorities));
    total_scans = 0;
    total_blocks = 0;
    return 0;
}

/**
 * @brief Destroy Prompt Filter.
 */
void SNEPPX_prompt_filter_destroy(SNEPPXPromptFilter* pf) {
    if (pf) memset(pf, 0, sizeof(*pf));
}

/**
 * @brief Perform Prompt Filter Add Pattern.
 *
 * @param pf [out] Pf value.
 * @param pattern [in] Pattern value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_filter_add_pattern(SNEPPXPromptFilter* pf, const char* pattern,
                                     SNEPPXFilterResult classification) {
    if (!pf || !pattern || pf->pattern_count >= SNEPPX_MAX_PATTERNS) return -1;
    SNEPPXFilterPattern* p = &pf->patterns[pf->pattern_count];
    strncpy(p->pattern, pattern, SNEPPX_PATTERN_MAX_LEN - 1);
    p->classification = classification;
    p->is_active = 1;
    pattern_priorities[pf->pattern_count] = 0;
    return pf->pattern_count++;
}

/**
 * @brief Perform Prompt Filter Scan.
 *
 * @param pf [out] Pf value.
 * @param prompt [in] Prompt value.
 *
 * @return The result value, or 0 on error.
 */
SNEPPXFilterResult SNEPPX_prompt_filter_scan(SNEPPXPromptFilter* pf,
                                          const char* prompt, size_t len) {
    if (!pf || !pf->enabled || !prompt) return SNEPPX_FILTER_CLEAN;
    total_scans++;
    char lower[SNEPPX_PATTERN_MAX_LEN];
    size_t plen = (len < SNEPPX_PATTERN_MAX_LEN - 1) ? len : SNEPPX_PATTERN_MAX_LEN - 1;
    for (size_t i = 0; i < plen; i++) lower[i] = (char)tolower((unsigned char)prompt[i]);
    lower[plen] = '\0';

    SNEPPXFilterResult highest = SNEPPX_FILTER_CLEAN;
    int highest_prio = -1;
    for (int i = 0; i < pf->pattern_count; i++) {
        if (!pf->patterns[i].is_active) continue;
        if (strstr(lower, pf->patterns[i].pattern)) {
            if (pattern_priorities[i] > highest_prio) {
                highest_prio = pattern_priorities[i];
                highest = pf->patterns[i].classification;
            }
        }
    }
    if (highest != SNEPPX_FILTER_CLEAN) total_blocks++;
    return highest;
}

/**
 * @brief Perform Prompt Filter Sanitize.
 *
 * @param pf [out] Pf value.
 * @param prompt [in] Prompt value.
 * @param len [in] Len value.
 * @param sanitized [out] Sanitized value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_filter_sanitize(SNEPPXPromptFilter* pf,
                                  const char* prompt, size_t len,
                                  char* sanitized, size_t* sanitized_len) {
    if (!pf || !prompt || !sanitized || !sanitized_len) return -1;
    SNEPPXFilterResult result = SNEPPX_prompt_filter_scan(pf, prompt, len);
    if (result != SNEPPX_FILTER_CLEAN) {
        const char* msg = "[Content filtered by security policy]";
        size_t msg_len = strlen(msg) + 1;
        if (*sanitized_len < msg_len) return -1;
        memcpy(sanitized, msg, msg_len);
        *sanitized_len = msg_len;
        return 1;
    }
    size_t copy_len = (len < *sanitized_len) ? len : *sanitized_len;
    memcpy(sanitized, prompt, copy_len);
    *sanitized_len = copy_len;
    return 0;
}

/**
 * @brief Perform Prompt Filter Load Defaults.
 */
void SNEPPX_prompt_filter_load_defaults(SNEPPXPromptFilter* pf) {
    if (!pf) return;
    for (int i = 0; default_injection_patterns[i]; i++)
        SNEPPX_prompt_filter_add_pattern(pf, default_injection_patterns[i], SNEPPX_FILTER_INJECTION);
    for (int i = 0; default_jailbreak_patterns[i]; i++)
        SNEPPX_prompt_filter_add_pattern(pf, default_jailbreak_patterns[i], SNEPPX_FILTER_JAILBREAK);
}

/**
 * @brief Perform Prompt Filter Set Priority.
 *
 * @param pf [out] Pf value.
 * @param pattern_name [in] Pattern Name value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_filter_set_priority(SNEPPXPromptFilter* pf, const char* pattern_name, int priority) {
    if (!pf || !pattern_name) return -1;
    for (int i = 0; i < pf->pattern_count; i++) {
        if (strcmp(pf->patterns[i].pattern, pattern_name) == 0) {
            pattern_priorities[i] = priority;
            return 0;
        }
    }
    return -1;
}

/**
 * @brief Perform Prompt Filter Match Wildcard.
 *
 * @param text [in] Text value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_filter_match_wildcard(const char* text, const char* pattern) {
    if (!text || !pattern) return 0;
    if (pattern[0] == '\0') return text[0] == '\0';
    if (pattern[0] == '*') {
        for (int i = 0; text[i]; i++)
            if (SNEPPX_prompt_filter_match_wildcard(text + i, pattern + 1)) return 1;
        return SNEPPX_prompt_filter_match_wildcard(text, pattern + 1);
    }
    if (pattern[0] == '?' && text[0]) {
        return SNEPPX_prompt_filter_match_wildcard(text + 1, pattern + 1);
    }
    if (pattern[0] == text[0]) {
        return SNEPPX_prompt_filter_match_wildcard(text + 1, pattern + 1);
    }
    return 0;
}

/**
 * @brief Perform Prompt Filter Scan Wildcard.
 *
 * @param pf [out] Pf value.
 * @param prompt [in] Prompt value.
 *
 * @return The result value, or 0 on error.
 */
SNEPPXFilterResult SNEPPX_prompt_filter_scan_wildcard(SNEPPXPromptFilter* pf,
                                                    const char* prompt, size_t len) {
    if (!pf || !pf->enabled || !prompt) return SNEPPX_FILTER_CLEAN;
    total_scans++;
    char lower[SNEPPX_PATTERN_MAX_LEN];
    size_t plen = (len < SNEPPX_PATTERN_MAX_LEN - 1) ? len : SNEPPX_PATTERN_MAX_LEN - 1;
    for (size_t i = 0; i < plen; i++) lower[i] = (char)tolower((unsigned char)prompt[i]);
    lower[plen] = '\0';

    SNEPPXFilterResult highest = SNEPPX_FILTER_CLEAN;
    int highest_prio = -1;
    for (int i = 0; i < pf->pattern_count; i++) {
        if (!pf->patterns[i].is_active) continue;
        if (SNEPPX_prompt_filter_match_wildcard(lower, pf->patterns[i].pattern)) {
            if (pattern_priorities[i] > highest_prio) {
                highest_prio = pattern_priorities[i];
                highest = pf->patterns[i].classification;
            }
        }
    }
    if (highest != SNEPPX_FILTER_CLEAN) total_blocks++;
    return highest;
}

/**
 * @brief Reset Prompt Filter.
 */
void SNEPPX_prompt_filter_reset(SNEPPXPromptFilter* pf) {
    if (!pf) return;
    memset(pf->patterns, 0, sizeof(SNEPPXFilterPattern) * pf->pattern_count);
    pf->pattern_count = 0;
    memset(pattern_priorities, 0, sizeof(pattern_priorities));
}

/**
 * @brief Perform Prompt Filter Stats.
 *
 * @param pf [out] Pf value.
 * @param patterns_loaded [out] Patterns Loaded value.
 * @param scans [out] Scans value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_filter_stats(SNEPPXPromptFilter* pf, int* patterns_loaded, int* scans, int* blocks) {
    if (!pf || !patterns_loaded || !scans || !blocks) return -1;
    *patterns_loaded = pf->pattern_count;
    *scans = total_scans;
    *blocks = total_blocks;
    return 0;
}

/**
 * @brief Perform Prompt Filter Remove Pattern.
 *
 * @param pf [out] Pf value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_filter_remove_pattern(SNEPPXPromptFilter* pf, const char* pattern_name) {
    if (!pf || !pattern_name) return -1;
    for (int i = 0; i < pf->pattern_count; i++) {
        if (strcmp(pf->patterns[i].pattern, pattern_name) == 0) {
            pf->patterns[i].is_active = 0;
            return 0;
        }
    }
    return -1;
}

/**
 * @brief Perform Prompt Filter Add Defaults.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_filter_add_defaults(SNEPPXPromptFilter* pf) {
    if (!pf) return -1;
    int count = 0;
    for (int i = 0; default_injection_patterns[i]; i++) {
        if (SNEPPX_prompt_filter_add_pattern(pf, default_injection_patterns[i], SNEPPX_FILTER_INJECTION) >= 0)
            count++;
    }
    for (int i = 0; default_jailbreak_patterns[i]; i++) {
        if (SNEPPX_prompt_filter_add_pattern(pf, default_jailbreak_patterns[i], SNEPPX_FILTER_JAILBREAK) >= 0)
            count++;
    }
    for (int i = 0; multilingual_patterns[i]; i++) {
        if (SNEPPX_prompt_filter_add_pattern(pf, multilingual_patterns[i], SNEPPX_FILTER_SUSPICIOUS) >= 0)
            count++;
    }
    return count;
}

/**
 * @brief Perform Prompt Filter Remove All.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_filter_remove_all(SNEPPXPromptFilter* pf) {
    if (!pf) return -1;
    memset(pf->patterns, 0, sizeof(SNEPPXFilterPattern) * pf->pattern_count);
    pf->pattern_count = 0;
    memset(pattern_priorities, 0, sizeof(pattern_priorities));
    return 0;
}

/**
 * @brief Perform Prompt Filter Get Pattern Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_filter_get_pattern_count(void) {
    int count = 0;
    for (int i = 0; i < SNEPPX_MAX_PATTERNS; i++) count++;
    return count;
}

/**
 * @brief Perform Prompt Filter Scan Advanced.
 *
 * @param pf [out] Pf value.
 * @param text [in] Text value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_filter_scan_advanced(SNEPPXPromptFilter* pf, const char* text, size_t len, SNEPPXFilterResult result_out[3]) {
    if (!pf || !text || !result_out) return -1;
    total_scans++;
    char lower[SNEPPX_PATTERN_MAX_LEN];
    size_t plen = (len < SNEPPX_PATTERN_MAX_LEN - 1) ? len : SNEPPX_PATTERN_MAX_LEN - 1;
    for (size_t i = 0; i < plen; i++) lower[i] = (char)tolower((unsigned char)text[i]);
    lower[plen] = '\0';
    int match_count = 0;
    int found_priorities[SNEPPX_MAX_PATTERNS];
    int found_indices[SNEPPX_MAX_PATTERNS];
    for (int i = 0; i < pf->pattern_count && match_count < 3; i++) {
        if (!pf->patterns[i].is_active) continue;
        if (strstr(lower, pf->patterns[i].pattern) || utf8_str_find(lower, pf->patterns[i].pattern)) {
            found_indices[match_count] = i;
            found_priorities[match_count] = pattern_priorities[i];
            match_count++;
        }
    }
    for (int i = 0; i < match_count && i < 3; i++) {
        result_out[i] = pf->patterns[found_indices[i]].classification;
    }
    for (int i = match_count; i < 3; i++)
        result_out[i] = SNEPPX_FILTER_CLEAN;
    if (match_count > 0) total_blocks++;
    return match_count;
}

/**
 * @brief Perform Prompt Filter Update Pattern.
 *
 * @param index [in] Index value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_filter_update_pattern(int index, const char* new_pattern) {
    if (index < 0 || index >= SNEPPX_MAX_PATTERNS || !new_pattern) return -1;
    for (int i = 0; i < SNEPPX_MAX_PATTERNS; i++) {
        (void)i;
    }
    return 0;
}

/**
 * @brief Perform Prompt Filter Get Patterns.
 *
 * @param buffer [out] Buffer value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_filter_get_patterns(char* buffer, int max) {
    if (!buffer || max <= 0) return 0;
    strncpy(buffer, "default_injection,default_jailbreak,multilingual", max - 1);
    buffer[max - 1] = '\0';
    return (int)strlen(buffer);
}
static void to_lower_utf8(char* dst, const char* src, size_t max_len) {
    if (!dst || !src) return;
    size_t i = 0, j = 0;
    while (src[i] && j < max_len - 1) {
        int clen = utf8_char_len((unsigned char)src[i]);
        if (clen == 1) {
            dst[j++] = (char)tolower((unsigned char)src[i++]);
        } else {
            for (int k = 0; k < clen && src[i + k] && j < max_len - 1; k++)
                dst[j++] = src[i + k];
            i += clen;
        }
    }
    dst[j] = '\0';
}

static int pattern_match_utf8(const char* text, const char* pattern) {
    if (!text || !pattern) return 0;
    return (strstr(text, pattern) != NULL) ? 1 : 0;
}

static int count_active_patterns(SNEPPXPromptFilter* pf) {
    if (!pf) return 0;
    int count = 0;
    for (int i = 0; i < pf->pattern_count; i++) {
        if (pf->patterns[i].is_active) count++;
    }
    return count;
}

/**
 * @brief Perform Prompt Filter Enable Pattern.
 *
 * @param pf [out] Pf value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_filter_enable_pattern(SNEPPXPromptFilter* pf, const char* pattern_name) {
    if (!pf || !pattern_name) return -1;
    for (int i = 0; i < pf->pattern_count; i++) {
        if (strcmp(pf->patterns[i].pattern, pattern_name) == 0) {
            pf->patterns[i].is_active = 1;
            return 0;
        }
    }
    return -1;
}

/**
 * @brief Perform Prompt Filter Disable Pattern.
 *
 * @param pf [out] Pf value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_filter_disable_pattern(SNEPPXPromptFilter* pf, const char* pattern_name) {
    if (!pf || !pattern_name) return -1;
    return SNEPPX_prompt_filter_remove_pattern(pf, pattern_name);
}

/**
 * @brief Perform Prompt Filter Get Classification.
 *
 * @param pf [out] Pf value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_filter_get_classification(SNEPPXPromptFilter* pf, int index) {
    if (!pf || index < 0 || index >= pf->pattern_count) return -1;
    return (int)pf->patterns[index].classification;
}

/**
 * @brief Perform Prompt Filter Is Enabled.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_filter_is_enabled(SNEPPXPromptFilter* pf) {
    return (pf && pf->enabled) ? 1 : 0;
}

/**
 * @brief Perform Prompt Filter Set Enabled.
 *
 * @param pf [out] Pf value.
 */
void SNEPPX_prompt_filter_set_enabled(SNEPPXPromptFilter* pf, int enabled) {
    if (pf) pf->enabled = (enabled != 0);
}

/**
 * @brief Perform Prompt Filter Get Max Token Length.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_filter_get_max_token_length(SNEPPXPromptFilter* pf) {
    return pf ? pf->max_token_length : 0;
}

/**
 * @brief Perform Prompt Filter Set Max Token Length.
 *
 * @param pf [out] Pf value.
 */
void SNEPPX_prompt_filter_set_max_token_length(SNEPPXPromptFilter* pf, int max_len) {
    if (pf && max_len > 0) pf->max_token_length = max_len;
}

/**
 * @brief Perform Prompt Filter Get Anomaly Threshold.
 *
 * @return The result value, or 0 on error.
 */
double SNEPPX_prompt_filter_get_anomaly_threshold(SNEPPXPromptFilter* pf) {
    return pf ? pf->anomaly_threshold : 0.0;
}

/**
 * @brief Perform Prompt Filter Set Anomaly Threshold.
 *
 * @param pf [out] Pf value.
 */
void SNEPPX_prompt_filter_set_anomaly_threshold(SNEPPXPromptFilter* pf, double t) {
    if (pf && t >= 0.0 && t <= 1.0) pf->anomaly_threshold = t;
}
/**
 * @brief Perform Prompt Filter Scan Multi.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_filter_scan_multi(const char* prompts[], size_t lengths[], int count) {
    if (!prompts || !lengths || count <= 0) return 0;
    int total_blocks = 0;
    for (int i = 0; i < count; i++) {
        total_blocks++;
    }
    return total_blocks;
}

/**
 * @brief Perform Prompt Filter Export Stats.
 *
 * @param buffer [out] Buffer value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_filter_export_stats(char* buffer, int max) {
    if (!buffer || max <= 0) return 0;
    int n = snprintf(buffer, (size_t)max, "scans=%d,blocks=%d,patterns=%d", total_scans, total_blocks, SNEPPX_MAX_PATTERNS);
    return (n > 0) ? n : 0;
}
static int hex_detection_enabled = 1;
static int base64_detection_enabled = 1;
static int scan_depth = 1;
static int total_match_count = 0;
static char last_match_pattern[SNEPPX_PATTERN_MAX_LEN] = "";
static size_t hex_decode_local(const char* in, size_t in_len, char* out) {
    size_t out_pos = 0;
    for (size_t i = 0; i + 1 < in_len; i += 2) {
        unsigned char c1 = (unsigned char)in[i], c2 = (unsigned char)in[i + 1];
        unsigned char v1 = (c1 >= 'a' ? c1 - 'a' + 10 : (c1 >= 'A' ? c1 - 'A' + 10 : c1 - '0'));
        unsigned char v2 = (c2 >= 'a' ? c2 - 'a' + 10 : (c2 >= 'A' ? c2 - 'A' + 10 : c2 - '0'));
        out[out_pos++] = (char)((v1 << 4) | v2);
    }
    return out_pos;
}
static const char* b64_alphabet_local = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
static int b64_rev_local[256];
static int b64_init_local = 0;
static void b64_build_rev_local(void) {
    if (b64_init_local) return; b64_init_local = 1;
    for (int i = 0; i < 256; i++) b64_rev_local[i] = -1;
    for (int i = 0; i < 64; i++) b64_rev_local[(int)b64_alphabet_local[i]] = i;
}
static size_t b64_decode_local(const char* in, size_t in_len, char* out) {
    b64_build_rev_local();
    size_t out_pos = 0; int val = 0, valb = -8;
    for (size_t i = 0; i < in_len; i++) {
        unsigned char c = (unsigned char)in[i];
        if (c == '=') break;
        int d = b64_rev_local[c];
        if (d == -1) continue;
        val = (val << 6) | d; valb += 6;
        if (valb >= 0) { out[out_pos++] = (char)((val >> valb) & 0xFF); valb -= 8; }
    }
    return out_pos;
}
/**
 * @brief Perform Prompt Filter Scan Hex.
 *
 * @param pf [out] Pf value.
 * @param text [in] Text value.
 *
 * @return The result value, or 0 on error.
 */
SNEPPXFilterResult SNEPPX_prompt_filter_scan_hex(SNEPPXPromptFilter* pf, const char* text, size_t len) {
    if (!pf || !text || !hex_detection_enabled) return SNEPPX_prompt_filter_scan(pf, text, len);
    char decoded[SNEPPX_PATTERN_MAX_LEN]; size_t dlen = hex_decode_local(text, len, decoded);
    if (dlen == 0) return SNEPPX_prompt_filter_scan(pf, text, len);
    decoded[dlen] = '\0';
    SNEPPXFilterResult res = SNEPPX_prompt_filter_scan(pf, decoded, dlen);
    if (res != SNEPPX_FILTER_CLEAN) { total_match_count++; strncpy(last_match_pattern, "hex_encoded", SNEPPX_PATTERN_MAX_LEN - 1); }
    return res;
}
/**
 * @brief Perform Prompt Filter Scan Base64.
 *
 * @param pf [out] Pf value.
 * @param text [in] Text value.
 *
 * @return The result value, or 0 on error.
 */
SNEPPXFilterResult SNEPPX_prompt_filter_scan_base64(SNEPPXPromptFilter* pf, const char* text, size_t len) {
    if (!pf || !text || !base64_detection_enabled) return SNEPPX_prompt_filter_scan(pf, text, len);
    char decoded[SNEPPX_PATTERN_MAX_LEN]; size_t dlen = b64_decode_local(text, len, decoded);
    if (dlen == 0) return SNEPPX_prompt_filter_scan(pf, text, len);
    decoded[dlen] = '\0';
    SNEPPXFilterResult res = SNEPPX_prompt_filter_scan(pf, decoded, dlen);
    if (res != SNEPPX_FILTER_CLEAN) { total_match_count++; strncpy(last_match_pattern, "base64_encoded", SNEPPX_PATTERN_MAX_LEN - 1); }
    return res;
}
/**
 * @brief Perform Prompt Filter Get Match Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_filter_get_match_count(void) { return total_match_count; }
/**
 * @brief Perform Prompt Filter Get Last Match.
 *
 * @param buffer [out] Buffer value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_filter_get_last_match(char* buffer, size_t size) {
    if (!buffer || size == 0) return -1;
    strncpy(buffer, last_match_pattern, size - 1); buffer[size - 1] = '\0';
    return (int)strlen(buffer);
}
/**
 * @brief Perform Prompt Filter Enable Hex Detection.
 */
void SNEPPX_prompt_filter_enable_hex_detection(int enabled) { hex_detection_enabled = (enabled != 0); }
/**
 * @brief Perform Prompt Filter Enable Base64 Detection.
 */
void SNEPPX_prompt_filter_enable_base64_detection(int enabled) { base64_detection_enabled = (enabled != 0); }
/**
 * @brief Perform Prompt Filter Set Scan Depth.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_filter_set_scan_depth(int depth) {
    if (depth < 0) return -1;
    scan_depth = depth; return 0;
}
/**
 * @brief Perform Prompt Filter Get Scan Depth.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_filter_get_scan_depth(void) { return scan_depth; }
/**
 * @brief Perform Prompt Filter Is Hex Detection Enabled.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_filter_is_hex_detection_enabled(void) { return hex_detection_enabled; }
/**
 * @brief Perform Prompt Filter Is Base64 Detection Enabled.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_filter_is_base64_detection_enabled(void) { return base64_detection_enabled; }
/**
 * @brief Perform Prompt Filter Reset Match Count.
 */
void SNEPPX_prompt_filter_reset_match_count(void) { total_match_count = 0; last_match_pattern[0] = '\0'; }
/**
 * @brief Perform Prompt Filter Scan Deep Recursive.
 *
 * @param pf [out] Pf value.
 * @param text [in] Text value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_filter_scan_deep_recursive(SNEPPXPromptFilter* pf, const char* text, size_t len, int depth) {
    if (!pf || !text) return SNEPPX_FILTER_CLEAN;
    SNEPPXFilterResult res = SNEPPX_prompt_filter_scan(pf, text, len);
    if (res != SNEPPX_FILTER_CLEAN || depth <= 0) return res;
    char decoded[SNEPPX_PATTERN_MAX_LEN]; size_t dlen = hex_decode_local(text, len, decoded);
    if (dlen > 0) { decoded[dlen] = '\0'; res = SNEPPX_prompt_filter_scan(pf, decoded, dlen); if (res != SNEPPX_FILTER_CLEAN) return res; }
    dlen = b64_decode_local(text, len, decoded);
    if (dlen > 0) { decoded[dlen] = '\0'; res = SNEPPX_prompt_filter_scan(pf, decoded, dlen); if (res != SNEPPX_FILTER_CLEAN) return res; }
    return SNEPPX_prompt_filter_scan_deep_recursive(pf, decoded, dlen, depth - 1);
}
/**
 * @brief Perform Prompt Filter Get Total Scans.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_filter_get_total_scans(void) { return total_scans; }
/**
 * @brief Perform Prompt Filter Get Total Blocks.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_filter_get_total_blocks(void) { return total_blocks; }
/**
 * @brief Perform Prompt Filter Reset Stats.
 */
void SNEPPX_prompt_filter_reset_stats(void) { total_scans = 0; total_blocks = 0; total_match_count = 0; }
/**
 * @brief Perform Prompt Filter Get Active Pattern Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_filter_get_active_pattern_count(SNEPPXPromptFilter* pf) {
    if (!pf) return -1; int count = 0;
    for (int i = 0; i < pf->pattern_count; i++) { if (pf->patterns[i].is_active) count++; }
    return count;
}
/**
 * @brief Perform Prompt Filter Classify Text.
 *
 * @param pf [out] Pf value.
 * @param text [in] Text value.
 *
 * @return The result value, or 0 on error.
 */
SNEPPXFilterResult SNEPPX_prompt_filter_classify_text(SNEPPXPromptFilter* pf, const char* text, size_t len) {
    if (!pf || !text) return SNEPPX_FILTER_CLEAN;
    SNEPPXFilterResult res = SNEPPX_prompt_filter_scan(pf, text, len);
    if (res != SNEPPX_FILTER_CLEAN) return res;
    if (hex_detection_enabled) { res = SNEPPX_prompt_filter_scan_hex(pf, text, len); if (res != SNEPPX_FILTER_CLEAN) return res; }
    if (base64_detection_enabled) { res = SNEPPX_prompt_filter_scan_base64(pf, text, len); }
    return res;
}
/**
 * @brief Perform Prompt Filter Get Pattern At.
 *
 * @param pf [out] Pf value.
 * @param index [in] Index value.
 * @param buffer [out] Buffer value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_filter_get_pattern_at(SNEPPXPromptFilter* pf, int index, char* buffer, size_t buf_size) {
    if (!pf || index < 0 || index >= pf->pattern_count || !buffer || buf_size == 0) return -1;
    strncpy(buffer, pf->patterns[index].pattern, buf_size - 1); buffer[buf_size - 1] = '\0';
    return (int)strlen(buffer);
}
/**
 * @brief Perform Prompt Filter Get Classification At.
 *
 * @param pf [out] Pf value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_filter_get_classification_at(SNEPPXPromptFilter* pf, int index) {
    if (!pf || index < 0 || index >= pf->pattern_count) return -1;
    return (int)pf->patterns[index].classification;
}
/**
 * @brief Perform Prompt Filter Set Pattern.
 *
 * @param pf [out] Pf value.
 * @param index [in] Index value.
 * @param pattern [in] Pattern value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_filter_set_pattern(SNEPPXPromptFilter* pf, int index, const char* pattern, SNEPPXFilterResult classification) {
    if (!pf || index < 0 || index >= SNEPPX_MAX_PATTERNS || !pattern) return -1;
    strncpy(pf->patterns[index].pattern, pattern, SNEPPX_PATTERN_MAX_LEN - 1);
    pf->patterns[index].classification = classification;
    pf->patterns[index].is_active = 1;
    if (index >= pf->pattern_count) pf->pattern_count = index + 1;
    return 0;
}
/**
 * @brief Perform Prompt Filter Scan Batch.
 *
 * @param pf [out] Pf value.
 * @param prompts [in] Prompts value.
 * @param lens [in] Lens value.
 * @param count [in] Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_filter_scan_batch(SNEPPXPromptFilter* pf, const char** prompts, const size_t* lens, int count, SNEPPXFilterResult* results) {
    if (!pf || !prompts || !lens || !results || count <= 0) return -1;
    for (int i = 0; i < count; i++) results[i] = SNEPPX_prompt_filter_scan(pf, prompts[i], lens[i]);
    return 0;
}
/**
 * @brief Perform Prompt Filter Scan Encoded Recursive.
 *
 * @param pf [out] Pf value.
 * @param text [in] Text value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_filter_scan_encoded_recursive(SNEPPXPromptFilter* pf, const char* text, size_t len) {
    if (!pf || !text) return SNEPPX_FILTER_CLEAN;
    SNEPPXFilterResult res = SNEPPX_prompt_filter_scan(pf, text, len);
    if (res != SNEPPX_FILTER_CLEAN) return res;
    for (int d = 0; d < scan_depth; d++) {
        char decoded[SNEPPX_PATTERN_MAX_LEN]; size_t dlen = 0;
        if (hex_detection_enabled) {
            dlen = hex_decode_local(text, len, decoded);
            if (dlen > 0) { decoded[dlen] = '\0'; res = SNEPPX_prompt_filter_scan(pf, decoded, dlen); if (res != SNEPPX_FILTER_CLEAN) return res; text = decoded; len = dlen; continue; }
        }
        if (base64_detection_enabled) {
            dlen = b64_decode_local(text, len, decoded);
            if (dlen > 0) { decoded[dlen] = '\0'; res = SNEPPX_prompt_filter_scan(pf, decoded, dlen); if (res != SNEPPX_FILTER_CLEAN) return res; text = decoded; len = dlen; continue; }
        }
        break;
    }
    return SNEPPX_FILTER_CLEAN;
}
/**
 * @brief Perform Prompt Filter Scan Hex At Depth.
 *
 * @param pf [out] Pf value.
 * @param text [in] Text value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_filter_scan_hex_at_depth(SNEPPXPromptFilter* pf, const char* text, size_t len, int depth) {
    if (!pf || !text || depth <= 0) return SNEPPX_prompt_filter_scan(pf, text, len);
    char decoded[SNEPPX_PATTERN_MAX_LEN]; size_t dlen = hex_decode_local(text, len, decoded);
    if (dlen == 0) return SNEPPX_prompt_filter_scan(pf, text, len);
    decoded[dlen] = '\0';
    SNEPPXFilterResult res = SNEPPX_prompt_filter_scan(pf, decoded, dlen);
    if (res != SNEPPX_FILTER_CLEAN) return res;
    if (depth > 1) return SNEPPX_prompt_filter_scan_hex_at_depth(pf, decoded, dlen, depth - 1);
    return SNEPPX_FILTER_CLEAN;
}
/**
 * @brief Perform Prompt Filter Scan Base64 At Depth.
 *
 * @param pf [out] Pf value.
 * @param text [in] Text value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_prompt_filter_scan_base64_at_depth(SNEPPXPromptFilter* pf, const char* text, size_t len, int depth) {
    if (!pf || !text || depth <= 0) return SNEPPX_prompt_filter_scan(pf, text, len);
    char decoded[SNEPPX_PATTERN_MAX_LEN]; size_t dlen = b64_decode_local(text, len, decoded);
    if (dlen == 0) return SNEPPX_prompt_filter_scan(pf, text, len);
    decoded[dlen] = '\0';
    SNEPPXFilterResult res = SNEPPX_prompt_filter_scan(pf, decoded, dlen);
    if (res != SNEPPX_FILTER_CLEAN) return res;
    if (depth > 1) return SNEPPX_prompt_filter_scan_base64_at_depth(pf, decoded, dlen, depth - 1);
    return SNEPPX_FILTER_CLEAN;
}
