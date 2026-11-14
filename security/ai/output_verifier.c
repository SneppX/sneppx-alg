#include "output_verifier.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

/*
 * SNEPPX - Output Verifier
 *
 * WHAT
 *   Output Verifier.
 *
 * CONCEPT
 *   Provides the Output Verifier.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef struct {
    uint64_t total_checks;
    uint64_t total_blocks;
    uint64_t pii_hits;
    uint64_t profanity_hits;
    uint64_t toxicity_hits;
    uint64_t bias_hits;
} SNEPPXOutputVerifierStats;

#define SNEPPX_MAX_ALLOWED_TOPICS 64
#define SNEPPX_PII_EMAIL 1
#define SNEPPX_PII_PHONE 2
#define SNEPPX_PII_SSN 3

static const char* default_blocked_topics[] = {
    "how to make weapons",
    "illegal activities",
    "child exploitation",
    "self-harm",
    "suicide methods",
    "hate speech",
    "discrimination",
    NULL
};

static const char* profanity_list[] = {
    "fuck", "shit", "ass", "bitch", "damn", "crap", "dick", "bastard",
    NULL
};

static char allowed_topics[SNEPPX_MAX_ALLOWED_TOPICS][SNEPPX_TOPIC_MAX_LEN];
static int allowed_topic_count = 0;
static int verifier_block_count = 0;

static int pii_detection_enabled = 1;
static int total_checks = 0;
static int total_pii_hits = 0;
static int total_profanity_hits = 0;
static int total_toxicity_hits = 0;
static int total_bias_hits = 0;

/**
 * @brief Initialize Output Verifier.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_init(SNEPPXS5Verifier* ov) {
    if (!ov) return -1;
    memset(ov, 0, sizeof(*ov));
    ov->toxicity_threshold = 0.7;
    ov->bias_threshold = 0.6;
    ov->check_factual_consistency = 1;
    ov->max_output_length = 8192;
    for (int i = 0; default_blocked_topics[i]; i++)
        SNEPPX_output_verifier_add_blocked_topic(ov, default_blocked_topics[i]);
    allowed_topic_count = 0;
    verifier_block_count = 0;
    pii_detection_enabled = 1;
    total_checks = 0;
    total_pii_hits = 0;
    total_profanity_hits = 0;
    total_toxicity_hits = 0;
    total_bias_hits = 0;
    return 0;
}

/**
 * @brief Destroy Output Verifier.
 */
void SNEPPX_output_verifier_destroy(SNEPPXS5Verifier* ov) {
    if (ov) memset(ov, 0, sizeof(*ov));
}

/**
 * @brief Perform Output Verifier Add Blocked Topic.
 *
 * @param ov [out] Ov value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_add_blocked_topic(SNEPPXS5Verifier* ov, const char* topic) {
    if (!ov || !topic || ov->topic_count >= SNEPPX_MAX_TOPIC_BLOCKLIST) return -1;
    SNEPPXBlockedTopic* bt = &ov->topics[ov->topic_count];
    strncpy(bt->topic, topic, SNEPPX_TOPIC_MAX_LEN - 1);
    bt->is_blocked = 1;
    return ov->topic_count++;
}

/**
 * @brief Perform Output Verifier Check.
 *
 * @param ov [out] Ov value.
 * @param output [in] Output value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_check(SNEPPXS5Verifier* ov, const char* output, size_t len) {
    if (!ov || !output) return 0;
    total_checks++;
    char lower[512];
    size_t clen = (len < sizeof(lower) - 1) ? len : sizeof(lower) - 1;
    for (size_t i = 0; i < clen; i++) lower[i] = (char)tolower((unsigned char)output[i]);
    lower[clen] = '\0';

    for (int i = 0; i < allowed_topic_count; i++) {
        if (strstr(lower, allowed_topics[i]))
            return 0;
    }

    for (int i = 0; i < ov->topic_count; i++) {
        if (ov->topics[i].is_blocked && strstr(lower, ov->topics[i].topic)) {
            verifier_block_count++;
            total_toxicity_hits++;
            return 1;
        }
    }
    return 0;
}

/**
 * @brief Perform Output Verifier Sanitize.
 *
 * @param ov [out] Ov value.
 * @param output [in] Output value.
 * @param len [in] Len value.
 * @param safe_output [out] Safe Output value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_sanitize(SNEPPXS5Verifier* ov,
                                    const char* output, size_t len,
                                    char* safe_output, size_t* safe_len) {
    if (!ov || !output || !safe_output || !safe_len) return -1;
    if (SNEPPX_output_verifier_check(ov, output, len)) {
        const char* msg = "[Output blocked by content policy]";
        size_t msg_len = strlen(msg) + 1;
        if (*safe_len < msg_len) return -1;
        memcpy(safe_output, msg, msg_len);
        *safe_len = msg_len;
        return 1;
    }
    size_t copy_len = (len < *safe_len) ? len : *safe_len;
    memcpy(safe_output, output, copy_len);
    *safe_len = copy_len;
    return 0;
}

/**
 * @brief Perform Output Verifier Add Allowed Topic.
 *
 * @param ov [out] Ov value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_add_allowed_topic(SNEPPXS5Verifier* ov, const char* topic) {
    (void)ov;
    if (!topic || allowed_topic_count >= SNEPPX_MAX_ALLOWED_TOPICS) return -1;
    strncpy(allowed_topics[allowed_topic_count], topic, SNEPPX_TOPIC_MAX_LEN - 1);
    allowed_topic_count++;
    return 0;
}

/**
 * @brief Perform Output Verifier Reset Topics.
 */
void SNEPPX_output_verifier_reset_topics(void) {
    allowed_topic_count = 0;
}

/**
 * @brief Perform Output Verifier Get Stats.
 *
 * @param ov [out] Ov value.
 * @param topic_count [out] Topic Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_get_stats(SNEPPXS5Verifier* ov, int* topic_count, int* block_count) {
    if (!ov || !topic_count || !block_count) return -1;
    *topic_count = ov->topic_count;
    *block_count = verifier_block_count;
    return 0;
}

/**
 * @brief Perform Output Verifier Check Full.
 *
 * @param ov [out] Ov value.
 * @param output [in] Output value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_check_full(SNEPPXS5Verifier* ov, const char* output, size_t len) {
    if (!ov || !output) return 0;
    if (SNEPPX_output_verifier_check(ov, output, len)) return 1;

    char lower[512];
    size_t clen = (len < sizeof(lower) - 1) ? len : sizeof(lower) - 1;
    for (size_t i = 0; i < clen; i++) lower[i] = (char)tolower((unsigned char)output[i]);
    lower[clen] = '\0';

    int has_at = 0, has_dot = 0;
    for (size_t i = 0; i < clen; i++) {
        if (lower[i] == '@') has_at = 1;
        if (lower[i] == '.') has_dot = 1;
    }
    int digit_count = 0;
    for (size_t i = 0; i < clen; i++) {
        if (lower[i] >= '0' && lower[i] <= '9') digit_count++;
    }
    if (has_at && has_dot && digit_count > 0) {
        verifier_block_count++;
        return 4;
    }
    if (digit_count >= 7) {
        verifier_block_count++;
        return 5;
    }
    return 0;
}

static int has_email_pattern(const char* text, size_t len) {
    int at_pos = -1;
    for (size_t i = 0; i < len; i++) {
        if (text[i] == '@') { at_pos = (int)i; break; }
    }
    if (at_pos <= 0) return 0;
    int has_dot_after = 0;
    for (size_t i = (size_t)(at_pos + 1); i < len; i++) {
        if (text[i] == '.') has_dot_after = 1;
    }
    return has_dot_after;
}

static int has_phone_pattern(const char* text, size_t len) {
    int digits = 0;
    for (size_t i = 0; i < len && digits < 10; i++) {
        if (text[i] >= '0' && text[i] <= '9') digits++;
    }
    return (digits >= 10) ? 1 : 0;
}

static int has_ssn_pattern(const char* text, size_t len) {
    int digit_groups = 0;
    int cur_digits = 0;
    for (size_t i = 0; i < len; i++) {
        if (text[i] >= '0' && text[i] <= '9') {
            cur_digits++;
        } else if (text[i] == '-' || text[i] == ' ') {
            if (cur_digits > 0) digit_groups++;
            cur_digits = 0;
        } else {
            cur_digits = 0;
        }
    }
    if (cur_digits > 0) digit_groups++;
    return (digit_groups >= 3) ? 1 : 0;
}

/**
 * @brief Perform Output Verifier Check Pii.
 *
 * @param text [in] Text value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_check_pii(const char* text, size_t len) {
    if (!text || len == 0 || !pii_detection_enabled) return 0;
    total_checks++;
    char lower[512];
    size_t clen = (len < sizeof(lower) - 1) ? len : sizeof(lower) - 1;
    for (size_t i = 0; i < clen; i++) lower[i] = (char)tolower((unsigned char)text[i]);
    lower[clen] = '\0';
    if (has_email_pattern(lower, clen)) { total_pii_hits++; return SNEPPX_PII_EMAIL; }
    if (has_phone_pattern(lower, clen)) { total_pii_hits++; return SNEPPX_PII_PHONE; }
    if (has_ssn_pattern(lower, clen)) { total_pii_hits++; return SNEPPX_PII_SSN; }
    return 0;
}

/**
 * @brief Perform Output Verifier Check Profanity.
 *
 * @param text [in] Text value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_check_profanity(const char* text, size_t len) {
    if (!text || len == 0) return 0;
    total_checks++;
    char lower[512];
    size_t clen = (len < sizeof(lower) - 1) ? len : sizeof(lower) - 1;
    for (size_t i = 0; i < clen; i++) lower[i] = (char)tolower((unsigned char)text[i]);
    lower[clen] = '\0';
    for (int i = 0; profanity_list[i]; i++) {
        if (strstr(lower, profanity_list[i])) {
            total_profanity_hits++;
            return 1;
        }
    }
    return 0;
}

/**
 * @brief Perform Output Verifier Set Toxicity Threshold.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_set_toxicity_threshold(double t) {
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;
    for (int i = 0; i < SNEPPX_MAX_TOPIC_BLOCKLIST; i++) (void)i;
    return 0;
}

/**
 * @brief Perform Output Verifier Set Bias Threshold.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_set_bias_threshold(double t) {
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;
    for (int i = 0; i < SNEPPX_MAX_TOPIC_BLOCKLIST; i++) (void)i;
    return 0;
}

/**
 * @brief Perform Output Verifier Query Stats.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_query_stats(SNEPPXOutputVerifierStats* stats) {
    if (!stats) return -1;
    memset(stats, 0, sizeof(*stats));
    stats->total_checks = total_checks;
    stats->total_blocks = verifier_block_count;
    stats->pii_hits = total_pii_hits;
    stats->profanity_hits = total_profanity_hits;
    stats->toxicity_hits = total_toxicity_hits;
    stats->bias_hits = total_bias_hits;
    return 0;
}

/**
 * @brief Perform Output Verifier Reset Stats.
 */
void SNEPPX_output_verifier_reset_stats(void) {
    total_checks = 0;
    verifier_block_count = 0;
    total_pii_hits = 0;
    total_profanity_hits = 0;
    total_toxicity_hits = 0;
    total_bias_hits = 0;
}

/**
 * @brief Perform Output Verifier Enable Pii Detection.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_enable_pii_detection(int enabled) {
    pii_detection_enabled = (enabled != 0);
    return 0;
}
static int has_url_pattern(const char* text, size_t len) {
    const char* prefixes[] = {"http://", "https://", "ftp://", "www.", NULL};
    for (int p = 0; prefixes[p]; p++) {
        if (strstr(text, prefixes[p])) return 1;
    }
    return 0;
}

static int digit_sequence_length(const char* text, size_t len) {
    int max_digits = 0, cur = 0;
    for (size_t i = 0; i < len; i++) {
        if (text[i] >= '0' && text[i] <= '9') { cur++; if (cur > max_digits) max_digits = cur; }
        else cur = 0;
    }
    return max_digits;
}

static int keyword_density(const char* text, size_t len, const char** keywords) {
    char lower[512];
    size_t clen = (len < sizeof(lower) - 1) ? len : sizeof(lower) - 1;
    for (size_t i = 0; i < clen; i++) lower[i] = (char)tolower((unsigned char)text[i]);
    lower[clen] = '\0';
    int matches = 0;
    for (int k = 0; keywords[k]; k++) {
        const char* pos = lower;
        while ((pos = strstr(pos, keywords[k])) != NULL) { matches++; pos++; }
    }
    return matches;
}

/**
 * @brief Perform Output Verifier Is Pii Detection Enabled.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_is_pii_detection_enabled(void) {
    return pii_detection_enabled;
}

/**
 * @brief Perform Output Verifier Get Total Checks.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_get_total_checks(void) {
    return total_checks;
}

/**
 * @brief Perform Output Verifier Get Block Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_get_block_count(void) {
    return verifier_block_count;
}

/**
 * @brief Perform Output Verifier Get Profanity Hits.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_get_profanity_hits(void) {
    return total_profanity_hits;
}

/**
 * @brief Perform Output Verifier Get Toxicity Hits.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_get_toxicity_hits(void) {
    return total_toxicity_hits;
}

/**
 * @brief Perform Output Verifier Get Bias Hits.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_get_bias_hits(void) {
    return total_bias_hits;
}

/**
 * @brief Perform Output Verifier Get Toxicity Threshold.
 *
 * @return The result value, or 0 on error.
 */
double SNEPPX_output_verifier_get_toxicity_threshold(SNEPPXS5Verifier* ov) {
    return ov ? ov->toxicity_threshold : 0.0;
}

/**
 * @brief Perform Output Verifier Get Bias Threshold.
 *
 * @return The result value, or 0 on error.
 */
double SNEPPX_output_verifier_get_bias_threshold(SNEPPXS5Verifier* ov) {
    return ov ? ov->bias_threshold : 0.0;
}

/**
 * @brief Perform Output Verifier Get Allowed Topic Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_get_allowed_topic_count(void) {
    return allowed_topic_count;
}

/**
 * @brief Perform Output Verifier Add Allowed Topic By Index.
 *
 * @param ov [out] Ov value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_add_allowed_topic_by_index(SNEPPXS5Verifier* ov, int index) {
    (void)ov; (void)index;
    return -1;
}

/**
 * @brief Perform Output Verifier Remove Allowed Topic.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_remove_allowed_topic(const char* topic) {
    if (!topic) return -1;
    for (int i = 0; i < allowed_topic_count; i++) {
        if (strcmp(allowed_topics[i], topic) == 0) {
            for (int j = i; j < allowed_topic_count - 1; j++)
                strncpy(allowed_topics[j], allowed_topics[j + 1], SNEPPX_TOPIC_MAX_LEN);
            allowed_topic_count--;
            return 0;
        }
    }
    return -1;
}

/**
 * @brief Perform Output Verifier Blocked Topic Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_blocked_topic_count(SNEPPXS5Verifier* ov) {
    return ov ? ov->topic_count : 0;
}

/**
 * @brief Perform Output Verifier Get Blocked Topic.
 *
 * @param ov [out] Ov value.
 *
 * @return Pointer on success, NULL on error.
 */
const char* SNEPPX_output_verifier_get_blocked_topic(SNEPPXS5Verifier* ov, int index) {
    if (!ov || index < 0 || index >= ov->topic_count) return NULL;
    return ov->topics[index].topic;
}

/**
 * @brief Perform Output Verifier Unblock Topic.
 *
 * @param ov [out] Ov value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_unblock_topic(SNEPPXS5Verifier* ov, const char* topic) {
    if (!ov || !topic) return -1;
    for (int i = 0; i < ov->topic_count; i++) {
        if (strcmp(ov->topics[i].topic, topic) == 0) {
            ov->topics[i].is_blocked = 0;
            return 0;
        }
    }
    return -1;
}
static int content_filtering_enabled = 1;
static int total_pii_items_found = 0;
/**
 * @brief Perform Output Verifier Check Pii Emails.
 *
 * @param text [in] Text value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_check_pii_emails(const char* text, size_t len) {
    if (!text || len == 0) return 0;
    char lower[512]; size_t clen = (len < sizeof(lower) - 1) ? len : sizeof(lower) - 1;
    for (size_t i = 0; i < clen; i++) lower[i] = (char)tolower((unsigned char)text[i]);
    lower[clen] = '\0';
    int at_pos = -1;
    for (size_t i = 0; i < clen; i++) { if (lower[i] == '@') { at_pos = (int)i; break; } }
    if (at_pos <= 0) return 0;
    int has_dot_after = 0;
    for (size_t i = (size_t)(at_pos + 1); i < clen; i++) { if (lower[i] == '.') { has_dot_after = 1; break; } }
    int has_prefix = 0;
    for (int i = 0; i < at_pos && i < 20; i++) { if (lower[i] >= 'a' && lower[i] <= 'z') has_prefix = 1; }
    if (has_dot_after && has_prefix) { total_pii_items_found++; return 1; }
    return 0;
}
/**
 * @brief Perform Output Verifier Check Pii Phones.
 *
 * @param text [in] Text value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_check_pii_phones(const char* text, size_t len) {
    if (!text || len == 0) return 0;
    int digits = 0;
    for (size_t i = 0; i < len && digits < 10; i++) {
        char c = text[i];
        if (c >= '0' && c <= '9') digits++;
    }
    if (digits >= 10) { total_pii_items_found++; return 1; }
    return 0;
}
/**
 * @brief Perform Output Verifier Check Pii Ssn.
 *
 * @param text [in] Text value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_check_pii_ssn(const char* text, size_t len) {
    if (!text || len == 0) return 0;
    int digit_groups = 0, cur_digits = 0;
    for (size_t i = 0; i < len; i++) {
        if (text[i] >= '0' && text[i] <= '9') cur_digits++;
        else if (text[i] == '-' || text[i] == ' ') { if (cur_digits > 0) digit_groups++; cur_digits = 0; }
        else cur_digits = 0;
    }
    if (cur_digits > 0) digit_groups++;
    if (digit_groups >= 3) { total_pii_items_found++; return 1; }
    return 0;
}
/**
 * @brief Perform Output Verifier Check Pii All.
 *
 * @param text [in] Text value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_check_pii_all(const char* text, size_t len) {
    if (!text || len == 0) return 0;
    int found = 0;
    if (SNEPPX_output_verifier_check_pii_emails(text, len)) found = 1;
    if (SNEPPX_output_verifier_check_pii_phones(text, len)) found = 1;
    if (SNEPPX_output_verifier_check_pii_ssn(text, len)) found = 1;
    return found;
}
/**
 * @brief Perform Output Verifier Get Pii Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_get_pii_count(void) { return total_pii_items_found; }
/**
 * @brief Perform Output Verifier Enable Content Filtering.
 */
void SNEPPX_output_verifier_enable_content_filtering(int enabled) { content_filtering_enabled = (enabled != 0); }
/**
 * @brief Perform Output Verifier Is Content Filtering Enabled.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_is_content_filtering_enabled(void) { return content_filtering_enabled; }
/**
 * @brief Perform Output Verifier Check Pii Credit Cards.
 *
 * @param text [in] Text value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_check_pii_credit_cards(const char* text, size_t len) {
    if (!text || len == 0) return 0;
    int digit_run = 0;
    for (size_t i = 0; i < len; i++) {
        if (text[i] >= '0' && text[i] <= '9') { digit_run++; if (digit_run >= 16) { total_pii_items_found++; return 1; } }
        else if (text[i] == '-' || text[i] == ' ') { if (digit_run < 4) digit_run = 0; }
        else digit_run = 0;
    }
    return 0;
}
/**
 * @brief Perform Output Verifier Check Pii Urls.
 *
 * @param text [in] Text value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_check_pii_urls(const char* text, size_t len) {
    if (!text || len == 0) return 0;
    const char* prefixes[] = {"http://", "https://", "ftp://", "www.", NULL};
    char lower[512]; size_t clen = (len < sizeof(lower) - 1) ? len : sizeof(lower) - 1;
    for (size_t i = 0; i < clen; i++) lower[i] = (char)tolower((unsigned char)text[i]);
    lower[clen] = '\0';
    for (int p = 0; prefixes[p]; p++) { if (strstr(lower, prefixes[p])) { total_pii_items_found++; return 1; } }
    return 0;
}
/**
 * @brief Perform Output Verifier Check Pii Ip Addresses.
 *
 * @param text [in] Text value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_check_pii_ip_addresses(const char* text, size_t len) {
    if (!text || len == 0) return 0;
    int octets = 0, digits = 0;
    for (size_t i = 0; i < len; i++) {
        if (text[i] >= '0' && text[i] <= '9') { digits++; }
        else if (text[i] == '.') { if (digits > 0) octets++; digits = 0; }
        else { octets = 0; digits = 0; }
        if (octets >= 3 && digits > 0) { total_pii_items_found++; return 1; }
    }
    return 0;
}
/**
 * @brief Perform Output Verifier Check Toxicity.
 *
 * @param text [in] Text value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_check_toxicity(const char* text, size_t len) {
    if (!text || len == 0) return 0;
    char lower[512]; size_t clen = (len < sizeof(lower) - 1) ? len : sizeof(lower) - 1;
    for (size_t i = 0; i < clen; i++) lower[i] = (char)tolower((unsigned char)text[i]);
    lower[clen] = '\0';
    const char* toxic[] = {"hate", "kill", "die", "murder", "stupid", "idiot", "racist", "sexist", NULL};
    int hits = 0;
    for (int t = 0; toxic[t]; t++) { if (strstr(lower, toxic[t])) hits++; }
    return hits;
}
/**
 * @brief Perform Output Verifier Check Bias.
 *
 * @param text [in] Text value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_check_bias(const char* text, size_t len) {
    if (!text || len == 0) return 0;
    char lower[512]; size_t clen = (len < sizeof(lower) - 1) ? len : sizeof(lower) - 1;
    for (size_t i = 0; i < clen; i++) lower[i] = (char)tolower((unsigned char)text[i]);
    lower[clen] = '\0';
    const char* bias_terms[] = {"all men", "all women", "they always", "never", "always", "everyone knows", NULL};
    int hits = 0;
    for (int b = 0; bias_terms[b]; b++) { if (strstr(lower, bias_terms[b])) hits++; }
    return hits;
}
/**
 * @brief Perform Output Verifier Check Full Extended.
 *
 * @param ov [out] Ov value.
 * @param output [in] Output value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_check_full_extended(SNEPPXS5Verifier* ov, const char* output, size_t len, int* flags) {
    if (!ov || !output || !flags) return -1;
    *flags = 0;
    if (SNEPPX_output_verifier_check(ov, output, len)) *flags |= 1;
    if (SNEPPX_output_verifier_check_pii_all(output, len)) *flags |= 2;
    if (SNEPPX_output_verifier_check_toxicity(output, len) > 0) *flags |= 4;
    if (SNEPPX_output_verifier_check_pii_credit_cards(output, len)) *flags |= 8;
    return *flags != 0 ? 1 : 0;
}
/**
 * @brief Perform Output Verifier Get Pii Hits.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_get_pii_hits(void) { return total_pii_hits; }
/**
 * @brief Perform Output Verifier Get Pii Items Found.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_get_pii_items_found(void) { return total_pii_items_found; }
/**
 * @brief Perform Output Verifier Reset Pii Count.
 */
void SNEPPX_output_verifier_reset_pii_count(void) { total_pii_items_found = 0; }
/**
 * @brief Perform Output Verifier Check Pii Custom.
 *
 * @param text [in] Text value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_check_pii_custom(const char* text, size_t len, const char* patterns[], int pattern_count) {
    if (!text || !patterns || pattern_count <= 0) return 0;
    char lower[512]; size_t clen = (len < sizeof(lower) - 1) ? len : sizeof(lower) - 1;
    for (size_t i = 0; i < clen; i++) lower[i] = (char)tolower((unsigned char)text[i]);
    lower[clen] = '\0';
    for (int p = 0; p < pattern_count; p++) { if (strstr(lower, patterns[p])) { total_pii_items_found++; return 1; } }
    return 0;
}
/**
 * @brief Perform Output Verifier Check Pii Batch.
 *
 * @param texts [in] Texts value.
 * @param lens [in] Lens value.
 * @param count [in] Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_check_pii_batch(const char** texts, const size_t* lens, int count, int* results) {
    if (!texts || !lens || !results || count <= 0) return -1;
    for (int i = 0; i < count; i++) results[i] = SNEPPX_output_verifier_check_pii_all(texts[i], lens[i]);
    return 0;
}
/**
 * @brief Perform Output Verifier Check Profanity Batch.
 *
 * @param texts [in] Texts value.
 * @param lens [in] Lens value.
 * @param count [in] Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_check_profanity_batch(const char** texts, const size_t* lens, int count, int* results) {
    if (!texts || !lens || !results || count <= 0) return -1;
    for (int i = 0; i < count; i++) results[i] = SNEPPX_output_verifier_check_profanity(texts[i], lens[i]);
    return 0;
}
/**
 * @brief Perform Output Verifier Check Consistency.
 *
 * @param ov [out] Ov value.
 * @param output [in] Output value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_check_consistency(SNEPPXS5Verifier* ov, const char* output, size_t len) {
    if (!ov || !output) return 0;
    if (!ov->check_factual_consistency) return 0;
    char lower[512]; size_t clen = (len < sizeof(lower) - 1) ? len : sizeof(lower) - 1;
    for (size_t i = 0; i < clen; i++) lower[i] = (char)tolower((unsigned char)output[i]);
    lower[clen] = '\0';
    const char* contradict_terms[] = {"but on the other hand", "however", "contradict", "actually", NULL};
    for (int c = 0; contradict_terms[c]; c++) { if (strstr(lower, contradict_terms[c])) { return 1; } }
    return 0;
}
/**
 * @brief Perform Output Verifier Get Check Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_get_check_count(void) { return total_checks; }
/**
 * @brief Perform Output Verifier Get Profanity Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_get_profanity_count(void) { return total_profanity_hits; }
/**
 * @brief Perform Output Verifier Get Toxicity Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_get_toxicity_count(void) { return total_toxicity_hits; }
/**
 * @brief Perform Output Verifier Get Bias Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_get_bias_count(void) { return total_bias_hits; }
/**
 * @brief Perform Output Verifier Set Pii Detection.
 */
void SNEPPX_output_verifier_set_pii_detection(int enabled) { pii_detection_enabled = enabled; }
/**
 * @brief Perform Output Verifier Get Verifier Block Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_get_verifier_block_count(void) { return verifier_block_count; }
/**
 * @brief Perform Output Verifier Check All.
 *
 * @param ov [out] Ov value.
 * @param output [in] Output value.
 * @param len [in] Len value.
 * @param toxicity_flag [out] Toxicity Flag value.
 * @param pii_flag [out] Pii Flag value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_check_all(SNEPPXS5Verifier* ov, const char* output, size_t len, int* toxicity_flag, int* pii_flag, int* profanity_flag) {
    if (!ov || !output || !toxicity_flag || !pii_flag || !profanity_flag) return -1;
    *toxicity_flag = SNEPPX_output_verifier_check(ov, output, len);
    *pii_flag = SNEPPX_output_verifier_check_pii_all(output, len);
    *profanity_flag = SNEPPX_output_verifier_check_profanity(output, len);
    return (*toxicity_flag || *pii_flag || *profanity_flag) ? 1 : 0;
}
/**
 * @brief Perform Output Verifier Check Full Report.
 *
 * @param ov [out] Ov value.
 * @param output [in] Output value.
 * @param len [in] Len value.
 * @param report [out] Report value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_check_full_report(SNEPPXS5Verifier* ov, const char* output, size_t len, char* report, size_t report_size) {
    if (!ov || !output || !report || report_size == 0) return -1;
    int t, p, prof;
    SNEPPX_output_verifier_check_all(ov, output, len, &t, &p, &prof);
    int n = snprintf(report, report_size, "toxicity=%d pii=%d profanity=%d", t, p, prof);
    return (n < 0) ? -1 : ((size_t)n < report_size ? n : (int)(report_size - 1));
}
/**
 * @brief Perform Output Verifier Sanitize Pii.
 *
 * @param text [in] Text value.
 * @param len [in] Len value.
 * @param safe [out] Safe value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_sanitize_pii(const char* text, size_t len, char* safe, size_t* safe_len) {
    if (!text || !safe || !safe_len) return -1;
    size_t cap = *safe_len; *safe_len = 0; size_t pos = 0;
    const char redacted[] = "[REDACTED]"; size_t rlen = strlen(redacted);
    for (size_t i = 0; i < len && pos < cap; i++) {
        if (text[i] == '@') {
            size_t start = i; while (start > 0 && text[start-1] != ' ' && text[start-1] != '\t' && text[start-1] != '\n' && text[start-1] != '\r') start--;
            size_t end = i; while (end < len && text[end] != ' ' && text[end] != '\t' && text[end] != '\n' && text[end] != '\r') end++;
            if (pos + rlen <= cap) { memcpy(safe + pos, redacted, rlen); pos += rlen; }
            i = end; if (i > 0) i--;
        } else { if (pos < cap) safe[pos++] = text[i]; }
    }
    *safe_len = pos; if (pos < cap) safe[pos] = 0;
    return 0;
}
/**
 * @brief Perform Output Verifier Sanitize Profanity.
 *
 * @param text [in] Text value.
 * @param len [in] Len value.
 * @param safe [out] Safe value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_output_verifier_sanitize_profanity(const char* text, size_t len, char* safe, size_t* safe_len) {
    if (!text || !safe || !safe_len) return -1;
    size_t cap = *safe_len; *safe_len = 0; size_t pos = 0;
    const char redacted[] = "****"; size_t rlen = strlen(redacted);
    char lower[512]; size_t clen = (len < sizeof(lower) - 1) ? len : sizeof(lower) - 1;
    for (size_t i = 0; i < clen; i++) lower[i] = (char)tolower((unsigned char)text[i]);
    lower[clen] = '\0';
    for (size_t i = 0; i < len && pos < cap; i++) {
        int matched = 0;
        for (int p = 0; profanity_list[p]; p++) {
            size_t plen = strlen(profanity_list[p]);
            if (i + plen <= len) {
                int match = 1;
                for (size_t j = 0; j < plen; j++) { if ((char)tolower((unsigned char)text[i+j]) != profanity_list[p][j]) { match = 0; break; } }
                if (match) { if (pos + rlen <= cap) { memcpy(safe + pos, redacted, rlen); pos += rlen; } i += plen - 1; matched = 1; break; }
            }
        }
        if (!matched) { if (pos < cap) safe[pos++] = text[i]; }
    }
    *safe_len = pos; if (pos < cap) safe[pos] = 0;
    return 0;
}


/* Compatibility wrappers for the documented S5 verifier API. */
int SNEPPX_s5_verifier_init(SNEPPXS5Verifier* ov) {
    return SNEPPX_output_verifier_init(ov);
}

void SNEPPX_s5_verifier_destroy(SNEPPXS5Verifier* ov) {
    SNEPPX_output_verifier_destroy(ov);
}

int SNEPPX_s5_verifier_add_blocked_topic(SNEPPXS5Verifier* ov, const char* topic) {
    return SNEPPX_output_verifier_add_blocked_topic(ov, topic);
}

int SNEPPX_s5_verifier_check(SNEPPXS5Verifier* ov, const char* output, size_t len) {
    return SNEPPX_output_verifier_check(ov, output, len);
}

int SNEPPX_s5_verifier_sanitize(SNEPPXS5Verifier* ov,
                                const char* output, size_t len,
                                char* safe_output, size_t* safe_len) {
    return SNEPPX_output_verifier_sanitize(ov, output, len, safe_output, safe_len);
}
