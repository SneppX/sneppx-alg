#include "rlhf_safety.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

#define RLHF_MAX_POLICIES 64
#define RLHF_MAX_INPUT 4096
#define RLHF_MAX_FEATURES 512

/*
 * SNEPPX - Rlhf Safety
 *
 * WHAT
 *   Rlhf Safety.
 *
 * CONCEPT
 *   Provides the Rlhf Safety.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef struct {
    char name[64];
    double weight;
    rlhf_rule_type_t rule_type;
    double threshold;
    uint8_t active;
    uint64_t violations;
} rlhf_policy_t;

typedef struct {
    double harm_score;
    double bias_score;
    double factual_score;
    double helpfulness_score;
    double honesty_score;
    double overall_score;
    char reasoning[512];
} rlhf_scoring_t;

static rlhf_policy_t policies[RLHF_MAX_POLICIES];
static int policy_count = 0;
static rlhf_config_t rlhf_config = {
    .harm_threshold = 0.7,
    .bias_threshold = 0.6,
    .factual_threshold = 0.4,
    .helpfulness_weight = 0.3,
    .harmlessness_weight = 0.4,
    .honesty_weight = 0.2,
    .helpfulness_weight = 0.1,
    .enable_refusal = 1,
    .enable_critique = 1,
    .enable_correction = 1
};
static uint64_t total_refusals = 0;
static uint64_t total_corrections = 0;

static double rlhf_detect_harm(const char *text, size_t len) {
    double score = 0.0;
    const char *harm_patterns[] = {
        "bomb", "explosive", "weapon", "kill", "murder", "suicide",
        "terrorist", "hack", "malware", "ransomware", "virus",
        "child abuse", "grooming", "exploit child",
        "nuclear", "biological weapon", "chemical weapon",
        "methods to", "steps to", "instructions for",
        "how to harm", "how to kill", "how to make a bomb",
        "self-harm", "self harm", "cutting", "eating disorder",
        "hate speech", "racial slurs", "discriminat",
        NULL
    };
    for (int i = 0; harm_patterns[i]; i++) {
        if (strstr(text, harm_patterns[i])) score += 0.15;
    }
    return score > 1.0 ? 1.0 : score;
}

static double rlhf_detect_bias(const char *text, size_t len) {
    double score = 0.0;
    const char *bias_patterns[] = {
        "all [a-z]+ are", "every [a-z]+ is",
        "they always", "they never", "these people",
        "inferior", "superior race",
        "gender stereotype", "racial stereotype",
        "men should", "women should",
        "boys don't", "girls can't",
        NULL
    };
    for (int i = 0; bias_patterns[i]; i++) {
        if (strstr(text, bias_patterns[i])) score += 0.12;
    }
    return score > 1.0 ? 1.0 : score;
}

static double rlhf_check_factual(const char *text, size_t len) {
    double score = 0.0;
    const char *hedge_words[] = {
        "maybe", "perhaps", "possibly", "might", "could be",
        "I think", "I believe", "in my opinion", "I'm not sure",
        "I don't know", "uncertain", "not verified",
        NULL
    };
    for (int i = 0; hedge_words[i]; i++) {
        if (strstr(text, hedge_words[i])) score += 0.1;
    }
    return score > 1.0 ? 1.0 : score;
}

/**
 * @brief Perform Rlhf Add Policy.
 *
 * @param name [in] Name value.
 * @param weight [in] Weight value.
 * @param rule_type [in] Rule Type value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rlhf_add_policy(const char *name, double weight, rlhf_rule_type_t rule_type, double threshold) {
    if (!name || policy_count >= RLHF_MAX_POLICIES) return -1;
    strncpy(policies[policy_count].name, name, 63);
    policies[policy_count].weight = weight;
    policies[policy_count].rule_type = rule_type;
    policies[policy_count].threshold = threshold;
    policies[policy_count].active = 1;
    policies[policy_count].violations = 0;
    return policy_count++;
}

/**
 * @brief Perform Rlhf Remove Policy.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rlhf_remove_policy(int policy_id) {
    if (policy_id < 0 || policy_id >= policy_count) return -1;
    policies[policy_id].active = 0;
    return 0;
}

/**
 * @brief Perform Rlhf Score.
 *
 * @param input [in] Input value.
 * @param output [in] Output value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rlhf_score(const char *input, const char *output, rlhf_score_t *score) {
    if (!input || !output || !score) return -1;
    memset(score, 0, sizeof(rlhf_score_t));
    score->harmfulness = rlhf_detect_harm(output, strlen(output));
    score->bias = rlhf_detect_bias(output, strlen(output));
    score->factuality = rlhf_check_factual(output, strlen(output));
    score->helpfulness = 1.0 - score->factuality * 0.3;
    score->honesty = 1.0 - score->factuality * 0.2;
    score->overall = (score->helpfulness * rlhf_config.helpfulness_weight) +
                     ((1.0 - score->harmfulness) * rlhf_config.harmlessness_weight) +
                     (score->honesty * rlhf_config.honesty_weight);
    score->should_refuse = (score->harmfulness > rlhf_config.harm_threshold ||
                           score->bias > rlhf_config.bias_threshold);
    if (score->should_refuse && rlhf_config.enable_refusal) total_refusals++;
    return 0;
}

/**
 * @brief Perform Rlhf Check Policies.
 *
 * @param text [in] Text value.
 * @param violations [out] Violations value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rlhf_check_policies(const char *text, rlhf_violation_t *violations, int max_violations) {
    if (!text || !violations) return -1;
    int found = 0;
    for (int i = 0; i < policy_count && found < max_violations; i++) {
        if (!policies[i].active) continue;
        double score = 0;
        switch (policies[i].rule_type) {
            case RLHF_RULE_HARM: score = rlhf_detect_harm(text, strlen(text)); break;
            case RLHF_RULE_BIAS: score = rlhf_detect_bias(text, strlen(text)); break;
            case RLHF_RULE_FACTUAL: score = rlhf_check_factual(text, strlen(text)); break;
            case RLHF_RULE_CUSTOM: score = rlhf_detect_harm(text, strlen(text)) * 0.5; break;
        }
        if (score > policies[i].threshold) {
            strncpy(violations[found].policy_name, policies[i].name, 63);
            violations[found].score = score;
            violations[found].threshold = policies[i].threshold;
            violations[found].severity = (score > 0.8) ? RLHF_SEV_CRITICAL :
                                          (score > 0.6) ? RLHF_SEV_HIGH : RLHF_SEV_MEDIUM;
            policies[i].violations++;
            found++;
        }
    }
    return found;
}

/**
 * @brief Perform Rlhf Generate Refusal.
 *
 * @param out [out] Out value.
 * @param out_len [in] Out Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rlhf_generate_refusal(char *out, size_t out_len, rlhf_score_t *score) {
    if (!out || !score) return -1;
    const char *refusals[] = {
        "I cannot generate this response as it may be harmful.",
        "I apologize, but I cannot comply with this request as it violates safety guidelines.",
        "This request appears to ask for potentially harmful content. I must refuse.",
        "I'm not able to help with this request. Please rephrase your question.",
        "I cannot provide this information as it could be misused.",
    };
    int idx = rand() % 5;
    snprintf(out, out_len, "%s", refusals[idx]);
    return 0;
}

/**
 * @brief Perform Rlhf Generate Critique.
 *
 * @param output [in] Output value.
 * @param critique [out] Critique value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rlhf_generate_critique(const char *output, char *critique, size_t critique_len) {
    if (!output || !critique) return -1;
    double harm = rlhf_detect_harm(output, strlen(output));
    double bias = rlhf_detect_bias(output, strlen(output));
    double factual = rlhf_check_factual(output, strlen(output));
    int pos = 0;
    pos += snprintf(critique + pos, critique_len - pos, "Critique of model output:\n");
    if (harm > 0.3) pos += snprintf(critique + pos, critique_len - pos,
        "- Contains potentially harmful content (score: %.2f)\n", harm);
    if (bias > 0.3) pos += snprintf(critique + pos, critique_len - pos,
        "- Contains biased language (score: %.2f)\n", bias);
    if (factual > 0.3) pos += snprintf(critique + pos, critique_len - pos,
        "- May contain unverified claims (score: %.2f)\n", factual);
    if (harm <= 0.3 && bias <= 0.3 && factual <= 0.3)
        pos += snprintf(critique + pos, critique_len - pos, "- Output appears safe and factual.\n");
    return pos;
}

/**
 * @brief Perform Rlhf Correct Output.
 *
 * @param output [in] Output value.
 * @param corrected [out] Corrected value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rlhf_correct_output(const char *output, char *corrected, size_t corrected_len) {
    if (!output || !corrected) return -1;
    int pos = 0;
    for (size_t i = 0; i < strlen(output) && pos < (int)corrected_len - 1; i++) {
        corrected[pos++] = output[i];
    }
    corrected[pos] = 0;
    total_corrections++;
    return pos;
}

/**
 * @brief Perform Rlhf Update Config.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rlhf_update_config(const rlhf_config_t *config) {
    if (!config) return -1;
    memcpy(&rlhf_config, config, sizeof(rlhf_config_t));
    return 0;
}

/**
 * @brief Perform Rlhf Get Config.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rlhf_get_config(rlhf_config_t *config) {
    if (!config) return -1;
    memcpy(config, &rlhf_config, sizeof(rlhf_config_t));
    return 0;
}

/**
 * @brief Perform Rlhf Get Stats.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rlhf_get_stats(rlhf_stats_t *stats) {
    if (!stats) return -1;
    stats->total_refusals = total_refusals;
    stats->total_corrections = total_corrections;
    stats->active_policies = 0;
    stats->total_violations = 0;
    for (int i = 0; i < policy_count; i++) {
        if (policies[i].active) stats->active_policies++;
        stats->total_violations += policies[i].violations;
    }
    stats->config = rlhf_config;
    return 0;
}

/**
 * @brief Reset Rlhf.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rlhf_reset(void) {
    policy_count = 0;
    total_refusals = 0;
    total_corrections = 0;
    memset(policies, 0, sizeof(policies));
    return 0;
}

/**
 * @brief Perform Rlhf Add Default Policies.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rlhf_add_default_policies(void) {
    SNEPPX_rlhf_add_policy("harm", 0.4, RLHF_RULE_HARM, 0.7);
    SNEPPX_rlhf_add_policy("bias", 0.3, RLHF_RULE_BIAS, 0.6);
    SNEPPX_rlhf_add_policy("factual", 0.2, RLHF_RULE_FACTUAL, 0.4);
    SNEPPX_rlhf_add_policy("custom_safety", 0.1, RLHF_RULE_CUSTOM, 0.5);
    return 0;
}
