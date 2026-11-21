#ifndef SNEPPX_RLHF_SAFETY_H
#define SNEPPX_RLHF_SAFETY_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

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


typedef enum {
    RLHF_RULE_HARM = 0,
    RLHF_RULE_BIAS,
    RLHF_RULE_FACTUAL,
    RLHF_RULE_CUSTOM
} rlhf_rule_type_t;

typedef enum {
    RLHF_SEV_LOW = 0,
    RLHF_SEV_MEDIUM,
    RLHF_SEV_HIGH,
    RLHF_SEV_CRITICAL
} rlhf_severity_t;

typedef struct {
    double harm_threshold;
    double bias_threshold;
    double factual_threshold;
    double helpfulness_weight;
    double harmlessness_weight;
    double honesty_weight;
    double enable_refusal;
    double enable_critique;
    double enable_correction;
} rlhf_config_t;

typedef struct {
    double harmfulness;
    double bias;
    double factuality;
    double helpfulness;
    double honesty;
    double overall;
    int should_refuse;
} rlhf_score_t;

typedef struct {
    char policy_name[64];
    double score;
    double threshold;
    rlhf_severity_t severity;
} rlhf_violation_t;

typedef struct {
    uint64_t total_refusals;
    uint64_t total_corrections;
    int active_policies;
    uint64_t total_violations;
    rlhf_config_t config;
} rlhf_stats_t;

/**
 * @brief Perform Rlhf Add Policy.
 *
 * @param name [in] Name value.
 * @param weight [in] Weight value.
 * @param rule_type [in] Rule Type value.
 * @param threshold [in] Threshold value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rlhf_add_policy(const char *name, double weight, rlhf_rule_type_t rule_type, double threshold);
/**
 * @brief Perform Rlhf Remove Policy.
 *
 * @param policy_id [in] Policy Id value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rlhf_remove_policy(int policy_id);
/**
 * @brief Perform Rlhf Score.
 *
 * @param input [in] Input value.
 * @param output [in] Output value.
 * @param score [out] Score value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rlhf_score(const char *input, const char *output, rlhf_score_t *score);
/**
 * @brief Perform Rlhf Check Policies.
 *
 * @param text [in] Text value.
 * @param violations [out] Violations value.
 * @param max_violations [in] Max Violations value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rlhf_check_policies(const char *text, rlhf_violation_t *violations, int max_violations);
/**
 * @brief Perform Rlhf Generate Refusal.
 *
 * @param out [out] Out value.
 * @param out_len [in] Out Len value.
 * @param score [out] Score value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rlhf_generate_refusal(char *out, size_t out_len, rlhf_score_t *score);
/**
 * @brief Perform Rlhf Generate Critique.
 *
 * @param output [in] Output value.
 * @param critique [out] Critique value.
 * @param critique_len [in] Critique Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rlhf_generate_critique(const char *output, char *critique, size_t critique_len);
/**
 * @brief Perform Rlhf Correct Output.
 *
 * @param output [in] Output value.
 * @param corrected [out] Corrected value.
 * @param corrected_len [in] Corrected Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rlhf_correct_output(const char *output, char *corrected, size_t corrected_len);
/**
 * @brief Perform Rlhf Update Config.
 *
 * @param config [in] Config value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rlhf_update_config(const rlhf_config_t *config);
/**
 * @brief Perform Rlhf Get Config.
 *
 * @param config [out] Config value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rlhf_get_config(rlhf_config_t *config);
/**
 * @brief Perform Rlhf Get Stats.
 *
 * @param stats [out] Stats value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rlhf_get_stats(rlhf_stats_t *stats);
/**
 * @brief Reset Rlhf.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rlhf_reset(void);
/**
 * @brief Perform Rlhf Add Default Policies.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rlhf_add_default_policies(void);


#ifdef __cplusplus
}
#endif
#endif
