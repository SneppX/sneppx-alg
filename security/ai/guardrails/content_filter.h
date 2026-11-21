#ifndef SNEPPX_AI_GUARDRAILS_CONTENT_FILTER_H
#define SNEPPX_AI_GUARDRAILS_CONTENT_FILTER_H

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_AI_GUARDRAIL_MAX_RULES 4096
#define SNEPPX_AI_GUARDRAIL_MAX_PATTERNS 256
#define SNEPPX_AI_GUARDRAIL_MAX_CATEGORIES 128
#define SNEPPX_AI_GUARDRAIL_SCORE_THRESHOLD 0.85f

/*
 * SNEPPX - Content Filter
 *
 * WHAT
 *   Content Filter.
 *
 * CONCEPT
 *   Provides the Content Filter.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_AI_CATEGORY_HATE_SPEECH,
    SNEPPX_AI_CATEGORY_VIOLENCE,
    SNEPPX_AI_CATEGORY_SEXUAL_CONTENT,
    SNEPPX_AI_CATEGORY_HARASSMENT,
    SNEPPX_AI_CATEGORY_SELF_HARM,
    SNEPPX_AI_CATEGORY_PERSONAL_IDENTIFIABLE,
    SNEPPX_AI_CATEGORY_FINANCIAL_INFO,
    SNEPPX_AI_CATEGORY_HEALTH_INFO,
    SNEPPX_AI_CATEGORY_LEGAL_ADVICE,
    SNEPPX_AI_CATEGORY_MEDICAL_ADVICE,
    SNEPPX_AI_CATEGORY_MALICIOUS_CODE,
    SNEPPX_AI_CATEGORY_TERRORISM,
    SNEPPX_AI_CATEGORY_CHILD_SAFETY,
    SNEPPX_AI_CATEGORY_MISINFORMATION,
    SNEPPX_AI_CATEGORY_PROMPT_INJECTION,
    SNEPPX_AI_CATEGORY_JAILBREAK_ATTEMPT,
    SNEPPX_AI_CATEGORY_SYSTEM_LEAK,
    SNEPPX_AI_CATEGORY_UNAUTHORIZED_ACTIONS,
    SNEPPX_AI_CATEGORY_CUSTOM
} SNEPPXAIGuardrailCategory;

typedef enum {
    SNEPPX_AI_GUARDRAIL_ACTION_BLOCK,
    SNEPPX_AI_GUARDRAIL_ACTION_FLAG,
    SNEPPX_AI_GUARDRAIL_ACTION_LOG,
    SNEPPX_AI_GUARDRAIL_ACTION_REDACT,
    SNEPPX_AI_GUARDRAIL_ACTION_MASK,
    SNEPPX_AI_GUARDRAIL_ACTION_REWRITE,
    SNEPPX_AI_GUARDRAIL_ACTION_ESCALATE
} SNEPPXAIGuardrailAction;

typedef struct {
    SNEPPXAIGuardrailCategory category;
    SNEPPXAIGuardrailAction action;
    uint8_t* regex_pattern;
    size_t pattern_len;
    uint8_t* keywords[SNEPPX_AI_GUARDRAIL_MAX_PATTERNS];
    size_t keyword_lens[SNEPPX_AI_GUARDRAIL_MAX_PATTERNS];
    uint32_t num_keywords;
    float threshold;
    uint8_t* redaction_template;
    size_t redaction_template_len;
    uint8_t enabled : 1;
    uint8_t use_ml : 1;
} SNEPPXAIGuardrailRule;

typedef struct {
    SNEPPXAIGuardrailRule rules[SNEPPX_AI_GUARDRAIL_MAX_RULES];
    uint32_t num_rules;
    uint8_t* input_buffer;
    size_t input_len;
    uint8_t* output_buffer;
    size_t output_len;
    uint8_t* detected_categories[SNEPPX_AI_GUARDRAIL_MAX_CATEGORIES];
    size_t detected_category_lens[SNEPPX_AI_GUARDRAIL_MAX_CATEGORIES];
    uint32_t num_detected;
    float confidence_scores[SNEPPX_AI_GUARDRAIL_MAX_CATEGORIES];
    uint32_t total_flags;
    uint32_t total_blocks;
    uint8_t blocked : 1;
    uint8_t modified : 1;
} SNEPPXAIGuardrailResult;

int snepx_ai_guardrail_add_rule(SNEPPXAIGuardrailRule* rules, uint32_t* num_rules, const SNEPPXAIGuardrailRule* rule);
int snepx_ai_guardrail_remove_rule(SNEPPXAIGuardrailRule* rules, uint32_t* num_rules, uint32_t rule_index);
int snepx_ai_guardrail_filter_input(const SNEPPXAIGuardrailRule* rules, uint32_t num_rules, const uint8_t* input, size_t input_len, SNEPPXAIGuardrailResult* result);
int snepx_ai_guardrail_filter_output(const SNEPPXAIGuardrailRule* rules, uint32_t num_rules, const uint8_t* output, size_t output_len, SNEPPXAIGuardrailResult* result);
int snepx_ai_guardrail_detect_prompt_injection(const uint8_t* text, size_t text_len, float* score);
int snepx_ai_guardrail_detect_jailbreak(const uint8_t* text, size_t text_len, float* score);
int snepx_ai_guardrail_redact_pii(uint8_t* text, size_t* text_len);
int snepx_ai_guardrail_scan_output(const uint8_t* output, size_t output_len, SNEPPXAIGuardrailCategory* categories, uint32_t* num_categories);
int snepx_ai_guardrail_result_destroy(SNEPPXAIGuardrailResult* result);

#endif
