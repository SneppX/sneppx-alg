#ifndef SNEPPX_S5_EXTENSIONS_H
#define SNEPPX_S5_EXTENSIONS_H
/*
 * SNEPPX - S5 Extensions
 *
 * WHAT
 *   S5 Extensions.
 *
 * CONCEPT
 *   Provides the S5 Extensions.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/* S5 AI Sanitizer extensions: semantic injection (NLP), multi-lang jailbreak,
   encoded attack decoder, token anomaly scoring, model inversion defense,
   membership inference, data extraction prevention, training data sanitization,
   model watermarking, adversarial perturbation, factuality scorer, bias measurement,
   prompt policy engine */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_S5_MAX_EMBEDDING 256
#define SNEPPX_S5_MAX_ENCODED_PATTERNS 64

/* Semantic injection detection via embedding similarity */
typedef struct {
    double known_attack_embeddings[SNEPPX_S5_MAX_EMBEDDING][8];
    int attack_count;
    double threshold;
} SNEPPXSemanticInjectionDetector;

/**
 * @brief Initialize Semantic Injection.
 *
 * @param sid [out] Sid value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_semantic_injection_init(SNEPPXSemanticInjectionDetector* sid);
/**
 * @brief Perform Semantic Injection Add Attack.
 *
 * @param sid [out] Sid value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_semantic_injection_add_attack(SNEPPXSemanticInjectionDetector* sid, const double embedding[8]);
/**
 * @brief Perform Semantic Injection Score.
 *
 * @param sid [out] Sid value.
 * @param score [out] Score value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_semantic_injection_score(SNEPPXSemanticInjectionDetector* sid, const double embedding[8], double* score);

/* Multi-language jailbreak detection */
/**
 * @brief Perform Ml Jailbreak Detect.
 *
 * @param text [in] Text value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_ml_jailbreak_detect(const char* text, size_t len);

/* Encoded attack decoder (base64/hex/rot13) */
/**
 * @brief Perform Encoded Attack Decode.
 *
 * @param input [in] Input value.
 * @param in_len [in] In Len value.
 * @param output [out] Output value.
 * @param out_len [out] Out Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_encoded_attack_decode(const char* input, size_t in_len, char* output, size_t* out_len);
/**
 * @brief Perform Encoded Attack Scan.
 *
 * @param text [in] Text value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_encoded_attack_scan(const char* text, size_t len);

/* Token-level anomaly scoring */
/**
 * @brief Perform Token Anomaly Score.
 *
 * @param token_ids [in] Token Ids value.
 * @param token_count [in] Token Count value.
 * @param expected_probs [in] Expected Probs value.
 *
 * @return The result value, or 0 on error.
 */
double SNEPPX_token_anomaly_score(const uint32_t* token_ids, size_t token_count, const double* expected_probs);

/* Model inversion defense */
typedef struct {
    double noise_scale;
    int gradient_clipping;
    double clip_norm;
} SNEPPXModelInversionDefense;

/**
 * @brief Initialize Model Inversion.
 *
 * @param mid [out] Mid value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_model_inversion_init(SNEPPXModelInversionDefense* mid);
/**
 * @brief Apply Model Inversion.
 *
 * @param mid [out] Mid value.
 * @param gradients [out] Gradients value.
 * @param grad_count [in] Grad Count value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_model_inversion_apply(SNEPPXModelInversionDefense* mid, double* gradients, size_t grad_count);

/* Membership inference defense */
/**
 * @brief Perform Membership Inference Defense.
 *
 * @param logits [out] Logits value.
 * @param logit_count [in] Logit Count value.
 * @param epsilon [in] Epsilon value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_membership_inference_defense(double* logits, size_t logit_count, double epsilon);

/* Data extraction prevention */
/**
 * @brief Perform Data Extraction Prevent.
 *
 * @param output [in] Output value.
 * @param len [in] Len value.
 * @param contains_sensitive [out] Contains Sensitive value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_data_extraction_prevent(const char* output, size_t len, int* contains_sensitive);

/* Training data sanitization */
/**
 * @brief Perform Training Sanitize.
 *
 * @param text [in] Text value.
 * @param len [in] Len value.
 * @param sanitized [out] Sanitized value.
 * @param sanitized_len [out] Sanitized Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_training_sanitize(const char* text, size_t len, char* sanitized, size_t* sanitized_len);

/* Model watermarking */
typedef struct {
    uint8_t watermark[32];
    int embedded;
} SNEPPXModelWatermark;

/**
 * @brief Initialize Watermark.
 *
 * @param mw [out] Mw value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_watermark_init(SNEPPXModelWatermark* mw);
/**
 * @brief Perform Watermark Embed.
 *
 * @param mw [out] Mw value.
 * @param weights [out] Weights value.
 * @param weight_count [in] Weight Count value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_watermark_embed(SNEPPXModelWatermark* mw, double* weights, size_t weight_count);
/**
 * @brief Verify Watermark.
 *
 * @param mw [out] Mw value.
 * @param weights [in] Weights value.
 * @param weight_count [in] Weight Count value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_watermark_verify(SNEPPXModelWatermark* mw, const double* weights, size_t weight_count);

/* Adversarial input perturbation */
/**
 * @brief Perform Adversarial Smooth.
 *
 * @param input [out] Input value.
 * @param input_dim [in] Input Dim value.
 * @param epsilon [in] Epsilon value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_adversarial_smooth(double* input, size_t input_dim, double epsilon);

/* Output factuality scorer */
/**
 * @brief Perform Factuality Score.
 *
 * @param statement [in] Statement value.
 * @param reference [in] Reference value.
 *
 * @return The result value, or 0 on error.
 */
double SNEPPX_factuality_score(const char* statement, const char* reference);

/* Bias measurement */
typedef struct {
    double demographic_parity;
    double equalized_odds;
    int measured;
} SNEPPXBiasMetrics;

/**
 * @brief Perform Bias Measure.
 *
 * @param bm [out] Bm value.
 * @param predictions [in] Predictions value.
 * @param sensitive_attr [in] Sensitive Attr value.
 * @param n [in] N value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_bias_measure(SNEPPXBiasMetrics* bm, const double* predictions, const int* sensitive_attr, size_t n);

/* Prompt policy engine (RBAC) */
typedef struct {
    char policies[16][256];
    int policy_count;
    int enabled;
} SNEPPXPromptPolicy;

/**
 * @brief Initialize Prompt Policy.
 *
 * @param pp [out] Pp value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_prompt_policy_init(SNEPPXPromptPolicy* pp);
/**
 * @brief Add Prompt Policy.
 *
 * @param pp [out] Pp value.
 * @param policy_rule [in] Policy Rule value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_prompt_policy_add(SNEPPXPromptPolicy* pp, const char* policy_rule);
/**
 * @brief Perform Prompt Policy Enforce.
 *
 * @param pp [out] Pp value.
 * @param prompt [in] Prompt value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_prompt_policy_enforce(SNEPPXPromptPolicy* pp, const char* prompt, size_t len);

#ifdef __cplusplus
}
#endif
#endif
