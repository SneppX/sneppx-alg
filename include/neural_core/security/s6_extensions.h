#ifndef SNEPPX_S6_EXTENSIONS_H
#define SNEPPX_S6_EXTENSIONS_H
/*
 * SNEPPX - S6 Extensions
 *
 * WHAT
 *   S6 Extensions.
 *
 * CONCEPT
 *   Provides the S6 Extensions.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/* S6 extensions: HSM-backed keys, Shamir's Secret Sharing, key ceremony,
   auto rotation scheduler, web dashboard, threat viz, policy editor DSL,
   compliance report generator */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_SHAMIR_MAX_SHARES 16
#define SNEPPX_SHAMIR_MIN_SHARES 5

/* HSM-backed key storage */
typedef struct {
    int hsm_connected;
    uint8_t session_handle[32];
} SNEPPXHSMKeyStore;

/**
 * @brief Initialize Hsm.
 *
 * @param hsm [out] Hsm value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_hsm_init(SNEPPXHSMKeyStore* hsm);
/**
 * @brief Perform Hsm Store Key.
 *
 * @param hsm [out] Hsm value.
 * @param key_id [in] Key Id value.
 * @param key_data [in] Key Data value.
 * @param key_len [in] Key Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_hsm_store_key(SNEPPXHSMKeyStore* hsm, const uint8_t* key_id, const uint8_t* key_data, size_t key_len);
/**
 * @brief Perform Hsm Load Key.
 *
 * @param hsm [out] Hsm value.
 * @param key_id [in] Key Id value.
 * @param key_data [out] Key Data value.
 * @param key_len [out] Key Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_hsm_load_key(SNEPPXHSMKeyStore* hsm, const uint8_t* key_id, uint8_t* key_data, size_t* key_len);
/**
 * @brief Perform Hsm Delete Key.
 *
 * @param hsm [out] Hsm value.
 * @param key_id [in] Key Id value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_hsm_delete_key(SNEPPXHSMKeyStore* hsm, const uint8_t* key_id);

/* Shamir's Secret Sharing */
/**
 * @brief Perform Shamir Split.
 *
 * @param secret [in] Secret value.
 * @param secret_len [in] Secret Len value.
 * @param n [in] N value.
 * @param k [in] K value.
 * @param shares [out] Shares value.
 * @param share_lens [out] Share Lens value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_shamir_split(const uint8_t* secret, size_t secret_len, int n, int k, uint8_t** shares, size_t* share_lens);
/**
 * @brief Perform Shamir Reconstruct.
 *
 * @param shares [out] Shares value.
 * @param share_lens [out] Share Lens value.
 * @param k [in] K value.
 * @param secret [out] Secret value.
 * @param secret_len [out] Secret Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_shamir_reconstruct(uint8_t** shares, size_t* share_lens, int k, uint8_t* secret, size_t* secret_len);

/* Key ceremony workflow */
typedef struct {
    int participants_required;
    int participants_present;
    int approved;
} SNEPPXKeyCeremony;

/**
 * @brief Initialize Key Ceremony.
 *
 * @param kc [out] Kc value.
 * @param participants [in] Participants value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_key_ceremony_init(SNEPPXKeyCeremony* kc, int participants);
/**
 * @brief Perform Key Ceremony Participant Approve.
 *
 * @param kc [out] Kc value.
 * @param token [in] Token value.
 * @param token_len [in] Token Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_key_ceremony_participant_approve(SNEPPXKeyCeremony* kc, const uint8_t* token, size_t token_len);
/**
 * @brief Perform Key Ceremony Execute.
 *
 * @param kc [out] Kc value.
 * @param generated_key [out] Generated Key value.
 * @param key_len [in] Key Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_key_ceremony_execute(SNEPPXKeyCeremony* kc, uint8_t* generated_key, size_t key_len);

/* Auto rotation scheduler */
typedef struct {
    uint64_t rotation_interval_seconds;
    uint64_t last_rotation;
    int auto_rotate;
} SNEPPXKeyRotationScheduler;

/**
 * @brief Initialize Key Rotation.
 *
 * @param ks [out] Ks value.
 * @param interval_seconds [in] Interval Seconds value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_key_rotation_init(SNEPPXKeyRotationScheduler* ks, uint64_t interval_seconds);
/**
 * @brief Perform Key Rotation Check.
 *
 * @param ks [out] Ks value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_key_rotation_check(SNEPPXKeyRotationScheduler* ks);

/* Web dashboard stub */
/**
 * @brief Initialize Security Dashboard.
 *
 * @param listen_addr [in] Listen Addr value.
 * @param port [in] Port value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_security_dashboard_init(const char* listen_addr, int port);
/**
 * @brief Update Security Dashboard.
 *
 * @param json_payload [in] Json Payload value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_security_dashboard_update(const char* json_payload);

/* Threat visualization */
/**
 * @brief Initialize Threat Viz.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_threat_viz_init(void);
/**
 * @brief Perform Threat Viz Add Edge.
 *
 * @param from [in] From value.
 * @param to [in] To value.
 * @param label [in] Label value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_threat_viz_add_edge(const char* from, const char* to, const char* label);

/* Policy editor DSL */
typedef struct {
    char rules[32][256];
    int rule_count;
} SNEPPXPolicyDSL;

/**
 * @brief Initialize Policy Dsl.
 *
 * @param dsl [out] Dsl value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_policy_dsl_init(SNEPPXPolicyDSL* dsl);
/**
 * @brief Perform Policy Dsl Add Rule.
 *
 * @param dsl [out] Dsl value.
 * @param rule [in] Rule value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_policy_dsl_add_rule(SNEPPXPolicyDSL* dsl, const char* rule);
/**
 * @brief Perform Policy Dsl Compile.
 *
 * @param dsl [out] Dsl value.
 * @param bytecode [out] Bytecode value.
 * @param bytecode_len [out] Bytecode Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_policy_dsl_compile(SNEPPXPolicyDSL* dsl, uint8_t* bytecode, size_t* bytecode_len);

/* Compliance report generator */
/**
 * @brief Perform Compliance Report.
 *
 * @param report_type [in] Report Type value.
 * @param output_path [in] Output Path value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_compliance_report(const char* report_type, const char* output_path);

#ifdef __cplusplus
}
#endif
#endif
