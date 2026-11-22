#ifndef SNEPPX_OUTPUT_VERIFIER_H
#define SNEPPX_OUTPUT_VERIFIER_H
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

/*
 * S5 AI Sanitizer — Output Verification
 * Validates AI outputs for toxicity, bias, factual consistency,
 * and policy compliance before delivery.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define SNEPPX_MAX_TOPIC_BLOCKLIST 128
#define SNEPPX_TOPIC_MAX_LEN 64

    typedef struct
    {
        char topic[SNEPPX_TOPIC_MAX_LEN];
        int is_blocked;
    } SNEPPXBlockedTopic;

    typedef struct
    {
        SNEPPXBlockedTopic topics[SNEPPX_MAX_TOPIC_BLOCKLIST];
        int topic_count;
        double toxicity_threshold;
        double bias_threshold;
        int check_factual_consistency;
        int max_output_length;
    } SNEPPXS5Verifier;

    /*
     * The S5 verifier is implemented under the SNEPPX_output_verifier_*
     * names (see security/ai/output_verifier.c); declare them here so the
     * C test suite compiles as C++ without implicit declarations.
     */
    int SNEPPX_output_verifier_init(SNEPPXS5Verifier* ov);
    void SNEPPX_output_verifier_destroy(SNEPPXS5Verifier* ov);
    int SNEPPX_output_verifier_check(SNEPPXS5Verifier* ov, const char* output, size_t len);
    int SNEPPX_output_verifier_add_blocked_topic(SNEPPXS5Verifier* ov, const char* topic);
    int SNEPPX_output_verifier_sanitize(SNEPPXS5Verifier* ov, const char* output, size_t len, char* safe_output,
                                        size_t* safe_len);

    /**
     * @brief Initialize S5 Verifier.
     *
     * @param ov [out] Ov value.
     *
     * @return 0 on success, -1 on error.
     */
    int SNEPPX_s5_verifier_init(SNEPPXS5Verifier* ov);
    /**
     * @brief Destroy S5 Verifier.
     *
     * @param ov [out] Ov value.
     */
    void SNEPPX_s5_verifier_destroy(SNEPPXS5Verifier* ov);
    /**
     * @brief Perform S5 Verifier Add Blocked Topic.
     *
     * @param ov [out] Ov value.
     * @param topic [in] Topic value.
     *
     * @return 0 on success, -1 on error.
     */
    int SNEPPX_s5_verifier_add_blocked_topic(SNEPPXS5Verifier* ov, const char* topic);
    /**
     * @brief Perform S5 Verifier Check.
     *
     * @param ov [out] Ov value.
     * @param output [in] Output value.
     * @param len [in] Len value.
     *
     * @return 0 on success, -1 on error.
     */
    int SNEPPX_s5_verifier_check(SNEPPXS5Verifier* ov, const char* output, size_t len);
    /**
     * @brief Perform S5 Verifier Sanitize.
     *
     * @param ov [out] Ov value.
     * @param output [in] Output value.
     * @param len [in] Len value.
     * @param safe_output [out] Safe Output value.
     * @param safe_len [out] Safe Len value.
     *
     * @return 0 on success, -1 on error.
     */
    int SNEPPX_s5_verifier_sanitize(SNEPPXS5Verifier* ov, const char* output, size_t len, char* safe_output,
                                    size_t* safe_len);

#ifdef __cplusplus
}
#endif
#endif
