#ifndef SNEPPX_MODEL_CHECKING_H
#define SNEPPX_MODEL_CHECKING_H
/*
 * SNEPPX - Model Checking
 *
 * WHAT
 *   Model Checking.
 *
 * CONCEPT
 *   Provides the Model Checking.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/*
 * S8 Formal Verification — Model Checking Framework
 * Lightweight model checking for crypto primitives, data flow, and invariants.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_MODEL_MAX_STATES 1024
#define SNEPPX_MODEL_MAX_TRANSITIONS 4096
#define SNEPPX_MODEL_PROP_MAX_LEN 256

typedef struct {
    uint32_t state_id;
    uint32_t next_states[8];
    int next_count;
    int is_accepting;
    int is_error;
} SNEPPXModelState;

typedef struct {
    SNEPPXModelState states[SNEPPX_MODEL_MAX_STATES];
    int state_count;
    int initial_state;
    char property[SNEPPX_MODEL_PROP_MAX_LEN];
} SNEPPXFormalModel;

typedef struct {
    int total_states;
    int reachable_states;
    int deadlock_states;
    int error_states;
    int property_satisfied;
    char counterexample[SNEPPX_MODEL_PROP_MAX_LEN];
} SNEPPXModelCheckResult;

/**
 * @brief Initialize Model.
 *
 * @param model [out] Model value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_model_init(SNEPPXFormalModel* model);
/**
 * @brief Perform Model Add State.
 *
 * @param model [out] Model value.
 * @param state_id [in] State Id value.
 * @param is_accepting [in] Is Accepting value.
 * @param is_error [in] Is Error value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_model_add_state(SNEPPXFormalModel* model, uint32_t state_id, int is_accepting, int is_error);
/**
 * @brief Perform Model Add Transition.
 *
 * @param model [out] Model value.
 * @param from [in] From value.
 * @param to [in] To value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_model_add_transition(SNEPPXFormalModel* model, uint32_t from, uint32_t to);
/**
 * @brief Perform Model Set Property.
 *
 * @param model [out] Model value.
 * @param property [in] Property value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_model_set_property(SNEPPXFormalModel* model, const char* property);
/**
 * @brief Perform Model Check.
 *
 * @param model [out] Model value.
 *
 * @return The result value, or 0 on error.
 */
SNEPPXModelCheckResult SNEPPX_model_check(SNEPPXFormalModel* model);
int  SNEPPX_model_verify_invariant(SNEPPXFormalModel* model, int (*invariant)(uint32_t state_id));
/**
 * @brief Perform Model Get Counterexample.
 *
 * @param model [in] Model value.
 * @param trace [out] Trace array of state IDs (initial -> error).
 * @param max_len [in] Max length of trace array.
 *
 * @return Number of states in trace, or 0 if no counterexample, -1 on error.
 */
int  SNEPPX_model_get_counterexample(SNEPPXFormalModel* model, uint32_t* trace, int max_len);

#ifdef __cplusplus
}
#endif
#endif
