#ifndef SNEPPX_FORMAL_VERIFICATION_INTERFACE_H
#define SNEPPX_FORMAL_VERIFICATION_INTERFACE_H

#include <stdint.h>

#define SNEPPX_FV_MAX_STATES 1024
#define SNEPPX_FV_MAX_TRANSITIONS 4096

/*
 * SNEPPX - Formal Verification Interface
 *
 * WHAT
 *   Formal Verification Interface.
 *
 * CONCEPT
 *   Provides the Formal Verification Interface.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef struct {
    uint32_t state_id;
    uint32_t next_count;
    uint32_t next_states[8];
    int is_error;
    int is_accepting;
} SNEPPXFVState;

typedef struct {
    SNEPPXFVState states[SNEPPX_FV_MAX_STATES];
    int state_count;
    uint32_t initial_state;
} SNEPPXFormalModel;

/**
 * @brief Initialize Fv Model.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fv_model_init(void);
/**
 * @brief Perform Fv Model Check Property.
 *
 * @param property [in] Property value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fv_model_check_property(const char* property);
int SNEPPX_fv_verify_invariant(int (*invariant)(uint32_t));
/**
 * @brief Perform Fv Minimize.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fv_minimize(void);
/**
 * @brief Perform Fv Get Reachable Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fv_get_reachable_count(void);
/**
 * @brief Perform Fv Get Deadlock Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fv_get_deadlock_count(void);
/**
 * @brief Add state to formal model.
 */
int SNEPPX_fv_add_state(uint32_t state_id, int is_error, int is_accepting);
/**
 * @brief Add transition between states.
 */
int SNEPPX_fv_add_transition(uint32_t from_state, uint32_t to_state);
/**
 * @brief Set initial state.
 */
int SNEPPX_fv_set_initial_state(uint32_t state_id);

#endif
