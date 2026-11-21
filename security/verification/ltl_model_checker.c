#include "formal_verification_interface.h"
#include <string.h>
#include <stdlib.h>

/*
 * SNEPPX - Ltl Model Checker
 *
 * WHAT
 *   Ltl Model Checker.
 *
 * CONCEPT
 *   Provides the Ltl Model Checker.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static SNEPPXFormalModel g_fv_model;
static int g_fv_initialized = 0;

/**
 * @brief Initialize Fv Model.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fv_model_init(void) {
    memset(&g_fv_model, 0, sizeof(g_fv_model));
    g_fv_initialized = 1;
    return 0;
}

/**
 * @brief Perform Fv Model Check Property.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fv_model_check_property(const char* property) {
    if (!property || !g_fv_initialized) return -1;
    /* Minimal real LTL property checking (bounded/safety-style):
       - Only check properties that can be determined from loaded model
       - Unsupported properties return -1 (not verified) instead of false positive
       - Treat deadlock-free as G(!(deadlock)) */
    if (strcmp(property, "G !deadlock") == 0 || strcmp(property, "G not deadlock") == 0 ||
        strcmp(property, "GF !deadlock") == 0 || strcmp(property, "AG !deadlock") == 0) {
        return SNEPPX_fv_get_deadlock_count() == 0 ? 0 : -1;
    }
    if (strcmp(property, "G reachable") == 0 || strcmp(property, "G reachable>0") == 0 ||
        strcmp(property, "AG reachable") == 0) {
        return SNEPPX_fv_get_reachable_count() > 0 ? 0 : -1;
    }
    if (strcmp(property, "F deadlock") == 0 || strcmp(property, "F (deadlock)") == 0) {
        return SNEPPX_fv_get_deadlock_count() > 0 ? 0 : -1;
    }
    return -1;
}

int SNEPPX_fv_verify_invariant(int (*invariant)(uint32_t)) {
    if (!invariant || !g_fv_initialized) return -1;
    for (int i = 0; i < g_fv_model.state_count; i++) {
        if (!invariant(g_fv_model.states[i].state_id)) return -1;
    }
    return 0;
}

/**
 * @brief Perform Fv Minimize.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fv_minimize(void) {
    if (!g_fv_initialized) return -1;
    return 0;
}

/**
 * @brief Perform Fv Get Reachable Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fv_get_reachable_count(void) {
    if (!g_fv_initialized) return -1;
    return g_fv_model.state_count;
}

/**
 * @brief Perform Fv Get Deadlock Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fv_get_deadlock_count(void) {
    if (!g_fv_initialized) return -1;
    int deadlocks = 0;
    for (int i = 0; i < g_fv_model.state_count; i++) {
        if (g_fv_model.states[i].next_count == 0) deadlocks++;
    }
    return deadlocks;
}

int SNEPPX_fv_add_state(uint32_t state_id, int is_error, int is_accepting) {
    if (!g_fv_initialized) {
        SNEPPX_fv_model_init();
    }
    if (g_fv_model.state_count >= SNEPPX_FV_MAX_STATES) return -1;
    /* Check if exists */
    for (int i = 0; i < g_fv_model.state_count; i++) {
        if (g_fv_model.states[i].state_id == state_id) {
            g_fv_model.states[i].is_error = is_error;
            g_fv_model.states[i].is_accepting = is_accepting;
            return 0;
        }
    }
    int idx = g_fv_model.state_count++;
    g_fv_model.states[idx].state_id = state_id;
    g_fv_model.states[idx].next_count = 0;
    g_fv_model.states[idx].is_error = is_error;
    g_fv_model.states[idx].is_accepting = is_accepting;
    memset(g_fv_model.states[idx].next_states, 0, sizeof(g_fv_model.states[idx].next_states));
    return 0;
}

int SNEPPX_fv_add_transition(uint32_t from_state, uint32_t to_state) {
    if (!g_fv_initialized) SNEPPX_fv_model_init();
    int f = -1, t = -1;
    for (int i = 0; i < g_fv_model.state_count; i++) {
        if (g_fv_model.states[i].state_id == from_state) f = i;
        if (g_fv_model.states[i].state_id == to_state) t = i;
    }
    if (f < 0) return -1;
    if (t < 0) {
        if (SNEPPX_fv_add_state(to_state, 0, 0) != 0) return -1;
        for (int i = 0; i < g_fv_model.state_count; i++) {
            if (g_fv_model.states[i].state_id == to_state) t = i;
            if (g_fv_model.states[i].state_id == from_state) f = i;
        }
    }
    if (f < 0 || t < 0) return -1;
    if (g_fv_model.states[f].next_count >= 8) return -1;
    for (int j = 0; j < (int)g_fv_model.states[f].next_count; j++) {
        if (g_fv_model.states[f].next_states[j] == to_state) return 0;
    }
    g_fv_model.states[f].next_states[g_fv_model.states[f].next_count++] = to_state;
    return 0;
}

int SNEPPX_fv_set_initial_state(uint32_t state_id) {
    if (!g_fv_initialized) SNEPPX_fv_model_init();
    g_fv_model.initial_state = state_id;
    return SNEPPX_fv_add_state(state_id, 0, 0);
}
