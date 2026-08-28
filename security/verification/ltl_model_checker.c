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
    /* Minimal real LTL: only assert properties that are actually verified
       against the loaded model. Previously this was a no-op returning 0 for
       ANY input (false assurance). Unsupported/unknown properties are reported
       as NOT verified (non-zero) rather than falsely claimed to hold. */
    if (strcmp(property, "G !deadlock") == 0 || strcmp(property, "G not deadlock") == 0) {
        return SNEPPX_fv_get_deadlock_count() == 0 ? 0 : -1;
    }
    if (strcmp(property, "G reachable") == 0 || strcmp(property, "G reachable>0") == 0) {
        return SNEPPX_fv_get_reachable_count() > 0 ? 0 : -1;
    }
    return -1;
}

int SNEPPX_fv_verify_invariant(int (*invariant)(uint32_t)) {
    if (!invariant || !g_fv_initialized) return 0;
    for (int i = 0; i < g_fv_model.state_count; i++) {
        if (!invariant(g_fv_model.states[i].state_id)) return 0;
    }
    return 1;
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
    if (!g_fv_initialized) return 0;
    int deadlocks = 0;
    for (int i = 0; i < g_fv_model.state_count; i++) {
        if (g_fv_model.states[i].next_count == 0) deadlocks++;
    }
    return deadlocks;
}
