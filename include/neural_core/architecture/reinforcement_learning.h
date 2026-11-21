#ifndef SNEPPX_REINFORCEMENT_LEARNING_H
#define SNEPPX_REINFORCEMENT_LEARNING_H
#include <stddef.h>
#include <stdbool.h>
/*
 * SNEPPX - Reinforcement Learning
 *
 * WHAT
 *   Reinforcement Learning.
 *
 * CONCEPT
 *   RL configuration and training API for the SNEPPX-Algo system.
 *
 * ROLE
 *   Declares the RL API for reinforcement learning operations within the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal module).
 */


#ifdef __cplusplus
extern "C" {
#endif

typedef struct SNEPPXRLAgent SNEPPXRLAgent;

/**
 * @brief Create Rl.
 *
 * @param state_dim [in] State Dim value.
 * @param action_dim [in] Action Dim value.
 * @param hidden_dim [in] Hidden Dim value.
 * @param lr [in] Lr value.
 * @param gamma [in] Gamma value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXRLAgent* SNEPPX_rl_create(size_t state_dim, size_t action_dim, size_t hidden_dim,
    float lr, float gamma);
/**
 * @brief Destroy Rl.
 *
 * @param agent [out] Agent value.
 */
void SNEPPX_rl_destroy(void* agent);
/**
 * @brief Perform Rl Select Action.
 *
 * @param agent [out] Agent value.
 * @param state [in] State value.
 * @param action [out] Action value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rl_select_action(void* agent, const float* state, float* action);
/**
 * @brief Update Rl.
 *
 * @param agent [out] Agent value.
 * @param state [in] State value.
 * @param action [in] Action value.
 * @param reward [in] Reward value.
 * @param next_state [in] Next State value.
 * @param done [in] Done value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rl_update(void* agent, const float* state, const float* action, float reward,
    const float* next_state, int done);

#ifdef __cplusplus
}
#endif
#endif
