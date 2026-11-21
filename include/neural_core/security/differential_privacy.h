#ifndef SNEPPX_DIFFERENTIAL_PRIVACY_H
#define SNEPPX_DIFFERENTIAL_PRIVACY_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

/*
 * SNEPPX - Differential Privacy
 *
 * WHAT
 *   Differential Privacy.
 *
 * CONCEPT
 *   Provides the Differential Privacy.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef struct {
    double epsilon;
    double delta;
} dp_budget_t;

typedef struct {
    int id;
    double epsilon;
    double delta;
    double sensitivity;
    char name[64];
} dp_mechanism_t;

typedef struct {
    int id;
    int indices[64];
    int num_mechanisms;
    double total_epsilon;
    double total_delta;
} dp_composition_t;

typedef struct {
    double total_epsilon_consumed;
    double total_delta_consumed;
    int num_mechanisms;
    int num_compositions;
    double budget_remaining_epsilon;
    double budget_remaining_delta;
} dp_stats_t;

/**
 * @brief Perform Dp Laplace Mech.
 *
 * @param value [in] Value value.
 * @param sensitivity [in] Sensitivity value.
 * @param epsilon [in] Epsilon value.
 *
 * @return The result value, or 0 on error.
 */
double SNEPPX_dp_laplace_mech(double value, double sensitivity, double epsilon);
/**
 * @brief Perform Dp Gaussian Mech.
 *
 * @param value [in] Value value.
 * @param sensitivity [in] Sensitivity value.
 * @param epsilon [in] Epsilon value.
 * @param delta [in] Delta value.
 *
 * @return The result value, or 0 on error.
 */
double SNEPPX_dp_gaussian_mech(double value, double sensitivity, double epsilon, double delta);
/**
 * @brief Perform Dp Register Mechanism.
 *
 * @param mech [out] Mech value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dp_register_mechanism(dp_mechanism_t *mech);
/**
 * @brief Perform Dp Compose.
 *
 * @param comp [out] Comp value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dp_compose(dp_composition_t *comp);
/**
 * @brief Perform Dp Sequential Compose.
 *
 * @param epsilons [out] Epsilons value.
 * @param deltas [out] Deltas value.
 * @param n [in] N value.
 * @param total_eps [out] Total Eps value.
 * @param total_del [out] Total Del value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dp_sequential_compose(double *epsilons, double *deltas, int n, double *total_eps, double *total_del);
/**
 * @brief Perform Dp Advanced Compose.
 *
 * @param epsilons [out] Epsilons value.
 * @param deltas [out] Deltas value.
 * @param n [in] N value.
 * @param delta [in] Delta value.
 * @param total_eps [out] Total Eps value.
 * @param total_del [out] Total Del value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dp_advanced_compose(double *epsilons, double *deltas, int n, double delta, double *total_eps, double *total_del);
/**
 * @brief Get Dp Set Bud.
 *
 * @param epsilon [in] Epsilon value.
 * @param delta [in] Delta value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dp_set_budget(double epsilon, double delta);
/**
 * @brief Get Dp Check Bud.
 *
 * @param epsilon [in] Epsilon value.
 * @param delta [in] Delta value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dp_check_budget(double epsilon, double delta);
/**
 * @brief Get Dp Reset Bud.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dp_reset_budget(void);
/**
 * @brief Perform Dp Dpsgd Step.
 *
 * @param params [out] Params value.
 * @param gradients [in] Gradients value.
 * @param n [in] N value.
 * @param lr [in] Lr value.
 * @param epsilon [in] Epsilon value.
 * @param delta [in] Delta value.
 * @param clip_norm [in] Clip Norm value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dp_dpsgd_step(double *params, const double *gradients, int n, double lr, double epsilon, double delta, double clip_norm);
/**
 * @brief Perform Dp Noise Layer.
 *
 * @param data [out] Data value.
 * @param n [in] N value.
 * @param sensitivity [in] Sensitivity value.
 * @param epsilon [in] Epsilon value.
 * @param delta [in] Delta value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dp_noise_layer(double *data, int n, double sensitivity, double epsilon, double delta);
/**
 * @brief Perform Dp Get Stats.
 *
 * @param stats [out] Stats value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dp_get_stats(dp_stats_t *stats);


#ifdef __cplusplus
}
#endif
#endif
