#include "differential_privacy.h"
#include "cryptographic_random_generator.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DP_MAX_MECHANISMS 64
#define DP_MAX_COMPOSITIONS 256

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


static dp_mechanism_t mechanisms[DP_MAX_MECHANISMS];
static int mech_count = 0;
static dp_composition_t compositions[DP_MAX_COMPOSITIONS];
static int comp_count = 0;
static double total_epsilon_consumed = 0.0;
static double total_delta_consumed = 0.0;
static dp_budget_t global_budget = {1.0, 1e-5};
static int budget_initialized = 0;

/**
 * @brief Perform Dp Laplace Mech.
 *
 * @param value [in] Value value.
 * @param sensitivity [in] Sensitivity value.
 *
 * @return The result value, or 0 on error.
 */
double SNEPPX_dp_laplace_mech(double value, double sensitivity, double epsilon) {
    double scale = sensitivity / epsilon;
    double u = ((double)rand() / RAND_MAX) - 0.5;
    return value - scale * (u > 0 ? log(1 - 2 * u) : -log(1 + 2 * u));
}

/**
 * @brief Perform Dp Gaussian Mech.
 *
 * @param value [in] Value value.
 * @param sensitivity [in] Sensitivity value.
 * @param epsilon [in] Epsilon value.
 *
 * @return The result value, or 0 on error.
 */
double SNEPPX_dp_gaussian_mech(double value, double sensitivity, double epsilon, double delta) {
    double sigma = sensitivity * sqrt(2 * log(1.25 / delta)) / epsilon;
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    double z = sqrt(-2 * log(u1)) * cos(2 * 3.141592653589793 * u2);
    return value + sigma * z;
}

/**
 * @brief Perform Dp Register Mechanism.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dp_register_mechanism(dp_mechanism_t *mech) {
    if (mech_count >= DP_MAX_MECHANISMS) return -1;
    mechanisms[mech_count++] = *mech;
    return mech_count - 1;
}

/**
 * @brief Perform Dp Compose.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dp_compose(dp_composition_t *comp) {
    if (comp_count >= DP_MAX_COMPOSITIONS) return -1;
    double eps_sum = 0.0, del_sum = 0.0;
    for (int i = 0; i < comp->num_mechanisms; i++) {
        eps_sum += mechanisms[comp->indices[i]].epsilon;
        del_sum += mechanisms[comp->indices[i]].delta;
    }
    comp->total_epsilon = eps_sum;
    comp->total_delta = del_sum;
    comp->id = comp_count;
    compositions[comp_count++] = *comp;
    total_epsilon_consumed += eps_sum;
    total_delta_consumed += del_sum;
    return comp->id;
}

/**
 * @brief Perform Dp Sequential Compose.
 *
 * @param epsilons [out] Epsilons value.
 * @param deltas [out] Deltas value.
 * @param n [in] N value.
 * @param total_eps [out] Total Eps value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dp_sequential_compose(double *epsilons, double *deltas, int n, double *total_eps, double *total_del) {
    *total_eps = 0; *total_del = 0;
    for (int i = 0; i < n; i++) {
        *total_eps += epsilons[i];
        *total_del += deltas[i];
    }
    return 0;
}

/**
 * @brief Perform Dp Advanced Compose.
 *
 * @param epsilons [out] Epsilons value.
 * @param deltas [out] Deltas value.
 * @param n [in] N value.
 * @param delta [in] Delta value.
 * @param total_eps [out] Total Eps value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dp_advanced_compose(double *epsilons, double *deltas, int n, double delta, double *total_eps, double *total_del) {
    *total_eps = 2 * sqrt(n * log(1.0 / delta));
    for (int i = 0; i < n; i++) *total_eps += epsilons[i] * (exp(epsilons[i]) - 1);
    *total_del = delta;
    return 0;
}

/**
 * @brief Get Dp Set Bud.
 *
 * @param epsilon [in] Epsilon value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dp_set_budget(double epsilon, double delta) {
    global_budget.epsilon = epsilon;
    global_budget.delta = delta;
    budget_initialized = 1;
    return 0;
}

/**
 * @brief Get Dp Check Bud.
 *
 * @param epsilon [in] Epsilon value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dp_check_budget(double epsilon, double delta) {
    if (!budget_initialized) return 0;
    return (total_epsilon_consumed + epsilon <= global_budget.epsilon &&
            total_delta_consumed + delta <= global_budget.delta) ? 0 : -1;
}

/**
 * @brief Get Dp Reset Bud.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dp_reset_budget(void) {
    total_epsilon_consumed = 0;
    total_delta_consumed = 0;
    return 0;
}

/**
 * @brief Perform Dp Dpsgd Step.
 *
 * @param params [out] Params value.
 * @param gradients [in] Gradients value.
 * @param n [in] N value.
 * @param lr [in] Lr value.
 * @param epsilon [in] Epsilon value.
 * @param delta [in] Delta value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dp_dpsgd_step(double *params, const double *gradients, int n, double lr, double epsilon, double delta, double clip_norm) {
    double norm = 0;
    for (int i = 0; i < n; i++) norm += gradients[i] * gradients[i];
    norm = sqrt(norm);
    double scale = (norm > clip_norm) ? clip_norm / norm : 1.0;
    double sigma = clip_norm * sqrt(2 * log(1.25 / delta)) / epsilon;
    for (int i = 0; i < n; i++) {
        double u1 = (double)rand() / RAND_MAX;
        double u2 = (double)rand() / RAND_MAX;
        double noise = sqrt(-2 * log(u1)) * cos(2 * 3.141592653589793 * u2) * sigma;
        params[i] -= lr * (gradients[i] * scale + noise);
    }
    return 0;
}

/**
 * @brief Perform Dp Noise Layer.
 *
 * @param data [out] Data value.
 * @param n [in] N value.
 * @param sensitivity [in] Sensitivity value.
 * @param epsilon [in] Epsilon value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dp_noise_layer(double *data, int n, double sensitivity, double epsilon, double delta) {
    double sigma = sensitivity * sqrt(2 * log(1.25 / delta)) / epsilon;
    for (int i = 0; i < n; i++) {
        double u1 = (double)rand() / RAND_MAX;
        double u2 = (double)rand() / RAND_MAX;
        data[i] += sqrt(-2 * log(u1)) * cos(2 * 3.141592653589793 * u2) * sigma;
    }
    return 0;
}

/**
 * @brief Perform Dp Get Stats.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dp_get_stats(dp_stats_t *stats) {
    if (!stats) return -1;
    stats->total_epsilon_consumed = total_epsilon_consumed;
    stats->total_delta_consumed = total_delta_consumed;
    stats->num_mechanisms = mech_count;
    stats->num_compositions = comp_count;
    stats->budget_remaining_epsilon = budget_initialized ? global_budget.epsilon - total_epsilon_consumed : -1;
    stats->budget_remaining_delta = budget_initialized ? global_budget.delta - total_delta_consumed : -1;
    return 0;
}
