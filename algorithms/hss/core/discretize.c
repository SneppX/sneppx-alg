#include "hierarchical_state_space.h"
#include <string.h>
#include <math.h>
/*
 * SNEPPX - HSS Discretization
 *
 * WHAT
 *   HSS Discretization.
 *
 * CONCEPT
 *   Converts continuous-time SSM matrices (A, B) to discrete-time equivalents using Zero-Order Hold or Bilinear (Tustin) transform.
 *
 * ROLE
 *   Implements the HSS pipeline stage that discretizes the state-space model before the selective scan.
 *
 * REFERENCES
 *   None (internal algorithm).
 */



/**
 * @brief Perform Hss Discretize Zoh.
 *
 * @param A [in] A value.
 * @param B [in] B value.
 * @param state_dim [in] State Dim value.
 * @param dt [in] Dt value.
 * @param Ad [out] Ad value.
 */
void SNEPPX_hss_discretize_zoh(const float* A, const float* B, int state_dim, float dt, float* Ad, float* Bd) {
    for (int i = 0; i < state_dim; i++) {
        for (int j = 0; j < state_dim; j++) {
            float a_ij = A[i * state_dim + j];
            if (i == j) {
                Ad[i * state_dim + j] = 1.0f + a_ij * dt;
            } else {
                Ad[i * state_dim + j] = a_ij * dt;
            }
        }
    }
    for (int i = 0; i < state_dim; i++) {
        for (int j = 0; j < state_dim; j++) {
            Bd[i * state_dim + j] = B[i * state_dim + j] * dt;
        }
    }
}

/**
 * @brief Perform Hss Discretize Bilinear.
 *
 * @param A [in] A value.
 * @param B [in] B value.
 * @param state_dim [in] State Dim value.
 * @param dt [in] Dt value.
 * @param Ad [out] Ad value.
 */
void SNEPPX_hss_discretize_bilinear(const float* A, const float* B, int state_dim, float dt, float* Ad, float* Bd) {
    float half_dt = 0.5f * dt;
    for (int i = 0; i < state_dim; i++) {
        for (int j = 0; j < state_dim; j++) {
            float a_ij = A[i * state_dim + j];
            if (i == j) {
                Ad[i * state_dim + j] = (1.0f + half_dt * a_ij) / (1.0f - half_dt * a_ij);
            } else {
                Ad[i * state_dim + j] = (half_dt * a_ij) / (1.0f - half_dt * A[i * state_dim + i]);
            }
        }
    }
    for (int i = 0; i < state_dim; i++) {
        for (int j = 0; j < state_dim; j++) {
            Bd[i * state_dim + j] = B[i * state_dim + j] * dt / (1.0f - half_dt * A[i * state_dim + i]);
        }
    }
}

/**
 * @brief Perform Hss Hierarchical Levels.
 *
 * @param state_dim [in] State Dim value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hss_hierarchical_levels(int state_dim, int min_dim) {
    if (state_dim <= min_dim || min_dim <= 0) return 0;
    int levels = 0;
    int current = state_dim;
    while (current > min_dim) {
        current = current / 2;
        levels++;
    }
    return levels;
}

/**
 * @brief Perform Hss Discretize.
 */
void SNEPPX_hss_discretize(SNEPPXHSSLayer* layer) {
    size_t s_dim = layer->A->shape[0];
    size_t i_dim = layer->B->shape[1];

    size_t shape_s[] = {s_dim, s_dim};
    size_t shape_si[] = {s_dim, i_dim};

    if (layer->A_bar) SNEPPX_tensor_destroy(layer->A_bar);
    if (layer->B_bar) SNEPPX_tensor_destroy(layer->B_bar);

    layer->A_bar = SNEPPX_tensor_zeros(shape_s, 2, SNEPPX_FLOAT32);
    layer->B_bar = SNEPPX_tensor_zeros(shape_si, 2, SNEPPX_FLOAT32);
    if (!layer->A_bar || !layer->B_bar) return;

    float* A = (float*)layer->A->data;
    float* B = (float*)layer->B->data;
    float* dt = (float*)layer->dt->data;
    float* A_bar = (float*)layer->A_bar->data;
    float* B_bar = (float*)layer->B_bar->data;

    for (size_t i = 0; i < s_dim; i++) {
        for (size_t j = 0; j < s_dim; j++) {
            float a_ij = A[i * s_dim + j];
            if (i == j) {
                A_bar[i * s_dim + j] = 1.0f + a_ij * dt[i];
            } else {
                A_bar[i * s_dim + j] = a_ij * dt[i];
            }
        }
    }

    for (size_t i = 0; i < s_dim; i++) {
        for (size_t j = 0; j < i_dim; j++) {
            B_bar[i * i_dim + j] = B[i * i_dim + j] * dt[i];
        }
    }
}
