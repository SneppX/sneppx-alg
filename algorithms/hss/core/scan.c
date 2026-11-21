#include "hierarchical_state_space.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>
/*
 * SNEPPX - HSS Associative Scan (Blelloch)
 *
 * WHAT
 *   HSS Associative Scan (Blelloch).
 *
 * CONCEPT
 *   Parallel associative scan using the Blelloch algorithm for HSS state recurrences.
 *
 * ROLE
 *   Implements the associative scan that parallelizes the HSS recurrence h_t = A h_{t-1} + B x_t into O(log n) steps.
 *
 * REFERENCES
 *   None (internal algorithm).
 */



typedef struct {
    float* A;  /* state_dim × state_dim  (row-major) */
    float* b;  /* state_dim vector */
} AssocPair;

static void assoc_combine(AssocPair* out, const AssocPair* a, const AssocPair* b, size_t s_dim) {
    /* out = b ∘ a   where (A2,b2)∘(A1,b1) = (A2·A1, A2·b1 + b2) */
    for (size_t i = 0; i < s_dim; i++) {
        for (size_t j = 0; j < s_dim; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < s_dim; k++)
                sum += b->A[i * s_dim + k] * a->A[k * s_dim + j];
            out->A[i * s_dim + j] = sum;
        }
    }
    for (size_t i = 0; i < s_dim; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < s_dim; j++)
            sum += b->A[i * s_dim + j] * a->b[j];
        out->b[i] = sum + b->b[i];
    }
}

/**
 * @brief Perform Hss Scan.
 *
 * @param layer [in] Layer value.
 * @param x_seq [in] X Seq value.
 * @param h_seq [out] H Seq value.
 */
void SNEPPX_hss_scan(const SNEPPXHSSLayer* layer, const SNEPPXTensor* x_seq, SNEPPXTensor* h_seq, SNEPPXTensor* y_seq) {
    size_t seq_len = x_seq->shape[0];
    size_t i_dim = x_seq->shape[1];
    size_t s_dim = layer->A_bar->shape[0];
    size_t o_dim = layer->C->shape[0];

    float* A_bar = (float*)layer->A_bar->data;
    float* B_bar = (float*)layer->B_bar->data;
    float* C = (float*)layer->C->data;
    float* D = (float*)layer->D->data;
    float* x_data = (float*)x_seq->data;
    float* h_data = (float*)layer->h->data;
    float* h_seq_data = (float*)h_seq->data;
    float* y_seq_data = (float*)y_seq->data;

    memset(h_data, 0, s_dim * sizeof(float));

    for (size_t t = 0; t < seq_len; t++) {
        float* x_t = &x_data[t * i_dim];

        float h_next[4096];
        for (size_t i = 0; i < s_dim; i++) {
            float sum = 0.0f;
            for (size_t j = 0; j < s_dim; j++) {
                sum += A_bar[i * s_dim + j] * h_data[j];
            }
            for (size_t k = 0; k < i_dim; k++) {
                sum += B_bar[i * i_dim + k] * x_t[k];
            }
            h_next[i] = sum;
        }

        memcpy(h_data, h_next, s_dim * sizeof(float));

        for (size_t i = 0; i < s_dim; i++) {
            h_seq_data[t * s_dim + i] = h_data[i];
        }

        for (size_t i = 0; i < o_dim; i++) {
            float y = 0.0f;
            for (size_t j = 0; j < s_dim; j++) {
                y += C[i * s_dim + j] * h_data[j];
            }
            for (size_t k = 0; k < i_dim; k++) {
                y += D[i * i_dim + k] * x_t[k];
            }
            y_seq_data[t * o_dim + i] = y;
        }
    }
}

/**
 * @brief Perform Hss Parallel Scan.
 *
 * @param layer [in] Layer value.
 * @param x_seq [in] X Seq value.
 * @param h_seq [out] H Seq value.
 */
void SNEPPX_hss_parallel_scan(const SNEPPXHSSLayer* layer, const SNEPPXTensor* x_seq, SNEPPXTensor* h_seq, SNEPPXTensor* y_seq) {
    size_t seq_len = x_seq->shape[0];
    size_t i_dim = x_seq->shape[1];
    size_t s_dim = layer->A_bar->shape[0];
    size_t o_dim = layer->C->shape[0];

    float* A_bar = (float*)layer->A_bar->data;
    float* B_bar = (float*)layer->B_bar->data;
    float* C = (float*)layer->C->data;
    float* D = (float*)layer->D->data;
    float* x_data = (float*)x_seq->data;
    float* h_seq_data = (float*)h_seq->data;
    float* y_seq_data = (float*)y_seq->data;

    if (seq_len == 0 || s_dim > 1024) { SNEPPX_hss_scan(layer, x_seq, h_seq, y_seq); return; }

    /* Allocate associative-pair array */
    size_t pair_size = s_dim * s_dim + s_dim;
    float* pair_buf = (float*)malloc(seq_len * pair_size * sizeof(float));
    if (!pair_buf) { SNEPPX_hss_scan(layer, x_seq, h_seq, y_seq); return; }

    /* Phase 1: initialise elements */
    for (size_t t = 0; t < seq_len; t++) {
        float* A_t = pair_buf + t * pair_size;
        float* b_t = A_t + s_dim * s_dim;
        memcpy(A_t, A_bar, s_dim * s_dim * sizeof(float));
        memset(b_t, 0, s_dim * sizeof(float));
        float* x_t = x_data + t * i_dim;
        for (size_t i = 0; i < s_dim; i++)
            for (size_t k = 0; k < i_dim; k++)
                b_t[i] += B_bar[i * i_dim + k] * x_t[k];
    }

    /* Phase 2: up-sweep (Blelloch) */
    int n = (int)seq_len;
    AssocPair pair_ap, curb_ap;
    float* cur_A = (float*)malloc(s_dim * s_dim * sizeof(float));
    float* cur_b = (float*)malloc(s_dim * sizeof(float));
    if (!cur_A || !cur_b) { free(pair_buf); free(cur_A); free(cur_b); SNEPPX_hss_scan(layer, x_seq, h_seq, y_seq); return; }

    for (int stride = 1; stride < n; stride *= 2) {
        for (int idx = 0; idx < n - stride; idx += 2 * stride) {
            int left = idx, right = idx + stride;
            float* A_l = pair_buf + left * pair_size;
            float* b_l = A_l + s_dim * s_dim;
            float* A_r = pair_buf + right * pair_size;
            float* b_r = A_r + s_dim * s_dim;
            memcpy(cur_A, A_r, s_dim * s_dim * sizeof(float));
            memcpy(cur_b, b_r, s_dim * sizeof(float));
            pair_ap.A = A_l; pair_ap.b = b_l;
            curb_ap.A = cur_A; curb_ap.b = cur_b;
            assoc_combine(&pair_ap, &pair_ap, &curb_ap, s_dim);
            memcpy(A_r, cur_A, s_dim * s_dim * sizeof(float));
            memcpy(b_r, cur_b, s_dim * sizeof(float));
        }
    }

    /* Phase 3: down-sweep (set last = identity, then propagate back) */
    {
        float* A_last = pair_buf + (n - 1) * pair_size;
        float* b_last = A_last + s_dim * s_dim;
        memset(A_last, 0, s_dim * s_dim * sizeof(float));
        for (size_t i = 0; i < s_dim; i++) A_last[i * s_dim + i] = 1.0f;
        memset(b_last, 0, s_dim * sizeof(float));
    }

    for (int stride = n / 2; stride > 0; stride /= 2) {
        for (int idx = 0; idx < n - stride; idx += 2 * stride) {
            int left = idx, right = idx + stride;
            float* A_l = pair_buf + left * pair_size;
            float* b_l = A_l + s_dim * s_dim;
            float* A_r = pair_buf + right * pair_size;
            float* b_r = A_r + s_dim * s_dim;
            memcpy(cur_A, A_l, s_dim * s_dim * sizeof(float));
            memcpy(cur_b, b_l, s_dim * sizeof(float));
            pair_ap.A = A_r; pair_ap.b = b_r;
            curb_ap.A = cur_A; curb_ap.b = cur_b;
            assoc_combine(&pair_ap, &pair_ap, &curb_ap, s_dim);
            memcpy(A_l, cur_A, s_dim * s_dim * sizeof(float));
            memcpy(b_l, cur_b, s_dim * sizeof(float));
        }
    }

    free(cur_A); free(cur_b);

    /* Extract hidden states and compute outputs */
    float h_prev[1024];
    memset(h_prev, 0, s_dim * sizeof(float));
    for (size_t t = 0; t < seq_len; t++) {
        float* A_t = pair_buf + t * pair_size;
        float* b_t = A_t + s_dim * s_dim;
        float h_t[1024];
        memset(h_t, 0, s_dim * sizeof(float));
        for (size_t i = 0; i < s_dim; i++) {
            float sum = 0.0f;
            for (size_t j = 0; j < s_dim; j++)
                sum += A_t[i * s_dim + j] * h_prev[j];
            h_t[i] = sum + b_t[i];
        }
        memcpy(h_prev, h_t, s_dim * sizeof(float));
        memcpy(h_seq_data + t * s_dim, h_t, s_dim * sizeof(float));
        for (size_t i = 0; i < o_dim; i++) {
            float y = 0.0f;
            for (size_t j = 0; j < s_dim; j++)
                y += C[i * s_dim + j] * h_t[j];
            for (size_t k = 0; k < i_dim; k++)
                y += D[i * i_dim + k] * x_data[t * i_dim + k];
            y_seq_data[t * o_dim + i] = y;
        }
    }

    free(pair_buf);
}
