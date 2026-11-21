#include "sparse_expert_routing.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <math.h>
/*
 * SNEPPX - SER Expert Network
 *
 * WHAT
 *   SER Expert Network.
 *
 * CONCEPT
 *   Individual expert network forward pass in the Mixture-of-Experts architecture.
 *
 * ROLE
 *   Each expert is a small feed-forward network that processes a subset of tokens routed to it by the gating function.
 *
 * REFERENCES
 *   None (internal algorithm).
 */



/**
 * @brief Perform Ser Config Default.
 *
 * @return The result value, or 0 on error.
 */
SNEPPXSERConfig SNEPPX_ser_config_default(void) {
    SNEPPXSERConfig cfg;
    cfg.num_experts = 8;
    cfg.num_active = 2;
    cfg.input_dim = 512;
    cfg.expert_dim = 2048;
    cfg.output_dim = 512;
    cfg.top_k_method = SNEPPX_TOPK_GREEDY;
    cfg.load_balance_coef = 0.01f;
    cfg.dropout_rate = 0.0f;
    cfg.use_mlp_gater = 0;
    cfg.gater_hidden_dim = 64;
    return cfg;
}

static unsigned long lcg_next(unsigned long state) {
    return state * 1103515245UL + 12345UL;
}

static void fill_randn(float* data, size_t n, unsigned long* state, float scale) {
    for (size_t i = 0; i < n; i += 2) {
        *state = lcg_next(*state);
        float u1 = (float)((*state >> 16) & 0x7FFF) / 32767.0f;
        *state = lcg_next(*state);
        float u2 = (float)((*state >> 16) & 0x7FFF) / 32767.0f;
        float r = sqrtf(-2.0f * logf(u1 + 1e-10f));
        float theta = 2.0f * 3.14159265f * u2;
        data[i] = r * cosf(theta) * scale;
        if (i + 1 < n) {
            data[i + 1] = r * sinf(theta) * scale;
        }
    }
}

/**
 * @brief Create Expert.
 *
 * @param config [in] Config value.
 * @param seed [in] Seed value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXExpert* SNEPPX_expert_create(const SNEPPXSERConfig* config, unsigned int seed, SNEPPXActivation activation) {
    SNEPPXExpert* expert = (SNEPPXExpert*)SNEPPX_malloc(sizeof(SNEPPXExpert), 64);
    if (!expert) return NULL;
    memset(expert, 0, sizeof(SNEPPXExpert));

    unsigned long state = seed;
    size_t i_dim = config->input_dim;
    size_t e_dim = config->expert_dim;
    size_t o_dim = config->output_dim;

    size_t shape_ei[] = {e_dim, i_dim};
    size_t shape_oe[] = {o_dim, e_dim};
    size_t shape_e1[] = {e_dim};
    size_t shape_o1[] = {o_dim};

    expert->w1 = SNEPPX_tensor_create(shape_ei, 2, SNEPPX_FLOAT32);
    expert->w2 = SNEPPX_tensor_create(shape_oe, 2, SNEPPX_FLOAT32);
    expert->b1 = SNEPPX_tensor_zeros(shape_e1, 1, SNEPPX_FLOAT32);
    expert->b2 = SNEPPX_tensor_zeros(shape_o1, 1, SNEPPX_FLOAT32);
    expert->activation = activation;

    if (expert->w1 && expert->w2 && expert->b1 && expert->b2) {
        float scale_w1 = sqrtf(2.0f / (float)i_dim);
        float scale_w2 = sqrtf(2.0f / (float)e_dim);
        fill_randn((float*)expert->w1->data, expert->w1->size, &state, scale_w1);
        fill_randn((float*)expert->w2->data, expert->w2->size, &state, scale_w2);
    }

    return expert;
}

/**
 * @brief Destroy Expert.
 */
void SNEPPX_expert_destroy(SNEPPXExpert* expert) {
    if (!expert) return;
    if (expert->w1) SNEPPX_tensor_destroy(expert->w1);
    if (expert->w2) SNEPPX_tensor_destroy(expert->w2);
    if (expert->b1) SNEPPX_tensor_destroy(expert->b1);
    if (expert->b2) SNEPPX_tensor_destroy(expert->b2);
    SNEPPX_free(expert, sizeof(SNEPPXExpert));
}
