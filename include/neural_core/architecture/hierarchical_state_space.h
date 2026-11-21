#ifndef SNEPPX_HSS_H
#define SNEPPX_HSS_H

#ifdef __cplusplus
extern "C" {
#endif

#include "multidimensional_tensor_engine.h"
#include "automatic_differentiation_framework.h"
#include <stddef.h>
/*
 * SNEPPX - Hierarchical State Space (HSS)
 *
 * WHAT
 *   Hierarchical State Space (HSS).
 *
 * CONCEPT
 *   HSS model configuration, layer structs, and forward declaration API.
 *
 * ROLE
 *   Declares the HSS module API used by the system architecture and the HSS core implementation.
 *
 * REFERENCES
 *   None (internal module).
 */



typedef struct {
    size_t state_dim;
    size_t input_dim;
    size_t output_dim;
    size_t num_layers;
    size_t seq_len;
    float dt_min;
    float dt_max;
    int use_hierarchical;
    int use_parallel_scan;
} SNEPPXHSSConfig;

typedef struct {
    SNEPPXTensor* A;
    SNEPPXTensor* B;
    SNEPPXTensor* C;
    SNEPPXTensor* D;
    SNEPPXTensor* dt;
    SNEPPXTensor* h;
    SNEPPXTensor* x_proj;
    SNEPPXTensor* x_proj_bias;
    SNEPPXTensor* A_bar;
    SNEPPXTensor* B_bar;
} SNEPPXHSSLayer;

typedef struct {
    SNEPPXHSSLayer** layers;
    SNEPPXHSSConfig config;
    SNEPPXTensor** norm_gamma;
    SNEPPXTensor** norm_beta;
} SNEPPXHSSModel;

/**
 * @brief Perform Hss Config Default.
 *
 * @return The result value, or 0 on error.
 */
SNEPPXHSSConfig SNEPPX_hss_config_default(void);
/**
 * @brief Create Hss Layer.
 *
 * @param config [in] Config value.
 * @param seed [in] Seed value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXHSSLayer* SNEPPX_hss_layer_create(const SNEPPXHSSConfig* config, unsigned int seed);
/**
 * @brief Destroy Hss Layer.
 *
 * @param layer [out] Layer value.
 */
void SNEPPX_hss_layer_destroy(SNEPPXHSSLayer* layer);
/**
 * @brief Create Hss Model.
 *
 * @param config [in] Config value.
 * @param seed [in] Seed value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXHSSModel* SNEPPX_hss_model_create(const SNEPPXHSSConfig* config, unsigned int seed);
/**
 * @brief Destroy Hss Model.
 *
 * @param model [out] Model value.
 */
void SNEPPX_hss_model_destroy(SNEPPXHSSModel* model);
/**
 * @brief Run the forward pass for Hss.
 *
 * @param model [out] Model value.
 * @param input [in] Input value.
 * @param output [out] Output value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hss_forward(SNEPPXHSSModel* model, const SNEPPXTensor* input, SNEPPXTensor** output);
/**
 * @brief Perform Hss Discretize Zoh.
 *
 * @param A [in] A value.
 * @param B [in] B value.
 * @param state_dim [in] State Dim value.
 * @param dt [in] Dt value.
 * @param Ad [out] Ad value.
 * @param Bd [out] Bd value.
 */
void SNEPPX_hss_discretize_zoh(const float* A, const float* B, int state_dim, float dt, float* Ad, float* Bd);
/**
 * @brief Perform Hss Discretize Bilinear.
 *
 * @param A [in] A value.
 * @param B [in] B value.
 * @param state_dim [in] State Dim value.
 * @param dt [in] Dt value.
 * @param Ad [out] Ad value.
 * @param Bd [out] Bd value.
 */
void SNEPPX_hss_discretize_bilinear(const float* A, const float* B, int state_dim, float dt, float* Ad, float* Bd);
/**
 * @brief Perform Hss Hierarchical Levels.
 *
 * @param state_dim [in] State Dim value.
 * @param min_dim [in] Min Dim value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hss_hierarchical_levels(int state_dim, int min_dim);
/**
 * @brief Perform Hss Discretize.
 *
 * @param layer [out] Layer value.
 */
void SNEPPX_hss_discretize(SNEPPXHSSLayer* layer);
/**
 * @brief Perform Hss Step.
 *
 * @param layer [in] Layer value.
 * @param x [in] X value.
 * @param h_next [out] H Next value.
 */
void SNEPPX_hss_step(const SNEPPXHSSLayer* layer, const SNEPPXTensor* x, SNEPPXTensor* h_next);
/**
 * @brief Perform Hss Scan.
 *
 * @param layer [in] Layer value.
 * @param x_seq [in] X Seq value.
 * @param h_seq [out] H Seq value.
 * @param y_seq [out] Y Seq value.
 */
void SNEPPX_hss_scan(const SNEPPXHSSLayer* layer, const SNEPPXTensor* x_seq, SNEPPXTensor* h_seq, SNEPPXTensor* y_seq);
/**
 * @brief Perform Hss Parallel Scan.
 *
 * @param layer [in] Layer value.
 * @param x_seq [in] X Seq value.
 * @param h_seq [out] H Seq value.
 * @param y_seq [out] Y Seq value.
 */
void SNEPPX_hss_parallel_scan(const SNEPPXHSSLayer* layer, const SNEPPXTensor* x_seq, SNEPPXTensor* h_seq, SNEPPXTensor* y_seq);
/**
 * @brief Perform Hss Hierarchical Scan.
 *
 * @param layer [in] Layer value.
 * @param x_seq [in] X Seq value.
 * @param y_seq [out] Y Seq value.
 */
void SNEPPX_hss_hierarchical_scan(const SNEPPXHSSLayer* layer, const SNEPPXTensor* x_seq, SNEPPXTensor* y_seq);
/**
 * @brief Perform Hss Get Params.
 *
 * @param model [in] Model value.
 * @param out_params [out] Out Params value.
 * @param max_params [in] Max Params value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_hss_get_params(const SNEPPXHSSModel* model, SNEPPXTensor** out_params, size_t max_params);
/**
 * @brief Perform Hss Build Train Graph.
 *
 * @param model [out] Model value.
 * @param tape [out] Tape value.
 * @param input_var [out] Input Var value.
 * @param weight_vars [out] Weight Vars value.
 * @param num_weights [in] Num Weights value.
 * @param output_var [out] Output Var value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hss_build_train_graph(SNEPPXHSSModel* model, SNEPPXTape* tape,
                               SNEPPXVariable* input_var,
                               SNEPPXVariable** weight_vars, size_t num_weights,
                               SNEPPXVariable** output_var);


#ifdef __cplusplus
}
#endif
#endif /* SNEPPX_HSS_H */
