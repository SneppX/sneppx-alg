#ifndef SNEPPX_ARCH_H
#define SNEPPX_ARCH_H

#ifdef __cplusplus
extern "C" {
#endif

#include "hierarchical_state_space.h"
#include "sparse_expert_routing.h"
#include "adversarial_robustness_certification.h"
#include "neural_programming_engine.h"
#include "fractal_memory_orchestrator.h"
#include "multi_head_attention_module.h"
/*
 * SNEPPX - System Architecture Definitions
 *
 * WHAT
 *   System Architecture Definitions.
 *
 * CONCEPT
 *   Unified configuration and model structs for the HSS-SER-ARC-NPE-FM pipeline.
 *
 * ROLE
 *   Top-level architecture header that wires all algorithm modules together.
 *
 * REFERENCES
 *   None (internal architecture).
 */



typedef struct {
    SNEPPXHSSConfig hss_config;
    SNEPPXSERConfig ser_config;
    SNEPPXARCConfig arc_config;
    SNEPPXNPEConfig npe_config;
    SNEPPXFMConfig fm_config;
    SNEPPXAttentionConfig attention_config;
    int enable_attention;
    int enable_hss;
    int enable_ser;
    int enable_arc;
    int enable_npe;
    int enable_fm;
    size_t input_dim;
    size_t output_dim;
    size_t vocab_size;
    unsigned int seed;
} SNEPPXArchConfig;

typedef struct {
    SNEPPXHSSModel* hss_model;
    SNEPPXSERModel* ser_model;
    SNEPPXARCLayer* arc_layer;
    SNEPPXNPEVM* npe_vm;
    SNEPPXNPEProgram* npe_program;
    SNEPPXFMController* fm_controller;
    SNEPPXAttentionWeights* attention;
    SNEPPXTensor* embed_weight;
    SNEPPXTensor* unembed_weight;
    SNEPPXArchConfig config;
} SNEPPXModel;

/**
 * @brief Perform Arch Config Default.
 *
 * @return The result value, or 0 on error.
 */
SNEPPXArchConfig SNEPPX_arch_config_default(void);
/**
 * @brief Perform Arch Has Avx.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_arch_has_avx(void);
/**
 * @brief Perform Arch Has Avx2.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_arch_has_avx2(void);
/**
 * @brief Perform Arch Has Neon.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_arch_has_neon(void);
/**
 * @brief Perform Arch Num Cores.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_arch_num_cores(void);
/**
 * @brief Perform Arch Cache Line Size.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_arch_cache_line_size(void);
/**
 * @brief Create Model.
 *
 * @param config [in] Config value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXModel* SNEPPX_model_create(const SNEPPXArchConfig* config);
/**
 * @brief Destroy Model.
 *
 * @param model [out] Model value.
 */
void SNEPPX_model_destroy(SNEPPXModel* model);
/**
 * @brief Run the forward pass for Model.
 *
 * @param model [out] Model value.
 * @param input [in] Input value.
 * @param output [out] Output value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_model_forward(SNEPPXModel* model, const SNEPPXTensor* input, SNEPPXTensor** output);
/**
 * @brief Perform Model Get Params.
 *
 * @param model [in] Model value.
 * @param out_params [out] Out Params value.
 * @param max_params [in] Max Params value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_model_get_params(const SNEPPXModel* model, SNEPPXTensor** out_params, size_t max_params);
/**
 * @brief Perform Model Build Train Graph.
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
int SNEPPX_model_build_train_graph(SNEPPXModel* model, SNEPPXTape* tape,
                                  SNEPPXVariable* input_var,
                                  SNEPPXVariable** weight_vars, size_t num_weights,
                                  SNEPPXVariable** output_var);


#ifdef __cplusplus
}
#endif
#endif /* SNEPPX_ARCH_H */
