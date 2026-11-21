#ifndef SNEPPX_SER_H
#define SNEPPX_SER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "multidimensional_tensor_engine.h"
#include "automatic_differentiation_framework.h"
#include <stddef.h>
/*
 * SNEPPX - Sparse Expert Routing (SER)
 *
 * WHAT
 *   Sparse Expert Routing (SER).
 *
 * CONCEPT
 *   SER model configuration, gating, and routing API declarations.
 *
 * ROLE
 *   Declares the SER module API used by the system architecture and the SER core implementation.
 *
 * REFERENCES
 *   None (internal module).
 */



typedef enum {
    SNEPPX_ACT_RELU,
    SNEPPX_ACT_GELU,
    SNEPPX_ACT_SWISH
} SNEPPXActivation;

typedef enum {
    SNEPPX_TOPK_GREEDY,
    SNEPPX_TOPK_NOISY
} SNEPPXTopKMethod;

typedef struct {
    size_t num_experts;
    size_t num_active;
    size_t input_dim;
    size_t expert_dim;
    size_t output_dim;
    SNEPPXTopKMethod top_k_method;
    float load_balance_coef;
    float dropout_rate;
    int use_mlp_gater;
    size_t gater_hidden_dim;
} SNEPPXSERConfig;

typedef struct {
    SNEPPXTensor* w1;
    SNEPPXTensor* w2;
    SNEPPXTensor* b1;
    SNEPPXTensor* b2;
    SNEPPXActivation activation;
} SNEPPXExpert;

typedef struct {
    SNEPPXExpert** experts;
    SNEPPXTensor* router;
    SNEPPXTensor* router_bias;
    SNEPPXTensor* gater_w1;
    SNEPPXTensor* gater_b1;
    SNEPPXTensor* gater_w2;
    SNEPPXTensor* gater_b2;
    SNEPPXSERConfig config;
    size_t expert_capacity;
} SNEPPXSERLayer;

typedef struct {
    SNEPPXSERLayer** layers;
    size_t num_layers;
    SNEPPXSERConfig config;
} SNEPPXSERModel;

/**
 * @brief Perform Ser Config Default.
 *
 * @return The result value, or 0 on error.
 */
SNEPPXSERConfig SNEPPX_ser_config_default(void);
/**
 * @brief Create Expert.
 *
 * @param config [in] Config value.
 * @param seed [in] Seed value.
 * @param activation [in] Activation value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXExpert* SNEPPX_expert_create(const SNEPPXSERConfig* config, unsigned int seed, SNEPPXActivation activation);
/**
 * @brief Destroy Expert.
 *
 * @param expert [out] Expert value.
 */
void SNEPPX_expert_destroy(SNEPPXExpert* expert);
/**
 * @brief Create Ser Layer.
 *
 * @param config [in] Config value.
 * @param seed [in] Seed value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXSERLayer* SNEPPX_ser_layer_create(const SNEPPXSERConfig* config, unsigned int seed);
/**
 * @brief Destroy Ser Layer.
 *
 * @param layer [out] Layer value.
 */
void SNEPPX_ser_layer_destroy(SNEPPXSERLayer* layer);
/**
 * @brief Create Ser Model.
 *
 * @param config [in] Config value.
 * @param seed [in] Seed value.
 * @param num_layers [in] Num Layers value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXSERModel* SNEPPX_ser_model_create(const SNEPPXSERConfig* config, unsigned int seed, size_t num_layers);
/**
 * @brief Destroy Ser Model.
 *
 * @param model [out] Model value.
 */
void SNEPPX_ser_model_destroy(SNEPPXSERModel* model);
/**
 * @brief Perform Ser Route.
 *
 * @param layer [out] Layer value.
 * @param input [in] Input value.
 * @param gate_weights [out] Gate Weights value.
 * @param expert_indices [out] Expert Indices value.
 */
void SNEPPX_ser_route(SNEPPXSERLayer* layer, const SNEPPXTensor* input, SNEPPXTensor** gate_weights, int** expert_indices);
/**
 * @brief Perform Ser Route Mlp Gated.
 *
 * @param layer [out] Layer value.
 * @param input [in] Input value.
 * @param gate_weights [out] Gate Weights value.
 * @param expert_indices [out] Expert Indices value.
 */
void SNEPPX_ser_route_mlp_gated(SNEPPXSERLayer* layer, const SNEPPXTensor* input, SNEPPXTensor** gate_weights, int** expert_indices);
/**
 * @brief Run the forward pass for Ser Gate.
 *
 * @param layer [in] Layer value.
 * @param input [in] Input value.
 * @param gate_weights [out] Gate Weights value.
 * @param expert_indices [out] Expert Indices value.
 * @param gate_logits [out] Gate Logits value.
 * @param temperature [in] Temperature value.
 */
void SNEPPX_ser_gate_forward(const SNEPPXSERLayer* layer, const SNEPPXTensor* input,
                           SNEPPXTensor** gate_weights, int** expert_indices,
                           SNEPPXTensor** gate_logits, float temperature);
/**
 * @brief Run the forward pass for Ser Expert.
 *
 * @param expert [in] Expert value.
 * @param input [in] Input value.
 * @param output [out] Output value.
 */
void SNEPPX_ser_expert_forward(const SNEPPXExpert* expert, const SNEPPXTensor* input, SNEPPXTensor* output);
/**
 * @brief Run the forward pass for Ser.
 *
 * @param layer [out] Layer value.
 * @param input [in] Input value.
 * @param output [out] Output value.
 */
void SNEPPX_ser_forward(SNEPPXSERLayer* layer, const SNEPPXTensor* input, SNEPPXTensor** output);
/**
 * @brief Perform Ser Load Balance Loss.
 *
 * @param gate_weights [in] Gate Weights value.
 * @param expert_indices [in] Expert Indices value.
 * @param num_tokens [in] Num Tokens value.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_ser_load_balance_loss(const SNEPPXTensor* gate_weights, const int* expert_indices, size_t num_tokens);
/**
 * @brief Perform Ser Z Loss.
 *
 * @param gate_logits [in] Gate Logits value.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_ser_z_loss(const SNEPPXTensor* gate_logits);
/**
 * @brief Perform Ser Aux Loss.
 *
 * @param gate_weights [in] Gate Weights value.
 * @param expert_indices [in] Expert Indices value.
 * @param gate_logits [in] Gate Logits value.
 * @param num_tokens [in] Num Tokens value.
 * @param load_balance_coef [in] Load Balance Coef value.
 * @param z_loss_coef [in] Z Loss Coef value.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_ser_aux_loss(const SNEPPXTensor* gate_weights, const int* expert_indices,
                        const SNEPPXTensor* gate_logits, size_t num_tokens,
                        float load_balance_coef, float z_loss_coef);
/**
 * @brief Perform Ser Expert Capacity Balance.
 *
 * @param gate_weights [out] Gate Weights value.
 * @param expert_indices [out] Expert Indices value.
 * @param num_tokens [in] Num Tokens value.
 * @param num_active [in] Num Active value.
 * @param expert_capacity [in] Expert Capacity value.
 */
void SNEPPX_ser_expert_capacity_balance(SNEPPXTensor* gate_weights, int* expert_indices,
                                      size_t num_tokens, size_t num_active,
                                      size_t expert_capacity);
/**
 * @brief Perform Ser Importance Loss.
 *
 * @param gate_weights [in] Gate Weights value.
 * @param expert_indices [in] Expert Indices value.
 * @param num_experts [in] Num Experts value.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_ser_importance_loss(const SNEPPXTensor* gate_weights, const int* expert_indices, int num_experts);
/**
 * @brief Perform Ser Aux Load Balance.
 *
 * @param gate_weights [in] Gate Weights value.
 * @param expert_indices [in] Expert Indices value.
 * @param num_experts [in] Num Experts value.
 * @param num_tokens [in] Num Tokens value.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_ser_aux_load_balance(const SNEPPXTensor* gate_weights, const int* expert_indices,
                                 int num_experts, size_t num_tokens);
/**
 * @brief Perform Ser Get Params.
 *
 * @param model [in] Model value.
 * @param out_params [out] Out Params value.
 * @param max_params [in] Max Params value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_ser_get_params(const SNEPPXSERModel* model, SNEPPXTensor** out_params, size_t max_params);
/**
 * @brief Perform Ser Build Train Graph.
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
int SNEPPX_ser_build_train_graph(SNEPPXSERModel* model, SNEPPXTape* tape,
                               SNEPPXVariable* input_var,
                               SNEPPXVariable** weight_vars, size_t num_weights,
                               SNEPPXVariable** output_var);


#ifdef __cplusplus
}
#endif
#endif /* SNEPPX_SER_H */
