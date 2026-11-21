#ifndef SNEPPX_NPE_H
#define SNEPPX_NPE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "multidimensional_tensor_engine.h"
#include "automatic_differentiation_framework.h"
#include <stddef.h>
/*
 * SNEPPX - Neural Programming Engine (NPE)
 *
 * WHAT
 *   Neural Programming Engine (NPE).
 *
 * CONCEPT
 *   NPE VM, compiler, and program API declarations.
 *
 * ROLE
 *   Declares the NPE module API for the 16-register neural VM with 32 opcodes.
 *
 * REFERENCES
 *   None (internal module).
 */



typedef enum {
    SNEPPX_NOP,
    SNEPPX_LOAD,
    SNEPPX_STORE,
    SNEPPX_ADD,
    SNEPPX_SUB,
    SNEPPX_MUL,
    SNEPPX_DIV,
    SNEPPX_MATMUL,
    SNEPPX_RELU,
    SNEPPX_SOFTMAX,
    SNEPPX_LAYERNORM,
    SNEPPX_ATTENTION,
    SNEPPX_BRANCH,
    SNEPPX_HALT,
    SNEPPX_NEG,
    SNEPPX_EXP,
    SNEPPX_LOG,
    SNEPPX_SQRT,
    SNEPPX_POW,
    SNEPPX_SIN,
    SNEPPX_COS,
    SNEPPX_TANH,
    SNEPPX_SIGMOID,
    SNEPPX_GELU,
    SNEPPX_SILU,
    SNEPPX_DROPOUT,
    SNEPPX_CONV2D,
    SNEPPX_POOL2D,
    SNEPPX_BATCHNORM,
    SNEPPX_EMBEDDING,
    SNEPPX_CROSSENTROPY,
    SNEPPX_MSE,
    SNEPPX_CONCAT,
    SNEPPX_SPLIT
} SNEPPXNPEOpCode;

typedef struct {
    int opcode;
    int dest_reg;
    int src_reg_a;
    int src_reg_b;
    int immediate;
    int shape_a[2];
    int shape_b[2];
} SNEPPXNPEInstruction;

typedef struct {
    SNEPPXNPEInstruction* instructions;
    size_t num_instructions;
    size_t max_instructions;
    SNEPPXTensor* registers[16];
    SNEPPXTensor* memory;
    size_t pc;
    SNEPPXTensor* param_w1;
    SNEPPXTensor* param_b1;
    SNEPPXTensor* param_w2;
    SNEPPXTensor* param_b2;
} SNEPPXNPEProgram;

typedef struct SNEPPXNPEJITProfile SNEPPXNPEJITProfile;

typedef struct {
    SNEPPXNPEProgram* program;
    SNEPPXNPEInstruction* execution_trace;
    size_t trace_length;
    size_t max_trace;
    size_t step_limit;
    SNEPPXNPEJITProfile* jit_profile;
} SNEPPXNPEVM;

typedef struct {
    size_t max_program_length;
    size_t register_count;
    size_t step_limit;
    int verification_mode;
    int trace_execution;
    int jit_enabled;
    size_t jit_hot_threshold;
} SNEPPXNPEConfig;

/**
 * @brief Perform Npe Config Default.
 *
 * @return The result value, or 0 on error.
 */
SNEPPXNPEConfig SNEPPX_npe_config_default(void);
/**
 * @brief Create Npe Program.
 *
 * @param max_instructions [in] Max Instructions value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXNPEProgram* SNEPPX_npe_program_create(size_t max_instructions);
/**
 * @brief Destroy Npe Program.
 *
 * @param prog [out] Prog value.
 */
void SNEPPX_npe_program_destroy(SNEPPXNPEProgram* prog);
/**
 * @brief Perform Npe Program Append.
 *
 * @param prog [out] Prog value.
 * @param inst [in] Inst value.
 */
void SNEPPX_npe_program_append(SNEPPXNPEProgram* prog, SNEPPXNPEInstruction inst);
/**
 * @brief Create Npe Vm.
 *
 * @param config [in] Config value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXNPEVM* SNEPPX_npe_vm_create(const SNEPPXNPEConfig* config);
/**
 * @brief Destroy Npe Vm.
 *
 * @param vm [out] Vm value.
 */
void SNEPPX_npe_vm_destroy(SNEPPXNPEVM* vm);
/**
 * @brief Load Npe Vm.
 *
 * @param vm [out] Vm value.
 * @param prog [out] Prog value.
 */
void SNEPPX_npe_vm_load(SNEPPXNPEVM* vm, SNEPPXNPEProgram* prog);
/**
 * @brief Run Npe Vm.
 *
 * @param vm [out] Vm value.
 * @param input [out] Input value.
 * @param output [out] Output value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_npe_vm_run(SNEPPXNPEVM* vm, SNEPPXTensor* input, SNEPPXTensor** output);
/**
 * @brief Perform Npe Vm Step.
 *
 * @param vm [out] Vm value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_npe_vm_step(SNEPPXNPEVM* vm);
/**
 * @brief Perform Npe Compile Attention.
 *
 * @param seq_len [in] Seq Len value.
 * @param dim [in] Dim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXNPEProgram* SNEPPX_npe_compile_attention(size_t seq_len, size_t dim);
/**
 * @brief Perform Npe Compile Mlp.
 *
 * @param dim [in] Dim value.
 * @param hidden_dim [in] Hidden Dim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXNPEProgram* SNEPPX_npe_compile_mlp(size_t dim, size_t hidden_dim);
/**
 * @brief Perform Npe Verify Program.
 *
 * @param prog [in] Prog value.
 * @param error_msg [out] Error Msg value.
 * @param error_len [out] Error Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_npe_verify_program(const SNEPPXNPEProgram* prog, char** error_msg, size_t* error_len);

// Training graph support
/**
 * @brief Perform Npe Get Params.
 *
 * @param prog [out] Prog value.
 * @param out_params [out] Out Params value.
 * @param max_params [in] Max Params value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_npe_get_params(SNEPPXNPEProgram* prog, SNEPPXTensor** out_params, size_t max_params);
/**
 * @brief Perform Npe Build Train Graph.
 *
 * @param prog [out] Prog value.
 * @param tape [out] Tape value.
 * @param input_var [out] Input Var value.
 * @param weight_vars [out] Weight Vars value.
 * @param num_weights [in] Num Weights value.
 * @param output_var [out] Output Var value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_npe_build_train_graph(SNEPPXNPEProgram* prog, SNEPPXTape* tape,
                                SNEPPXVariable* input_var,
                                SNEPPXVariable** weight_vars, size_t num_weights,
                                SNEPPXVariable** output_var);

// JIT profiling data
typedef struct SNEPPXNPEJITProfile {
    size_t op_frequency[32];
    size_t op_latency[32];
    size_t total_instructions;
    size_t hot_threshold;
    int   is_profiling;
} SNEPPXNPEJITProfile;

/**
 * @brief Create Npe Jit Profile.
 *
 * @param hot_threshold [in] Hot Threshold value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXNPEJITProfile* SNEPPX_npe_jit_profile_create(size_t hot_threshold);
/**
 * @brief Destroy Npe Jit Profile.
 *
 * @param profile [out] Profile value.
 */
void SNEPPX_npe_jit_profile_destroy(SNEPPXNPEJITProfile* profile);
/**
 * @brief Perform Npe Jit Record.
 *
 * @param profile [out] Profile value.
 * @param opcode [in] Opcode value.
 * @param latency_us [in] Latency Us value.
 */
void SNEPPX_npe_jit_record(SNEPPXNPEJITProfile* profile, int opcode, float latency_us);
/**
 * @brief Perform Npe Jit Compile.
 *
 * @param profile [out] Profile value.
 * @param original [in] Original value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXNPEProgram* SNEPPX_npe_jit_compile(SNEPPXNPEJITProfile* profile, const SNEPPXNPEProgram* original);
/**
 * @brief Perform Npe Jit Specialize.
 *
 * @param prog [in] Prog value.
 * @param batch [in] Batch value.
 * @param seq_len [in] Seq Len value.
 * @param dim [in] Dim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXNPEProgram* SNEPPX_npe_jit_specialize(const SNEPPXNPEProgram* prog, size_t batch, size_t seq_len, size_t dim);
/**
 * @brief Perform Npe Jit Fuse.
 *
 * @param prog [in] Prog value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXNPEProgram* SNEPPX_npe_jit_fuse(const SNEPPXNPEProgram* prog);
/**
 * @brief Perform Npe Jit Constant Fold.
 *
 * @param prog [in] Prog value.
 * @param memory [in] Memory value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXNPEProgram* SNEPPX_npe_jit_constant_fold(const SNEPPXNPEProgram* prog, const SNEPPXTensor* memory);
/**
 * @brief Perform Npe Jit Dce.
 *
 * @param prog [in] Prog value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXNPEProgram* SNEPPX_npe_jit_dce(const SNEPPXNPEProgram* prog);
/**
 * @brief Perform Npe Jit Optimize.
 *
 * @param profile [out] Profile value.
 * @param prog [in] Prog value.
 * @param memory [in] Memory value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXNPEProgram* SNEPPX_npe_jit_optimize(SNEPPXNPEJITProfile* profile, const SNEPPXNPEProgram* prog, const SNEPPXTensor* memory);
/**
 * @brief Perform Npe Vm Optimize.
 *
 * @param vm [out] Vm value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_npe_vm_optimize(SNEPPXNPEVM* vm);


#ifdef __cplusplus
}
#endif
#endif /* SNEPPX_NPE_H */
