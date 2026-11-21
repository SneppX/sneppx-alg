#ifndef SNEPPX_NPE_CUDA_CUH
#define SNEPPX_NPE_CUDA_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../../../kernel/cuda/common.cuh"

/*
 * SNEPPX - Npe Cuda
 *
 * WHAT
 *   Npe Cuda.
 *
 * CONCEPT
 *   Provides the Npe Cuda.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


// NPE (Neural Program Engine) CUDA Kernels
// - Neural VM instruction dispatch
// - Tensor program execution
// - Differentiable interpreter

#ifdef __cplusplus
extern "C" {
#endif

// NPE instruction types
typedef enum {
    SNEPPX_NPE_INPUT_ADD = 0,
    SNEPPX_NPE_INPUT_MUL = 1,
    SNEPPX_NPE_INPUT_MATMUL = 2,
    SNEPPX_NPE_INPUT_RELU = 3,
    SNEPPX_NPE_INPUT_CONV2D = 4,
    SNEPPX_NPE_INPUT_REDUCE_SUM = 5,
    SNEPPX_NPE_INPUT_RESHAPE = 6,
    SNEPPX_NPE_INPUT_BRANCH = 7,
    SNEPPX_NPE_INPUT_CALL = 8,
    SNEPPX_NPE_INPUT_RET = 9,
} SNEPPX_NPE_Opcode;

// NPE program (flat instruction list for GPU)
typedef struct {
    int* opcodes;
    int* operands;       // [num_instrs][max_operands]
    int* program_counter;
    int num_instructions;
    int max_operands;
    bool use_registers;
} SNEPPX_NPE_Program;

// Neural VM state on device
typedef struct {
    half* registers;
    half* memory;
    half* stack;
    int num_registers;
    int memory_size;
    int stack_size;
} SNEPPX_NPE_VMState;

// Execute a compiled NPE program on GPU
SNEPPX_CudaError sneppx_cuda_npe_execute(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_NPE_Program* program,
    SNEPPX_NPE_VMState* vm_state,
    const half* input,
    half* output,
    int batch_size
);

// Compile NPE program (host side, generates GPU-ready instruction format)
SNEPPX_CudaError sneppx_cuda_npe_compile(
    SNEPPX_CudaStream_t stream,
    const int* ir_opcodes,
    const int* ir_operands,
    int num_instructions,
    SNEPPX_NPE_Program** program
);

// Destroy NPE program
SNEPPX_CudaError sneppx_cuda_npe_destroy_program(
    SNEPPX_NPE_Program* program
);

#ifdef __cplusplus
}
#endif

#endif
