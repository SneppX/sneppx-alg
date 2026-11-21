#include "npe_cuda.cuh"
#include <cooperative_groups.h>

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


namespace cg = cooperative_groups;

// NPE instruction dispatch kernel
__global__ void npe_execute_kernel(
    const int* opcodes,
    const int* operands,
    half* registers,
    half* memory,
    const half* input,
    half* output,
    int num_instructions,
    int max_operands,
    int batch_size
) {
    int batch_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    // PC initialized per batch
    int pc = 0;
    
    while (pc < num_instructions) {
        int op = opcodes[pc];
        int* ops = const_cast<int*>(&operands[pc * max_operands]);
        
        // Decode operands
        int dst = ops[0];
        int src1 = ops[1];
        int src2 = ops[2];
        int imm = ops[3];
        
        // Execute instruction (tid processes one element)
        switch (op) {
            case 0: { // ADD
                float a = __half2float(registers[src1 * blockDim.x + tid]);
                float b = __half2float(registers[src2 * blockDim.x + tid]);
                registers[dst * blockDim.x + tid] = __float2half_rn(a + b);
                break;
            }
            case 1: { // MUL
                float a = __half2float(registers[src1 * blockDim.x + tid]);
                float b = __half2float(registers[src2 * blockDim.x + tid]);
                registers[dst * blockDim.x + tid] = __float2half_rn(a * b);
                break;
            }
            case 2: { // MATMUL (simplified dot product)
                float sum = 0.0f;
                // Thread 0 handles the dot product for this batch
                if (tid == 0) {
                    for (int k = 0; k < imm; k++) {
                        sum += __half2float(registers[src1 * blockDim.x + k]) *
                               __half2float(registers[src2 * blockDim.x + k]);
                    }
                    registers[dst * blockDim.x] = __float2half_rn(sum);
                }
                break;
            }
            case 3: { // RELU
                float x = __half2float(registers[src1 * blockDim.x + tid]);
                registers[dst * blockDim.x + tid] = __float2half_rn(fmaxf(0.0f, x));
                break;
            }
            case 4: { // CONV2D (placeholder)
                registers[dst * blockDim.x + tid] = registers[src1 * blockDim.x + tid];
                break;
            }
            case 5: { // REDUCE_SUM
                float val = __half2float(registers[src1 * blockDim.x + tid]);
                for (int offset = 16; offset > 0; offset >>= 1) {
                    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
                }
                if (tid == 0) registers[dst * blockDim.x] = __float2half_rn(val);
                break;
            }
            case 6: { // RESHAPE (no-op, just copy)
                registers[dst * blockDim.x + tid] = registers[src1 * blockDim.x + tid];
                break;
            }
            case 7: { // BRANCH (conditional jump)
                float cond = __half2float(registers[src1 * blockDim.x]);
                if (cond > 0.0f) {
                    pc = imm;
                    continue;
                }
                break;
            }
            case 8: { // CALL (push PC, jump)
                // Stack not implemented for GPU kernel simplicity
                break;
            }
            case 9: { // RET
                pc = num_instructions; // Halt
                continue;
            }
        }
        pc++;
    }
    
    // Write output register
    output[batch_idx * blockDim.x + tid] = registers[0 * blockDim.x + tid];
}

SNEPPX_CudaError sneppx_cuda_npe_execute(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_NPE_Program* program,
    SNEPPX_NPE_VMState* vm_state,
    const half* input,
    half* output,
    int batch_size
) {
    if (!program || !vm_state || !input || !output) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    dim3 grid(1, batch_size);
    dim3 block(256);
    
    npe_execute_kernel<<<grid, block, 0, stream>>>(
        program->opcodes, program->operands,
        vm_state->registers, vm_state->memory,
        input, output,
        program->num_instructions, program->max_operands,
        batch_size
    );
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

SNEPPX_CudaError sneppx_cuda_npe_compile(
    SNEPPX_CudaStream_t stream,
    const int* ir_opcodes,
    const int* ir_operands,
    int num_instructions,
    SNEPPX_NPE_Program** program
) {
    if (!ir_opcodes || !ir_operands || !program) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    SNEPPX_NPE_Program* p = (SNEPPX_NPE_Program*)malloc(sizeof(SNEPPX_NPE_Program));
    if (!p) return SNEPPX_CUDA_ERROR_OUT_OF_MEMORY;
    
    p->num_instructions = num_instructions;
    p->max_operands = 4;
    p->use_registers = true;
    
    cudaMallocAsync(&p->opcodes, num_instructions * sizeof(int), stream);
    cudaMallocAsync(&p->operands, num_instructions * 4 * sizeof(int), stream);
    cudaMallocAsync(&p->program_counter, sizeof(int), stream);
    
    cudaMemcpyAsync(p->opcodes, ir_opcodes, num_instructions * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(p->operands, ir_operands, num_instructions * 4 * sizeof(int), cudaMemcpyHostToDevice, stream);
    
    int zero = 0;
    cudaMemcpyAsync(p->program_counter, &zero, sizeof(int), cudaMemcpyHostToDevice, stream);
    
    *program = p;
    return SNEPPX_CUDA_SUCCESS;
}

SNEPPX_CudaError sneppx_cuda_npe_destroy_program(SNEPPX_NPE_Program* program) {
    if (!program) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    cudaFree(program->opcodes);
    cudaFree(program->operands);
    cudaFree(program->program_counter);
    free(program);
    return SNEPPX_CUDA_SUCCESS;
}
