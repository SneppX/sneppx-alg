# Code Walkthrough — NPE (Neural Programming Engine)

NPE is the fourth stage of the SNEPPX-Alg pipeline. Unlike the other stages
(which are plain tensor layers), NPE executes *programs* on a small virtual
machine: 16 registers, 32 opcodes, a program counter, and a flat tensor
memory. Neural subroutines (attention, MLP) are *compiled* into this bytecode
and executed by the VM.

- Public header: `include/neural_core/architecture/neural_programming_engine.h`
- Implementation: `algorithms/npe/core/`
- CUDA extensions: `algorithms/npe/cuda/`

## The instruction set

`SNEPPXNPEInstruction` is a fixed-size struct: opcode, destination register,
two source registers, an immediate, and shape fields.

```c
typedef struct {
    int opcode;     // SNEPPXNPEOpCode
    int dest_reg;
    int src_reg_a;
    int src_reg_b;
    int immediate;
    int shape_a[2]; // tensor shape hint
    int shape_b[2];
} SNEPPXNPEInstruction;
```

The 32 opcodes (`SNEPPXNPEOpCode`) cover:

- **Memory**: `LOAD`, `STORE`
- **Arithmetic**: `ADD`, `SUB`, `MUL`, `DIV`, `NEG`, `EXP`, `LOG`, `SQRT`,
  `POW`, `SIN`, `COS`
- **Tensor/neural**: `MATMUL`, `RELU`, `SOFTMAX`, `LAYERNORM`, `ATTENTION`,
  `DROPOUT`, `CONV2D`, `POOL2D`, `BATCHNORM`, `EMBEDDING`
- **Activations/loss**: `TANH`, `SIGMOID`, `GELU`, `SILU`, `CROSSENTROPY`,
  `MSE`, `CONCAT`, `SPLIT`
- **Control**: `BRANCH`, `HALT`, `NOP`

## Program and VM structure

```c
SNEPPXNPEProgram* prog = SNEPPX_npe_program_create(64);   // capacity
SNEPPX_npe_program_append(prog, inst);                    // build the program
int rc = SNEPPX_npe_verify_program(prog, &err, &err_len); // static validation
```

A program holds its instruction array plus the parameter tensors used by the
neural opcodes (`param_w1/b1/w2/b2`). The VM adds an execution trace and a JIT
profile:

```c
SNEPPXNPEConfig cfg = SNEPPX_npe_config_default();
SNEPPXNPEVM* vm = SNEPPX_npe_vm_create(&cfg);
SNEPPX_npe_vm_load(vm, prog);
int rc = SNEPPX_npe_vm_run(vm, input, &output);   // run to HALT
```

`SNEPPX_npe_vm_step` executes a single instruction (fetch → decode → execute)
so the VM can be paused and inspected between steps.

## Compiling neural ops

`algorithms/npe/core/compiler.c` lowers high-level ops to VM bytecode:

- `SNEPPX_npe_compile_attention(seq_len, dim)` emits a sequence of
  `LOAD → MATMUL → SOFTMAX → MATMUL → STORE → HALT` that performs scaled
  attention over the register file.
- `SNEPPX_npe_compile_mlp(dim, hidden_dim)` emits a two-layer MLP
  (`MATMUL → GELU/RELU → MATMUL → STORE`).

## JIT optimization

`algorithms/npe/core/jit_pipeline.c` implements a profiling JIT:

| Pass | Function | What it does |
|------|----------|--------------|
| Profiling | `SNEPPX_npe_jit_record` | records opcode frequency + latency in `SNEPPXNPEJITProfile` |
| Specialize | `SNEPPX_npe_jit_specialize` | specializes a program for fixed batch/seq/dim |
| Fuse | `SNEPPX_npe_jit_fuse` | fuses adjacent element-wise ops |
| Constant fold | `SNEPPX_npe_jit_constant_fold` | evaluates compile-time constant instructions |
| Dead code | `SNEPPX_npe_jit_dce` | removes instructions whose results are unused |
| Optimize | `SNEPPX_npe_jit_optimize` | runs the full pipeline: profile → specialize → fold → dce → fuse |
| VM entry | `SNEPPX_npe_vm_optimize` | JIT-optimizes the loaded program before execution |

## Training graph

Like the other stages, NPE programs can be differentiated:

```c
int rc = SNEPPX_npe_build_train_graph(prog, tape,
                                      input_var, weight_vars, num_weights,
                                      &output_var);
```

`SNEPPX_npe_get_params()` flattens the program's parameter tensors.

## Minimal example

```c
#include "neural_programming_engine.h"

SNEPPXNPEProgram* prog = SNEPPX_npe_compile_mlp(8, 16);

SNEPPXNPEConfig cfg = SNEPPX_npe_config_default();
SNEPPXNPEVM* vm = SNEPPX_npe_vm_create(&cfg);
SNEPPX_npe_vm_load(vm, prog);

SNEPPXTensor* input = /* build (1, 8) tensor */;
SNEPPXTensor* output = NULL;
SNEPPX_npe_vm_run(vm, input, &output);

SNEPPX_npe_vm_destroy(vm);
SNEPPX_npe_program_destroy(prog);
```

## Public API summary

- Program: `SNEPPX_npe_program_create/destroy/append`,
  `SNEPPX_npe_verify_program`.
- VM: `SNEPPX_npe_vm_create/destroy/load/run/step/optimize`.
- Compiler: `SNEPPX_npe_compile_attention`, `SNEPPX_npe_compile_mlp`.
- JIT: `SNEPPX_npe_jit_profile_create/destroy`, `SNEPPX_npe_jit_record`,
  `SNEPPX_npe_jit_compile`, `SNEPPX_npe_jit_specialize`, `SNEPPX_npe_jit_fuse`,
  `SNEPPX_npe_jit_constant_fold`, `SNEPPX_npe_jit_dce`,
  `SNEPPX_npe_jit_optimize`.
- Training: `SNEPPX_npe_get_params`, `SNEPPX_npe_build_train_graph`.
