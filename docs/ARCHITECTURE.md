# SNEPPX-Algo Architecture

## Documentation Standards

This project uses a four-layer commenting standard defined in [COMMENTING.md](COMMENTING.md). All source files must have Layer 1 file headers and public `SNEPPX_*` functions must have Layer 4 Doxygen docs. See [STYLE_GUIDE.md](STYLE_GUIDE.md) for code formatting rules.

## Overview

SNEPPX-Algo implements a composite AI algorithm with cryptographic integrity. The system processes inputs through five sequential algorithm components, each wrapped in ten security layers.

```
                     ┌──────────────────────────────────────┐
                     |         Security Layer (S0-S9)        |
                     |  Crypto · Secure Mem · Obfuscation   |
                     |  Monitor · Network · AI Sanitizer    |
                     |  Key Vault · Updates · Formal · Pen. |
                     └──────────────────────────────────────┘
                                     |
                     ┌───────────────▼──────────────────────┐
                     |         Algorithm Pipeline            |
                     |  HSS -> SER -> ARC -> NPE -> FM       |
                     |  (SSM) (MoE) (Guard) (VM)  (Fed Mem) |
                     └──────────────────────────────────────┘
                                     |
                      ┌───────────────▼──────────────────────┐
                      |         Integrity Layer               |
                      |  ZK Proofs (opt-in) · Formal Safety   |
                      |  On-Device Attestation                |
                      └──────────────────────────────────────┘
```

## Algorithm Pipeline

### HSS — Hierarchical State Space

**Layer**: `algorithms/hss/core/`
**Files**: `layer.c`, `forward_train.c`, `cuda/hss_cuda.cu`

HSS is a multi-layer state space model that processes sequences by maintaining a hidden state updated per timestep.

#### Mathematical Formulation

For each layer l at timestep k:

```
x_k  in R^{d_model}     (input at timestep k)
h^l_k                   (hidden state for layer l, timestep k)
y^l_k                   (output for layer l, timestep k)

Transition matrices:
A^l  in R^{d_state x d_state}
B^l  in R^{d_state x d_model}
C^l  in R^{d_model x d_state}
D^l  in R^{d_model x d_model}     (step size)

Discretization (zero-order hold):
A_bar = exp(D * A)
B_bar = (D * A)^{-1} * (exp(D * A) - I) * D * B

State update:
h^l_k = A_bar * h^l_{k-1} + B_bar * x^l_k

Output:
y^l_k = C^l * h^l_k

Merge multi-layer outputs:
y_k = Sum_l W^l * y^l_k + x_k     (residual connection)
```

The sequential scan processes timesteps sequentially. The Blelloch tree-based parallel scan over state dimension is enabled by default on CPU (set `use_parallel_scan=0` in config to disable). A CUDA parallel scan is pending.

#### Code Structure

```c
typedef struct {
    int32_t d_model;         // Input/output dimension
    int32_t d_state;         // Hidden state dimension (default: d_model/4)
    int32_t num_layers;      // Number of HSS layers
    int32_t use_parallel_scan; // 1 = Blelloch parallel scan (default)
    float dt_min;            // Minimum step size (default: 0.001)
    float dt_max;            // Maximum step size (default: 0.1)
    int32_t seed;            // Random seed for init
} SNEPPXHSSConfig;

SNEPPXHSSConfig SNEPPX_hss_config_default(void);
SNEPPXHSSLayer* SNEPPX_hss_layer_create(const SNEPPXHSSConfig* config, unsigned int seed);
void SNEPPX_hss_layer_destroy(SNEPPXHSSLayer* layer);
SNEPPXHSSModel* SNEPPX_hss_model_create(const SNEPPXHSSConfig* config, unsigned int seed);
void SNEPPX_hss_model_destroy(SNEPPXHSSModel* model);
int SNEPPX_hss_forward(SNEPPXHSSModel* model, const SNEPPXTensor* input, SNEPPXTensor** output);
int SNEPPX_hss_build_train_graph(SNEPPXHSSModel* model, SNEPPXTape* tape, SNEPPXVariable* input, SNEPPXVariable** params, int n_params, SNEPPXVariable** output);
```

### SER — Sparse Expert Routing

**Layer**: `algorithms/ser/core/`
**Files**: `layer.c`, `gater.c`, `forward_train.c`, `cuda/ser_cuda.cu`

SER implements a Mixture-of-Experts layer with top-k routing and optional learned MLP gating.

#### Mathematical Formulation

```
Input:  x in R^{batch x seq x d_model}
Experts: E_1, ..., E_n where E_i : R^{d_model} -> R^{d_model}
Router:  r(x) = Softmax(W_r * x + b_r)     W_r in R^{n x d_model}

Top-k selection (k << n):
indices = topk(r(x), k)
weights = Softmax(r(x)[indices])

Output:
y = Sum_{i=1}^{k} weights_i * E_{indices_i}(x)
```

**Learned MLP Gater** (opt-in, `use_mlp_gater=1`): replaces the linear router with a 2-layer MLP:
```
r(x) = Softmax(W2 * ReLU(W1 * x + b1) + b2)
```

Load-balancing loss:
```
L_balance = alpha * n * Sum_{i=1}^{n} f_i * P_i

where:
f_i = fraction of tokens routed to expert i
P_i = average router probability for expert i
alpha = balancing coefficient (default: 0.01)
```

#### Code Structure

```c
typedef struct {
    int32_t d_model;          // Input/output dimension
    int32_t d_expert;         // Expert hidden dimension (default: 4*d_model)
    int32_t num_experts;      // Total number of experts (default: 8)
    int32_t top_k;            // Number of active experts per token (default: 2)
    int32_t use_mlp_gater;    // 1 = MLP gater, 0 = linear router
    int32_t gater_hidden_dim;  // MLP gater hidden size (default: 64)
    int32_t seed;             // Random seed for expert init
} SNEPPXSERConfig;

SNEPPXSERConfig SNEPPX_ser_config_default(void);
SNEPPXSERModel* SNEPPX_ser_model_create(SNEPPXSERConfig* config, unsigned int seed);
int SNEPPX_ser_forward(SNEPPXSERLayer* layer, const SNEPPXTensor* input, SNEPPXTensor** output);
float SNEPPX_ser_load_balance_loss(const SNEPPXSERLayer* layer, const SNEPPXTensor* routing_weights);
int SNEPPX_ser_route_mlp_gated(SNEPPXSERLayer* layer, const SNEPPXTensor* input, SNEPPXTensor** routing_weights);
void SNEPPX_ser_model_destroy(SNEPPXSERModel* model);
```

### ARC — Adversarial Robustness Core

**Layer**: `algorithms/arc/core/`
**Files**: `layer.c`, `forward_train.c`, `cuda/arc_cuda.cu`

ARC provides a three-layer defense-with-verification pipeline plus adversarial training.

#### Defense Layers

1. **Input Guard**: z-score anomaly detection across the input.

```c
// For each feature dimension j:
z_j = (x_j - mu_j) / sigma_j
// Flag if |z_j| > threshold (default: 3.0)
```

2. **Gradient Obfuscation**: Noise injection + gradient clipping.

```c
// During training:
g' = g + N(0, sigma^2 * |g|)     // noise proportional to gradient magnitude
g'' = clamp(g', -gamma, gamma)     // gradient clipping threshold gamma
```

3. **Output Verifier**: Consistency check with cosine similarity + temporal smoothing.

```c
// Cosine similarity:
sim = y_hat * y_prev / (|y_hat| * |y_prev|)
// Detect if sim drops below threshold tau_adv (default: 0.5)
// Temporal smoothing:
y_smooth = beta * y_prev + (1 - beta) * y_hat     // with beta in [0.7, 0.99]
```

#### Attack Simulation

ARC includes simulation for:
- **FGSM** (Fast Gradient Sign Method): `x_adv = x + eps * sign(grad_x L(x, y))`
- **PGD** (Projected Gradient Descent): Multi-step FGSM with projection
- **C&W** (Carlini-Wagner): Optimization-based attack: `min ||delta|| + c * f(x + delta)`

#### Adversarial Training Graph

`SNEPPX_arc_build_adversarial_train_graph` constructs a shared-weight dual forward graph:
1. Clean forward on original input (requires_grad=1)
2. FGSM perturbation generated from clean loss gradients
3. Adversarial forward on perturbed input (adversarial variable requires_grad=0)
4. Combined loss = clean_loss + lambda * adv_loss

#### Code Structure

```c
typedef struct {
    float z_threshold;        // Z-score anomaly threshold (default: 3.0)
    float noise_std;          // Gradient noise std deviation (default: 0.01)
    float grad_clip;          // Gradient clipping threshold (default: 1.0)
    float sim_threshold;      // Cosine similarity threshold (default: 0.5)
    float temporal_beta;      // Temporal smoothing coefficient (default: 0.9)
    float attack_epsilon;     // FGSM perturbation size (default: 0.1)
    int32_t seed;             // Random seed
} SNEPPXARCConfig;

SNEPPXARCConfig SNEPPX_arc_config_default(void);
SNEPPXARCModel* SNEPPX_arc_model_create(SNEPPXARCConfig* config);
void SNEPPX_arc_forward(SNEPPXARCModel* model, SNEPPXTensor* input,
                        SNEPPXTensor** output, SNEPPXTensor** anomaly_scores);
void SNEPPX_arc_attack_fgsm(SNEPPXARCModel* model, SNEPPXTensor* input,
                            SNEPPXTensor* gradient, float epsilon);
int SNEPPX_arc_build_adversarial_train_graph(SNEPPXARCModel* model, SNEPPXTape* tape,
    SNEPPXVariable* input, SNEPPXVariable** weights, int n_weights,
    SNEPPXVariable** clean_out, SNEPPXVariable** adv_out);
void SNEPPX_arc_model_destroy(SNEPPXARCModel* model);
```

### NPE — Neural Program Executor

**Layer**: `algorithms/npe/core/`
**Files**: `vm.c`, `compiler.c`, `jit_pipeline.c`, `cuda/npe_cuda.cu`

NPE is a 16-register virtual machine with 15+ opcodes for executing neural network operations. Includes a JIT pipeline with multiple optimization passes.

#### Virtual Machine

```
Registers: R0-R15, each holds an SNEPPXTensor*
Program Counter: PC
Halt Flag: H
JIT Profile: records executed opcodes for hot-path optimization

Opcodes:
  0x00  NOP                No operation
  0x01  LOAD rd, imm       Load immediate into register
  0x02  MOV rd, rs         Move register
  0x03  MATMUL rd, rs1, rs2    Matrix multiply
  0x04  ADD  rd, rs1, rs2      Element-wise add
  0x05  RELU rd, rs            ReLU activation
  0x06  SOFTMAX rd, rs         Softmax over last dim
  0x07  LAYERNORM rd, rs       Layer normalization
  0x08  SIGMOID rd, rs         Sigmoid activation
  0x09  SUM rd, rs             Sum all elements
  0x0A  MEAN rd, rs            Mean of elements
  0x0D  ATTENTION rd, q, k, v  Scaled dot-product attention
  0x10  COMPUTE rd, rs         Compute opcode (extended)
  0x11  HALT                   Stop execution
  0x13  TANH rd, rs            Tanh activation

Fused opcodes (JIT output):
  0x80000000  MATMUL_RELU    MATMUL + RELU fused
  0x40000000  MATMUL_ADD     MATMUL + ADD fused (bias in low byte)
  0x20000000  MATMUL_ADD_RELU  MATMUL + ADD + RELU fused
  0x80000001  ADD_RELU       ADD + RELU fused
```

#### JIT Pipeline

The JIT pipeline (`jit_pipeline.c`) composes four optimization passes:

```
Input Program
    |
    v
[DCE] — Dead Code Elimination: removes NOPs and unused writes
    |
    v
[Constant Folding] — evaluates constant expressions at compile time
    |
    v
[Fusion] — merges adjacent compatible ops:
     MATMUL + ADD + RELU -> MATMUL_ADD_RELU (bit 0x20000000)
     MATMUL + RELU       -> MATMUL_RELU     (bit 0x80000000)
     MATMUL + ADD        -> MATMUL_ADD      (bit 0x40000000, bias reg in low byte)
     ADD + RELU          -> ADD_RELU        (bit 0x80000001)
    |
    v
[Compile] — final program assembly
    |
    v
Optimized Program
```

**Auto-JIT Mode**: The VM records executed opcodes in a profile during `vm_step`. When the hot threshold is reached (configurable, default 100 executions), `vm_run` automatically triggers `SNEPPX_npe_vm_optimize` before continuing.

#### Code Structure

```c
typedef struct {
    int32_t num_registers;     // Number of registers (default: 16)
    int32_t max_program_size;  // Maximum program length (default: 1024)
    int32_t jit_hot_threshold; // Auto-JIT trigger count (default: 100)
} SNEPPXNPEConfig;

SNEPPXNPEConfig SNEPPX_npe_config_default(void);
SNEPPXNPEModel* SNEPPX_npe_model_create(SNEPPXNPEConfig* config);
int32_t SNEPPX_npe_load_program(SNEPPXNPEModel* model, uint8_t* program, int32_t size);
void SNEPPX_npe_execute(SNEPPXNPEModel* model, SNEPPXTensor** outputs, int32_t num_outputs);
int32_t SNEPPX_npe_verify(SNEPPXNPEModel* model);

// JIT API
SNEPPXNPEProgram* SNEPPX_npe_jit_pipeline_compose(SNEPPXNPEProgram* input);
int SNEPPX_npe_vm_optimize(SNEPPXNPEModel* model);
int SNEPPX_npe_jit_fuse(SNEPPXNPEProgram* prog);
int SNEPPX_npe_compile_mlp(size_t input_dim, size_t hidden_dim);
int SNEPPX_npe_compile_attention(size_t dim);

void SNEPPX_npe_model_destroy(SNEPPXNPEModel* model);
```

### FM — Federated Memory

**Layer**: `algorithms/fm/core/`
**Files**: `fm.c`, `sync.c`, `cuda/fm_cuda.cu`

FM provides distributed memory with per-node memory banks, trust-weighted synchronization, and NCCL sync bridge.

#### Memory Bank

Each node maintains a fixed-size memory bank with:
- **Euclidean similarity** for nearest-neighbor retrieval
- **LRU eviction** when the bank is full
- **Learnable read/write** operations

#### Synchronization Protocol

```
Trust-weighted all-reduce:
theta_global = Sum_i trust_score_i * theta_i / Sum_j trust_score_j

Differential privacy noise:
theta_shared = theta_global + Lap(0, epsilon)     // Laplace noise
epsilon = noise_scale / sensitivity

Gradient compression:
Select top-k gradients by magnitude (random sampling fallback)
```

#### NCCL Sync Bridge

`SNEPPX_fm_sync_nccl` integrates with NCCL using a callback pattern:

```c
typedef int (*SNEPPXFMSyncCallback)(const float* data, size_t size, void* context);

int SNEPPX_fm_sync_nccl(SNEPPXFMController* ctrl,
    SNEPPXFMSyncCallback callback, void* context);
```

The callback receives the aggregated gradient data and a user-provided context pointer. The bridge is NCCL-header-free — no compile-time dependency on NCCL.

#### Code Structure

```c
typedef struct {
    int32_t memory_size;        // Number of memory slots per node (default: 1024)
    int32_t d_model;            // Memory vector dimension
    float lr;                   // Learning rate for memory writes
    float diff_privacy_noise;   // Differential privacy noise scale
    float compression_ratio;    // Gradient compression ratio (default: 0.01)
} SNEPPXFMConfig;

SNEPPXFMConfig SNEPPX_fm_config_default(void);
SNEPPXFMModel* SNEPPX_fm_model_create(SNEPPXFMConfig* config, int32_t node_id, int32_t seed);
void SNEPPX_fm_write(SNEPPXFMModel* model, SNEPPXTensor* key, SNEPPXTensor* value);
void SNEPPX_fm_read(SNEPPXFMModel* model, SNEPPXTensor* query, SNEPPXTensor** output);
int SNEPPX_fm_sync_all_reduce(SNEPPXFMController* ctrl, SNEPPXFMNode** nodes, size_t num_nodes, float privacy_epsilon);
int SNEPPX_fm_sync_nccl(SNEPPXFMController* ctrl, SNEPPXFMSyncCallback callback, void* context);
int SNEPPX_fm_forward(SNEPPXFMController* ctrl, size_t node_id, const SNEPPXTensor* input, SNEPPXTensor** output);
void SNEPPX_fm_model_destroy(SNEPPXFMModel* model);
```

## Trainer + CUDA Accelerated Optimization

**Files**: `kernel/train/trainer.c`, `kernel/train/trainer_cuda.c`, `kernel/train/trainer_cuda.h`

The trainer provides a CPU training loop with an optional CUDA-accelerated optimizer path.

### Architecture

```
Training Step:
  1. Forward pass (HSS -> SER -> ARC -> NPE -> FM)  [CPU]
  2. Compute loss (MSE + load balance + ARC regularization)  [CPU]
  3. Backward pass (autodiff tape)  [CPU]
  4. [if use_cuda_optimizer] Transfer params/grads to GPU async  [CUDA]
  5. Optimizer step (SGD/AdamW)  [CPU or GPU]
  6. [if use_cuda_optimizer] Transfer updated params back to CPU  [CUDA]
  7. Secure memory wipe of intermediate tensors  [CPU]
```

### CUDA Optimizer Bridge

```c
int SNEPPX_trainer_cuda_available(void);       // Runtime device detection
int SNEPPX_trainer_cuda_init(size_t mem_pool_size);  // Allocates device pool + stream + cublas handle
void SNEPPX_trainer_cuda_shutdown(void);       // Frees all device resources
int SNEPPX_trainer_cuda_transfer(void* dst, const void* src, size_t bytes);  // Async H2D
int SNEPPX_trainer_cuda_to_host(void* dst, const void* src, size_t bytes);   // Async D2H
int SNEPPX_trainer_cuda_optimizer_step(float* params, const float* grads, size_t n,
    float lr, float weight_decay);             // cuBLAS SGD or AdamW

// Config flag:
typedef struct {
    // ... other fields ...
    int32_t use_cuda_optimizer;    // 1 = CUDA optimizer path (default: 0)
} SNEPPXTrainConfig;
```

When `SNEPPX_HAS_CUDA` is not defined, all `trainer_cuda` functions return clean stubs (error codes / 0), and the trainer falls back to CPU optimization.

## Security Layers

### S0 — Cryptographic Core

**Status: Complete**
**Files**: `security/crypto/c/` and `include/neural_core/security/`

Production-grade cryptographic primitives:

| Primitive | Implementation | Status |
|-----------|---------------|--------|
| Ed25519 | RFC 8032 | Verified, 304/306 vectors pass |
| X25519 | RFC 7748 | Full DH exchange |
| ChaCha20-Poly1305 | RFC 8439 | AEAD encrypt/decrypt |
| SHA-3 | FIPS 202 | 224/256/384/512 |
| SHA-256 | FIPS 180-4 | General-purpose hashing |
| BLAKE3 | Reference impl | Fast hashing |
| Argon2id | RFC 9106 | KDF with timing defense |
| Secure Random | OS CPRG | Entropy source abstraction |
| Kyber-512/768/1024 | FIPS 203 (ML-KEM) | PQ key encapsulation |
| Dilithium-2/3/5 | FIPS 204 (ML-DSA) | PQ digital signatures |
| SPHINCS+-128/192/256 | FIPS 205 (SLH-DSA) | Stateless PQ signatures |

### S1 — Secure Memory

**Status: Complete**
**Files**: `security/memory/` and `include/neural_core/security/`

| Feature | Description |
|---------|-------------|
| Guard Pages | RW pages with PROT_NONE adjacent to detect overflow |
| Canaries | Stack-based overflow detection |
| ASLR | Heap entropy via Windows VirtualAlloc / Linux mmap randomization |
| Locked Memory | mlock/VirtualLock to prevent swapping to disk |
| Secure Wipe | memset with compiler barrier to prevent removal |
| Constant-Time Ops | memcmp variant for secret comparison |
| Memory Leak Detector | Tracks allocations and reports unfreed blocks on shutdown |

### S2 — Obfuscation Engine

**Status: Complete**
**Files**: `security/cpp/` and `security/obfuscation/`

Control flow flattening, string encryption, instruction substitution, opaque predicates, code virtualization, anti-debug, binary substitution, junk code insertion, constant unfolding, IAT protection, SEH/VEH obfuscation, TLS callback obfuscation, anti-dump, multi-VM diversity, instruction scheduling randomization.

### S3 — Behavioral Monitor

**Status: Complete**
**Files**: `security/monitor/`

Integrity monitoring (CRC32), container breakout detection, frequency analysis, timing analysis, anomaly detection (statistical baseline comparison).

### S4 — Network Security

**Status: Complete**
**Files**: `security/network/`

DDoS mitigation (SYN flood detection, rate limiting, connection tracking, IP blacklisting), transport security (TLS-compatible handshake padding, traffic analysis resistance), identity management (certificate pinning, peer fingerprint verification).

### S5 — AI Sanitizer

**Status: Complete**
**Files**: `security/ai/`

Prompt injection detection, differential privacy (Laplace mechanism), data poisoning defense (gradient outlier detection), RLHF safety (reward model validation, preference alignment checking, harmful output filtering).

### S6 — Security UI / Key Vault

**Status: Complete**
**Files**: `security/ui/`

Audit logging (structured JSON), key vault (in-memory encrypted store with PIN protection).

### S7 — Secure Updates

**Status: Complete**
**Files**: `security/updates/`

Container security (OCI layer verification), signed update bundles (Ed25519), rollback protection (monotonic version counter), staged rollout.

### S8 — Formal Verification

**Status: Complete**
**Files**: `security/formal/`

Model checking (bounded state-space exploration), invariant verification (pre/post-condition checking), symbolic execution (NPE bytecode), container breakout detection (state-machine rules).

### S9 — Penetration Testing

**Status: Complete**
**Files**: `security/pentest/`

Network fuzzer (protocol-aware, mutation strategies), self-audit (internal consistency checks), security report generation, CTF utilities.

## Foundation Components

### Tensor Core

**Files**: `kernel/tensor/` and `include/neural_core/kernel/`

Multi-dimensional array with row-major layout. Supports 12 data types.

```c
typedef enum {
    SNEPPX_BOOL,       // sizeof(int8_t)
    SNEPPX_INT8,       // sizeof(int8_t)
    SNEPPX_INT16,      // sizeof(int16_t)
    SNEPPX_INT32,      // sizeof(int32_t)
    SNEPPX_INT64,      // sizeof(int64_t)
    SNEPPX_UINT8,      // sizeof(uint8_t)
    SNEPPX_FLOAT16,    // sizeof(uint16_t) — IEEE 754 half
    SNEPPX_BFLOAT16,   // sizeof(uint16_t) — Brain float
    SNEPPX_FLOAT32,    // sizeof(float) — default
    SNEPPX_FLOAT64,    // sizeof(double)
    SNEPPX_FLOAT8,     // sizeof(uint8_t) — FP8 E4M3/E5M2
    SNEPPX_COMPLEX64,  // sizeof(float) * 2
    SNEPPX_COMPLEX128, // sizeof(double) * 2
} SNEPPXDType;
```

**Supported ops (50+):**
- Creation: zeros, ones, full, randn, arange, identity, from_buffer, copy
- Shape: reshape, transpose, permute, expand, squeeze, unsqueeze, slice, pad
- Arithmetic: add, sub, mul, div, pow, sqrt, neg, abs
- Matmul: matmul, batch_matmul
- Reduction: sum, mean, var, max, min, argmax, argmin
- Comparison: eq, ne, lt, gt, le, ge
- Unary: exp, log, sin, cos, tan, sigmoid, relu, softmax, gelu, layer_norm
- Format I/O: safetensors, numpy (.npy/.npz), PyTorch (.pth), ONNX

### Autodiff

**Files**: `kernel/autodiff/` and `include/neural_core/kernel/`

Tape-based reverse-mode automatic differentiation. Builds a computation graph of operations, then traverses it in reverse topological order to compute gradients.

```c
SNEPPXTape* SNEPPX_tape_create(void);
void SNEPPX_tape_backward(SNEPPXTape* tape, SNEPPXVariable* loss);
void SNEPPX_tape_zero_grad(SNEPPXTape* tape);
float SNEPPX_tape_global_norm(SNEPPXTape* tape);
void SNEPPX_tape_clip_grad_norm(SNEPPXTape* tape, float max_norm);
```

### Optimizer

**Files**: `kernel/optimizer/` and `include/neural_core/kernel/`

CPU optimizers: SGD (momentum, weight decay), AdamW, RMSprop, Adagrad. LR schedulers: StepLR, Exponential, Cosine, ReduceOnPlateau.

### Memory

**Files**: `kernel/memory/` and `include/neural_core/kernel/`

Aligned allocation on aligned boundaries. Free list with best-fit for sizes > 4KB. Optional guard pages on request.

### Thread Pool

**Files**: `kernel/thread_pool/` and `include/neural_core/kernel/`

Single-threaded fallback. Interface supports work queue and tasks. Real parallelism deferred.

## Data Flow

```
Input (SNEPPXTensor)
    |
    v
+---------------------+
|  HSS Forward         |
|  (parallel scan)     |
|  Output: (B, S, D)   |
+---------+-----------+
          |
          v
+---------------------+
|  SER Forward         |
|  (top-k routing +    |
|   MLP gater)         |
|  Output: (B, S, D)   |
+---------+-----------+
          |
          v
+---------------------+
|  ARC Forward         |
|  (guard + filter)    |
|  Output: (B, S, D)   |
|  + anomaly_scores    |
+---------+-----------+
          |
          v
+---------------------+
|  NPE Execute         |
|  (VM program,        |
|   JIT-optimized)     |
|  Output: (B, S, D)   |
+---------+-----------+
          |
          v
+---------------------+
|  FM Read             |
|  (memory lookup,     |
|   NCCL sync)         |
|  Output: (B, S, D)   |
+---------+-----------+
          |
          v
      Output (SNEPPXTensor)
```

## Training Flow

```
Forward pass (HSS -> SER -> ARC -> NPE -> FM)
    |
    v
Compute loss (MSE + load balance + ARC regularization)
    |
    v
Backward pass (autodiff tape-based reverse-topological gradient computation)
    |
    v
[if use_cuda_optimizer] Transfer params/grads to GPU (async H2D)
    |
    v
Optimizer step (SGD / AdamW on CPU, or cuBLAS SGD / AdamW on GPU)
    |
    v
[if use_cuda_optimizer] Transfer params back to CPU (async D2H)
    |
    v
Secure memory wipe of intermediate tensors
```

## Integrity Layer (future / opt-in)

Zero-knowledge proofs of correct inference (`SNEPPX_BUILD_ZK` enables a Schnorr proof over Curve25519 reference backend). Formal safety verification of constraint compliance. On-device attestation for distributed verification.
