# Design

## Principles

1. **Security-first**: Every layer has defensive measures. No trust boundaries
   are crossed. Security is embedded into the architecture, not bolted on.

2. **O(n log n)**: All sequence modeling operations scale as O(n log n) using
   Blelloch parallel associative scan. No quadratic attention.

3. **Verifiable**: All execution paths are deterministic and verifiable. The
   Neural Program Executor (NPE) provides bytecode-level verification and a
   JIT pipeline with DCE, constant folding, and fusion passes.

4. **Federated**: Learning happens across distributed memory banks with
   differential privacy guarantees. NCCL sync bridge enables multi-node
   coordination.

5. **Deterministic**: Given the same inputs and seed, all operations produce
   identical outputs. No non-determinism in forward passes.

6. **Hybrid Execution**: Neuro-symbolic architecture combining neural components
   (HSS, SER) with a symbolic program executor (NPE) with JIT optimization.

7. **GPU Acceleration**: Optional CUDA backend for optimizer steps (SGD/AdamW)
   via cuBLAS, with clean CPU fallback when CUDA is unavailable.

## Component Interaction

The pipeline processes data through five components in sequence:

```
Input -> HSS -> SER -> ARC -> NPE -> FM -> Output
```

### Data Flow

1. **HSS (Hierarchical State Space)**
   - Input: Sequence of vectors (seq_len x input_dim)
   - Output: Sequence of hidden states (seq_len x state_dim)
   - Encodes temporal dependencies via Blelloch parallel associative scan
   - Hierarchical layers capture multi-scale patterns

2. **SER (Sparse Expert Routing)**
   - Input: Hidden states from HSS
   - Output: Expert-weighted representations
   - Routes each token to top-k experts via linear or MLP gater
   - Load balancing loss prevents expert collapse

3. **ARC (Adversarial Robustness Core)**
   - Input: SER output
   - Output: Sanitized representations
   - Z-score anomaly detection on inputs
   - Gradient obfuscation against adversarial attacks
   - Output consistency verification
   - Adversarial training graph with FGSM perturbation

4. **NPE (Neural Program Executor)**
   - Input: ARC output (or raw tensor for compiled programs)
   - Output: Verified computation result
   - Executes bytecode programs through a 16-register VM
   - JIT pipeline optimizes hot programs (DCE -> constant fold -> fuse -> compile)
   - Auto-JIT triggers optimization at configurable hot threshold

5. **FM (Federated Memory)**
   - Input: NPE output
   - Output: Memory-augmented representations
   - Key-value memory banks with temporal decay
   - NCCL sync bridge for distributed gradient aggregation
   - Distributed sync with differential privacy

## Error Handling

- Fail fast: detect errors at the earliest point
- No silent errors: all error paths return error codes
- Log and abort: unrecoverable errors log diagnostic info before aborting
- Error codes: consistent integer return codes (0 = success, nonzero = error)
- Input validation: all public functions validate parameters

## Memory Management

- Pool allocator: `SNEPPX_malloc` / `SNEPPX_free` with alignment support
- Reference counting for shared tensor data
- No garbage collection in hot paths
- Memory pools for fixed-size allocations (tensor metadata, small buffers)
- Secure wipe for sensitive data (keys, gradients) on deallocation
- Guard pages around security-critical allocations
- Canary values for buffer overflow detection
- CUDA device memory pool with async H2D/D2H transfers

## Thread Safety

- Immutable tensors: tensor data is read-only after creation
- Lock-free atomics for reference counts and shared state
- Mutex fallback for complex operations
- Thread-local storage for per-thread state (random seeds, temp buffers)
- No global mutable state

## CUDA Architecture

```
Trainer (CPU loop)
    |
    v
[if use_cuda_optimizer]
    |
    v
trainer_cuda_transfer()      # async H2D copy to device pool
    |
    v
trainer_cuda_optimizer_step() # cuBLAS SGEMM / cuBLAS Saxpy for SGD
    |                          # AdamW: cuBLAS + host element-wise fallback
    v
trainer_cuda_to_host()       # async D2H copy
    |
    v
Continue CPU training loop
```

All CUDA functions have clean stubs when `SNEPPX_HAS_CUDA` is undefined.

## NPE JIT Pipeline

```
Input Program (bytecode)
    |
    v
[DCE] --- removes NOPs and dead register writes
    |
    v
[Constant Fold] --- evaluates constant expressions at compile time
    |
    v
[Fusion] --- merges adjacent compatible ops:
  - MATMUL + ADD + RELU -> MATMUL_ADD_RELU (bit 0x20000000)
  - MATMUL + RELU       -> MATMUL_RELU     (bit 0x80000000)
  - MATMUL + ADD        -> MATMUL_ADD      (bit 0x40000000)
  - ADD + RELU          -> ADD_RELU        (bit 0x80000001)
    |
    v
[Compile] --- final optimized program
    |
    v
Output Program
```

## Security Architecture

```
S0: Crypto primitives (Ed25519, ChaCha20-Poly1305, SHA-3, BLAKE3, Kyber, Dilithium, SPHINCS+)
S1: Secure memory (guard pages, canaries, ASLR, locked memory, leak detector)
S2: Obfuscation (control flow flattening, string encryption, instruction substitution, code virtualization)
S3: Behavioral monitoring (integrity CRC, container breakout, anomaly detection)
S4: Network security (DDoS mitigation, transport security, identity management)
S5: AI sanitizer (prompt injection, differential privacy, RLHF safety, poisoning detection)
S6: Key vault (in-memory encrypted store, audit logging)
S7: Secure updates (container security, signed bundles, rollback protection)
S8: Formal verification (model checking, symbolic execution, invariants)
S9: Penetration testing (network fuzzer, self-audit, CTF utilities)
```

Every component includes defensive measures appropriate to its role:
- ARC: input sanitization, adversarial training, and output verification
- NPE: bytecode verification before execution, JIT optimization with provenance
- FM: differential privacy on gradient shares, NCCL callback authentication
- HSS/SER: constant-time operations where timing could leak information
