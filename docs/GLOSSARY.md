# Glossary

| Term | Definition |
|------|------------|
| **ARC** | Adversarial Robustness Core -- three-layer defense pipeline (input guard, gradient obfuscation, output verifier + attack simulation) |
| **ARI** | Abstract Representation Intermediate -- intermediate representation used by NPE compilers |
| **ASLR** | Address Space Layout Randomization -- heap entropy via randomized memory allocation |
| **Autodiff** | Automatic differentiation -- tape-based reverse-mode gradient computation |
| **BDFL** | Benevolent Dictator For Life -- project governance model |
| **Blelloch** | Tree-based parallel prefix scan algorithm used by HSS for O(log n) state updates |
| **BPE** | Byte-Pair Encoding -- subword tokenization algorithm |
| **C&W** | Carlini-Wagner -- optimization-based adversarial attack |
| **Canary** | Stack-based overflow detection value with generation counter |
| **cuBLAS** | NVIDIA's CUDA Basic Linear Algebra Subprograms library |
| **CUDA** | NVIDIA's parallel computing platform, used for GPU-accelerated optimizer |
| **DCE** | Dead Code Elimination -- JIT pass that removes NOPs and unused instructions |
| **FGSM** | Fast Gradient Sign Method -- single-step adversarial attack |
| **FM** | Federated Memory -- multi-node memory banks with trust-weighted sync + NCCL bridge |
| **HSS** | Hierarchical State Space -- multi-layer SSM with O(n log n) Blelloch parallel scan |
| **JIT** | Just-In-Time compilation -- runtime program optimization pipeline (DCE -> constant fold -> fuse -> compile) |
| **KV-Cache** | Key-Value Cache -- cached attention keys/values for autoregressive generation |
| **MHA** | Multi-Head Attention -- attention with multiple parallel heads |
| **MLP Gater** | Learned MLP-based router for SER (2-layer MLP replacing linear router) |
| **MoE** | Mixture of Experts -- sparse routing with top-k expert activation |
| **NCCL** | NVIDIA Collective Communications Library -- multi-GPU synchronization |
| **NPE** | Neural Program Executor -- 16-register VM with 15+ opcodes + JIT pipeline |
| **PGD** | Projected Gradient Descent -- iterative adversarial attack |
| **RoPE** | Rotary Position Embedding -- relative position encoding for attention |
| **RNG** | Random Number Generator -- secure entropy source |
| **SER** | Sparse Expert Routing -- MoE layer with learned gating (linear or MLP) |
| **SSM** | State Space Model -- sequence model with latent state evolution |
| **ZKP** | Zero-Knowledge Proof -- cryptographic proof without revealing inputs |
| **ZOH** | Zero-Order Hold -- discretization method for continuous-time SSMs |
