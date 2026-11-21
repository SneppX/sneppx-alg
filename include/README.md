# C/C++ Headers

The `include/` directory contains all public C/C++ header files for the
SNEPPX-ALG engine, organized under `include/neural_core/`.

## Structure

```
include/neural_core/
├── kernel/          # Core kernel headers
│   ├── tensor.h, tensor_expr.h, simd_gemm.h
│   ├── automatic_differentiation_framework.h
│   ├── gradient_optimization_suite.h
│   ├── differentiable_training_pipeline.h
│   ├── polymorphic_memory_allocator.h
│   ├── concurrent_workload_dispatch.h
│   ├── quantization.h (INT8/FP8/AWQ/GPTQ)
│   ├── checkpoint.h, heartbeat.h, elastic.h
│   ├── profiler.h, logger.h
│   └── model_zoo.h
├── architecture/    # Algorithm headers
│   ├── hierarchical_state_space.h
│   ├── sparse_expert_routing.h
│   ├── adversarial_robustness_certification.h
│   ├── neural_programming_engine.h
│   ├── fractal_memory_orchestrator.h
│   └── distributed.h, advanced_arch.h
└── security/        # Security headers
    ├── cryptographic_primitives_bundle.h (umbrella)
    ├── kyber.h, dilithium.h, sphincsplus.h
    ├── chacha20.h, poly1305.h, aead.h
    ├── ed25519.h, x25519.h
    ├── sha2.h, sha3.h, blake3.h
    ├── argon2.h, hkdf.h, pbkdf2.h
    ├── secure_mem.h, canary.h, aslr.h
    ├── container_breakout.h, integrity_monitor.h
    ├── transport_security.h, ddos_mitigation.h
    ├── firewall.h
    ├── prompt_filter.h, output_verifier.h, rlhf_safety.h
    ├── signed_update.h, container_security.h
    └── model_checking.h, network_fuzzer.h
```

## Include Convention

Headers use short-form paths relative to the project root:
```c
#include "multidimensional_tensor_engine.h"        // from include/neural_core/kernel/
#include "cryptographic_primitives_bundle.h"        // from include/neural_core/security/
```

CMake targets add the appropriate `include/` subdirectories to the include path.
