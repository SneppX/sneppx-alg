# SNEPPX-Algo Project Documentation (v1.0.0)

This file contains practical documentation and quick start guides for working with the SNEPPX-Algo project.

## Quick Start Guide

### 5-Minute Setup

```bash
# Clone the project
git clone https://github.com/ammar49-cyber/sneppx-alg.git
cd sneppx-alg

# Configure build
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# Build everything
cmake --build build --config Release -j$(nproc)

# Test everything
cd build && ctest -C Release --output-on-failure

# Install Python bindings
pip install -e bindings/python

# You're ready to go!
```

### Python Usage

```python
# Import the high-level API
from SneppX_ALG import Tensor, Linear, AdamW, TrainConfig, Trainer

# Create a simple neural network
model = Linear(1024, 512)
optimizer = AdamW(model.parameters(), lr=0.001)

# Training loop
x = Tensor.randn(32, 1024)
y = Tensor.randn(32, 512)
config = TrainConfig(learning_rate=0.001, batch_size=32, max_steps=100)
trainer = Trainer(model, config)
trainer.fit(x, y)

# Inference
output = model(x)
```

### Model Zoo — from_pretrained

```python
from SneppX_ALG.interface_bindings.model_zoo import (
    get_model_config, from_pretrained, build_model_from_config, ModelHub
)

# Look up a preset
config = get_model_config("llama3", "8B")
print(config.hidden_size)  # 4096

# Get model info
model_info = build_model_from_config(config)
print(model_info["layers"])        # 32
print(model_info["param_str"])     # "8.0B"

# Use ModelHub for full lifecycle
hub = ModelHub()
hub.save_pretrained("llama-2-7b", "./my_model")
```

### C/C++ Integration

```c
#include "include/neural_core/tensor.h"
#include <stdio.h>

int main() {
    // Create a tensor
    SNEPPX_Tensor* t = sneppx_tensor_create(2, 3);
    
    // Fill with zeros
    sneppx_tensor_fill(t, 0.0f);
    
    // Operations
    sneppx_tensor_print(t);
    
    // Clean up
    sneppX_tensor_free(t);
    return 0;
}
```

### Model Zoo C API

```c
#include <neural_core/model_zoo/model_config.h>
#include <neural_core/model_zoo/registry.h>
#include <neural_core/model_zoo/weights.h>
#include <neural_core/model_zoo/model_card.h>

// Create config from preset
ModelConfig *cfg = model_config_llama2_7b();
printf("Hidden size: %ld\n", cfg->hidden_size);

// Serialize to JSON
char *json = model_config_to_json(cfg, 1);

// Register model
ModelRegistry *reg = model_registry_create();
model_registry_register(reg, "my-model", "1.0", "transformer",
                        "desc", "author", "MIT", "", "", "", 1);

// Create weights
WeightCollection *wc = weight_collection_create();
int64_t shape[] = {4096, 4096};
float data[1] = {0.0f};
weight_collection_add(wc, "weight", shape, 2, "f32", data, sizeof(data), 0);

// Create model card
ModelCard *card = model_card_create();
model_card_set_name(card, "my-model");
model_card_set_version(card, "1.0");
model_card_add_tag(card, "nlp");

// Cleanup
model_config_destroy(cfg);
model_registry_destroy(reg);
weight_collection_destroy(wc);
model_card_destroy(card);
free(json);
```

### Rust Bindings

```rust
use neural_core_algo::Tensor;

fn main() {
    // Create a tensor
    let mut t = Tensor::new(vec![2, 3]);
    
    // Use it
    let ones = Tensor::ones(vec![2, 3]);
    
    println!("Shape: {:?}", ones.shape());
    
    // Clean up happens automatically
}
```

## Build Configuration Options

### Debug vs Release

**Debug Build (for development):**

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build --config Debug
```

**Release Build (for production):**

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

### Special Build Options

```bash
# Build with Python bindings
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSNEPPX_BUILD_PYTHON=ON

# Build with CUDA support
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSNEPPX_BUILD_CUDA=ON

# Build with all security layers
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSNEPPX_BUILD_SECURITY=ON

# Build opt-in reference backends (real computation; OFF by default)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSNEPPX_BUILD_VULKAN=ON
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSNEPPX_BUILD_TPU=ON
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSNEPPX_BUILD_HTTP=ON
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSNEPPX_BUILD_ZK=ON

# Build with tests
-DSNEPPX_BUILD_TESTS=ON

# Build with benchmarks
-DSNEPPX_BUILD_BENCHMARKS=ON
```

## Python Module Structure

### Core Modules (8 phases)

#### Phase 1: Tensor Engine & Autodiff

| Module | Purpose |
|--------|---------|
| `tensor.py` | Tensor creation, manipulation, operations |
| `autograd.py` | Automatic differentiation |
| `autograd_ops.py` | Differentiable operations |

#### Phase 2: Neural Network Building Blocks

| Module | Purpose |
|--------|---------|
| `nn.py` | Core neural network layers (Linear, Conv2d, etc.) |
| `attention.py` | Attention mechanisms |
| `activations.py` | Activation functions |

#### Phase 3: Optimization & Training

| Module | Purpose |
|--------|---------|
| `optim.py` | Optimizers (Adam, SGD, etc.) |
| `optim_extra.py` | Additional optimizers |
| `optim_advanced.py` | Advanced optimization |

#### Phase 4: Data Management

| Module | Purpose |
|--------|---------|
| `data.py` | Data loading utilities |
| `data_loader.py` | Dataset loaders |
| `tokenizer.py` | Tokenization for NLP |

#### Phase 5: Checkpoints & State Management

| Module | Purpose |
|--------|---------|
| `checkpoint.py` | Model saving/loading |
| `checkpoint_manager.py` | Checkpoint management |
| `experiment_tracker.py` | Experiment tracking |

#### Phase 6: Quantization & Compression

| Module | Purpose |
|--------|---------|
| `quantization.py` | Model quantization (INT8, FP8, AWQ, GPTQ) |
| `pruning.py` | Model pruning |
| `distillation.py` | Knowledge distillation |

#### Phase 7: Profiling & Deployment

| Module | Purpose |
|--------|---------|
| `profiler.py` | Performance profiling with NVTX markers |
| `benchmark.py` | Benchmarking |
| `serve_cli.py` | CLI for serving models |
| `train_cli.py` | CLI for training |

#### Phase 8: Model Zoo

| Module | Purpose |
|--------|---------|
| `model_zoo.py` | Model presets, from_pretrained, ModelHub, weight converters |

### Advanced Interfaces

#### Low-level C Bindings

```python
from SneppX_ALG import _neural_engine_bridge as ax

# Access low-level C APIs
t = ax.SNEPPX_Tensor_create(2, 3)
ax.SNEPPX_Tensor_fill(t, 1.0f)
```

#### Security Interfaces

```python
from SneppX_ALG import (
    crypto_hash,
    crypto_encrypt,
    crypto_decrypt,
    generate_keypair,
    sign_data,
    verify_signature,
)

# Cryptographic operations
keypair = generate_keypair()
encrypted = crypto_encrypt(b"data", keypair.public_key)
decrypted = crypto_decrypt(encrypted, keypair.private_key)
```

#### Distributed Training

```python
from SneppX_ALG import (
    distributed,
    zero,
    nccl,
    checkpoint_distributed,
)

# Distributed training setup
distributed.init({"world_size": 4, "rank": 0})
zero_optimizer = zero.ZeroOptimizer(model.parameters())
```

## Architecture Overview

### Core Components

1. **Kernel Layer** (`kernel/`)
   - Tensor operations (SIMD, CUDA)
   - Autodiff engine
   - Optimizers
   - Training loop
   - CUDA backend (GEMM, attention, memory pool, RNG)

2. **Algorithm Layer** (`algorithms/`)
   - HSS (Hierarchical State Spaces / Mamba-2)
   - SER (Sparse Expert Routing / MoE)
   - ARC (Adversarial Robustness Certification)
   - NPE (Neural Program Engine)
   - FM (Factorized Manifolds / Federated Memory)

3. **Security Layer** (`security/`)
   - S0-S9 security layers (21,984+ LOC)
   - Post-quantum crypto (Kyber, Dilithium, SPHINCS+)
   - Code obfuscation
   - Behavioral monitoring, RLHF safety

4. **Network Layer** (`net/`)
   - Distributed training (NCCL)
   - ZeRO-1/2/3, pipeline/tensor/expert parallelism
   - Elastic training with fault tolerance
   - Checkpoint coordinator

5. **Model Zoo** (`algorithms/model_zoo/`)
   - Model configs, registry, weights, cards
   - C/C++ RAII wrappers
   - Python ModelHub

### Supported Architectures

#### Large Language Models (LLMs)

- LLaMA 2/3 (7B, 13B, 70B)
- Mistral 7B
- Qwen2 (7B, 72B)
- DeepSeek V2 (Lite, Full)

#### Advanced Architectures

- **Differential Attention** — λ-scaled subtracted QK pairs
- **Multi-head Latent Attention (MLA)** — DeepSeek-style absorbed KV projection
- **FlexAttention** — Block-sparse with mask modulation
- **Mamba-2** — Selective SSM with HiPPO initialization
- **Mixture of Depth** — Token-level expert routing
- **YaRN** — NTK-aware RoPE scaling with ramp interpolation
- **ALiBi** — Attenuated linear bias position encoding

#### Custom Pipeline Algorithms

- HSS/Mamba: Hierarchical state spaces
- SER/MoE: Sparse expert routing
- ARC: Adversarial robustness training (FGSM, PGD, CW)
- NPE: Neural program extraction with JIT pipeline
- FM: Fractal memory / federated averaging

### Hardware Support

#### Current
- x86-64 (AVX2, AVX-512)
- CUDA 12.x (NVIDIA GPUs, tensor-core GEMM, Flash Attention)
- ROCm (AMD GPUs)
- Vulkan, TPU — opt-in reference backends
- HTTP, ZK — opt-in reference backends
- Metal, oneAPI — reference backends

#### Planned
- ARMv8-A (NEON)
- NPU (Qualcomm, etc.)

## API Reference

### Python API

#### Core Types

| Class | Description |
|-------|-------------|
| `Tensor` | Multi-dimensional array |
| `Linear` | Fully connected layer |
| `Model` | Base neural network class |
| `Optimizer` | Base optimizer |
| `Trainer` | Training loop wrapper |
| `ModelConfig` | Unified model config dataclass |
| `ModelHub` | Model download/cache/save manager |

#### Key Functions

| Function | Purpose |
|----------|---------|
| `Tensor.create()` | Create a tensor |
| `Tensor.zeros()` | Create zero tensor |
| `Tensor.ones()` | Create one tensor |
| `Tensor.randn()` | Create random tensor |
| `optim.SGD()` | SGD optimizer |
| `optim.Adam()` | Adam optimizer |
| `optim.AdamW()` | AdamW optimizer |
| `get_model_config()` | Look up model config by name/size |
| `from_pretrained()` | Download/cache model weights |
| `build_model_from_config()` | Build model info dict from config |
| `convert_hf_to_sneppx()` | Convert HF weights to SNEPPX format |

### C API

Key C functions (from `include/neural_core/`):

- `sneppx_tensor_create()` - Create tensor
- `sneppx_tensor_free()` - Free tensor
- `sneppx_tensor_fill()` - Fill with values
- `sneppx_tensor_print()` - Print tensor
- `sneppx_tensor_add()` - Add tensors
- `sneppx_tensor_multiply()` - Multiply tensors

#### Model Zoo C API

| Function | Header | Purpose |
|----------|--------|---------|
| `model_config_llama2_7b()` | `model_config.h` | Create LLaMA-2 7B config |
| `model_config_to_json()` | `model_config.h` | Serialize config to JSON |
| `model_config_from_json()` | `model_config.h` | Parse JSON to config |
| `model_registry_create()` | `registry.h` | Create model registry |
| `model_registry_register()` | `registry.h` | Register a model |
| `model_registry_search()` | `registry.h` | Search models by name |
| `weight_collection_create()` | `weights.h` | Create weight collection |
| `weight_collection_add()` | `weights.h` | Add weight tensor |
| `weight_tensor_quantize_int8()` | `weights.h` | Quantize to INT8 |
| `model_card_create()` | `model_card.h` | Create model card |
| `model_card_to_json()` | `model_card.h` | Serialize card to JSON |
| `model_card_save()` | `model_card.h` | Save card to file |
| `model_card_validate()` | `model_card.h` | Validate card fields |

### Rust API

Rust bindings provide safe, idiomatic access:

```rust
use neural_core_algo::{Tensor, Linear, TrainingLoop};

fn main() {
    // Tensor operations
    let mut t = Tensor::zeros(vec![2, 3]);
    t.fill(1.0);
    
    // Neural network
    let model = Linear::new(1024, 512);
    
    // Training loop
    let mut trainer = TrainingLoop::new(model, 0.001);
    trainer.train(&t);
}
```

## Development Workflow

### Setting Up Development Environment

```bash
# Clone the repo
git clone https://github.com/ammar49-cyber/sneppx-alg.git
cd sneppx-alg

# Install Python dependencies
pip install -r requirements.txt
pip install -e bindings/python

# Install Rust toolchain (if needed)
rustup toolchain install stable
rustup component add rust-src

# Setup subprojects
rustup toolchain install nightly
```

### Making Changes

1. **Always create a feature branch**
2. **Follow the project's commit message conventions**
3. **Test locally before creating PRs**
4. **All tests must pass**

### Testing

#### Python Tests

```bash
# Run Python tests
$env:PYTHONPATH = "bindings/python"
python -m pytest tests/python/ -v

# Model Zoo Python tests
python bindings/python/SneppX_ALG/interface_bindings/tests/test_model_config.py
python bindings/python/SneppX_ALG/interface_bindings/tests/test_integration.py

# Quick test
python -m pytest tests/python/test_tensor.py -v

# Model Zoo C tests
.\build\algorithms\model_zoo\test_model_config.exe
.\build\algorithms\model_zoo\test_model_registry.exe
.\build\algorithms\model_zoo\test_model_weights.exe
.\build\algorithms\model_zoo\test_model_card.exe
.\build\algorithms\model_zoo\test_model_factory.exe
.\build\algorithms\model_zoo\test_integration.exe
```

#### C/C++ Tests

```bash
cd build && ctest -C Release --output-on-failure

# Individual test
ctest -C Release -R test_tensor -v
```

#### Rust Tests

```bash
cd net/distributed && cargo test
cd lib/rust && cargo test
```

#### Performance Benchmarks

```bash
cd build && ./benchmarks/benchmark_suite --mode=full
```

## Troubleshooting

### Common Issues

#### Import Errors

```bash
# Error: Could not import 'SneppX_ALG'
$ pip install sneppx-alg==1.0.0
```

#### Build Failures

```bash
# Error: C++11 not found
# Fix: Install g++ >= 7
apt-get update && apt-get install g++
```

#### CUDA Errors

```bash
# Error: No CUDA device
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSNEPPX_BUILD_CUDA=OFF
```

### Getting Help

If you run into issues:

1. **Check the docs**: https://github.com/ammar49-cyber/sneppx-alg
2. **Search issues**: Look for similar problems
3. **Create a new issue**: Include:
   - Reproduction steps
   - Error messages
   - Environment info
   - Code snippets

## Roadmap

### Completed (v1.0.0)

- [x] All 8 phases: Tensor Engine, Neural Networks, Optimization, Data Pipeline, Checkpointing, Quantization, Profiling, Model Zoo
- [x] Distributed training: ZeRO-1/2/3, Pipeline, Tensor/Expert Parallel, Elastic Training, Fault Tolerance
- [x] Advanced architectures: Differential Attention, MLA, FlexAttention, Mamba-2, MoD, YaRN, ALiBi
- [x] CUDA backend: Tensor-core GEMM, Flash Attention v2/v3, Autodiff, Memory Pool, RNG
- [x] Security: S0-S9 complete (21,984+ LOC)
- [x] from_pretrained: Model Hub with caching, presets for LLaMA/Mistral/Qwen2/DeepSeek V2
- [x] Quantization: INT8, FP8, AWQ, GPTQ (C + CUDA + Python)

### Next (v1.1+)

- [ ] Model serving: vLLM/TensorRT-LLM integration
- [ ] LoRA/QLoRA fine-tuning
- [ ] LM Evaluation Harness integration
- [ ] pip-installable wheel
- [ ] ARMv8-A (NEON) support
- [ ] NPU (Qualcomm, etc.) support
- [ ] Federated learning
- [ ] Security formal verification
- [ ] Production-ready tooling

## References

### Tools & Libraries

- **CMake** - Build system
- **pybind11** - Python bindings
- **tokio** - Asynchronous runtime
- **serde** - Serialization
- **tracing** - Logging
- **clap** - CLI argument parsing

## License

MIT License

## Copyright

© 2024-2026 Ammar [SNEPPX] - algosneppx@gmail.com

---

_Generated with SNEPPX-Alg v1.0.0_
