# SNEPPX-Alg Examples

This directory contains demonstration programs and usage examples for the SNEPPX-Alg project.

## Overview

The examples directory showcases different aspects of the SNEPPX-Alg system, from basic tensor operations to complete training workflows. Each example is designed to be self-contained and can be built independently.

## Examples Structure

### Core Algorithms

#### hss
- **HSS Demo**: Demonstrates hierarchical state space model using HMM for sequence modeling with variable-length inputs
- **Files**: `hss_demo.c` (main), `hss.h` (API)

#### ser
- **SER Demo**: Shows sparse expert routing with MoE architecture, load balancing, and expert dispatch
- **Files**: `ser_demo.c` (main), `ser.h` (API)

#### arc
- **ARC Demo**: Adversarial Robustness Certification with input guarding, gradient obfuscation, and output verification
- **Files**: `arc_demo.c` (main), `arc.h` (API)

#### npe
- **NPE Demo**: Neural Program Executor demonstrating VM execution with 16 registers and 14 opcodes
- **Files**: `npe_demo.c` (main), `npe.h` (API)

#### fm
- **FM Demo**: Federated Memory with distributed memory banks, trust-weighted synchronization, and differential privacy
- **Files**: `fm_demo.c` (main), `fm.h` (API)

### C/C++ Examples

#### tensor_examples
- **Basic Ops**: Tensor creation, manipulation, and operations
- **Examples**: reshape, transpose, matmul, reduction operations

#### autodiff_examples
- **Gradients**: Automatic differentiation for simple neural networks
- **Examples**: forward/backward passes, gradient computation

#### optimizer_examples
- **Training**: Different optimizers (SGD, Adam, etc.)
- **Examples**: parameter updates, learning rate scheduling

#### security_examples
- **Crypto**: Post-quantum cryptographic operations
- **Examples**: Ed25519 signing, AES-GCM encryption

#### mixed_mode
- **CPU+GPU**: Interleaved CPU and CUDA operations
- **Examples**: Data transfer, kernel launches, async execution

### Python Examples

#### basic_examples
- **Tensor**: Basic tensor creation and operations
- **Model**: Simple neural network construction
- **Training**: Complete training loop example

#### advanced_examples
- **Distributed**: Multi-GPU/distributed training
- **Quantization**: Model quantization and inference
- **Security**: Secure inference with homomorphic operations

### Experimental Examples

#### arch_experiments
- **Architecture Search**: Automated neural architecture design
- **Hyperparameter Tuning**: Grid search and Bayesian optimization

#### benchmark_examples
- **Performance**: Core component benchmarks
- **Scale**: Scaling studies with large datasets

## Building and Running Examples

### C/C++ Examples

```bash
# Build all examples
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# Run a specific example
./examples/hss/hss_demo

# View help
./examples/hss/hss_demo --help
```

### Python Examples

```python
import SneppX_ALG as ax
import numpy as np

# Example: Simple neural network with training
t = ax.Tensor.randn(32, 1024)
y = ax.Tensor.randn(32, 512)

# Create model and train
model = ax.Linear(1024, 512)
optimizer = ax.AdamW(model.parameters(), lr=0.001)

for epoch in range(10):
    output = model(t)
    loss = ax.nn.functional.mse_loss(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
```

### Rust Examples

```rust
use neural_core_algo::Tensor;

fn main() {
    // Create tensor
    let mut t = Tensor::zeros(&[4, 8]);
    
    // Add one for demonstration
    let mut ones = Tensor::ones(&[4, 8]);
    t = t + ones;
    
    println!("Tensor shape: {:?}", t.shape());
}
```

## Example Categories

### Algorithm Demonstrations

These showcase the core 5-algorithm pipeline:

1. **HSS (Hierarchical State Space)**
   - Sequence modeling with HMM
   - O(n log n) temporal processing

2. **SER (Sparse Expert Routing)**
   - Mixture of Experts architecture
   - Load balancing via top-k routing

3. **ARC (Adversarial Robustness Core)**
   - Three-layer defense system
   - Input guard + gradient obfuscation + output verifier

4. **NPE (Neural Program Executor)**
   - 16-register VM with 14 opcodes
   - Static verification for program correctness

5. **FM (Federated Memory)**
   - Distributed memory banks with trust weighting
   - Differential privacy with gradient compression

### Integration Examples

These demonstrate system integration patterns:

1. **Security Integration**
   - Cryptographic operations in training loops
   - Secure memory allocation

2. **Distributed Training**
   - Multi-GPU coordination
   - Parameter synchronization

3. **Hardware Acceleration**
   - CPU-CUDA overlap
   - Heterogeneous computing

### Performance Examples

These showcase performance characteristics:

1. **Benchmark Suite**
   - Tensor core performance
   - Algorithm scaling studies

2. **Optimization Examples**
   - Vectorization strategies
   - Memory layout optimizations

## Example Viewer Script

A companion script allows browsing examples:

```bash
# View example documentation
./tools/example_viewer.sh [example_name]

# List all examples
./tools/example_viewer.sh list

# Get example info
./tools/example_viewer.sh info hss
```

## Contributing Examples

To add a new example:

1. Create a new directory under `examples/` with:
   - `main.c` (the main program)
   - `example.h` (API header, if needed)
   - `README.md` (example documentation)

2. Update `CMakeLists.txt` in the parent `examples/` directory

3. Add entry to `examples/README.md`

4. Test the example locally

## Example Status

### Production Ready (✅)

| Example | Language | Status | Description |
|---------|----------|--------|-------------|
| hss/hss_demo | C | ✅ | HMM-based sequence modeling |
| ser/ser_demo | C | ✅ | MoE with load balancing |
| arc/arc_demo | C | ✅ | Adversarial robustness demo |
| npe/npe_demo | C | ✅ | VM execution demo |
| fm/fm_demo | C | ✅ | Distributed memory demo |

### Development Ready (⚠️)

| Example | Language | Status | Description |
|---------|----------|--------|-------------|
| Python tensor_examples | Python | ⚠️ | Basic tensor operations |
| Python model_examples | Python | ⚠️ | Neural network construction |
| Python training_examples | Python | ⚠️ | Complete training loop |

### Planned (📋)

| Example | Language | Status | Description |
|---------|----------|--------|-------------|
| distributed_examples | C | 📋 | Multi-GPU coordination |
| gpu_examples | C | 📋 | CUDA kernel examples |
| security_examples | C | 📋 | Cryptographic operations |
| quantum_examples | Q# | 📋 | Post-quantum cryptography |

## Quick Reference

### Common Commands

```bash
# Build and test all examples
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
cd build && ctest -C Release

# Run all examples at once
./tools/run_all_examples.sh

# Profile examples
./tools/profile_examples.sh
```

### Example Execution

```bash
# Example execution patterns

# Single example
./examples/hss/hss_demo --data-file data.txt --output-dir results

# With validation
./examples/ser/ser_demo --validate --benchmark

# Benchmark mode
./examples/arc/arc_demo --benchmark --repeats 10

# Profile execution
valgrind --tool=massif ./examples/npe/npe_demo --profile
```

## Example Index

### By Algorithm

1. **HSS Examples**
   - `examples/hss/`: HMM and state space models

2. **SER Examples**
   - `examples/ser/`: Mixture of Experts

3. **ARC Examples**
   - `examples/arc/`: Adversarial robustness

4. **NPE Examples**
   - `examples/npe/`: Virtual machine execution

5. **FM Examples**
   - `examples/fm/`: Distributed memory

### By Language

1. **C/C++ Examples**
   - Core algorithm demonstrations
   - Performance benchmarks

2. **Python Examples**
   - High-level API usage
   - Data science workflows

3. **Rust Examples**
   - Systems programming patterns
   - Safety demonstrations

### By Complexity

1. **Beginner**
   - Tensor creation and manipulation
   - Simple model training

2. **Intermediate**
   - Distributed algorithms
   - Advanced optimization

3. **Advanced**
   - GPU acceleration
   - Security integration

## Conclusion

The examples directory provides comprehensive coverage of the SNEPPX-Alg system, from basic operations to advanced algorithms. Each example is designed to be educational, practical, and self-contained, making it easy to understand and extend the system.

Ready to dive in? Check out the `examples/README.md` in your preferred language category and start experimenting!