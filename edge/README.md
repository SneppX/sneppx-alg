# SNEPPX-Alg Edge Runtime

Lightweight inference engine for mobile (Android/iOS) and edge (ARM/x86) devices.

## Overview

Minimal binary size (<5MB) execution engine designed for resource-constrained environments.

- **Minimal Binary Size**: Optimized for ARM and x86 targets.
- **Quantization Support**: INT8, FP16 quantization.
- **Memory Optimization**: Arena allocation with reuse.
- **Minimal Op Subset**: Conv, ReLU, MaxPool, MatMul.

## Build

```bash
cd edge
mkdir build && cd build
cmake ..
cmake --build .
```

## Model Conversion

```bash
python sneppx-to-edge.py <input_model> <output_model.edge>
```
