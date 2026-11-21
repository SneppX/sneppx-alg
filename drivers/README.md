# Hardware Drivers

The `drivers/` directory contains hardware abstraction layers for GPU accelerators:
CUDA (NVIDIA), ROCm (AMD), and TPU (Google).

## CUDA Driver (`drivers/cuda/`)

Manages NVIDIA GPU devices: device enumeration, context creation, memory management,
kernel dispatch, and stream synchronization. Detects Hopper, Ampere, Volta,
and Pascal architectures.

## ROCm Driver (`drivers/rocm/`)

AMD ROCm abstraction layer for CDNA/RDNA GPUs. HIP-based API for kernel
launch and memory management. Conditional build with `-DSNEPPX_BUILD_ROCM=ON`.

## TPU Driver (`drivers/tpu/`)

Google TPU abstraction via PjRt API. Supports Cloud TPU v2-v5e and local
TPU runtime. Conditional build with `-DSNEPPX_BUILD_TPU=ON`.

## Usage

```python
from SneppX_ALG import CUDADriver, ROCmDriver, TPUDriver

# Enumerate devices
driver = CUDADriver()
devices = driver.enumerate()
print(devices[0].name, devices[0].compute_capability)
```
