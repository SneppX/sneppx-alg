---
hide:
  - toc
---

# Project scope and current limitations

SneppX-ALG is an open-source project exploring tensor operations, automatic differentiation, neural-network utilities, native C/C++ components, and security-related tooling.

It is an experimental codebase, not a drop-in replacement for PyTorch, TensorFlow, or JAX. The public Python tensor and neural-network APIs use NumPy-backed implementations for much of their functionality. Native code and optional backends have separate build and integration paths.

## What the repository contains

- Python tensor operations and automatic differentiation.
- Neural-network layers, losses, and optimizers.
- Model configuration, conversion, and inference utilities.
- Native C/C++ components and optional CUDA-related code.
- Distributed-computing and quantization modules.
- Security components documented across the S0–S9 architecture.

This list describes repository areas, not a guarantee that every feature is complete or suitable for production. Check the API reference, implementation, and tests for the exact component you plan to use.

## Known limitations

### Backend coverage

The public Python API is not uniformly backed by native code. Much of the Tensor and neural-network surface uses NumPy. Do not assume that installing the Python package enables native acceleration.

### Framework compatibility

SneppX-ALG does not provide full API or ecosystem compatibility with PyTorch, TensorFlow, or JAX. The [parity roadmap](roadmap/tensorflow-pytorch-parity.md) tracks areas that remain incomplete.

### Model support

Model configuration and model-zoo utilities do not imply that every listed model can be downloaded, loaded, trained, or served end to end. Check the relevant model guide and implementation before planning around a model.

### Distributed execution and accelerators

Availability depends on the backend, build configuration, and runtime environment. Read the distributed and build documentation before assuming multi-device execution or GPU acceleration.

### Security

Security modules have different implementation and validation levels. Experimental cryptographic code is not a substitute for a validated cryptographic library. The repository's tests and documentation do not constitute an independent security audit or certification. Start with the [security overview](security.md) and the [PQC interoperability report](pqc_kat.md).

## Choosing a tool

Use SneppX-ALG when you want to inspect or experiment with this codebase, work with the available Python APIs, or contribute to its native and security-related components.

Choose an established framework when your project requires mature GPU training, broad hardware support, extensive pretrained-model coverage, stable production serving, or a large third-party ecosystem.

## Useful references

- [Getting started](getting-started.md)
- [Python API reference](api/python.md)
- [Build guide](build.md)
- [Framework parity roadmap](roadmap/tensorflow-pytorch-parity.md)
- [Security documentation](security.md)
- [Contributing guide](contributing.md)
