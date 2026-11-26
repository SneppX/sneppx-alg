# SneppX-ALG

[![PyPI version](https://img.shields.io/pypi/v/sneppx-alg)](https://pypi.org/project/sneppx-alg/)
[![Python versions](https://img.shields.io/pypi/pyversions/sneppx-alg)](https://pypi.org/project/sneppx-alg/)
[![CI](https://github.com/SneppX/sneppx-alg/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SneppX/sneppx-alg/actions/workflows/ci.yml)
[![MIT License](https://img.shields.io/github/license/SneppX/sneppx-alg)](LICENSE)

SneppX-ALG is an experimental, open-source machine-learning framework with a native C/C++ core and Python bindings. It focuses on tensor operations, automatic differentiation, and neural-network building blocks, with supporting work in quantization, model tooling, and runtime security research.

**Project maturity: Alpha.** APIs and implementation coverage vary by subsystem and backend. SneppX-ALG is not a drop-in replacement for PyTorch, TensorFlow, or JAX. Review the implementation status and tests for the exact feature you plan to use.

## Release status

The latest package published on PyPI is **1.5.2**. The current `main` branch contains the **1.5.3-rc.1** release candidate; it has not been published to PyPI. Source-tree behavior may therefore differ from the published package. See the [packaging and release policy](docs/dev/python-packaging.md) before choosing an installation path.

## Install

Install the currently published Python package:

```bash
python -m pip install sneppx-alg
```

For a source checkout, install the bindings from the repository:

```bash
python -m pip install ./bindings/python
```

## What is in the project?

- **Tensor and autograd APIs:** tensor operations and reverse-mode automatic differentiation, exposed through Python bindings. Gradient coverage and optional native acceleration vary by operation.
- **Neural-network components:** layers, losses, optimizers, and transformer-related utilities, with compatibility and test coverage varying across modules.
- **Model and quantization tools:** configuration presets, conversion, quantization, and serving-related utilities. End-to-end pretrained-weight loading and some integrations remain in progress.
- **Native implementation:** C/C++ components and a Python/NumPy path for development on CPU-only systems.
- **Security research:** experimental S0–S9 components. These are not a security certification, not independently audited as a complete system, and not a basis for FIPS-compliance claims.
- **Distributed training:** experimental API and native-code work only. The current Python multi-rank process-group and collective APIs fail closed; DDP/FSDP/ZeRO should not be treated as working multi-rank training.

See the [framework parity roadmap](docs/roadmap/tensorflow-pytorch-parity.md), [distributed-training status](docs/distributed-guide.md), and [security documentation](docs/security.md) for current limitations.

## Build from source

Requirements include CMake 3.16 or newer, a C11/C++20 compiler (GCC 11+, Clang 14+, or MSVC 2022), and Python with NumPy for Python bindings.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

CUDA and other optional backends are not required for basic CPU development. See the [build guide](docs/build.md) for configuration options.

## Documentation

- [Documentation home](https://sneppx.github.io/sneppx-alg/latest/)
- [Getting started](https://sneppx.github.io/sneppx-alg/latest/getting-started/)
- [Python API reference](https://sneppx.github.io/sneppx-alg/latest/api/python/)
- [Architecture overview](docs/ARCHITECTURE.md)
- [Security status and audit history](docs/security.md)
- [Development guide](docs/DEVELOPMENT.md)

The [SneppX project organization](https://github.com/SneppX) hosts related open-source work.

## Codebase telemetry

The repository includes a dependency-free codebase telemetry (CTM) CLI:

```bash
sneppx-ctm .
sneppx-ctm . --format json --output ctm-report.json
```

It reports per-language LOC and bytes, skips generated and binary content, and exposes additional console scripts. See the [CTM tools documentation](docs/tools/ctm-tools.md).

## Security

Security-related components have different implementation and validation levels. Several areas remain experimental, partial, or incomplete; no complete-system independent audit or certification is claimed. Do not rely on these components as security boundaries without reviewing the [security status](docs/security.md), [historical self-audit](docs/security/audits/index.md), and relevant tests.

To report a suspected vulnerability, follow the [security policy](SECURITY.md) and use private disclosure.

## Contributing

Bug reports, feature requests, and contributions are welcome. Read the [development guide](docs/DEVELOPMENT.md) before making changes.

- [Issues](https://github.com/SneppX/sneppx-alg/issues)
- [Discussions](https://github.com/SneppX/sneppx-alg/discussions)
- [Contribution guide](CONTRIBUTING.md)

## License

Released under the [MIT License](LICENSE).
