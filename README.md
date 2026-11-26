# SneppX-ALG

[![PyPI version](https://img.shields.io/pypi/v/sneppx-alg)](https://pypi.org/project/sneppx-alg/)
[![Python versions](https://img.shields.io/pypi/pyversions/sneppx-alg)](https://pypi.org/project/sneppx-alg/)
[![CI](https://github.com/SneppX/sneppx-alg/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SneppX/sneppx-alg/actions/workflows/ci.yml)
[![MIT License](https://img.shields.io/github/license/SneppX/sneppx-alg)](LICENSE)

SneppX-ALG is an experimental open-source machine-learning framework with a
native C/C++ codebase and Python APIs for tensor operations, automatic
differentiation, and neural-network components. It is built for developers who
want to study, extend, and test a native machine-learning stack without
assuming full compatibility with a mature framework.

**Maturity: Alpha.** API coverage and validation vary by component and backend.
SneppX-ALG is not a drop-in replacement for PyTorch or TensorFlow and should
not be treated as a production security product.

## Install

```bash
python -m pip install sneppx-alg
```

For a source checkout, install the canonical Python distribution:

```bash
python -m pip install ./bindings/python
```

The latest published PyPI release is 1.5.2. Version 1.5.3 is an unpublished
release candidate and remains subject to the project's release checks.

## What is in the repository?

- **Tensor and autograd APIs:** Python tensor operations, reverse-mode
  differentiation, neural-network modules, losses, and optimizers. Feature
  coverage and numerical behavior are still being validated.
- **Native implementation:** C11/C++20 core libraries with optional native
  extensions and accelerator build options. A normal Python installation can
  use the NumPy-backed implementation.
- **Model and quantization utilities:** architecture configuration, conversion,
  and quantization tools. End-to-end pretrained-weight loading and some
  quantization/backend paths remain incomplete or experimental.
- **Repository tooling:** dependency-light codebase telemetry tools for
  scanning source files, measuring code, and checking documentation.
- **Security research components:** experimental cryptographic, memory,
  network, and verification code. These components are not independently
  audited or certified; post-quantum implementations are not fully
  interoperable with the finalized standards.

### Distributed training status

The Python distributed package currently supports single-process execution and
local dataset-index partitioning. Multi-rank process groups, collectives,
FSDP/ZeRO training, and multi-node execution are not validated as end-to-end
features. Unsupported multi-rank paths fail explicitly rather than silently
training with unsynchronized gradients. See the
[distributed support matrix](https://sneppx.github.io/sneppx-alg/latest/distributed-guide/).

## Build from source

Requirements include CMake 3.16+, a C11/C++20 toolchain, and Python for the
bindings.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

CUDA and other optional backends are not required for a basic CPU build. See
the [build guide](https://sneppx.github.io/sneppx-alg/latest/build/).

## Documentation

- [Documentation home](https://sneppx.github.io/sneppx-alg/)
- [Getting started](https://sneppx.github.io/sneppx-alg/latest/getting-started/)
- [Python API](https://sneppx.github.io/sneppx-alg/latest/api/python/)
- [Architecture and implementation status](docs/ARCHITECTURE.md)
- [Security limitations](docs/security_layers.md)
- [Framework roadmap](docs/roadmap/tensorflow-pytorch-parity.md)
- [Development guide](docs/DEVELOPMENT.md)

## Codebase telemetry

The repository includes a dependency-free CTM CLI:

```bash
sneppx-ctm .
sneppx-ctm . --format json --output ctm-report.json
```

See the [CTM documentation](ctm_tools/README.md).

## Security

Security-related code has different implementation and validation levels.
Do not rely on experimental components as security boundaries or for
compliance requirements. Review the security documentation and tests before
using these components.

To report a suspected vulnerability, follow [SECURITY.md](SECURITY.md) and use
private disclosure.

## Contributing

Bug reports, feature requests, and contributions are welcome. Read
[CONTRIBUTING.md](CONTRIBUTING.md) and the [development guide](docs/DEVELOPMENT.md)
before making changes.

- [Issues](https://github.com/SneppX/sneppx-alg/issues)
- [Discussions](https://github.com/SneppX/sneppx-alg/discussions)
- [Pull requests](https://github.com/SneppX/sneppx-alg/pulls)

## License

SneppX-ALG is released under the [MIT License](LICENSE).
