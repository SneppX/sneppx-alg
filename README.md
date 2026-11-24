# SneppX-ALG

[![PyPI version](https://img.shields.io/pypi/v/sneppx-alg)](https://pypi.org/project/sneppx-alg/)
[![Python versions](https://img.shields.io/pypi/pyversions/sneppx-alg)](https://pypi.org/project/sneppx-alg/)
[![CI](https://github.com/SneppX/sneppx-alg/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SneppX/sneppx-alg/actions/workflows/ci.yml)
[![MIT License](https://img.shields.io/github/license/SneppX/sneppx-alg)](LICENSE)

SneppX-ALG is an open-source AI framework engineered for performance, composability, and security from the ground up. A native C/C++ core powers a full tensor and automatic-differentiation engine, while Python bindings keep research and production workflows approachable. Neural-network building blocks, distributed-training primitives, quantization, and a layered post-quantum security stack make SneppX-ALG a serious foundation for next-generation AI systems.

**Project maturity: Alpha.** Core paths are tested and evolving; coverage varies across components and optional backends. Check the relevant documentation and tests before depending on a specific feature. SneppX-ALG is not yet a drop-in replacement for TensorFlow or PyTorch.

## Install

Install the Python package from PyPI:

```bash
python -m pip install sneppx-alg
```

For a source checkout, install the bindings from the repository:

```bash
python -m pip install ./bindings/python
```

## Features

- Tensor operations and automatic differentiation.
- Neural-network layers, losses, and optimizers.
- Transformer and model components.
- Quantization and distributed-computing components, with varying maturity.
- Python bindings and developer utilities.
- Security, monitoring, cryptography, and network-security components.

See the [TensorFlow and PyTorch parity roadmap](docs/roadmap/tensorflow-pytorch-parity.md) for current gaps and compatibility goals.

## Build from source

Requirements include CMake 3.16 or newer, a C11 and C++17-compatible toolchain, and Python with NumPy for Python support.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

CUDA and other optional backends are not required for a basic CPU build. See the [build guide](docs/build.md) for configuration options.

## Documentation

- [Live documentation](https://sneppx.github.io/sneppx-alg/)
- [Getting started](https://sneppx.github.io/sneppx-alg/latest/getting-started/)
- [Python API reference](https://sneppx.github.io/sneppx-alg/latest/api/python/)
- [Architecture](docs/ARCHITECTURE.md)
- [Security implementation notes](docs/security_layers.md)
- [Development guide](docs/DEVELOPMENT.md)

The [SneppX-OV project website](https://sneppx.github.io/sneppxov/) provides source-linked catalogues and a project overview.

## Codebase telemetry

The repository ships a dependency-free codebase telemetry & metrics (CTM) CLI:

```bash
sneppx-ctm .
sneppx-ctm . --format json --output ctm-report.json
```

It reports per-language LOC/bytes, skips generated and binary content, and
exposes console scripts such as `sneppx-ctm-functions`, `sneppx-ctm-large`, and
`sneppx-ctm-todo`. See the [CTM tools documentation](docs/tools/ctm-tools.md)
and the [CTM guide](ctm_tools/README.md) for the full surface.

## Security

Security-related components have different implementation and validation levels. They have not all received independent security audits or certification. Review the [security documentation](docs/security_layers.md) and the relevant tests before deployment.

To report a suspected vulnerability, follow the [security policy](SECURITY.md) and use private disclosure.

## Contributing

Bug reports, feature requests, and contributions are welcome. Read the [development guide](docs/DEVELOPMENT.md) before making changes.

- [Issues](https://github.com/SneppX/sneppx-alg/issues)
- [Discussions](https://github.com/SneppX/sneppx-alg/discussions)
- [Contribution guide](CONTRIBUTING.md)

## License

Released under the [MIT License](LICENSE).
