---
description: "SneppX-ALG documentation for the native C/C++ project and Python bindings, including installation, tensor operations, autograd, architecture, APIs, and project limitations."
---

# SneppX-ALG documentation

SneppX-ALG is an open-source AI and machine-learning project with a native C/C++ codebase and Python bindings. This documentation explains how to install and build the available components, use the documented APIs, understand the repository, and verify implementation status.

The project is in active development. The Python API, native code, optional backends, and security components do not all have the same implementation or test coverage. An architecture proposal or catalogue entry is not, by itself, evidence that a feature is complete or production-ready.

## Start here

- [Getting started](getting-started.md): install the Python package and run a small tensor example.
- [Source build](guide/quickstart.md): configure and build the native project.
- [Python API](api/python.md): public Python modules, examples, and backend notes.
- [Autograd guide](autograd-guide.md): reverse-mode and higher-order differentiation.
- [API explorer](api-explorer.md): browse the documented API.
- [Contributing](contributing.md): development setup and contribution workflow.

## Project areas

| Area | Documentation |
|---|---|
| Tensors and automatic differentiation | [Python API](api/python.md), [Autograd guide](autograd-guide.md) |
| Neural-network components | [Cookbook](cookbook/index.md) |
| Model and inference utilities | [Model zoo guide](guide/model_zoo.md), [Serving guide](guide/serving.md) |
| Quantization and distributed computing | [Quantization cookbook](cookbook/quantization.md), [Distributed guide](distributed-guide.md) |
| Security components | [Security overview](security.md), [security-layer notes](security_layers.md) |
| Native architecture | [Architecture overview](ARCHITECTURE.md), [pipeline notes](architecture/5-pipeline-architecture.md) |
| Repository tooling | [CTM tools](ctm_tools/README.md), [development workflow](DEVELOPMENT.md) |

These pages describe different parts of the repository. Check the relevant implementation and tests before depending on a specific backend, operation, or security control.

## Install the Python bindings

Install the package from its canonical Python project directory in a source checkout:

```bash
python -m pip install ./bindings/python
```

Or use the bindings directly from the repository:

```bash
PYTHONPATH=bindings/python python -c "from SneppX_ALG import Tensor; print(Tensor([1, 2, 3]))"
```

See [Python packaging](dev/python-packaging.md) for the distinction between the development workspace and the distributable package.

## Build the native project

The native build uses CMake and a compatible C/C++ toolchain:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build --output-on-failure
```

Build options and platform-specific requirements are documented in the [build guide](build.md).

## Before using it in a project

- Check whether the component you need is implemented and covered by tests.
- Confirm whether the Python path uses NumPy or an available native backend.
- Review known limitations in the relevant API and architecture pages.
- Treat security features according to their documented implementation and independent validation status.
- Do not assume compatibility with PyTorch, TensorFlow, JAX, or their model ecosystems.

## Project information

- [Source code and issue tracker](https://github.com/SneppX/sneppx-alg)
- [Release notes](changelog/index.md)
- [Framework parity roadmap](roadmap/tensorflow-pytorch-parity.md)
- [License](https://github.com/SneppX/sneppx-alg/blob/main/LICENSE)
