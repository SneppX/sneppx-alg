# SneppX-ALG

SneppX-ALG is an experimental AI and machine-learning framework with a native C/C++ core and Python APIs. It includes tensor operations, automatic differentiation, neural-network components, model utilities, and optional backend integrations.

The project is under active development. Feature coverage and maturity vary by subsystem. SneppX-ALG is not currently a drop-in replacement for PyTorch or TensorFlow.

## Features

- Native tensor and automatic differentiation APIs
- Neural-network layers, loss functions, and optimizers
- Transformer and other model components
- Model configuration and local checkpoint utilities
- Quantization and distributed-computing experiments
- Optional accelerator and serving backends
- Experimental security and cryptography components
- Python bindings and developer tools

See the [TensorFlow and PyTorch parity roadmap](docs/roadmap/tensorflow-pytorch-parity.md) for compatibility goals and known gaps.

## Requirements

- CMake 3.16 or newer
- A C11 and C++17 compatible compiler
- Python and NumPy for the Python package

## Build

Configure and build the native library:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Run the test suite:

```bash
cd build
ctest -C Release --output-on-failure
```

Optional backends can be enabled through CMake options. See the [quick-start guide](QUICKSTART.md) for details.

## Python

Python examples are available in [`examples/python_api/`](examples/python_api/). For example:

```bash
PYTHONPATH=bindings/python python examples/python_api/ex_01_linear_regression.py
```

The Python package provides a NumPy-backed tensor and neural-network API. See the [Python documentation](docs/python_features.md) for details.

## Documentation

- [Documentation index](docs/index.md)
- [Quick start](QUICKSTART.md)
- [Architecture](docs/ARCHITECTURE.md)
- [API reference](docs/API.md)
- [Python features](docs/python_features.md)
- [Security components](docs/security_layers.md)
- [TensorFlow and PyTorch parity roadmap](docs/roadmap/tensorflow-pytorch-parity.md)

## Contributing

Contributions, bug reports, and feature requests are welcome. Please read the [development guide](docs/DEVELOPMENT.md) before making changes.

- [Issues](https://github.com/SneppX/sneppx-alg/issues)
- [Discussions](https://github.com/SneppX/sneppx-alg/discussions)

## License

SneppX-ALG is licensed under the [MIT License](LICENSE).
