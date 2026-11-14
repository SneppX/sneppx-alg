# SneppX-ALG

SneppX-ALG is an open-source AI and machine-learning framework built around a native C/C++ core and Python APIs.

It provides tensor operations, automatic differentiation, neural-network components, model utilities, and security tools.

## Features

- Tensor operations and automatic differentiation
- Neural-network layers, losses, and optimizers
- Transformer and model components
- Quantization and distributed-computing tools
- Python bindings and developer utilities
- Security, monitoring, and firewall components

Feature coverage varies by subsystem. SneppX-ALG is not a drop-in replacement for TensorFlow or PyTorch. See the [parity roadmap](docs/roadmap/tensorflow-pytorch-parity.md).

## Requirements

- CMake 3.16 or newer
- A C11 and C++17 compatible compiler
- Python and NumPy for Python support

## Build

Configure the project with `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`.

Build with `cmake --build build --config Release`.

Run native tests with `ctest --test-dir build -C Release --output-on-failure`.

For optional backends and additional setup, see the [quick-start guide](QUICKSTART.md).

## Python

The Python package is located in `bindings/python/`.

Run the linear regression example with `PYTHONPATH=bindings/python python examples/python_api/ex_01_linear_regression.py`.

## Security

The project includes cryptography, memory-hardening, monitoring, network-security, audit, and firewall components.

Security features have different implementation and validation levels. Review the [security documentation](docs/security_layers.md) before deployment.

## Documentation

- [Documentation index](docs/index.md)
- [Quick start](QUICKSTART.md)
- [Architecture](docs/ARCHITECTURE.md)
- [API reference](docs/API.md)
- [Python features](docs/python_features.md)
- [Security](docs/security_layers.md)
- [Development guide](docs/DEVELOPMENT.md)

## Contributing

Bug reports, feature requests, and contributions are welcome. Read the [development guide](docs/DEVELOPMENT.md) before contributing.

- [Issues](https://github.com/SneppX/sneppx-alg/issues)
- [Discussions](https://github.com/SneppX/sneppx-alg/discussions)

## License

Released under the [MIT License](LICENSE).
