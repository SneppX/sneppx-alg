# SneppX-ALG

SneppX-ALG is an open-source AI and machine-learning framework with a native C/C++ core and Python APIs. It brings together tensor operations, automatic differentiation, neural-network components, model utilities, and security tooling.

## Features

- Tensor and automatic differentiation APIs
- Neural-network layers, losses, and optimizers
- Transformer and other model components
- Quantization and distributed-computing components
- Python bindings and developer utilities
- Security modules for cryptography, monitoring, network controls, and firewall policies

Feature coverage varies across subsystems. TensorFlow and PyTorch compatibility is incomplete; SneppX-ALG is not a drop-in replacement for either framework. See the [parity roadmap](docs/roadmap/tensorflow-pytorch-parity.md).

## Requirements

- CMake 3.16 or newer
- A C11 and C++17 compatible compiler
- Python and NumPy for the Python package

## Build

Build the native library:

\`\`\`bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
\`\`\`

Run the native tests:

\`\`\`bash
ctest --test-dir build -C Release --output-on-failure
\`\`\`

Optional backends can be enabled with their CMake options. See the [quick-start guide](QUICKSTART.md).

## Python

The Python package is in \`bindings/python/\`. Example:

\`\`\`python
import numpy as np
from SneppX_ALG import Tensor

a = Tensor(np.array([[1.0, 2.0]], dtype=np.float32))
b = Tensor(np.array([[3.0], [4.0]], dtype=np.float32))
print((a @ b).data)
\`\`\`

Run examples with:

\`\`\`bash
PYTHONPATH=bindings/python python examples/python_api/ex_01_linear_regression.py
\`\`\`

## Security

The project includes cryptography, memory-hardening, monitoring, network-security, audit, and firewall components. The Python firewall supports request filtering, rate limits, and configurable default-deny route policies.

Security features have different implementation and validation levels. They are not a guarantee of protection, a substitute for deployment security controls, or a claim of independent security certification. Review [security documentation](docs/security_layers.md) before deployment.

## Documentation

- [Documentation index](docs/index.md)
- [Quick start](QUICKSTART.md)
- [Architecture](docs/ARCHITECTURE.md)
- [API reference](docs/API.md)
- [Python features](docs/python_features.md)
- [Security](docs/security_layers.md)
- [Development guide](docs/DEVELOPMENT.md)

## Contributing

Contributions, bug reports, and feature requests are welcome. Please read the [development guide](docs/DEVELOPMENT.md) before submitting changes.

- [Issues](https://github.com/SneppX/sneppx-alg/issues)
- [Discussions](https://github.com/SneppX/sneppx-alg/discussions)

## License

SneppX-ALG is released under the [MIT License](LICENSE).
