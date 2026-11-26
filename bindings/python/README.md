# SneppX-ALG

SneppX-ALG is an experimental open-source machine-learning framework with a
native C/C++ codebase and Python APIs for tensor operations, automatic
differentiation, and neural-network components.

**Project maturity: Alpha.** API coverage and validation vary by component and
backend. This package is not a drop-in replacement for PyTorch or TensorFlow.

## Install

```bash
python -m pip install sneppx-alg
```

## Python quick start

```python
import numpy as np
from SneppX_ALG import Tensor

x = Tensor(np.array([2.0, 3.0], dtype=np.float32), requires_grad=True)
loss = (x * x).sum()
loss.backward()

print("loss:", loss.item())
print("gradient:", x.grad)
```

## Package scope

- Python tensor operations and autograd APIs, backed by NumPy when the native
  extension is unavailable.
- Neural-network modules, loss functions, optimizers, and selected training
  utilities.
- Architecture configuration, conversion, quantization, and repository tools;
  support varies by feature and backend.
- Optional native extensions and accelerator components that require separate
  builds and matching hardware.

## Important limitations

- Not every tensor operator, gradient, dtype, or backend has complete
  conformance coverage.
- Multi-rank process groups and collectives are not implemented in the current
  Python backend. Multi-rank distributed training, FSDP, and ZeRO must not be
  treated as supported.
- Model-zoo from_pretrained helpers currently resolve architecture
  configurations/skeletons; they do not provide general pretrained-weight
  downloading.
- Security components are experimental and have not received independent
  certification or a complete external audit. Post-quantum implementations
  have documented interoperability gaps and must not be marketed as FIPS
  validated.

See the [documentation](https://sneppx.github.io/sneppx-alg/) for installation,
API details, feature status, and the [distributed support matrix](https://sneppx.github.io/sneppx-alg/latest/distributed-guide/).

## Project links

- [Source repository](https://github.com/SneppX/sneppx-alg)
- [Documentation](https://sneppx.github.io/sneppx-alg/)
- [Issue tracker](https://github.com/SneppX/sneppx-alg/issues)
- [Discussions](https://github.com/SneppX/sneppx-alg/discussions)
- [License](https://github.com/SneppX/sneppx-alg/blob/main/LICENSE)
