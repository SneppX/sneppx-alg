# Migrating from JAX to SNEPPX-Algo

JAX developers will recognize the SNEPPX functional transforms API
(`interface_bindings/jit.py`): `jit`, `grad`, `value_and_grad`, `jacobian`,
`hessian`, `vmap`, plus a `Tracer`/`Trace` IR. These are the closest analogs
to `jax.jit`, `jax.grad`, etc., and they delegate to the same C backend
`Tensor` autograd tape.

## Installation

```bash
pip install sneppx-alg
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
$env:PYTHONPATH = "bindings/python"
```

## Functional transforms

| JAX | SNEPPX-Algo |
|-----|-------------|
| `jax.jit(f)` | `jit(f)` |
| `jax.grad(f)` | `grad(f)` |
| `jax.value_and_grad(f)` | `value_and_grad(f)` |
| `jax.jacfwd(f)` / `jax.jacrev(f)` | `jacobian(f)` |
| `jax.hessian(f)` | `hessian(f)` |
| `jax.vmap(f)` | `vmap(f)` |
| `jax.make_jaxpr` | `trace(f)` / `Tracer` |

```python
# JAX
import jax, jax.numpy as jnp
def loss_fn(params, x, y):
    preds = jax.vmap(lambda p, xi: jnp.dot(p, xi))(params, x)
    return jnp.mean((preds - y) ** 2)
grads = jax.grad(loss_fn)(params, x, y)

# SNEPPX
from SneppX_ALG import Tensor
from SneppX_ALG.interface_bindings.jit import jit, grad, vmap

def loss_fn(params, x, y):
    preds = vmap(lambda p, xi: (p @ xi))(params, x)
    return ((preds - y) ** 2).mean()
grads = grad(loss_fn)(params, x, y)
```

## Arrays & dtypes

| JAX | SNEPPX-Algo |
|-----|-------------|
| `jax.numpy` / `jax.Array` | `SneppX_ALG.Tensor` |
| `jnp.float32` | `"float32"` / `Dtype.FLOAT32` |
| `jnp.ones((4,8))` | `Tensor.ones((4, 8))` |
| `jnp.zeros((4,8))` | `Tensor.zeros((4, 8))` |
| `jnp.dot(a, b)` | `a @ b` (`__matmul__`) |
| `x.device()` | `x.device` (string) |
| `x.reshape(...)` | `x.reshape(...)` |
| `x.block_until_ready()` | sync is implicit on CPU; on CUDA use `synchronize()` |

```python
from SneppX_ALG import Tensor, Dtype
a = Tensor.ones((4, 8), dtype=Dtype.FLOAT32)
b = Tensor.randn((8, 16))
c = a @ b                       # shape (4, 16)
```

## Neural networks

SNEPPX does **not** ship a `jax.nn`-style module system tied to pytrees;
instead use the imperative `nn.Module` API or the Keras-compatible API.

| JAX ecosystem | SNEPPX-Algo |
|----------------|-------------|
| `flax.linen.Dense` | `Dense` (Keras API) / `Linear` (imperative) |
| `flax.linen.relu` / `sigmoid` / `gelu` | `GELU`, `SiLU`, `ReLU`, `Sigmoid` |
| `optax.adamw` | `AdamW` |
| `optax.piecewise_constant_schedule` | `StepLR` / `LinearWarmupCosineDecay` |
| `chex` | `Tensor` overloads (arithmetic, indexing, `reshape`) |

```python
from SneppX_ALG import Sequential, Dense, GELU, AdamW, Linear
model = Sequential(Dense(64, activation="relu"), Dense(10))
opt   = AdamW(model.parameters(), lr=1e-3)
```

## PRNG

| JAX | SNEPPX-Algo |
|-----|-------------|
| `jax.random.PRNGKey(0)` | `seed=` on `ModelConfig` / `HSSModel(..., seed=0)` |
| `jax.random.split` | handled internally per-model constructor |
| `jax.random.normal(key, shape)` | `Tensor.randn(shape)` |

## Devices & sharding

| JAX | SNEPPX-Algo |
|-----|-------------|
| `jax.devices()` | `cuda_is_available()`, `device_count()` |
| `jax.device_count()` | `device_count()` |
| `jax.lax.with_sharding_constraint` | `DistributedWrapper` + `DistributedSampler` |
| `jax.experimental.pjit` | `init_process_group(backend="nccl")` + `DistributedWrapper` |

```python
from SneppX_ALG import cuda_is_available, device_count, init_process_group, DistributedWrapper
if cuda_is_available():
    print("GPUs:", device_count())
init_process_group(backend="nccl")
model = DistributedWrapper(model, device="cuda")
```

## Gotchas

- SNEPPX `grad` is **reverse-mode** (VJP tape) like `jax.grad`, but operates on
  `Tensor` objects rather than pytrees of ndarrays.
- `vmap` is a thin NumPy-broadcasting shim — for production throughput, use the
  C/CUDA kernels (enable with `SNEPPX_BUILD_CUDA=ON`).
- PyTree registration is not supported; pass flat `Tensor` lists to `grad`/`jit`.
- Without the C backend, `grad`/`jit` fall back to a slow NumPy interpreter
  (functional, but not optimized) and `Trainer.fit` raises
  `RuntimeError: C backend not available`.
