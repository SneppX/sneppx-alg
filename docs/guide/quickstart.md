# Quick Start

Build SNEPPX-Alg from source, run a core demo, then train and serve a model in
three minutes. Everything here runs on CPU — no GPU required for this path.

## 1. Build the core

```bash
git clone https://github.com/SneppX/sneppx-alg.git
cd sneppx-alg
cmake --preset release
cmake --build build --config Release -j$(nproc)
cd build && ctest --output-on-failure
```

On Windows, use the Ninja generator (per the project note) or VS 2022:

```powershell
git clone https://github.com/SneppX/sneppx-alg.git; cd sneppx-alg
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

> Each pipeline component has a self-test binary: `test_hss`, `test_ser`,
> `test_arc`, `test_npe`, `test_fm`. Run one to confirm the build.

## 2. Install the Python bindings

```bash
cd ..
export PYTHONPATH="$PWD/bindings/python"
pip install -e bindings/python
python -c "from SneppX_ALG import Tensor, Linear, AdamW, Trainer; print('ok')"
```

The bindings fall back to a pure-NumPy engine when no native `_arix_c`/`_SNEPPX_c`
extension is present, so imports succeed on any Python 3.11+ machine.

## 3. Run your first tensor op

```python
from SneppX_ALG import Tensor, SNEPPXDtype

a = Tensor([[1, 2, 3],
            [4, 5, 6]], dtype=SNEPPXDtype.FLOAT32)
b = Tensor([[0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6]], dtype=SNEPPXDtype.FLOAT32)

# HSS (state space) projection: y = a @ b  +  gelu(b @ a.T)
y = a.matmul(b).add(b.matmul(a).transpose(0,1).gelu())
print(y.shape)      # (2, 2)
print(y.to_numpy())
```

## 4. Train a tiny classifier end-to-end

```python
from SneppX_ALG import Tensor, Linear, AdamW, Trainer, SNEPPXDtype
import numpy as np

# 2-class synthetic batch
X = Tensor(np.random.randn(64, 16).astype('float32'))
y = Tensor(np.random.randint(0, 2, size=(64,)).astype('int64'))

model = Linear(16, 2, dtype=SNEPPXDtype.FLOAT32)
opt   = AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)

for step in range(200):
    logits = model(X)                      # HSS -> SER logits
    loss   = logits.sparse_softmax_cross_entropy(y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    if step % 50 == 0:
        print(f"step {step:3d}  loss={loss.item():.4f}")
```

The same pipeline applies to the other four stages:

| Stage | What you do | Entry point |
|-------|-------------|-------------|
| HSS   | feature / state encoding | `Tensor` ops, `Linear` |
| SER   | MoE routing / mixing | `MixtureOfExperts`, `Transformer` |
| ARC   | adversarial guard, safety | `sneppx-analyze` CLI |
| NPE   | neural VM / bytecode | `sneppx-train --arch npe*` |
| FM    | memory / serving | `sneppx-serve` |

## 5. Serve a quantized model

```bash
sneppx-quantize  --model llama-3-8b \
                 --mode int4_awq   \
                 --out model.int4.safetensors

sneppx-serve     --model model.int4.safetensors \
                 --port 8000 \
                 --api-key "$(pass show sneppx-serve-key)"
```

```python
from SneppX_ALG.interface_bindings.serving_client import SNEPPXClient
c = SNEPPXClient("http://127.0.0.1:8000", api_key="...")
c.generate("Explain the S0-S9 security stack in two sentences.", max_tokens=64)
```

## 6. Go distributed (2 GPUs/CPU ranks)

```python
from SneppX_ALG.interface_bindings.distributed import init_distributed
from SneppX_ALG import Trainer
init_distributed(backend="nccl")        # or "gloo" / "mpi" on CPU
Trainer(...).fit(train_loader, epochs=3, parallel={"tp":2, "dp":1})
```

See `docs/tutorials/` for full notebooks on classification, generation, RLHF
fine-tuning, quantization+serving, MoE routing, security scanning, profiling,
data pipelines, and model conversion.
