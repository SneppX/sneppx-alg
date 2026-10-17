# Tutorial — Profiling & Benchmarks

**Notebook:** [`profiling_benchmarks.ipynb`](https://github.com/SneppX/sneppx-alg/blob/main/notebooks/profiling_benchmarks.ipynb)
([download](https://github.com/SneppX/sneppx-alg/blob/main/notebooks/profiling_benchmarks.ipynb))

## What you'll build

Instrument a forward+backward pass with the `Profiler`/`Timer`, track peak
memory with `MemoryTracker`, and run a regression benchmark with the
`sneppx-bench` CLI.

## Setup

```powershell
$env:PYTHONPATH = "bindings/python"
```

```python
import numpy as np, time
from SneppX_ALG import (
    Transformer, Linear, GELU, Tensor, AdamW,
    Profiler, Timer, MemoryTracker, get_profiler, timeit,
)
HAS_C = __import__("SneppX_ALG")._HAS_C_BACKEND
```

## 1. Profile a region

```python
model = Transformer(vocab_size=500, dim=256, num_heads=4, num_layers=4, ffn_dim=1024, max_seq_len=64)
x = Tensor.randn((8, 64))

prof = Profiler(enabled=True)
with Timer(prof, "forward"):
    out = model(x)
with Timer(prof, "loss"):
    loss = out.data.mean()      # pure-NumPy surrogate (no C backend needed)
prof.print_summary()
```

## 2. Global profiler + @timeit

```python
g = get_profiler(); g.enabled = True
g.reset()

@timeit(g)
def step(x):
    return model(x)

for _ in range(5):
    step(x)
print(g.to_json())             # structured JSON for dashboards
```

## 3. Memory tracking

```python
mt = MemoryTracker()
mt.start()
for _ in range(10):
    _ = model(x)
peak = mt.peak(); mt.checkpoint("after-10")
mt.stop()
print("peak bytes:", peak)
```

## 4. Benchmark with sneppx-bench (regression)

```bash
sneppx-bench run tests/python/test_tensor.py \
    --repeat 10 --save results/bench_$(git rev-parse --short HEAD).json
# Compare two runs:
sneppx-bench compare results/bench_a.json results/bench_b.json
```

Flags: `--repeat N`, `--warmup N`, `--filter <regex>`, `--save PATH`,
`--unit <ms|s|us>`. The tool tracks regressions against a local history under
`~/.sneppx/`.

## 5. CPU timing (no C backend) — fallback

```python
import numpy as np
from SneppX_ALG import Tensor

# When the C backend is absent we still get deterministic shapes; just time it.
t0 = time.perf_counter()
for _ in range(20):
    y = model(x)
dt = (time.perf_counter() - t0) / 20
print(f"fwd {dt*1000:.2f} ms")
```

## Key takeaways

- `Profiler` aggregates by named region; `Timer` is the context-manager form.
- `get_profiler()` returns the **global** profiler reused by `@timeit`.
- On CUDA, NVTX markers (`SNEPPX_USE_NVTX`) are emitted automatically — import
  into `nsys` / `nvtx` UIs.
- `sneppx-bench` stores results tagged with the git SHA for regression diffs.
- The C core also ships `kernel/profiler.c` (`SNEPPX_Profiler`) for C-side
  instrumentation; the Python `Profiler` is its mirror.

## Next steps

- Profile the quantized model — see
  [Quantization & Serving](quantization_serving.md).
- Read `docs/PROGRESS_TENSOR.md` and `docs/BENCHMARKS.md` for full perf tables.
