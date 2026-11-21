# Cookbook — Profiling

## 1. Time a region with the Profiler

**Intent:** Measure wall-clock time of a code block.

```python
from SneppX_ALG import Profiler, Timer

prof = Profiler(enabled=True)
with Timer(prof, "forward"):
    out = model(x)              # recorded under "forward"
with Timer(prof, "backward"):
    loss.backward()

prof.print_summary()
# Operation                          Calls    Total(s)   Avg(s)   Min(s)   Max(s)
# backward                                1     0.2456   0.2456   0.2456   0.2456
# forward                                 1     0.1123   0.1123   0.1123   0.1123
```

**Notes:** `Timer` is a context manager around `Profiler.record`. CPU-safe.

## 2. Decorate a function with @timeit

**Intent:** Profile a function call by name.

```python
from SneppX_ALG import get_profiler, timeit

prof = get_profiler(); prof.enabled = True

@timeit(prof)
def train_step(x, y):
    return (model(x) - y).pow(2).mean()

for batch in loader:
    train_step(*batch)
prof.print_summary()
```

**Notes:** `timeit` is the global-profiler equivalent of
`torch.profiler`'s `record_section`. The decorator appends to the global
`_GLOBAL_PROFILER`.

## 3. Track memory usage

**Intent:** Detect leaks and peak consumption.

```python
from SneppX_ALG import MemoryTracker

mt = MemoryTracker()
mt.start()
# ... allocate tensors / run forward ...
peak = mt.peak()              # bytes
mt.checkpoint("after-forward")  # named snapshot
mt.stop()
print(mt.report())            # text summary + growth between checkpoints
```

**Notes:** On CUDA, `MemoryTracker` queries `cudaMemGetInfo`; on CPU it reads
`/proc/self/status` (Linux) or `GetProcessMemoryInfo` (Windows). CPU-safe
fallback reads RSS.
