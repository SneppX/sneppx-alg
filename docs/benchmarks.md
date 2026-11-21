# Benchmarks

SNEPPX-Alg ships a small C benchmark harness (`tests/benchmark/`) that measures
the native tensor engine. Results below are from `benchmarks/baseline.json`
(captured on the reference machine; re-run locally with `cmake -B build
-DSNEPPX_BUILD_TESTS=ON && cmake --build build && ./build/tests/benchmark/bench_tensor`).

All times are **median of N runs** on a single thread (CPU), in milliseconds.

## Tensor create

| Shape | Time (ms) | min / max (ms) | runs |
|---|---|---|---|
| `[16, 16]` | 0.001 | 0.001 / 0.060 | 10000 |
| `[256, 256]` | 0.158 | 0.131 / 1.050 | 1000 |
| `[1024, 1024]` | 4.871 | 3.739 / 13.801 | 100 |

## Fill

| Shape | op | Time (ms) | min / max (ms) | runs |
|---|---|---|---|---|
| `[256, 256]` | `fill_f32` | 0.039 | 0.029 / 2.469 | 1000 |
| `[1024, 1024]` | `fill_f32` | 0.979 | 0.561 / 2.736 | 100 |

## Element-wise (shape `[256, 256]`)

| op | Time (ms) | min / max (ms) | runs |
|---|---|---|---|
| `add` | 0.200 | 0.166 / 2.022 | 1000 |
| `mul` | 0.206 | 0.166 / 1.740 | 1000 |

## Matmul

| Shapes | Time (ms) | min / max (ms) | runs |
|---|---|---|---|
| `[128,128] × [128,128]` | 3.236 | 2.485 / 5.544 | 500 |
| `[64,256] × [256,64]` | 1.552 | 1.254 / 4.055 | 500 |

## Slice / reshape (shape `[128, 128]`)

| op | Time (ms) | min / max (ms) | runs |
|---|---|---|---|
| `slice dim0 0:64` | 0.000 | 0.000 / 0.001 | 1000 |
| `reshape [16384]` | 0.000 | 0.000 / 0.001 | 1000 |
| `transpose 0,1` | 0.001 | 0.000 / 0.001 | 1000 |

## Unary (shape `[256, 256]`)

| op | Time (ms) | min / max (ms) | runs |
|---|---|---|---|
| `relu` | 0.343 | 0.216 / 1.847 | 500 |
| `sigmoid` | 1.334 | 0.925 / 4.387 | 500 |
| `softmax dim=1` | 1.378 | 0.994 / 3.567 | 500 |
| `sqrt` | 0.606 | 0.378 / 2.750 | 500 |

## Reduction (shape `[256, 256]`)

| op | Time (ms) | min / max (ms) | runs |
|---|---|---|---|
| `sum dim=0` | 0.061 | 0.044 / 0.806 | 500 |
| `mean dim=1` | 0.273 | 0.241 / 1.678 | 500 |

## Notes

- Single-threaded CPU reference path; opt-in backends (Vulkan, TPU, Metal, oneAPI)
  report their own numbers when enabled.
- The C tensor engine is the hot path; the Python `SneppX_ALG` package uses a
  NumPy-backed fallback for ergonomics and is not benchmarked here.
- Reproduce with `cmake -B build -DSNEPPX_BUILD_TESTS=ON && cmake --build build
  --config Release && ./build/tests/benchmark/bench_tensor`.
