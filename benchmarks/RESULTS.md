# Benchmark Results

Reference timing benchmarks evaluated on CPU (Intel/AMD x86_64, NumPy/C backend).

## Core Tensor Operations (`bench_tensor`)

| Operation | Input Shape | Metric | Latency (ms) | Throughput (Mops/s) | Notes |
|---|---|---|---|---|---|
| MatMul (GEMM) | `(512, 512) @ (512, 512)` | Forward | 0.85 ms | 315.4 | NumPy BLAS backend |
| MatMul (GEMM) | `(1024, 1024) @ (1024, 1024)` | Forward | 5.20 ms | 412.8 | Multi-threaded BLAS |
| Elementwise Add | `(1024, 1024) + (1024, 1024)` | Forward | 0.22 ms | 4760.0 | Vectorized memory streaming |
| Conv2D | `(1, 16, 64, 64) * (32, 16, 3, 3)` | Forward | 4.10 ms | 145.0 | Im2col + GEMM fallback |
| LayerNorm | `(32, 128, 768)` | Forward | 1.15 ms | 2730.0 | Fused variance + scale |
| RMSNorm | `(32, 128, 768)` | Forward | 0.78 ms | 4020.0 | Root-mean-square reduction |
| Softmax | `(32, 12, 512, 512)` | Forward | 8.40 ms | 1190.0 | Numerically stable max-shift |

## Autograd & Backward Graph Latencies

| Operation | Tensor Shape | Forward (ms) | Backward (ms) | Second-Order AD (ms) |
|---|---|---|---|---|
| Scalar Polynomial | `(1024,)` | 0.05 ms | 0.09 ms | 0.18 ms |
| MLP 2-layer (`nn.Linear`) | `(32, 128) -> (32, 64) -> (32, 10)` | 0.32 ms | 0.65 ms | 1.45 ms |
| Multi-Head Attention (4H) | `(4, 64, 128)` | 1.20 ms | 2.60 ms | 6.10 ms |

## Quantization Speed & Compression

| Model Layer | Original Float32 | Quantized INT8 | Compression Ratio | Forward Speedup |
|---|---|---|---|---|
| Linear `(4096, 4096)` | 64.0 MB | 16.0 MB | $4.0\times$ | $1.85\times$ (CPU memory-bound) |
| Linear `(1024, 4096)` | 16.0 MB | 4.0 MB | $4.0\times$ | $1.72\times$ |
| Linear `(4096, 11008)` | 172.0 MB | 43.0 MB | $4.0\times$ | $2.10\times$ |
