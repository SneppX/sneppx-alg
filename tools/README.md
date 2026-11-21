# Tools Directory

This directory contains standalone utility programs for development and analysis.

| Tool | Description | Status |
|------|-------------|--------|
| `performance_benchmarking_suite/` | Performance benchmarks for all components | Stable |
| `diagnostic_profiler/` | Runtime profiling and tracing | Planned |

## Building

Tools are built automatically as part of the main CMake build when `SNEPPX_BUILD_TOOLS=ON`.

## Usage

```bash
# Run benchmarks
./tools/performance_benchmarking_suite/bench_tensor
./tools/performance_benchmarking_suite/bench_autodiff
```
