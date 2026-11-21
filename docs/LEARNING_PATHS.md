# Learning Paths

Track-specific roadmaps for progressing from Explorer to Senior Contributor. Each path is organized by month with concrete goals.

---

## Python Track

*Prerequisites: 1+ year Python, familiarity with NumPy*

| Month | Goal | Deliverable |
|-------|------|-------------|
| 1 | Understand project structure, run tests | Build from source, run `ctest`, run Python test suite |
| 2 | Fix a documentation or typo bug | Submit a PR updating docstrings or README |
| 3 | Write a Python test | Add test coverage for an existing feature |
| 4 | Add a simple CLI flag or option | Modify CLI tool, update docs, write test |
| 5 | Implement a new Python binding | Wrap a C function in `bindings/python/` |
| 6 | Own a Python module | Take responsibility for one module in `bindings/python/SneppX_ALG/` |
| 7+ | Review Python PRs | Provide code reviews as a Senior contributor |

### Key Resources

- `docs/STYLE_GUIDE.md` — Python section (PEP 8, Black, Google-style docstrings)
- `bindings/python/` — existing bindings as reference
- `tests/python/` — test patterns and fixtures
- `docs/API.md` — full API reference

---

## C/C++ Core Track

*Prerequisites: 2+ years C, familiarity with CMake, memory management*

| Month | Goal | Deliverable |
|-------|------|-------------|
| 1 | Build from source, read ARCHITECTURE.md | Understand kernel/ layout, build all targets |
| 2 | Fix a kernel bug with a test | Identify bug in kernel/, write reproduction test, submit fix |
| 3 | Implement a new tensor operation | Add function to `kernel/tensor/`, expose in header, test in C and Python |
| 4 | Document a kernel module | Write developer docs for `kernel/autodiff/` or `kernel/memory/` |
| 5 | Optimize an existing kernel | Profile, optimize, measure speedup, document approach |
| 6 | Own a kernel module | Take ownership of `kernel/tensor/` or `kernel/optimizer/` |
| 7+ | Review C PRs, mentor | Provide thorough code reviews emphasizing memory safety |

### Key Resources

- `docs/ARCHITECTURE.md` — full system design
- `docs/STYLE_GUIDE.md` — C style rules
- `docs/DEVELOPMENT.md` — build and testing workflow
- `include/neural_core/kernel/` — public API headers
- `kernel/` — existing implementations

---

## Security Track

*Prerequisites: 3+ years in cryptography/security engineering*

| Month | Goal | Deliverable |
|-------|------|-------------|
| 1 | Build from source, read security/ READMEs | Understand S0-S9 layer architecture, run crypto tests |
| 2 | Implement test vectors for an existing cipher | Add Known Answer Tests (KATs) for Kyber, Dilithium, or SPHINCS+ |
| 3 | Audit a security module | Review a module for constant-time correctness, submit findings |
| 4 | Add a new cipher wrapper | Wrap a standard algorithm, implement interface in `security/crypto/` |
| 5 | Conduct a full threat model | Write STRIDE analysis for a subsystem |
| 6 | Own an S-layer | Take responsibility for one security layer (S0–S9) |
| 7+ | Mentor security contributors | Guide contributors through crypto-audit process |

### Key Resources

- `docs/ARCHITECTURE.md` (Security Layers section)
- `docs/security.md` — security overview
- `SECURITY.md` — vulnerability reporting
- `security/` — all S-layer implementations
- `tests/security/` — test suites

---

## CUDA Track

*Prerequisites: 2+ years CUDA C++, GPU architecture knowledge*

| Month | Goal | Deliverable |
|-------|------|-------------|
| 1 | Build CUDA backend, read CUDA kernels | `cmake -B build -G Ninja -DSNEPPX_BUILD_CUDA=ON`, run cuda_test_suite |
| 2 | Optimize an existing CUDA kernel | Profile with ncu, optimize occupancy/memory, measure speedup |
| 3 | Implement a new CUDA kernel | Write a fused kernel (e.g., bias+activation+GEMM), benchmark |
| 4 | Add CUDA kernel documentation | Document kernel strategy, grid/block sizing, shared mem usage |
| 5 | Extend test coverage for CUDA | Add edge-case tests (empty tensors, alignment edge cases) |
| 6 | Own a CUDA module | Take ownership of `kernel/cuda/attention_cuda.cu` or similar |
| 7+ | Review CUDA PRs, mentor | Provide reviews focused on warp divergence, memory coalescing |

### Key Resources

- `docs/ARCHITECTURE.md` (CUDA Accelerated Optimization section)
- `kernel/cuda/` — existing CUDA kernel implementations
- `tests/cuda_test_suite.cu` — CUDA test patterns
- NVIDIA CUDA Best Practices Guide

---

## Algorithms Track

*Prerequisites: ML theory, linear algebra, probability, statistics*

| Month | Goal | Deliverable |
|-------|------|-------------|
| 1 | Understand algorithm pipeline | Read `docs/ARCHITECTURE.md` (HSS, SER, ARC, NPE, FM sections) |
| 2 | Run algorithm tests | Execute all algorithm tests, understand failure modes |
| 3 | Implement a new feature for an algorithm | Add a new routing strategy to SER or a new defense to ARC |
| 4 | Write algorithm documentation | Document mathematical formulation, parameter choices, trade-offs |
| 5 | Benchmark algorithm performance | Create benchmark scripts, compare against baselines |
| 6 | Own an algorithm module | Take ownership of HSS, SER, ARC, NPE, or FM |
| 7+ | Propose new algorithm extensions | Write design docs for new algorithm variants |

### Key Resources

- `docs/ARCHITECTURE.md` — full algorithm descriptions
- `algorithms/` — source code for each component
- `docs/DESIGN.md` — design decisions and trade-offs
- `docs/GLOSSARY.md` — terminology reference

---

## Documentation Track

*Prerequisites: Technical writing ability, English proficiency*

| Month | Goal | Deliverable |
|-------|------|-------------|
| 1 | Read all existing documentation | Identify gaps, outdated sections, missing examples |
| 2 | Fix documentation issues | Submit PRs fixing typos, broken links, unclear explanations |
| 3 | Write API documentation | Document a module's API with examples |
| 4 | Create a tutorial | Write a step-by-step guide for a common workflow |
| 5 | Revise architecture docs | Update `docs/ARCHITECTURE.md` with new components |
| 6 | Own documentation quality | Maintain docs/ index, review doc PRs, enforce standards |

### Key Resources

- Existing docs in `docs/` directory
- `docs/STYLE_GUIDE.md` (Documentation Style section)
- Python source files (docstrings)
- C header files (API comments)

---

## Infrastructure Track

*Prerequisites: Build systems (CMake), packaging experience*

| Month | Goal | Deliverable |
|-------|------|-------------|
| 1 | Understand build system | Read `CMakeLists.txt`, understand presets and targets |
| 2 | Optimize build times | Profile build, identify bottlenecks, implement ccache/distcc |
| 3 | Containerize the project | Write Dockerfile for development and testing |
| 4 | Automate release process | Script the release workflow (version bump, changelog, publish) |
| 5 | Improve test infrastructure | Expand ctest suite, add sanitizer targets |
| 6 | Own infrastructure | Take ownership of build system, packaging, scripts |

### Key Resources

- `CMakeLists.txt` — build configuration
- `scripts/` — build and development scripts
- `cmake/` — CMake modules
