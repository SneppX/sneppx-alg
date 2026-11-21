# Code Review Guide

A guide to reviewing code in SNEPPX-Algo. All Senior+ contributors are expected to follow this guide when reviewing PRs.

---

## Review Checklist

For every PR, verify:

```md
[ ] Builds without warnings
[ ] All tests pass (ctest / pytest)
[ ] Linter clean (pre-commit run --all-files)
[ ] Test coverage ≥ 80% for new code
[ ] Signed commits (GPG/Ed25519)
[ ] Follows STYLE_GUIDE.md
[ ] Follows COMMENTING.md (Layer 1 headers + Layer 4 API docs on public functions)
[ ] `sneppx-format --docs` clean on modified files
[ ] No new clang-tidy warnings
[ ] CHANGELOG.md updated (if user-facing)
[ ] Documentation updated
[ ] All conversations resolved
```

---

## Review Levels

| Level | Responsibility | Required For |
|-------|---------------|--------------|
| L1 — Basic | Correctness, style, test coverage | All PRs |
| L2 — Deep | Performance, safety, concurrency | Kernel, CUDA, Distributed PRs |
| L3 — Security | Cryptography, constant-time, threat model | Security layer PRs (S0–S9) |

---

## What to Look For (by Subsystem)

### Tensor Core (`kernel/tensor/`)

- Memory safety: all allocations checked for NULL
- Bounds checking: no out-of-bounds access on shape/stride
- Type promotion: correct handling of mixed dtypes
- Broadcasting: correct broadcast rules (NumPy semantics)
- Error returns: all failure paths return error codes

### Autodiff (`kernel/autodiff/`)

- Tape correctness: graph traversal order is topological
- Gradient shapes: output gradient shapes match input shapes
- Gradient values: numerical gradient check for new ops
- Memory: intermediate activations freed after backward
- Edge cases: scalar gradients, zero-size tensors

### Optimizer (`kernel/optimizer/`)

- Numerical stability: no division by zero, NaN handling
- State management: momentum buffers initialized correctly
- LR scheduling: decay applied at correct step boundaries
- Weight decay: correct interaction with learning rate

### HSS / SER / ARC / NPE / FM (`algorithms/`)

- Mathematical correctness: matches documented formulas
- Numerical precision: float32 accumulation, overflow checks
- Configuration validation: invalid configs return errors
- Edge cases: empty sequences, single token, single expert

### CUDA Kernels (`kernel/cuda/`)

- Memory coalescing: global memory access patterns
- Bank conflicts: shared memory access patterns
- Occupancy: registers per thread, shared memory per block
- Synchronization: __syncthreads placement, no deadlocks
- Error checking: all CUDA API calls checked for errors
- Fallback: graceful degradation if no CUDA device

### Security Layer (`security/`)

- Constant-time: no secret-dependent branches or memory access
- Memory: secure wipe of sensitive data after use
- Side channels: no cache-timing leaks in crypto implementations
- Entropy: proper use of OS CPRNG, no predictable seeds
- Input validation: all inputs checked for length/range
- ASM correctness: MASM routines verified against C reference
- FIPS compliance: correct parameter sets for PQ algorithms

### Python Bindings (`bindings/python/`)

- Type hints: all public functions typed
- Error handling: C errors translated to Python exceptions
- Memory management: no leaks from C→Python bridge
- NumPy compatibility: correct dtype/contiguity handling

### Build System (`CMakeLists.txt`, `cmake/`)

- Option hygiene: new options documented with defaults
- Dependency management: version checks for required tools
- Cross-platform: Windows/Linux/macOS paths
- Generator agnostic: works with Ninja and Visual Studio

---

## Review Process

```
1. Read the PR description and linked issue/design doc
2. Clone branch locally for deep reviews (L2/L3)
3. Build and test locally before approving
4. Start with high-level architecture, then line-by-line
5. Leave clear, actionable comments
6. Approve only when all concerns are resolved
```

## How to Write Good Review Comments

**Bad**: "This is wrong."
**Good**: "This buffer isn't checked for NULL after allocation on line 42. If `malloc` fails, `tensor->data` will be NULL and line 45 will segfault. Please check the return value and propagate the error."

**Bad**: "Fix style."
**Good**: "Line 88: `SNEPPX_tensor_create ( ... )` has a space before the opening paren. Style guide says no space between function name and `(`."

---

## Review Tags

Use these tags in review comments:

| Tag | Meaning |
|-----|---------|
| `nit:` | Minor style issue, non-blocking |
| `blocking:` | Must fix before merge |
| `question:` | Clarification needed, not a defect |
| `suggestion:` | Alternative approach, not required |
| `praise:` | Highlight something well done |

---

## Security-Sensitive Review

All PRs touching `security/` or `include/neural_core/security/` require:

1. **L3 Security review** by a Senior+ contributor with security expertise
2. **Constant-time analysis** for any crypto changes
3. **Test vectors verified** for any new cryptographic primitive
4. **Side-channel review** if modifying existing crypto implementations

Security-sensitive files that always require an L3 review:

- `security/crypto/c/*.c`
- `security/crypto/asm/x86_64/*.asm`
- `security/memory/*.c`
- `include/neural_core/security/*.h`
- `tests/security/*.c`

---

## Review Responsibilities

As a reviewer, you are responsible for:

- Verifying correctness and safety
- Ensuring adequate test coverage
- Checking that documentation is updated
- Mentoring the contributor through feedback
- Responding within 48 hours (target)
- Merging only after all blocking comments are resolved
