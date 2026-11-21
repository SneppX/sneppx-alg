# SNEPPX-Algo Style Guide

## C Style

### Naming
- **Functions**: `snake_case` with `SNEPPX_` prefix: `SNEPPX_tensor_create`
- **Types**: `PascalCase` with `SNEPPX` prefix: `SNEPPXTensor`
- **Macros**: `UPPER_SNAKE_CASE`: `SNEPPX_FLOAT32`
- **Constants**: `UPPER_SNAKE_CASE`: `MAX_TOKEN_LEN`
- **File-scope statics**: `s_` prefix: `s_global_pool`

### Formatting
- **Indentation**: 4 spaces, no tabs
- **Line length**: 100 characters max
- **Braces**: K&R style (opening brace on same line)
- **Pointer alignment**: Right (`int* p`)
- **Spacing**:
  - One space after `if`, `for`, `while`, `do`, `switch`
  - No space between function name and `(`
  - Space around binary operators
  - No space around unary operators

### Headers
- Include guards: `#ifndef SNEPPX_MODULE_NAME_H` / `#define SNEPPX_MODULE_NAME_H`
- Standard headers first, then project headers, alphabetically
- Forward declarations preferred over includes when possible

### Example

```c
#ifndef SNEPPX_TENSOR_H
#define SNEPPX_TENSOR_H

#define SNEPPX_MAX_NDIM 8

typedef struct {
    float* data;
    size_t shape[SNEPPX_MAX_NDIM];
    size_t ndim;
    size_t size;
} SNEPPXTensor;

SNEPPXTensor* SNEPPX_tensor_create(const size_t* shape, size_t ndim, SNEPPXDtype dtype);
void SNEPPX_tensor_destroy(SNEPPXTensor* t);

#endif
```

### Error Handling
- Return `int` (0 = success, nonzero = error) for operations
- Return `NULL` for failed allocations
- Check all allocation returns
- Never assert in production code paths

### Memory
- Always use `SNEPPX_malloc` / `SNEPPX_free` with size tracking
- Parameter order: output params before input params where possible
- Document ownership transfer

## C++ Style

Used only for S2 obfuscation engine and pybind11 bindings.

- **Namespace**: `SNEPPX`
- **Class names**: PascalCase
- **Method names**: snake_case
- **Variable names**: snake_case
- **Member variables**: trailing underscore (`member_`)
- **RAII** everywhere
- No exceptions in hot paths
- `const` correctness
- No RTTI
- No `dynamic_cast`

## Python Style

- PEP 8 compliant
- Line length: 88 columns (Black default)
- Formatter: Black (required)
- Type hints on all public functions
- Docstrings: Google style
- Tests use pytest

### Example

```python
from typing import Optional


class Tensor:
    """Multi-dimensional array."""

    def __init__(self, data: np.ndarray) -> None:
        self._data = data.astype(np.float32)

    def numpy(self) -> np.ndarray:
        return self._data.copy()
```

## Commit Messages

```
component: Brief description in present tense

Longer description explaining motivation and approach.
Wrap at 72 characters.
```

- Prefix with component: `tensor:`, `hss:`, `ser:`, `arc:`, `npe:`, `fm:`,
  `security:`, `build:`, `docs:`, `tests:`, `python:`
- First line: max 72 chars
- Use imperative mood: "Fix", not "Fixed"
- GPG or Ed25519 commit signing required on `main` branch

## Documentation Style

- Markdown for all documentation files
- Code examples in fenced code blocks with language tag
- Tables for structured data
- Links use relative paths within the repository

## Naming Conventions

| Entity | Convention | Example |
|--------|------------|---------|
| C files | snake_case | `tensor_ops.c` |
| C headers | snake_case | `SNEPPX_tensor.h` |
| C functions | snake_case | `SNEPPX_tensor_create` |
| C structs | PascalCase | `SNEPPXTensor` |
| C enums | PascalCase | `SNEPPXDtype` |
| C macros | SCREAMING_SNAKE_CASE | `SNEPPX_MAX_NDIM` |
| C++ classes | PascalCase | `Tensor` |
| C++ methods | snake_case | `tensor_create` |
| Python classes | PascalCase | `Tensor` |
| Python functions | snake_case | `from_numpy` |
| Directories | snake_case | `kernel/tensor/` |

## Commenting Standard

See [COMMENTING.md](COMMENTING.md) for the four-layer commenting standard:
- Layer 1: File header blocks (WHAT/CONCEPT/ROLE/REFERENCES)
- Layer 2: Concept blocks before non-obvious algorithms
- Layer 3: Inline "why" comments
- Layer 4: Doxygen `@brief/@param/@return` on every public `SNEPPX_*` function
