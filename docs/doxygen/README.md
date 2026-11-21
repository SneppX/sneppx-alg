# Doxygen API Documentation

Generate HTML documentation from C/C++ code comments:

```powershell
# From repo root
doxygen docs/doxygen/Doxyfile

# Output: docs/doxygen/output/html/index.html
```

Requirements: [Doxygen](https://www.doxygen.nl/download.html) 1.9+ installed and on PATH.

# Doxygen API Documentation

Generate HTML documentation from C/C++ code comments:

```powershell
# From repo root
doxygen docs/doxygen/Doxyfile

# Output: docs/doxygen/output/html/index.html
```

Requirements: [Doxygen](https://www.doxygen.nl/download.html) 1.9+ installed and on PATH.

The configuration extracts docs from `include/neural_core/` (public headers). Private implementation details in `kernel/` are not included.

## Commenting Standard

All public `SNEPPX_*` functions must have Doxygen `@brief/@param/@return` blocks following the [COMMENTING.md](../COMMENTING.md) standard (Layer 4). File headers (Layer 1) are required on all source files. Run `sneppx-format --docs` to verify compliance.
