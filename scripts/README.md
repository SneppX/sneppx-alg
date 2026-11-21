# Scripts

Build, CI, and automation scripts for SNEPPX-ALG.

| Script | Description |
|--------|-------------|
| `dev-tools.ps1` | Windows: run the full SNEPPX tool chain (analyze, format, deps, stats, test) |
| `dev-tools.sh` | Linux/macOS: run the full SNEPPX tool chain |
| `run_sanitizers.sh` | Linux: ASan/UBSan builds + compute-sanitizer + ctest |
| `run_sanitizers.ps1` | Windows: ASan/UBSan builds + compute-sanitizer + ctest |
| `pre-commit.sh` | Git pre-commit hook (installed via `install-hooks.sh`) — includes sneppx tool checks |
| `verify-package.sh` | Verify release tarball checksums + signature |
| `stats.sh` | Legacy line-count stats (use `sneppx-stats` for richer output) |

## Dev Tool Chain

The 7 standalone SNEPPX developer tools (`sneppx-analyze`, `sneppx-format`,
`sneppx-deps`, `sneppx-stats`, `sneppx-test`, `sneppx-bench`, `sneppx-serve`)
are wired into the dev workflow.

Install all at once:

```powershell
pip install sneppx-toolkit[all]
```

Run the full chain:

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File scripts\dev-tools.ps1

# Linux/macOS
./scripts/dev-tools.sh
```

The chain runs:
1. Security scan (`sneppx-analyze`) over `kernel/`, `algorithms/`, `net/`, `security/`
2. Formatting / lint (`sneppx-format`)
3. Circular dependency check (`sneppx-deps`)
4. Code statistics (`sneppx-stats`, saved to `.sneppx/stats.json`)
5. Test suite (`sneppx-test` — or ctest fallback) excluding CUDA tests
6. Benchmarks (`sneppx-bench`, only with `--all`)

Tool configuration lives in `.sneppx-tools.json` at the repo root.

Sanitizer scripts perform:
1. CMake configure with `-DSNEPPX_ENABLE_ASAN=ON -DSNEPPX_ENABLE_UBSAN=ON`
2. Build all targets
3. Run ctest with output-on-failure
4. NVIDIA compute-sanitizer (memcheck/racecheck/initcheck) if CUDA available
