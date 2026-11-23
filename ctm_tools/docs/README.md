# CTM Tools Documentation

## Overview

Custom tools for codebase metrics analysis. The maintained implementation is the
dependency-free Python scanner `ctm_tools/py/ctm.py`. The C sources under
`ctm_tools/c/` are **unbuilt legacy prototypes** and must not be used for
measurements (see `ctm_tools/README.md`).

## Supported scanner (Python)

Run from the repository root:

```bash
python ctm_tools/py/ctm.py /path/to/repo
python ctm_tools/py/ctm.py /path/to/repo --format json --output ctm-report.json
python ctm_tools/py/ctm.py /path/to/repo --format csv
python ctm_tools/py/ctm.py /path/to/repo --format html --output ctm-report.html
```

Formats: `text` (default), `json` (schema version 1), `csv`, and `html`.
See `ctm_tools/README.md` for metric definitions, exclusions, and safety notes.

Regression tests:

```bash
python -m pytest tests/python/test_ctm_tools.py -q
```

## Legacy single-purpose Python scripts

These remain available for compatibility. Their exclusions and output contracts
are **not** uniform with `ctm.py`, so prefer the unified scanner for automation:

| Tool | Purpose |
|---|---|
| `ctm_count_total_loc.py` | Total physical lines |
| `ctm_count_total_files.py` | Total files |
| `ctm_count_total_dirs.py` | Total directories |
| `ctm_count_loc_by_file.py` | LOC per file |
| `ctm_count_loc_by_language.py` | LOC aggregated by language |
| `ctm_summary_report.py` | Combined summary |

Run any of them as `python ctm_tools/py/<tool>.py /path/to/repo`.
Additional counters are listed in `EXTENDED.md`.

## C prototypes (not built)

The sources in `ctm_tools/c/` are incomplete prototypes: several return
placeholder values, one does not compile, and none provides a portable
implementation. `ctm_tools/CMakeLists.txt` therefore exposes only a `ctm-help`
target. The root CMake option `SNEPPX_BUILD_TOOLS` builds `tools/`, **not**
`ctm_tools/`. Do not use the C files for measurements and do not add build
targets for them without a shared, tested cross-platform filesystem walker.

## Notes

- Tools ignore `.git`, `build`, `__pycache__`, `node_modules`, virtualenvs, and other generated/cache directories; the authoritative list is in `ctm.py`.
- Binary extensions and files containing NUL bytes are skipped by `ctm.py`.
- The Python implementations are cross-platform and tested on Windows.
