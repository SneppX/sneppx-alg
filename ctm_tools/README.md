# CTM: Codebase Telemetry & Metrics

CTM is a dependency-free repository scanner for repeatable codebase measurements.
It reports filesystem facts; it does not measure code quality, security, complexity,
or executable behavior.

## Quick start

From the repository root:

```bash
python ctm_tools/py/ctm.py .
python ctm_tools/py/ctm.py . --format json --output ctm-report.json
python ctm_tools/py/ctm.py . --format csv
python ctm_tools/py/ctm.py . --format html --output ctm-report.html
python ctm_tools/py/ctm.py . --include-other-text
```

The scanner requires Python 3.9 or newer and uses only the standard library.

## Metrics

| Field | Meaning |
|---|---|
| `files` | Count of readable, included text files |
| `directories` | Count of included descendant directories (the root is excluded) |
| `lines_of_code` | Physical text lines in included files, including blank/comment lines |
| `bytes` | Total byte size of included files |
| `languages` | Per-language file, line, and byte totals |
| `skipped_binary_or_symlink` | Binary files, configured binary extensions, and symlink files |
| `unreadable_files` | Files whose metadata could not be read |
| `git` | Tracked and untracked counts when the root is inside a Git worktree |

LOC is a physical line count, not logical statements or nonblank source lines.
Comment lines are included. Generated and vendor directories are excluded by
name; the exact default exclusions are defined in `ctm_tools/py/ctm.py`.

## Output formats

- `text`: human-readable summary (default)
- `json`: stable top-level schema with `schema_version: 1`
- `csv`: language rows followed by a TOTAL row
- `html`: self-contained responsive dashboard with escaped content and no external assets

JSON is the preferred format for dashboards and automation. Consumers should
check `schema_version` before assuming future fields are compatible.

## Exit behavior

- Exit code 0: scan and report succeeded.
- Non-zero: invalid root, invalid arguments, or output write failure.
- Git metadata is optional. If Git is unavailable or the directory is not a
  repository, the scan still succeeds and omits the `git` object.

## Safety and scope

- Does not execute scanned files.
- Does not follow directory symlinks.
- Skips common generated, cache, and dependency directories.
- Skips known binary extensions and files containing NUL bytes.
- Unknown extensions are ignored unless `--include-other-text` is supplied.
- Reads UTF-8 text. Non-UTF-8 files are counted as skipped, not silently
  decoded with lossy replacement.
- Scanning time and I/O scale with the repository size; file contents are
  counted incrementally to avoid loading an entire large file into memory.
  Do not point it at untrusted network mounts without considering the I/O cost.

## Utility tools

The unified scanner and ten focused additional tools share `ctm.py`'s inclusion
filters:

| Tool | Entry point | Purpose |
|---|---|---|
| `py/ctm.py` | `sneppx-ctm` | Unified scanner: files, dirs, LOC, bytes, languages, skips |
| `py/ctm_count_functions.py` | `sneppx-ctm-functions` | Heuristic function-definition counts by language |
| `py/ctm_count_headers.py` | `sneppx-ctm-headers` | C/C++ header and source file counts |
| `py/ctm_diff_loc.py` | `sneppx-ctm-diff` | Per-language LOC/bytes delta between two trees |
| `py/ctm_list_large_files.py` | `sneppx-ctm-large` | Largest included files with human-readable sizes |
| `py/ctm_recent_files.py` | `sneppx-ctm-recent` | Most recently modified included files |
| `py/ctm_todo_scan.py` | `sneppx-ctm-todo` | TODO/FIXME/HACK/XXX marker scan |
| `py/ctm_doc_links.py` | `sneppx-ctm-links` | Relative Markdown link checker |
| `py/ctm_duplicate_files.py` | `sneppx-ctm-duplicates` | Byte-identical file duplicate detection |
| `py/ctm_churn.py` | `sneppx-ctm-churn` | Git-history change frequency per file |
| `py/ctm_license_inventory.py` | `sneppx-ctm-license` | SPDX/heuristic license header + manifest inventory |
| `py/ctm_schema_check.py` | `sneppx-ctm-schema` | Validate CTM JSON reports against schema v1 |
| `py/ctm_encoding_report.py` | `sneppx-ctm-encoding` | BOM / invalid-UTF-8 / newline-style report |
| `py/ctm_ignore_audit.py` | `sneppx-ctm-ignore` | Explain why files are excluded by the default filters |
| `py/ctm_trend.py` | `sneppx-ctm-trend` | Record and chart successive CTM JSON reports |

The console scripts are registered in `pyproject.toml` under
`[project.scripts]` and point at the `ctm_tools.py` import package
(`ctm_tools.py.<module>:main`). The tools also run directly from source via
`python ctm_tools/py/<tool>.py ...`.

Each accepts a root directory, `--format text|json|csv` (plus `html` for the
unified scanner), `--output`, and `--include-other-text` where applicable. Their
counts are filesystem telemetry, not semantic analysis; function counting is a
conservative text heuristic.

## Regression tests

```bash
python -m pytest tests/python/test_ctm_tools.py tests/python/test_ctm_new_tools.py -q
```

## Legacy implementations

The older focused Python scripts remain available for compatibility, but their
output contracts and exclusions are not uniform. The C files under `ctm_tools/c/`
are legacy prototypes and are not built: several implementations return
placeholder values on non-Windows systems or unconditionally return zero.
Do not use those C binaries for measurements. The standalone CMake project
exposes only a help target until a shared, tested cross-platform C scanner is
implemented.

The unified `ctm.py` command is the supported implementation for new
automation because it provides consistent exclusions, machine-readable output,
and regression tests.
