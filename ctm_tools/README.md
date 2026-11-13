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
- Scans can still consume I/O and memory proportional to the size of text
  files. Do not point it at untrusted network mounts without considering that
  operational cost.

## Regression tests

```bash
python -m pytest tests/python/test_ctm_tools.py -q
```

## Existing focused tools

The older scripts in `ctm_tools/py/` remain available for compatibility.
The unified `ctm.py` command is the recommended starting point for new
automation because it provides one schema and consistent exclusion behavior.
