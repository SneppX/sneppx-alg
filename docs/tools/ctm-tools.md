---
description: "Reference for the CTM tools in SneppX-ALG: the unified repository scanner and its sneppx-ctm command-line utilities."
---

# CTM Tools

CTM (Codebase Telemetry & Metrics) provides a standard-library repository
scanner plus standalone focused utilities for codebase telemetry.

The supported implementation is the dependency-free Python scanner:

```bash
sneppx-ctm /path/to/repo
sneppx-ctm /path/to/repo --format json --output ctm-report.json
sneppx-ctm /path/to/repo --format csv
sneppx-ctm /path/to/repo --format html --output ctm-report.html
```

## Current tool surface

| Entry point | Purpose |
|---|---|
| `sneppx-ctm` | Unified repository scanner (text/json/csv/html reports, schema v1) |
| `sneppx-ctm-functions` | Heuristic function-definition counts by language |
| `sneppx-ctm-headers` | C/C++ header and source file counts |
| `sneppx-ctm-diff` | Per-language LOC/bytes delta between two trees |
| `sneppx-ctm-large` | Largest included files with human-readable sizes |
| `sneppx-ctm-recent` | Most recently modified included files |
| `sneppx-ctm-todo` | TODO/FIXME/HACK/XXX marker scan |
| `sneppx-ctm-links` | Relative Markdown link checker |
| `sneppx-ctm-duplicates` | Byte-identical file duplicate detection |
| `sneppx-ctm-churn` | Git-history change frequency per file |
| `sneppx-ctm-license` | SPDX/heuristic license header + manifest inventory |
| `sneppx-ctm-schema` | Validate CTM JSON reports against schema v1 |
| `sneppx-ctm-encoding` | BOM / invalid-UTF-8 / newline-style report |
| `sneppx-ctm-ignore` | Explain why files are excluded by the default filters |
| `sneppx-ctm-trend` | Record and chart successive CTM JSON reports |

All tools accept a root directory, `--format text|json|csv|html`, `--output`,
and, where relevant, `--include-other-text`. The standalone Python sources live
in `ctm_tools/py/` and the shipped behavior is shared through the
`ctm_tools.py` import package plus `sneppx-ctm*` console scripts registered in
`pyproject.toml`.

The C prototypes under `ctm_tools/c/` are unbuilt legacy scaffolding and are
not part of the supported tool surface.
