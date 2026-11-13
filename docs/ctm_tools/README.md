# CTM Tooling

CTM (Codebase Telemetry & Metrics) is the repository's metrics toolkit.
The recommended entry point is the unified, standard-library scanner:

```bash
python ctm_tools/py/ctm.py .
python ctm_tools/py/ctm.py . --format json --output ctm-report.json
python ctm_tools/py/ctm.py . --format csv
```

It reports included text files, descendant directories, physical lines,
bytes, language breakdowns, skipped binary/symlink files, and optional Git
tracked/untracked counts. It does not execute scanned files.

See the [complete CTM guide](../../ctm_tools/README.md) for metric definitions,
exclusions, JSON schema, output formats, limitations, and tests.
