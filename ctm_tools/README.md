# CTM Tools Suite

Custom tools for codebase metrics analysis (LOC, files, dirs, languages). Provides both C and Python implementations.

## Structure
- `c/` - C implementations (Windows-focused, with POSIX support planned)
- `py/` - Python implementations (cross-platform)
- `common/` - Shared utilities
- `scripts/` - Build/run helpers
- `docs/` - Documentation

## Core Tools
1. `ctm_count_total_loc` - Count total lines of code
2. `ctm_count_total_files` - Count total files
3. `ctm_count_total_dirs` - Count total directories
4. `ctm_count_loc_by_file` - LOC per file (sorted)
5. `ctm_count_loc_by_language` - LOC aggregated by language

## Extended Tools
- `ctm_count_files_by_ext` - Count files by extension
- `ctm_largest_files` - Find largest files by size
- `ctm_count_empty_files` - Count empty files
- `ctm_dir_sizes` - Directory sizes

## Usage (Python)
```bash
# Text output (default)
python ctm_tools/py/ctm_count_total_files.py /path/to/repo

# JSON output
python ctm_tools/py/ctm_count_total_files.py /path/to/repo --format json

# CSV output
python ctm_tools/py/ctm_count_loc_by_language.py /path/to/repo --format csv

# Write to file
python ctm_tools/py/ctm_count_total_loc.py /path/to/repo --format json --output metrics.json
```

## Build (C tools)
```bash
cd sneppx-alg
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ctm_count_total_files ctm_count_total_dirs ctm_count_total_loc ctm_count_loc_by_file
```

## Documentation
- Local: `ctm_tools/docs/`
- Main docs: `docs/ctm_tools/`
