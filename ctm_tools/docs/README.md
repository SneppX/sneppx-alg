# CTM Tools Documentation

## Overview
Custom tools for codebase metrics analysis. Both C and Python implementations provided.

## Core Tools (Implemented)

### 1. ctm_count_total_loc
Counts total lines of code in a directory tree (ignoring common build/cache dirs).

**Python:** ctm_tools/py/ctm_count_total_loc.py
`ash
python ctm_tools/py/ctm_count_total_loc.py /path/to/repo
`

**C (Windows):** ctm_tools/c/ctm_count_total_loc.c (build via CMake in ctm_tools/)

### 2. ctm_count_total_files
Counts total files.

**Python:** ctm_tools/py/ctm_count_total_files.py
`ash
python ctm_tools/py/ctm_count_total_files.py /path/to/repo
`

**C:** ctm_tools/c/ctm_count_total_files.c

### 3. ctm_count_total_dirs
Counts total directories.

**Python:** ctm_tools/py/ctm_count_total_dirs.py
`ash
python ctm_tools/py/ctm_count_total_dirs.py /path/to/repo
`

**C:** ctm_tools/c/ctm_count_total_dirs.c

### 4. ctm_count_loc_by_file
Lists LOC per file sorted by count.

**Python:** ctm_tools/py/ctm_count_loc_by_file.py
`ash
python ctm_tools/py/ctm_count_loc_by_file.py /path/to/repo
`

**C:** ctm_tools/c/ctm_count_loc_by_file.c

### 5. ctm_count_loc_by_language
Aggregates LOC and file counts by language.

**Python:** ctm_tools/py/ctm_count_loc_by_language.py
`ash
python ctm_tools/py/ctm_count_loc_by_language.py /path/to/repo
`

## Build (C tools)
`ash
cd sneppx-alg
cmake -B build -DSNEPPX_BUILD_TOOLS=ON  # or use ctm_tools/CMakeLists.txt standalone
# or: cmake -B build_ctm -S ctm_tools
`

## Notes
- Tools ignore .git, build, __pycache__, node_modules, venvs, etc.
- Binary/extensions like .pyc, .exe, .dll are skipped.
- Python versions are cross-platform and tested on Windows.
