# CTM Tools Suite - Extended Documentation

## Additional Tools

- ctm_count_files_by_ext.py - Count files grouped by extension
- ctm_largest_files.py - Find largest files by size
- ctm_count_empty_files.py - Count empty files
- ctm_dir_sizes.py - Directory sizes (top N)

All tools support passing root directory as first argument. Defaults to current directory.

## Usage Examples
`ash
python ctm_tools/py/ctm_count_files_by_ext.py .
python ctm_tools/py/ctm_largest_files.py . 30
python ctm_tools/py/ctm_count_empty_files.py .
python ctm_tools/py/ctm_dir_sizes.py . 
`

## Ignored Patterns
.git, .svn, .hg, __pycache__, node_modules, .venv, venv, build, dist, .next, .cache, .pytest_cache, .ruff_cache, plus binary extensions.
