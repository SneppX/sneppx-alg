# CTM Extended Tools

The legacy scripts in `ctm_tools/py/` remain available for focused reports:

- `ctm_count_comments_blank_code.py`: comment, blank, and code-line estimates
- `ctm_count_empty_files.py`: empty-file inventory
- `ctm_count_files_by_ext.py`: extension distribution
- `ctm_count_functions.py`: heuristic function-declaration count
- `ctm_count_headers.py`: header-file inventory
- `ctm_count_includes.py`: include/import inventory
- `ctm_diff_loc.py`: compare LOC snapshots
- `ctm_dir_sizes.py`: directory-size summary
- `ctm_find_large_files.py`, `ctm_largest_files.py`, `ctm_list_large_files.py`: large-file discovery
- `ctm_git_tracked_vs_untracked.py`: Git index/worktree counts
- `ctm_recent_files.py`: recently modified file listing
- `ctm_tree_stats.py`: directory-depth distribution

These scripts predate the unified scanner and may have different exclusions,
output contracts, and platform behavior. Treat heuristic results as estimates,
not authoritative source analysis. For consistent metrics and automation,
prefer [`ctm_tools/py/ctm.py`](../../ctm_tools/README.md).
