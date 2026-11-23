"""Importable CTM tool package.

The tools were originally written as standalone scripts that import their
siblings by bare module name (``import ctm``). When installed as a package,
this module registers those aliases so the same files work unchanged both as
scripts and as installed modules.
"""

from __future__ import annotations

import sys
from types import ModuleType

_MODULES = (
    "ctm",
    "ctm_find_large_files",
    "ctm_count_functions",
    "ctm_count_headers",
    "ctm_diff_loc",
    "ctm_list_large_files",
    "ctm_recent_files",
    "ctm_todo_scan",
    "ctm_doc_links",
    "ctm_duplicate_files",
    "ctm_churn",
    "ctm_license_inventory",
    "ctm_schema_check",
    "ctm_encoding_report",
    "ctm_ignore_audit",
    "ctm_trend",
)


def _alias(name: str, module: ModuleType) -> None:
    sys.modules.setdefault(name, module)


# Dependency order matters: ctm_list_large_files imports ctm_find_large_files.
for _name in _MODULES:
    _module = __import__(f"{__name__}.{_name}", fromlist=[_name])
    _alias(_name, _module)
    globals()[_name] = _module

__all__ = list(_MODULES)
