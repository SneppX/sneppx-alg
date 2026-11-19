"""Regression tests for MkDocs navigation validation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_mkdocs_nav import validate_group  # noqa: E402


def test_rejects_duplicate_sibling_labels() -> None:
    errors = validate_group(
        [{"Guides": "index.md"}, {"Guides": "getting-started.md"}],
        "nav",
        {},
    )

    assert any("duplicate sibling navigation label" in error for error in errors)


def test_rejects_duplicate_targets_across_groups() -> None:
    errors = validate_group(
        [
            {"Reference": [{"Home": "index.md"}]},
            {"Getting started": [{"Home": "index.md"}]},
        ],
        "nav",
        {},
    )

    assert any("duplicate navigation target 'index.md'" in error for error in errors)


def test_rejects_missing_targets() -> None:
    errors = validate_group(
        [{"Missing page": "this-page-does-not-exist.md"}],
        "nav",
        {},
    )

    assert any("target does not exist" in error for error in errors)


def test_rejects_paths_outside_docs_directory() -> None:
    errors = validate_group([{"Outside": "../README.md"}], "nav", {})

    assert any("target escapes the docs directory" in error for error in errors)
