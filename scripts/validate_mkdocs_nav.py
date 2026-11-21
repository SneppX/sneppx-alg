#!/usr/bin/env python3
"""Validate MkDocs navigation labels, targets, and repository containment."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCS = (ROOT / "docs").resolve()
CONFIG = ROOT / "mkdocs.yml"


def validate_group(
    entries: Any,
    group_name: str,
    seen_targets: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    if not isinstance(entries, list):
        return [f"{group_name}: navigation group must be a list"]

    seen_labels: set[str] = set()
    for index, entry in enumerate(entries):
        location = f"{group_name}[{index}]"
        if not isinstance(entry, dict) or len(entry) != 1:
            errors.append(f"{location}: each navigation item must contain exactly one label")
            continue

        label, target = next(iter(entry.items()))
        if not isinstance(label, str) or not label.strip():
            errors.append(f"{location}: navigation label must be a non-empty string")
            continue
        if label in seen_labels:
            errors.append(f"{location}: duplicate sibling navigation label {label!r}")
        seen_labels.add(label)

        child_group = f"{group_name}/{label}"
        if isinstance(target, list):
            errors.extend(validate_group(target, child_group, seen_targets))
            continue

        if not isinstance(target, str) or not target.strip():
            errors.append(f"{location}: target must be a non-empty path or nested group")
            continue
        if not target.endswith(".md"):
            errors.append(f"{location}: expected a Markdown target, got {target!r}")
            continue

        resolved = (DOCS / target).resolve()
        if resolved != DOCS and DOCS not in resolved.parents:
            errors.append(f"{location}: target escapes the docs directory: {target!r}")
            continue
        if not resolved.is_file():
            errors.append(f"{location}: target does not exist: {target!r}")
            continue

        previous = seen_targets.get(target)
        if previous is not None:
            errors.append(
                f"{location}: duplicate navigation target {target!r} "
                f"(already used at {previous})"
            )
        else:
            seen_targets[target] = location

    return errors


def main() -> int:
    try:
        config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        print(f"Cannot read valid MkDocs configuration: {exc}", file=sys.stderr)
        return 1

    if not isinstance(config, dict) or "nav" not in config:
        print("mkdocs.yml: missing navigation configuration", file=sys.stderr)
        return 1

    seen_targets: dict[str, str] = {}
    errors = validate_group(config["nav"], "nav", seen_targets)
    if errors:
        print("MkDocs navigation validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"MkDocs navigation passed: {len(seen_targets)} unique Markdown targets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
