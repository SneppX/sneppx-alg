#!/usr/bin/env python3
"""Validate CTM JSON reports against schema version 1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

INT_FIELDS = (
    "files",
    "directories",
    "lines_of_code",
    "bytes",
    "skipped_binary_or_symlink",
    "unreadable_files",
)
LANGUAGE_FIELDS = ("files", "loc", "bytes")


def validate(payload: Any) -> list[str]:
    """Return a sorted list of schema errors; empty means valid."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["report must be a JSON object"]

    if payload.get("tool") != "ctm":
        errors.append("tool must be 'ctm'")
    if payload.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    for field in ("root",):
        if not isinstance(payload.get(field), str) or not payload[field]:
            errors.append(f"{field} must be a non-empty string")
    for field in INT_FIELDS:
        value = payload.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"{field} must be a non-negative integer")

    languages = payload.get("languages")
    if not isinstance(languages, dict):
        errors.append("languages must be an object")
    else:
        for name, metrics in sorted(languages.items()):
            if not isinstance(metrics, dict):
                errors.append(f"languages.{name} must be an object")
                continue
            for field in LANGUAGE_FIELDS:
                value = metrics.get(field)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    errors.append(
                        f"languages.{name}.{field} must be a non-negative integer"
                    )
        for field, total in (
            ("files", "files"),
            ("loc", "lines_of_code"),
            ("bytes", "bytes"),
        ):
            summed = sum(
                metrics.get(field, 0)
                for metrics in languages.values()
                if isinstance(metrics, dict) and isinstance(metrics.get(field, 0), int)
            )
            if summed != payload.get(total):
                errors.append(
                    f"languages sum of {field} ({summed}) != {total} ({payload.get(total)})"
                )

    git = payload.get("git")
    if git is not None and not isinstance(git, dict):
        errors.append("git must be an object when present")
    return sorted(errors)


def check_files(paths: list[str | Path]) -> dict[str, Any]:
    """Load and validate each report file; never raises for invalid content."""
    results: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            errors = validate(payload)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors = [f"cannot load report: {exc}"]
        results.append({"path": str(path), "valid": not errors, "errors": errors})
    return {
        "tool": "ctm_schema_check",
        "reports": len(results),
        "valid": sum(1 for item in results if item["valid"]),
        "invalid": sum(1 for item in results if not item["valid"]),
        "results": results,
    }


def render(report: dict[str, Any], fmt: str) -> str:
    if fmt == "json":
        return json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    lines = [
        f"CTM schema check: {report['valid']}/{report['reports']} valid",
    ]
    for item in report["results"]:
        status = "OK" if item["valid"] else "INVALID"
        lines.append(f"  [{status}] {item['path']}")
        lines.extend(f"      - {error}" for error in item["errors"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate CTM JSON reports")
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    report = check_files(args.reports)
    rendered = render(report, args.format)
    if args.output:
        Path(args.output).expanduser().write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0 if report["invalid"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
