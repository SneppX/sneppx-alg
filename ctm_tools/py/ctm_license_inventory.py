#!/usr/bin/env python3
"""Inventory SPDX license headers and dependency manifests in included files."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import ctm

SPDX = re.compile(r"SPDX-License-Identifier\s*:\s*([A-Za-z0-9.+\-]+)")
LICENSE_WORDS = re.compile(
    r"\b(MIT License|Apache License|GNU General Public License|BSD License|"
    r"Mozilla Public License|ISC License|Unlicense)\b",
    re.IGNORECASE,
)
MANIFEST_NAMES = {
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "Pipfile",
    "poetry.lock",
    "package.json",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Cargo.toml",
    "Cargo.lock",
    "go.mod",
    "go.sum",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "Gemfile",
    "composer.json",
}
LICENSE_FILES = {"license", "license.txt", "license.md", "copying", "copying.txt"}
HEADER_LINES = 40


def license_inventory(root: str | Path, include_other_text: bool = False) -> dict[str, Any]:
    """Report SPDX/heuristic license headers and dependency manifests."""
    base = Path(root).expanduser().resolve()
    by_spdx: dict[str, int] = defaultdict(int)
    by_heuristic: dict[str, int] = defaultdict(int)
    files_with_license: list[dict[str, Any]] = []
    files_scanned = 0
    manifests: list[str] = []
    license_files: list[str] = []

    for path in ctm.iter_files(base, include_other_text=True):
        relative = path.relative_to(base).as_posix()
        name = path.name.lower()
        if name in LICENSE_FILES and len(path.parts) == len(base.parts) + 1:
            license_files.append(relative)
        if path.name in MANIFEST_NAMES and len(path.parts) == len(base.parts) + 1:
            manifests.append(relative)

    for path in ctm.text_files(base, include_other_text=include_other_text):
        files_scanned += 1
        spdx_found: list[str] = []
        heuristic_found: list[str] = []
        with path.open("r", encoding="utf-8", errors="strict") as stream:
            for index, line in enumerate(stream):
                if index >= HEADER_LINES:
                    break
                spdx_found.extend(SPDX.findall(line))
                heuristic_found.extend(LICENSE_WORDS.findall(line))
        if not spdx_found and not heuristic_found:
            continue
        entry: dict[str, Any] = {
            "path": path.relative_to(base).as_posix(),
            "spdx": sorted(set(spdx_found)),
            "heuristic": sorted(set(heuristic_found)),
        }
        files_with_license.append(entry)
        for identifier in entry["spdx"]:
            by_spdx[identifier] += 1
        for name in entry["heuristic"]:
            by_heuristic[name] += 1

    files_with_license.sort(key=lambda item: item["path"])
    return {
        "tool": "ctm_license_inventory",
        "root": str(base),
        "files_scanned": files_scanned,
        "files_with_license_header": len(files_with_license),
        "by_spdx": dict(sorted(by_spdx.items())),
        "by_heuristic": dict(sorted(by_heuristic.items())),
        "files": files_with_license,
        "manifests": sorted(manifests),
        "license_files": sorted(license_files),
    }


def render(report: dict[str, Any], fmt: str) -> str:
    if fmt == "json":
        return json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if fmt == "csv":
        from io import StringIO

        stream = StringIO()
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["path", "spdx", "heuristic"])
        for item in report["files"]:
            writer.writerow([item["path"], ",".join(item["spdx"]), ",".join(item["heuristic"])])
        return stream.getvalue()
    lines = [
        f"CTM license inventory: {report['root']}",
        f"Files scanned: {report['files_scanned']}",
        f"Files with license header: {report['files_with_license_header']}",
    ]
    lines.extend(f"  SPDX {name}: {count}" for name, count in report["by_spdx"].items())
    lines.extend(f"  heuristic {name}: {count}" for name, count in report["by_heuristic"].items())
    lines.append(f"  license files: {', '.join(report['license_files']) or 'n/a'}")
    lines.append(f"  manifests: {', '.join(report['manifests']) or 'n/a'}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inventory license headers and manifests")
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--format", choices=("text", "json", "csv"), default="text")
    parser.add_argument("--output")
    parser.add_argument("--include-other-text", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = license_inventory(args.root, args.include_other_text)
    except OSError as exc:
        parser.error(str(exc))
    rendered = render(report, args.format)
    if args.output:
        Path(args.output).expanduser().write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
