#!/usr/bin/env python3
"""Fail when a text file contains a credential literal.

This is a deliberately conservative scanner: it looks for unambiguous key
formats (AWS/GitHub/Slack/Stripe/Google/JWT/PEM) and for long string literals
assigned to credential-like variable names. Placeholder values used in docs,
examples, and tests are ignored. It is wired into pre-commit so accidental
credentials cannot enter the repository.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Iterator, List, Pattern, Tuple

RULES: List[Tuple[str, Pattern[str]]] = [
    ("pem-private-key", re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----")),
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("stripe-live-key", re.compile(r"\b(?:sk|rk)_live_[0-9A-Za-z]{16,}\b")),
    (
        "jwt",
        re.compile(
            r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
        ),
    ),
    (
        "credential-assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|"
            r"password|passwd)\b\s*[:=]\s*[\"']([A-Za-z0-9+/_=-]{16,})[\"']"
        ),
    ),
]

_PLACEHOLDER = re.compile(
    r"example|dummy|fake|placeholder|redact|changeme|your|sample|test|"
    r"[<>{}$]|^([A-Za-z0-9+/_=-])\1+$",
    re.I,
)


def scan_text(text: str, include_assignments: bool = True) -> Iterator[Tuple[int, str]]:
    """Yield ``(line_number, rule_name)`` for each credential-like match."""
    for lineno, line in enumerate(text.splitlines(), 1):
        for name, pattern in RULES:
            # Fixtures and examples deliberately contain fake credential
            # variables; the unambiguous key formats are still checked there.
            if name == "credential-assignment" and not include_assignments:
                continue
            for match in pattern.finditer(line):
                value = match.group(1) if match.groups() else match.group(0)
                if value and _PLACEHOLDER.search(value):
                    continue
                yield lineno, name


def scan_files(paths: Iterable[str]) -> List[str]:
    findings: List[str] = []
    for path in paths:
        normalized = path.replace("\\", "/")
        include_assignments = not (
            normalized.startswith("tests/")
            or normalized.startswith("docs/")
            or normalized.endswith(".example")
        )
        try:
            text = Path(path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, name in scan_text(text, include_assignments):
            findings.append(f"{path}:{lineno}: {name}")
    return findings


def _tracked_files() -> List[str]:
    raw = subprocess.check_output(["git", "ls-files", "-z"])
    return [p.decode("utf-8", "replace") for p in raw.split(b"\0") if p]


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files", nargs="*", help="files to scan (default: git ls-files)"
    )
    args = parser.parse_args(argv)
    findings = scan_files(args.files or _tracked_files())
    for finding in findings:
        print(f"secret scan: {finding}")
    if findings:
        print(
            f"secret scan: {len(findings)} potential credential(s) found",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
