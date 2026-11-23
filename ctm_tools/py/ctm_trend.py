#!/usr/bin/env python3
"""Store successive CTM JSON reports and show LOC/files trends over time."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HISTORY_NAME = "history.jsonl"
METRICS = ("files", "lines_of_code", "bytes")


def _history_path(store: str | Path) -> Path:
    base = Path(store).expanduser().resolve()
    return base / HISTORY_NAME


def add_snapshot(store: str | Path, report_path: str | Path) -> dict[str, Any]:
    """Append a snapshot extracted from a validated CTM JSON report."""
    path = Path(report_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"not a CTM schema-v1 report: {path}")
    entry: dict[str, Any] = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "root": str(payload.get("root", "")),
        "source": str(path),
    }
    for key in METRICS:
        value = payload.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"invalid {key} in report: {path}")
        entry[key] = value
    languages = payload.get("languages")
    if not isinstance(languages, dict):
        raise ValueError(f"invalid languages in report: {path}")
    entry["languages"] = {
        str(name): int(metrics.get("loc", 0))
        for name, metrics in languages.items()
        if isinstance(metrics, dict)
    }
    history = _history_path(store)
    history.parent.mkdir(parents=True, exist_ok=True)
    with history.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    return entry


def read_history(store: str | Path, limit: int = 0) -> list[dict[str, Any]]:
    """Load stored snapshots, skipping malformed lines."""
    if limit < 0:
        raise ValueError("limit must be non-negative")
    history = _history_path(store)
    if not history.exists():
        return []
    entries: list[dict[str, Any]] = []
    with history.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                entries.append(payload)
    if limit > 0:
        entries = entries[-limit:]
    return entries


def trend(store: str | Path, limit: int = 0) -> dict[str, Any]:
    """Return stored snapshots with deltas relative to the first sample."""
    entries = read_history(store, limit)
    first = entries[0] if entries else None
    snapshots: list[dict[str, Any]] = []
    for entry in entries:
        delta = {key: entry.get(key, 0) - first.get(key, 0) if first else 0 for key in METRICS}
        snapshots.append({**entry, "delta": delta})
    return {
        "tool": "ctm_trend",
        "store": str(_history_path(store).parent),
        "snapshots": len(entries),
        "history_file": str(_history_path(store)),
        "series": snapshots,
    }


def render(report: dict[str, Any], fmt: str) -> str:
    if fmt == "json":
        return json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if fmt == "csv":
        from io import StringIO

        stream = StringIO()
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["recorded_at", "root", "files", "loc", "bytes", "loc_delta"])
        for item in report["series"]:
            writer.writerow(
                [
                    item.get("recorded_at", ""),
                    item.get("root", ""),
                    item.get("files", 0),
                    item.get("lines_of_code", 0),
                    item.get("bytes", 0),
                    item.get("delta", {}).get("lines_of_code", 0),
                ]
            )
        return stream.getvalue()
    lines = [
        f"CTM trend: {report['store']}",
        f"Snapshots: {report['snapshots']}",
    ]
    values = [item.get("lines_of_code", 0) for item in report["series"]]
    maximum = max(values, default=0)
    for item, value in zip(report["series"], values):
        bar = "#" * round(40 * value / maximum) if maximum else ""
        lines.append(
            f"  {item.get('recorded_at', '')[:19]}  {value:>8} LOC  "
            f"{item['delta']['lines_of_code']:+d}  {bar}"
        )
    if not values:
        lines.append("  No snapshots recorded. Use --add <report.json>.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Track CTM metrics over time")
    parser.add_argument("store", help="directory holding history.jsonl")
    parser.add_argument("--add", metavar="REPORT", help="append a CTM JSON report")
    parser.add_argument("--limit", type=int, default=0, help="show last N snapshots")
    parser.add_argument("--format", choices=("text", "json", "csv"), default="text")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        if args.add:
            add_snapshot(args.store, args.add)
        report = trend(args.store, args.limit)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    rendered = render(report, args.format)
    if args.output:
        Path(args.output).expanduser().write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
