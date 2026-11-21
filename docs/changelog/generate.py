#!/usr/bin/env python3
"""Generate a markdown changelog from git tags + commit history.

This is a *manual* helper (no CI). Run it from the repository root to (re)build
``docs/changelog/index.md``::

    python docs/changelog/generate.py

It reads:
    * git tags (annotated or lightweight)
    * commit messages between consecutive tags (and since the latest tag)
    * the ``VERSION`` file for the current release label

For each release it emits:
    * a version header with the tag/commit date
    * categorized bullet lists (Features, Bug Fixes, Security, Performance,
      Internal) parsed from conventional-commit-style prefixes
      (``feat:``, ``fix:``, ``security:``, ``perf:``, ``refactor:`` …)
    * a "diff" link against the previous release using the GitHub compare
      viewer — a *visual diff* that renders as a button in the static docs

The emitted markdown is written to ``docs/changelog/index.md`` and also echoed
to stdout when ``--print`` is passed.

Usage:
    python docs/changelog/generate.py [--repo ammar49-cyber/sneppx-alg] [--print]
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# git helpers
# --------------------------------------------------------------------------- #
def _git(*args: str, check: bool = True) -> str:
    """Run a git command in REPO_ROOT and return stdout."""
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True,
        text=True,
        check=check,
    )
    return proc.stdout.strip()


def list_tags() -> list[str]:
    """Return tags sorted by version-descending (newest first)."""
    try:
        out = _git("tag", "--sort=-v:refname", check=True)
    except subprocess.CalledProcessError:
        return []
    return [t for t in out.splitlines() if t]


def latest_tag() -> str | None:
    tags = list_tags()
    return tags[0] if tags else None


def tag_date(tag: str) -> str:
    try:
        raw = _git("log", "-1", "--format=%cI", tag, check=True)
        return _iso_to_date(raw)
    except subprocess.CalledProcessError:
        return ""


def commits_since(base: str | None, head: str = "HEAD") -> list[str]:
    """Commit subjects between base..head (or all commits if base is None)."""
    rng = f"{base}..{head}" if base else head
    try:
        out = _git("log", "--pretty=format:%s", rng, check=True)
    except subprocess.CalledProcessError:
        return []
    return [c for c in out.splitlines() if c]


def current_version_label() -> str:
    """Label for the unreleased/next section: from VERSION file or git describe."""
    version_file = REPO_ROOT / "VERSION"
    if version_file.exists():
        first = version_file.read_text().strip().splitlines()[0]
        # VERSION holds e.g. "SneppX-ALG algo1.1.1"
        m = re.search(r"(\d+\.\d+\.\d+)", first)
        if m:
            return f"v{m.group(1)}"
        return first
    try:
        return f"v{_git('describe', '--tags', check=False) or '0.0.0'}"
    except subprocess.CalledProcessError:
        return "v0.0.0"


def unreleased_flag(tag: str | None) -> bool:
    """True if the current HEAD is ahead of the given tag (i.e. changes are
    unreleased)."""
    if tag is None:
        return True
    try:
        _git("merge-base", "--is-ancestor", tag, "HEAD", check=True)
    except subprocess.CalledProcessError:
        return False  # tag is ahead of HEAD (shouldn't happen in practice)
    # HEAD is descendant of tag — check there's actually a diff
    diff = _git("rev-list", f"{tag}..HEAD", "--count", check=False)
    return int(diff or "0") > 0


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
CATEGORIES = [
    ("Features", "feat"),
    ("Security", "security"),
    ("Bug Fixes", "fix"),
    ("Performance", "perf"),
    ("Refactor / Internal", "refactor"),
    ("Build / Tooling", "build"),
]


def categorize(commits: list[str]) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {name: [] for name, _ in CATEGORIES}
    buckets["Other"] = []
    for c in commits:
        handled = False
        for name, prefix in CATEGORIES:
            if c.lower().startswith(prefix + ":") or c.lower().startswith(prefix + "("):
                buckets[name].append(c)
                handled = True
                break
        if not handled:
            buckets["Other"].append(c)
    return buckets


def _iso_to_date(iso: str) -> str:
    iso = iso.strip()
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return iso


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def render_release(
    label: str,
    date: str,
    commits: list[str],
    repo: str,
    prev_tag: str | None,
) -> str:
    lines: list[str] = []
    lines.append(f"## {label}")
    if date:
        lines.append(f"**{date}**")
    else:
        lines.append("_unreleased_")
    lines.append("")
    if commits:
        buckets = categorize(commits)
        for name, _ in CATEGORIES:
            items = buckets[name]
            if items:
                lines.append(f"### {name}")
                lines.append("")
                for c in items:
                    lines.append(f"- {c}")
                lines.append("")
        if buckets["Other"]:
            lines.append("### Other")
            lines.append("")
            for c in buckets["Other"]:
                lines.append(f"- {c}")
            lines.append("")
    else:
        lines.append("_No tagged changes; showing commits up to this tag._")
        lines.append("")
    # visual-diff link against the previous release
    if prev_tag:
        url = f"https://github.com/{repo}/compare/{prev_tag}...{label}"
    else:
        url = f"https://github.com/{repo}/commits/{label}"
    lines.append(
        f"[:material-git-compare: `View diff vs previous`]({url}){{ .md-button }}"
    )
    lines.append("")
    return "\n".join(lines)


def build_changelog(repo: str) -> str:
    tags = list_tags()
    label_current = current_version_label()
    out: list[str] = []
    out.append("# Changelog\n")
    out.append(
        "Auto-generated from git tags + commit history. Regenerate with "
        "`python docs/changelog/generate.py`. No CI writes this file.\n"
    )
    out.append("<!-- prettier-ignore -->\n")
    out.append("[TOC]\n")

    prev_tag: str | None = None

    # 1) Unreleased section (HEAD ahead of latest tag)
    head_tag = tags[0] if tags else None
    unreleased = commits_since(head_tag)
    if unreleased:
        out.append(render_release("Unreleased", "", unreleased, repo, head_tag))
        prev_tag = head_tag

    # 2) Each tagged release, walking newest -> oldest
    for i, tag in enumerate(tags):
        date = tag_date(tag)
        prev = tags[i + 1] if i + 1 < len(tags) else None
        commits = commits_since(prev, tag)
        out.append(render_release(tag, date, commits, repo, prev))

    # Attach a footer note about the diff-link scheme
    out.append("---\n")
    out.append(
        "Each release header links to a **GitHub compare view** "
        "(`{prev}...{tag}`) which renders a side-by-side **visual diff** of "
        "the source tree between releases. Tag the repo (`git tag v1.2.0`) "
        "and re-run `python docs/changelog/generate.py` to refresh.\n"
    )
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default="ammar49-cyber/sneppx-alg",
        help="GitHub owner/repo for compare links",
    )
    parser.add_argument(
        "--print", action="store_true", help="Also print changelog to stdout"
    )
    args = parser.parse_args()

    md = build_changelog(args.repo)
    target = REPO_ROOT / "docs" / "changelog" / "index.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(md, encoding="utf-8")
    print(f"wrote {target} ({len(md.splitlines())} lines)")
    if args.print:
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
