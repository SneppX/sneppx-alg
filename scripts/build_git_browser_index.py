#!/usr/bin/env python3
"""Generate a static index of every Git branch, tag, tree, and commit."""

from __future__ import annotations
import argparse
import base64
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def git(*args: str, binary: bool = False):
    result = subprocess.run(
        ["git", *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=not binary,
    )
    return result.stdout


def identifier(name: str) -> str:
    return base64.urlsafe_b64encode(name.encode()).decode().rstrip("=")


def refs(prefix: str):
    output = git("for-each-ref", "--format=%(refname:short)%09%(objectname)", prefix)
    result = []
    for line in output.splitlines():
        name, sha = line.split("\t", 1)
        if prefix == "refs/remotes/origin":
            if name == "origin/HEAD":
                continue
            name = name.removeprefix("origin/")
        result.append((name, sha))
    return sorted(result, key=lambda item: item[0].casefold())


def tree(ref: str):
    raw = git("ls-tree", "-r", "-l", "-z", "--full-tree", ref, binary=True)
    entries = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, kind, sha, size = metadata.decode("ascii").split(" ", 3)
        entries.append(
            {
                "path": raw_path.decode("utf-8", "replace"),
                "mode": mode,
                "type": kind,
                "sha": sha,
                "size": None if size.strip() == "-" else int(size.strip()),
            }
        )
    return entries


def history(ref: str):
    output = git("log", "--format=%H%x09%an%x09%aI%x09%s", ref, "--")
    commits = []
    for line in output.splitlines():
        parts = line.split("\t", 3)
        if len(parts) == 4:
            sha, author, date, subject = parts
            commits.append(
                {"sha": sha, "author": author, "date": date, "subject": subject}
            )
    return commits


SEARCH_GLOBS = (
    "*.py",
    "*.pyi",
    "*.c",
    "*.h",
    "*.cc",
    "*.hh",
    "*.cpp",
    "*.hpp",
    "*.cxx",
    "*.hxx",
    "*.js",
    "*.jsx",
    "*.ts",
    "*.tsx",
    "*.mjs",
    "*.cjs",
    "*.java",
    "*.kt",
    "*.rs",
    "*.go",
    "*.swift",
    "*.cs",
    "*.rb",
    "*.php",
    "*.sh",
    "*.bash",
    "*.zsh",
    "*.ps1",
    "*.psm1",
    "*.sql",
    "*.json",
    "*.yaml",
    "*.yml",
    "*.toml",
    "*.ini",
    "*.cfg",
    "*.cmake",
    "CMakeLists.txt",
    "*.md",
    "*.html",
    "*.css",
    "*.scss",
    "*.xml",
    "*.proto",
    "*.graphql",
)


def parse_search_lines(output: str, max_lines: int = 12000, max_bytes: int = 1800000):
    results, used, truncated = [], 0, False
    for record in output.splitlines():
        path, separator, remainder = record.partition(":")
        line_number, separator2, content = remainder.partition(":")
        if not separator or not separator2 or not line_number.isdigit():
            continue
        item = {"path": path, "line": int(line_number), "text": content[:1200]}
        cost = len(json.dumps(item, ensure_ascii=False).encode("utf-8")) + 1
        if len(results) >= max_lines or used + cost > max_bytes:
            truncated = True
            break
        results.append(item)
        used += cost
    return results, truncated


def search_index(ref: str):
    return parse_search_lines(
        git("grep", "-I", "-n", "-E", "-e", ".", ref, "--", *SEARCH_GLOBS)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="docs/git-browser-data")
    parser.add_argument("--repository", default="SneppX/sneppx-alg")
    args = parser.parse_args()
    out = Path(args.output)
    trees, commits, search = out / "trees", out / "commits", out / "search"
    trees.mkdir(parents=True, exist_ok=True)
    commits.mkdir(parents=True, exist_ok=True)
    search.mkdir(parents=True, exist_ok=True)
    branches = []
    for name, sha in refs("refs/remotes/origin"):
        key = identifier(name)
        entries = tree(sha)
        log = history(sha)
        search_lines, search_truncated = search_index(sha)
        (search / (key + ".json")).write_text(
            json.dumps(
                {
                    "branch": name,
                    "commit": sha,
                    "truncated": search_truncated,
                    "lines": search_lines,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        (trees / (key + ".json")).write_text(
            json.dumps(
                {"branch": name, "commit": sha, "entries": entries},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        (commits / (key + ".json")).write_text(
            json.dumps(
                {"branch": name, "commits": log},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        branches.append(
            {
                "name": name,
                "id": key,
                "commit": sha,
                "file_count": len(entries),
                "commit_count": len(log),
                "search_count": len(search_lines),
                "search_truncated": search_truncated,
            }
        )
    tags = []
    for name, object_sha in refs("refs/tags"):
        sha = git("rev-parse", name + "^{commit}").strip()
        date = git("show", "-s", "--format=%cI", sha).strip()
        key = identifier("tag:" + name)
        entries = tree(sha)
        (trees / (key + ".json")).write_text(
            json.dumps(
                {"branch": name, "commit": sha, "entries": entries},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        tags.append(
            {
                "name": name,
                "id": key,
                "object": object_sha,
                "commit": sha,
                "date": date,
                "file_count": len(entries),
            }
        )
    manifest = {
        "repository": args.repository,
        "default_branch": "main",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "branches": branches,
        "tags": tags,
    }
    (out / "index.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Indexed {len(branches)} branches, {len(tags)} tags, "
        f"{sum(item['file_count'] for item in branches)} branch-file entries."
    )


if __name__ == "__main__":
    main()
