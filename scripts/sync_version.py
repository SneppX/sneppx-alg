#!/usr/bin/env python3
"""
Version synchronization script for SNEPPX-ALG.
Keeps VERSION file, pyproject.toml, CMakeLists.txt, and Python __version__ in sync.
"""
import re
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
VERSION_FILE = ROOT / "VERSION"
PYPROJECT_FILE = ROOT / "pyproject.toml"
CMAKE_FILE = ROOT / "CMakeLists.txt"
INIT_FILE = Path(__file__).parent / "bindings" / "python" / "SneppX_ALG" / "interface_bindings" / "__init__.py"

VERSION_PATTERN = re.compile(r'algo([0-9]+\.[0-9]+\.[0-9]+)')

def read_version() -> str:
    """Read version from VERSION file."""
    content = VERSION_FILE.read_text(encoding='utf-8')
    match = VERSION_PATTERN.search(content)
    if not match:
        raise ValueError("Could not parse version from VERSION file")
    return match.group(1)

def update_pyproject(version: str) -> None:
    """Update version in pyproject.toml."""
    content = Path(PYPROJECT_FILE).read_text(encoding='utf-8')
    new_content = re.sub(r'version\s*=\s*"[^"]+"', f'version = "{version}"', content)
    Path(PYPROJECT_FILE).write_text(new_content, encoding='utf-8')
    print(f"Updated pyproject.toml to {version}")

def update_cmake(version: str) -> None:
    """Update version in CMakeLists.txt."""
    content = Path(CMAKE_FILE).read_text(encoding='utf-8')
    new_content = re.sub(
        r'project\(cognitive_processing_system VERSION [0-9.]+',
        f'project(cognitive_processing_system VERSION {version}',
        content
    )
    Path(CMAKE_FILE).write_text(new_content, encoding='utf-8')
    print(f"Updated CMakeLists.txt to {version}")

def update_init(version: str) -> None:
    """Update __version__ in Python package."""
    if INIT_FILE.exists():
        content = INIT_FILE.read_text(encoding='utf-8')
        new_content = re.sub(r'__version__\s*=\s*"[^"]+"', f'__version__ = "{version}"', content)
        INIT_FILE.write_text(new_content, encoding='utf-8')
        print(f"Updated interface_bindings/__init__.py to {version}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python sync_version.py [major|minor|patch|rc|<version>]")
        print(f"Current version: {read_version()}")
        return 1

    action = sys.argv[1]
    current = read_version()
    major, minor, patch = map(int, current.split('.'))

    if action == "major":
        new_version = f"{major + 1}.0.0"
    elif action == "minor":
        new_version = f"{major}.{minor + 1}.0"
    elif action == "patch":
        new_version = f"{major}.{minor}.{patch + 1}"
    elif action == "rc":
        new_version = f"{major}.{minor}.{patch}-rc1"
    elif re.match(r'^\d+\.\d+\.\d+(?:-rc\d+)?$', action):
        new_version = action
    else:
        print(f"Unknown action: {action}")
        return 1

    print(f"Bumping version: {current} -> {new_version}")

    # Update VERSION file
    VERSION_FILE.write_text(
        f"SneppX-ALG algo{new_version}\n"
        f"Release Date: 2026-07-10\n"
        f"Status: Feature Complete — Vision Transformers, LLM Models, Benchmarking, LLM Architectures\n"
        f"Security: S0-S9 Complete — 21,809+ lines of security implementation\n"
        f"Build: cmake 3.16+, C11/C++20, Python 3.11+, macOS/Linux/Windows\n"
        f"Total Codebase: ~114,000 lines across 500+ files\n",
        encoding='utf-8'
    )

    update_pyproject(new_version)
    update_cmake(new_version)
    update_init(new_version)

    print(f"✅ Version synchronized to {new_version}")
    return 0

if __name__ == "__main__":
    sys.exit(main())