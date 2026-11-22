#!/usr/bin/env python3
"""Check that the two Python project files describe compatible build roots."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_PROJECT = ROOT / "pyproject.toml"
BINDINGS_PROJECT = ROOT / "bindings" / "python" / "pyproject.toml"
PACKAGE_DIR = ROOT / "bindings" / "python" / "SneppX_ALG"


def read_project(path: Path) -> dict:
    try:
        with path.open("rb") as stream:
            return tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise SystemExit(f"Cannot read {path.relative_to(ROOT)}: {exc}") from exc


def main() -> int:
    root_config = read_project(ROOT_PROJECT)
    bindings_config = read_project(BINDINGS_PROJECT)
    root_project = root_config.get("project", {})
    bindings_project = bindings_config.get("project", {})
    errors: list[str] = []

    for field in (
        "name",
        "version",
        "requires-python",
        "description",
        "readme",
        "license",
        "authors",
        "keywords",
        "classifiers",
        "dependencies",
        "urls",
        "scripts",
        "optional-dependencies",
    ):
        root_value = root_project.get(field)
        bindings_value = bindings_project.get(field)
        if not root_value or not bindings_value:
            errors.append(f"Both project files must declare project.{field}.")
        elif root_value != bindings_value:
            errors.append(
                f"project.{field} differs: root={root_value!r}, "
                f"bindings={bindings_value!r}"
            )

    root_build = root_config.get("build-system", {})
    bindings_build = bindings_config.get("build-system", {})
    for field in ("requires", "build-backend"):
        if root_build.get(field) != bindings_build.get(field):
            errors.append(
                f"build-system.{field} differs: root={root_build.get(field)!r}, "
                f"bindings={bindings_build.get(field)!r}"
            )

    root_package_data = (
        root_config.get("tool", {}).get("setuptools", {}).get("package-data")
    )
    bindings_package_data = (
        bindings_config.get("tool", {}).get("setuptools", {}).get("package-data")
    )
    if root_package_data != bindings_package_data:
        errors.append(
            "Setuptools package-data declarations differ between build roots."
        )

    root_where = (
        root_config.get("tool", {})
        .get("setuptools", {})
        .get("packages", {})
        .get("find", {})
        .get("where")
    )
    bindings_where = (
        bindings_config.get("tool", {})
        .get("setuptools", {})
        .get("packages", {})
        .get("find", {})
        .get("where")
    )
    if root_where != ["bindings/python"]:
        errors.append(
            "Root workspace package discovery must point to ['bindings/python']; "
            f"found {root_where!r}."
        )
    if bindings_where != ["."]:
        errors.append(
            "Standalone bindings discovery must use ['.']; "
            f"found {bindings_where!r}."
        )
    if not (PACKAGE_DIR / "__init__.py").is_file():
        errors.append("The canonical SneppX_ALG package entry point is missing.")

    if errors:
        print("Python packaging metadata validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(
        "Packaging metadata is aligned: "
        f"{bindings_project['name']} {bindings_project['version']} "
        f"(Python {bindings_project['requires-python']})."
    )
    print("Canonical PyPI build root: bindings/python")
    print("Repository root: development workspace compatibility build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
