# Python Packaging and Release

This repository has two Python project configurations. They serve different workflows and must not be treated as interchangeable release roots.

## Build-root responsibilities

| Configuration | Purpose | Release status |
|---|---|---|
| `pyproject.toml` at the repository root | Editable development install, repository-wide tooling, documentation dependencies, and workspace integration | Development workspace only; do not publish its artifacts |
| `bindings/python/pyproject.toml` | Standalone Python distribution containing the `SneppX_ALG` package | Canonical PyPI release project |

The published PyPI 1.5.2 metadata reflects the previous nested-project configuration. The 1.5.3 candidate aligns the nested project with the workspace metadata while keeping `bindings/python` as the canonical PyPI build root.

## Build and validate the release artifacts

Run from the repository root:

```bash
python -m pip install --upgrade build
python -m build --sdist --wheel --outdir dist bindings/python
```

Test both generated artifacts before publishing:

1. Create a clean virtual environment.
2. Install the wheel with runtime dependencies.
3. Import `SneppX_ALG` and confirm the installed distribution version.
4. Run the focused autograd regression suite.
5. Rebuild a wheel from the source distribution and repeat the install and tests.
6. Record the wheel and source archive SHA-256 hashes.

The GitHub Actions workflow **Python package artifact validation** first checks that the two project files agree on distribution name, version, and supported Python range, and that each package-discovery root points to the intended source tree. It then performs the build, rebuild, clean-install, and autograd regression checks for both project roots. The nested project is the release candidate; the root build is a development-workspace compatibility check.

## Versioning rules

- Keep the `version` field in both project files synchronized.
- Update the release notes and MkDocs navigation when preparing a release.
- The packaging validator enforces parity for release metadata, dependencies, URLs, console entry points, optional extras, and package data. The package-discovery roots intentionally remain different.
- Do not publish the root workspace artifact to PyPI.
- Do not reuse a published version number after changing package contents.

## Published 1.5.2 artifact discrepancy

The current-main package validation compares the published PyPI 1.5.2 wheel with the current nested build. The `SneppX_ALG/interface_bindings/autograd_ops.py` member differs:

| Artifact | File size | SHA-256 |
|---|---:|---|
| Current bindings build | 227,986 bytes | `0f52d8afb175fed681cfee73f091f4e0c40e6ea378a171fd7973c585f51928c0` |
| Published PyPI 1.5.2 | 151,545 bytes | `b896ad7c0e8655a8bc15aec7933aecd4d6b7319fe34585d5bb0ad055ca6fab21` |

The published file lacks current matrix/vector gradient and broadcast-gradient handling. The current source-built wheel and the wheel rebuilt from its source distribution both pass the 19 focused autograd regression tests. The published 1.5.2 test baseline still has two known failures in matrix multiplication gradients and multiple-output gradients.

A corrected release must use a new version number and be tested from the exact release artifacts. No replacement release has been published.

## Publishing policy

Publishing is a separate release action. This repository workflow validates artifacts but does not upload them to PyPI. Do not add PyPI credentials to the repository or workflow. A release must be explicitly approved, and the exact wheel and source archive must pass the release checks first.

## 1.5.3 release candidate

The next patch candidate is version 1.5.3. It contains the current-main
autograd fixes and aligned standalone distribution metadata. The candidate
must pass both generated-artifact regression runs and the independent
functional suite before publication. The 1.5.2 PyPI artifact is immutable
and remains the published baseline until a new version is explicitly
approved and uploaded. No PyPI credentials or upload workflow are added
by this preparation work.

The root VERSION file and bindings README badge identify this as 1.5.3-rc.1; CITATION.cff intentionally remains at the latest published version until release approval.
