---
name: Bug Report
about: Report a bug to help improve SNEPPX-Alg
title: "[Bug] "
labels: ["bug"]
assignees: []
---

## Description
A clear and concise description of the bug.

> Before filing, please check the [FAQ](https://sneppx.github.io/sneppx-alg/latest/FAQ/) and search
> [existing issues](https://github.com/SneppX/sneppx-alg/issues) to avoid duplicates.

## Reproduction Steps
1. Configure: `cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Debug -DSNEPPX_BUILD_TESTS=ON`
2. Build: `cmake --build .`
3. Run: `ctest --output-on-failure -R <test-name>`
4. Observe: ...

## Expected Behavior
What should happen.

## Actual Behavior
What actually happens. Include error messages, stack traces, or test failures.

## Environment
- OS: [e.g. Ubuntu 22.04, macOS 14, Windows 11]
- Compiler: [e.g. GCC 12, Clang 15, MSVC 2022]
- CMake version: [e.g. 3.25]
- Build type: [Debug / Release]
- SNEPPX-Alg version: [e.g. v1.5.0, commit hash]

## Additional Context
Logs, screenshots, or relevant test output.

> **Security issues**: Do NOT open public issues for security vulnerabilities. Report them privately via [GitHub Security Advisories](https://github.com/SneppX/sneppx-alg/security/advisories/new).
> **BDFL governance**: This project uses BDFL governance. Issues are tracked but not voted on.
