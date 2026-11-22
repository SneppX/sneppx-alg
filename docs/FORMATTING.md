# Repository formatting policy

This repository uses pinned formatters and validation hooks so formatting is reproducible across contributors and CI.

## Supported source formats

- C, C++, headers, and CUDA: `clang-format`, using the root `.clang-format`
- Python: Ruff lint fixes and Ruff formatting
- CMake: gersemi
- JavaScript, TypeScript, JSON, YAML, HTML, CSS, and Markdown: Prettier
- TOML: Taplo
- Shell scripts: shfmt, followed by ShellCheck validation
- Rust: `cargo fmt` for the root workspace and the security crate
- Go: gofmt
- Jupyter notebooks: nbqa with Ruff formatting for code cells
- All tracked text files: trailing-whitespace, end-of-file, and LF line-ending checks

Generated documentation, build output, binary assets, and third-party/vendor directories are not rewritten by source formatters. Assembly, Lean, and standalone C# files receive the repository-wide whitespace and line-ending checks; they are not passed through an unrelated formatter.

## Run locally

Install the required language toolchains, then run:

```sh
python -m pip install pre-commit
pre-commit run --all-files
```

The local-only `sneppx-docs` hook is a separate documentation-coverage check and requires the `sneppx-format` utility.

## Automation

The **Repository-wide formatting** workflow runs on pushes to `main`, weekly, and on manual dispatch. It formats a temporary candidate branch, runs the complete required CI workflow set against that exact tree, and fast-forwards `main` only when every workflow succeeds. It does not open pull requests.

The **Code Formatting Check** workflow runs on pushes and pull requests. It fails if any configured formatter would modify a tracked file. This prevents new formatting drift from entering the repository.

Formatting automation deliberately refuses to promote if `main` changes while validation is running. It never force-pushes or overwrites concurrent work.
