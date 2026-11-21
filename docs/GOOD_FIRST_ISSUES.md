# Good First Issues — sneppx-alg

New to the codebase? This page lists approachable, well-scoped tasks that are a
great way to make your first contribution. Each one is labeled
[`good first issue`](https://github.com/SneppX/sneppx-alg/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
on GitHub.

## How to pick one

1. Browse the [open good-first-issue list](https://github.com/SneppX/sneppx-alg/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).
2. Pick something that matches your skills (Python, C, docs, tests).
3. Comment "I'd like to work on this" so maintainers can assign it to you.
4. Fork, branch, implement, test, open a PR.

## Beginner-friendly areas

### Python bindings (`bindings/python/SneppX_ALG/`)
- Add unit tests for an untested module (see `tests/python/` for the pattern).
- Add a new `nn` module or `optim` optimizer following the existing wrappers.
- Improve docstrings / type hints.

### Documentation (`docs/`)
- Fix typos, clarify build instructions, add examples.
- Add a tutorial (see `docs/tutorials/` for the format).
- Improve the API reference (`docs/api/python.md`).

### Tests (`tests/python/`)
- Add edge-case tests for existing ops.
- Add regression tests for fixed bugs.

### Benchmarks (`benchmarks/`, `tests/benchmark/`)
- Add a new benchmark for an op not yet covered.
- Update `docs/benchmarks.md` with fresh numbers.

## Before you start

- Read [`CONTRIBUTING.md`](https://github.com/SneppX/sneppx-alg/blob/main/CONTRIBUTING.md) and [`CODE_OF_CONDUCT.md`](https://github.com/SneppX/sneppx-alg/blob/main/CODE_OF_CONDUCT.md).
- Build and run one test to confirm your environment works:
  ```powershell
  $env:PYTHONPATH = "bindings/python"
  python -m pytest tests/python/test_tensor.py -q -p no:cacheprovider
  ```
- Follow the commit-message convention: `<area>: <description>` (e.g. `optim: add load_state_dict to SparseAdam`).

## Getting help

- 💬 [Discussions](https://github.com/SneppX/sneppx-alg/discussions) — Q&A and ideas.
- 🐛 [Issues](https://github.com/SneppX/sneppx-alg/issues) — bugs and feature requests.
- 📚 [Contributor Tiers](CONTRIBUTOR_TIERS.md) — the full contribution ladder.

## Not sure where to start?

Look for issues labeled `good first issue` or `documentation`. If none fit,
open a discussion and ask — maintainers are happy to point you at something.
