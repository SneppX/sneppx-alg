# Contributing to SNEPPX-Algo

> *Security-first AI infrastructure requires trust, not just code. Every contributor earns their access through demonstrated competence.*

This guide describes the contribution framework, tier system, and workflow.

---

## Quick Start

```bash
# Clone
git clone https://github.com/SneppX/sneppx-alg.git
cd sneppx-alg

# Build
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# Test
cd build && ctest -C Release --output-on-failure
```

---

## Contributor Tiers

SNEPPX-Algo uses a **five-tier merit-based system**. See `docs/CONTRIBUTOR_TIERS.md` for full details.

| Tier | Role | Can Submit PRs | Can Merge |
|------|------|---------------|-----------|
| T1 | Explorer | No | No |
| T2 | Contributor | Yes | No |
| T3 | Senior | Yes | No (can approve) |
| T4 | Maintainer | Yes | Yes |
| T5 | Core | Yes | Yes (full access) |

All source files must follow the four-layer commenting standard in [docs/COMMENTING.md](docs/COMMENTING.md). Run `sneppx-format --docs` to verify before submitting.

Start at Tier 1 (Explorer) by reading the docs, building the project, and submitting an issue.

---

## Contribution Workflow

### 1. Ideation

Open an issue describing the bug or feature. For significant changes (Tier 3+), write a Design Document first at `docs/proposals/<feature>.md`.

### 2. Branch

```bash
git checkout -b feature/<track>-<name>
```

Track prefixes: `python`, `c-core`, `security`, `cuda`, `algo`, `docs`, `infra`

### 3. Develop

- Follow `docs/STYLE_GUIDE.md`
- Run linter: `pre-commit run --all-files`
- Run tests: `ctest -C Release --output-on-failure`
- Sign commits: `git commit -S`

### 4. Submit PR

Open a PR using the template at `docs/PR_TEMPLATE.md`. Reference the issue and design doc (if applicable).

### 5. Review

- Tier 2 contributors need 1 approval from a Senior+
- Tier 3+ contributors may also approve lower-tier PRs
- Security-related changes require an L3 security review
- See `docs/CODE_REVIEW_GUIDE.md` for detailed review criteria

### 6. Merge

Maintainer+ merges via squash or rebase. PR must have:

- All conversations resolved
- Signed commits
- Approval from required reviewers
- All tests passing locally

### 7. Post-Merge

- Update `CHANGELOG.md` if user-facing
- Name added to `AUTHORS` (first merged PR)

---

## Acceptance Criteria

Every PR must pass before merge:

```md
[ ] Builds without warnings
[ ] All tests pass
[ ] Linter clean
[ ] Test coverage ≥ 80% for new code
[ ] Signed commits
[ ] Documentation updated
[ ] Required approvals obtained
[ ] All conversations resolved
```

---

## Learning Paths

## Branching Strategy

See `docs/BRANCHING_STRATEGY.md` for the full branching model and branch naming conventions.

---

## Learning Paths

Choose your track and follow the roadmap in `docs/LEARNING_PATHS.md`:

- **Python** — Bindings, CLI, tests
- **C/C++ Core** — Kernel, tensor, optimizer, autodiff
- **Security** — Cryptography, hardening, formal verification
- **CUDA** — GPU kernels, flash attention, NCCL
- **Algorithms** — HSS, SER, ARC, NPE, FM
- **Documentation** — Technical writing, API docs
- **Infrastructure** — Build system, packaging, scripts

---

## Code Review

See `docs/CODE_REVIEW_GUIDE.md` for:

- Review checklist
- Per-subsystem review criteria
- Review levels (L1–L3)
- How to write good review comments

---

## Security

- All security-related PRs require an L3 security review
- Report vulnerabilities to algoSNEPPX@gmail.com
- See `SECURITY.md` for full security policy

---

## Governance

See `GOVERNANCE.md` for:

- BDFL model
- Maintainer responsibilities
- Promotion process

---

## Code of Conduct

See `CODE_OF_CONDUCT.md`. All contributors must follow it.

---

## License

By contributing, you agree that your contributions are licensed under the MIT License (see `LICENSE`).

---

**Questions?** Contact: algoSNEPPX@gmail.com
