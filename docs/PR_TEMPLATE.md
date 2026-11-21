# Pull Request Template

Use this template when submitting a PR to SNEPPX-Algo.

---

## Description

<!-- Briefly describe the change. Reference the issue/design doc. -->

Fixes #<!-- issue number -->
Closes #<!-- issue number -->

## Type of Change

- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change
- [ ] Documentation update
- [ ] Refactor / performance
- [ ] Security fix
- [ ] Test addition / improvement

## Contributor Tier

<!-- Required. Update as appropriate. -->

- [ ] Tier 1 — Explorer (issues/docs only)
- [ ] Tier 2 — Contributor
- [ ] Tier 3 — Senior Contributor
- [ ] Tier 4 — Maintainer
- [ ] Tier 5 — Core Maintainer

## Checklist

### Code Quality

- [ ] Builds without warnings (`cmake --build`)
- [ ] All tests pass (`ctest -C Release --output-on-failure`)
- [ ] Linter clean (`pre-commit run --all-files`)
- [ ] Test coverage ≥ 80% for new code
- [ ] No new clang-tidy warnings
- [ ] Follows `docs/STYLE_GUIDE.md`

### Security

- [ ] No secrets, credentials, or tokens in code
- [ ] No hardcoded cryptographic parameters
- [ ] Memory allocations checked for NULL
- [ ] Secure wipe of sensitive data (if applicable)
- [ ] Constant-time operations (if cryptographic)

### Documentation

- [ ] Public API documented with docstrings
- [ ] `CHANGELOG.md` updated (if user-facing)
- [ ] Relevant docs in `docs/` updated
- [ ] Design doc referenced (if Tier 3+)

### Commits

- [ ] Commits signed with GPG or Ed25519 key
- [ ] Conventional commit messages (`component: message`)
- [ ] Branch name: `feature/<track>-<name>` or `fix/<track>-<name>`

## Design Document

<!-- Link to design doc in docs/proposals/ if applicable -->

## Additional Context

<!-- Any additional information, screenshots, benchmark results -->

## Reviewer Request

<!-- Who should review this PR? What level of review is needed? -->

- [ ] L1 — Basic review (correctness, style)
- [ ] L2 — Deep review (performance, safety)
- [ ] L3 — Security review (crypto, constant-time)

---

*By submitting this PR, I confirm that my contributions are made under the MIT License.*
