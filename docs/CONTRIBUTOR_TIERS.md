# SNEPPX-Algo Contributor Tiers

> *Security-first AI infrastructure requires trust, not just code. Every contributor earns their access through demonstrated competence.*

This document defines the five-tier contribution ladder. Each tier has specific prerequisites, gates, and privileges.

---

## Overview

```
TIER    ROLE              PRs    REVIEWS   MERGE    ACCESS
T1      Explorer          0      0         No       Issues, discussions
T2      Contributor       1+     0         No       Submit PRs, vote on issues
T3      Senior            5+     10+       No       Approve PRs, mentor T1/T2
T4      Maintainer        100+   50+       Yes      Merge PRs, manage releases
T5      Core              Leadership        Yes      Full commit, governance
```

---

## Tier 1 — Explorer

**Goal**: Demonstrate basic understanding and good faith.

### Prerequisites

- Read `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `GOVERNANCE.md`
- Read `docs/ARCHITECTURE.md` (full system understanding)
- Build the project from source (`cmake -B build -G Ninja && cmake --build build`)
- Run the test suite (`ctest -C Release --output-on-failure`)

### Gates

| Gate | Description |
|------|-------------|
| G1.1 | Submit one valid issue (bug report with reproduction OR feature request with use case) |
| G1.2 | Have the issue acknowledged by a Maintainer+ |

### Privileges

- Report bugs, request features
- Comment on issues and discussions
- No PR submission rights

### Badge

**Tier 1 — Explorer**

*Time to complete: 1–3 days*

---

## Tier 2 — Contributor

**Goal**: Submit production-quality code that passes all gates.

### Prerequisites

- Hold Tier 1 for ≥ 14 days
- 2 accepted issue reports (your own or collaborative)
- Read `docs/STYLE_GUIDE.md`
- Read `docs/DEVELOPMENT.md`

### Gates

| Gate | Description |
|------|-------------|
| G2.1 | Submit a PR with ≥ 80% test coverage for new code |
| G2.2 | All CI checks pass (build, lint, test) |
| G2.3 | Zero clang-tidy/clang-format warnings |
| G2.4 | Documentation updated for any new public API |
| G2.5 | Commits signed with GPG or Ed25519 key |
| G2.6 | Reviewed and approved by 1 Senior+ contributor |
| G2.7 | PR merged by a Maintainer+ |

### Privileges

- Submit pull requests
- Assigned to issues
- Name listed in `AUTHORS`
- **Badge**: Tier 2 — Contributor

*Time to complete: 2–6 weeks per PR*

---

## Tier 3 — Senior Contributor

**Goal**: Take ownership of a module and mentor others.

### Prerequisites

- Hold Tier 2 for ≥ 90 days
- 5+ merged PRs (at least 2 in the module you seek ownership of)
- Demonstrated code review participation

### Gates

| Gate | Description |
|------|-------------|
| G3.1 | Write a Design Document (`docs/proposals/<feature>.md`) for a new feature or refactor |
| G3.2 | Design document approved by a Maintainer+ |
| G3.3 | Review 10+ PRs from Tier 2 contributors (≥ 5 with substantive comments) |
| G3.4 | Pass a **Security Review** — write a threat model for your module using STRIDE |
| G3.5 | Demonstrate merge conflict resolution, rebase, and `git bisect` |

### Privileges

- Approve PRs from Tier 1 and Tier 2 contributors
- Request changes on PRs
- Assigned module ownership in repository
- Access to project roadmap discussions
- **Badge**: Tier 3 — Senior

*Time to complete: 6–12 months*

---

## Tier 4 — Maintainer

**Goal**: Ensure quality and consistency across the project.

### Prerequisites

- Hold Tier 3 for ≥ 6 months
- 100+ commits in the project
- Lead one major feature from design → implementation → release

### Gates

| Gate | Description |
|------|-------------|
| G4.1 | Contribute to project roadmap in `docs/ROADMAP.md` |
| G4.2 | Nominated by a Core Maintainer and approved by BDFL |
| G4.3 | Pass an **Architecture Review** — explain full system architecture to BDFL |
| G4.4 | Write a release blog post for a minor version |

### Privileges

- Merge PRs (squash or rebase)
- Manage releases and versioning
- npm/PyPI publisher rights
- Set technical direction for modules
- **Badge**: Tier 4 — Maintainer

*Time to complete: 12–18 months as Senior*

---

## Tier 5 — Core Maintainer

**Goal**: Steer the project's future alongside the BDFL.

### Prerequisites

- Hold Tier 4 for ≥ 1 year
- 3+ consecutive years of project involvement
- Demonstrated leadership in 3+ major releases

### Gates

| Gate | Description |
|------|-------------|
| G5.1 | Contribute to governance documents (`GOVERNANCE.md`, `CONTRIBUTING.md`) |
| G5.2 | Unanimous approval from existing Core Maintainers + BDFL |

### Privileges

- Full commit access to all repositories
- Architectural decision-making power
- BDFL succession eligibility
- Vote on governance changes
- **Badge**: Tier 5 — Core

*Time to reach: 3+ years*

---

## Promotion Process

1. Contributor meets all gate criteria for the target tier
2. Opens a promotion request issue using the promotion template
3. Current tier reviewers verify each gate
4. Maintainer+ approves promotion
5. Tier badge is updated in `MAINTAINERS.md`
6. New privileges take effect immediately

## Demotion Process

A contributor may be demoted one tier for:

- Repeated violations of `CODE_OF_CONDUCT.md`
- Malicious code submissions (intentional vulnerabilities, backdoors)
- Extended inactivity (≥ 12 months without contribution)

Demotion requires approval from a Maintainer+ and notification to the contributor.

## Recognition

| Milestone | Recognition |
|-----------|-------------|
| First merged PR | Added to `AUTHORS` |
| Tier 2 reached | Contributor badge + issue/PR assignment rights |
| 10 merged PRs | Listed in `MAINTAINERS.md` as Active Contributor |
| Tier 3 reached | Senior badge + module ownership |
| Tier 4 reached | Maintainer badge + merge rights |
| Tier 5 reached | Core badge + governance rights |
| Security audit contribution | Listed in `SECURITY.md` hall of fame |
