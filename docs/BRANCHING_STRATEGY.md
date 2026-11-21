# Branching Strategy

SNEPPX-Algo follows a **track-based Git Flow** model adapted for security-first development. Every branch type maps to a contributor tier. All verification is manual — no CI/CD.

---

## Branch Overview

```
PERMANENT:
  main         Production-ready, tagged releases
  dev          Integration branch for active development

TEMPORARY:
  feature/*    Feature development per track
  release/*    Release stabilization
  hotfix/*     Urgent production fixes
  security/*   Security patches (L3 review)
  docs/*       Documentation-only changes
  experiment/* Spikes and research (no merge guarantee)
```

---

## Permanent Branches

### `main`

The `main` branch is **always deployable**. Every commit on `main` is a release candidate.

| Property | Value |
|----------|-------|
| Base | — |
| Protections | No direct pushes — merge only from `release/*`, `hotfix/*`, `security/*` |
| Merge approval | 2 T4+ approvals |
| Tags | Every merge creates a signed semver tag (`v<major>.<minor>.<patch>`) |
| Tier access | T4+ can merge, T5 has full access |

### `dev`

The `dev` branch is the **integration hub**. All feature branches merge here first.

| Property | Value |
|----------|-------|
| Base | `main` |
| Protections | No direct pushes — merge only from `feature/*`, `docs/*` |
| Merge approval | 1 T3+ approval for features, 1 T4+ for structural changes |
| Tier access | T3+ can merge |

---

## Temporary Branches

### Feature Branches (`feature/<track>-<name>`)

For developing new features. Created from `dev`, merged back via squash.

| Property | Value |
|----------|-------|
| Base | `dev` |
| Naming | `feature/<track>-<description>` |
| Track prefixes | `python`, `c-core`, `cuda`, `security`, `algo`, `infra`, `dist` |
| Review | `code-review` label triggers review assignment |
| Merge | Squash-merge to `dev` |
| Deletion | Delete after merge |
| Tier access | T2+ can push and open PRs |

### Release Branches (`release/v<major>.<minor>.<patch>`)

For stabilizing a release. Created from `dev` when feature freeze is declared.

| Property | Value |
|----------|-------|
| Base | `dev` |
| Naming | `release/v<major>.<minor>.<patch>` |
| Allowed changes | Bug fixes, docs, release config, version bumps |
| Merge | Merged to `main` (as a release commit) AND back to `dev` |
| Tags | Created on `main` after merge |
| Tier access | T3+ can push, T4+ can approve merges |

### Hotfix Branches (`hotfix/<name>`)

For urgent production fixes. Created from `main`, merged back to `main` and `dev`.

| Property | Value |
|----------|-------|
| Base | `main` |
| Naming | `hotfix/<short-description>` |
| Merge | Merged to `main` first, then `dev` |
| Tier access | T4+ only |

### Security Branches (`security/<cve-or-name>`)

For coordinated security patches. May use private forks for embargoed fixes.

| Property | Value |
|----------|-------|
| Base | `main` |
| Naming | `security/<CVE-ID>` or `security/<short-name>` |
| Merge | Merged to `main` first, then `dev` |
| Tier access | T4+ only, L3 security review required |
| Embargo | Private fork until coordinated disclosure date |

### Documentation Branches (`docs/<name>`)

For documentation-only changes.

| Property | Value |
|----------|-------|
| Base | `dev` |
| Merge | Squash-merge to `dev` |
| Tier access | T2+ |

### Experiment Branches (`experiment/<name>`)

For research, spikes, and throwaway code. No merge guarantee.

| Property | Value |
|----------|-------|
| Base | `dev` |
| Merge | Discard or rebase into a `feature/*` branch |
| Tier access | T2+ |

---

## Branch Lifecycle Diagram

```
feature/* ──┐
docs/* ─────┤
experiment/*─┤     ┌──────┐     ┌─────────┐     ┌──────┐
             └────►│ dev  │────►│release/*│────►│ main │
                   └──────┘     └─────────┘     └──┬───┘
                                                   │
                        ┌──────┐     ┌─────────┐   │
                        │ main │◄────│hotfix/* │   │
                        └──────┘     └─────────┘   │
                        ┌──────┐     ┌─────────┐   │
                        │ main │◄────│security/*│   │
                        └──────┘     └─────────┘   │
                                                   ▼
                                               (tagged
                                              release)
```

---

## Tier → Branch Mapping

| Tier | Can push to | Can merge to | Can approve to |
|------|------------|-------------|----------------|
| T1 Explorer | Fork only | — | — |
| T2 Contributor | `feature/*`, `docs/*`, `experiment/*` | — | — |
| T3 Senior | Above + `dev`, `release/*` | `dev` | `dev` |
| T4 Maintainer | Above + `main`, `hotfix/*`, `security/*` | `main` | All branches |
| T5 Core | All | All | All |



## Workflow Examples

### Standard Feature Development (T2 Contributor)

```bash
# Start from latest dev
git checkout dev
git pull
git checkout -b feature/python-new-optimizer

# Develop, commit, push
git commit -S -m "python: add fused AdamW optimizer"
git push origin feature/python-new-optimizer

# Open PR to dev using template at docs/PR_TEMPLATE.md
# After review and approval, squash-merge to dev
```

### Urgent Hotfix (T4 Maintainer)

```bash
git checkout main
git pull
git checkout -b hotfix/crypto-oob-read
# Fix, commit, push
git commit -S -m "security: fix out-of-bounds read in Kyber decapsulate"
git push origin hotfix/crypto-oob-read
# Open PR to main, after merge also PR to dev
```

### Coordinated Security Release (T4+ Maintainer)

```bash
# For embargoed fixes, create private fork
git checkout main
git checkout -b security/CVE-2026-1234
# Fix, commit with signed tags
git commit -S -m "security: patch constant-time violation in Ed25519"
# After coordinated disclosure, PR to main then dev
```

---

---

## See Also

- `CONTRIBUTING.md` — contribution workflow
- `docs/CONTRIBUTOR_TIERS.md` — tier definitions and gates
- `docs/DEVELOPMENT.md` — build and test workflow
