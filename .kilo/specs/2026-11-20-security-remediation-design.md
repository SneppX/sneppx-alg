# SneppX-ALG Security Remediation Design

**Date:** 2026-11-20  
**Target:** `SneppX/sneppx-alg`, default branch `main`  
**Status:** Approved for execution by repository owner

## Intent

Remediate the confirmed security defects from the Phase 3 investigation. The work must turn security-sensitive APIs into verifiable controls rather than names or placeholders that imply protection they do not provide.

## Goals

- Make HTTP authentication fail closed when state, credentials, or middleware registration is unavailable.
- Provide explicit, safe API-key provisioning and constant-time verification.
- Bind signed-update signatures to canonical metadata and every delivered payload byte.
- Reject malformed delta patches without out-of-bounds reads, writes, or partial output.
- Harden HTTP request parsing, filesystem containment, resource limits, and network exposure.
- Isolate browser-executed Python from the documentation page and its origin.
- Reduce CI/CD write permissions and dependency drift.
- Add regression tests and clear security limitations for functionality that cannot yet be implemented safely.

## Non-goals

- Do not claim that the project is fully secure or formally verified.
- Do not invent TPM, TUF, TLS, or hardware-backed guarantees where no real provider exists.
- Do not silently remove public APIs. Where an API cannot be made secure without a signature or key parameter, add a safe replacement and make the legacy API fail closed with migration documentation.
- Do not rewrite unrelated tensor, model, or optimizer functionality.

## Approaches considered

1. **Cosmetic hardening:** add input checks around existing logic. Rejected because it would leave fake signatures, weak hashes, and missing payload binding intact.
2. **Fail-closed disablement:** disable every questionable API. Safe as an emergency measure, but unnecessarily removes functionality where the repository already has Ed25519 and BLAKE3 primitives.
3. **Real cryptographic contracts with fail-closed legacy APIs:** use existing cryptographic primitives, define canonical signed data, verify payload bytes, and disable legacy interfaces that cannot express the necessary trust inputs. Selected because it restores real security semantics while preserving safe functionality.

## Architecture decisions

- Use the repository's existing Ed25519 and BLAKE3 implementations after validating them against published vectors and adversarial cases.
- Treat a missing trusted key, malformed metadata, invalid signature, invalid payload, failed auth initialization, or failed middleware registration as denial.
- Use explicit, in-memory API-key registration for the C HTTP auth component unless a real persistent store is implemented and tested. The existing `db_path` parameter must not be represented as a working database when it is not.
- Use a versioned, bounds-checked delta-patch format. Reject malformed and legacy ambiguous encodings rather than guessing.
- Keep public HTTP binding loopback-only by default. Public exposure requires a separately verified TLS termination layer and must be documented.
- Treat browser code as untrusted. It must not share a privileged page context or access same-origin application state.
- Preserve API compatibility where safe; introduce versioned APIs when required to carry public keys, signatures, or payload data.

## Constraints and execution rules

- Push directly to `main`; do not create pull requests.
- Never force-push or rewrite existing history.
- Make meaningful, independently verifiable commits.
- Before starting the next commit, all required checks for the previous `main` commit must be green.
- Add regression coverage for every security fix; never weaken tests to obtain green CI.
- Never commit credentials, private keys, tokens, or generated secrets.
- Preserve existing public behavior where it is safe; document any intentionally disabled legacy behavior.
- Report exact commit SHAs and actual workflow results. Do not claim a check passed without reading its result.

## Residual-risk policy

A security API that cannot authenticate its inputs or verify its outputs must return failure and be documented as unsupported. A successful build is not proof of cryptographic correctness. Cryptographic changes require published test vectors, malformed-input tests, and review of the full call path.
