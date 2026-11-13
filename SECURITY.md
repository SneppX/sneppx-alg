# Security Policy

## Supported Versions

| Version | Status | Level of Support |
|---------|--------|------------------|
| 1.5.x   | Current | Active feature, bug, and security updates |
| 1.0.x - 1.1.x | Maintained | Critical security patches only |
| < 1.0.0 | End of Life | Unsupported |

## Reporting a Vulnerability

Report security vulnerabilities privately. Do NOT open public issues.

- **GitHub Security Advisory (Preferred):** [Open a private advisory](https://github.com/SneppX/sneppx-alg/security/advisories/new)
- **Direct Maintainer Contact:** `algoSNEPPX@gmail.com`

Do not send secrets, private keys, or production credentials with a report. An authenticated disclosure channel is preferred.

Please include:
- Description of the vulnerability
- Steps to reproduce
- Affected versions and components
- Any proof-of-concept code
- Your contact information for follow-up

### Response Timeline

| Phase | Timeframe |
|-------|-----------|
| Acknowledgment | 48 hours |
| Initial assessment | 7 days |
| Critical fix release | 30 days |
| High fix release | 90 days |

## Security Status

| Phase | Status | Components |
|-------|--------|------------|
| S0 | Experimental / partial | Classical primitives and PQC code exist; PQC interoperability is incomplete and no FIPS validation is claimed |
| S1 | Partial; independent audit required | Memory-hardening and constant-time helpers exist, but the complete layer is not certified or proven safe |
| S2 | Known high-risk findings | RNG, timing, code-transformation, and memory-safety findings are documented; do not rely on S2 as a security boundary |
| S3 | Partial | Monitoring and anomaly-detection components exist; syscall filtering and runtime enforcement are incomplete |
| S4 | Experimental / partial | Network-security components exist; protocol configuration and deployment hardening require independent review |
| S5 | Partial | AI input/output filtering components exist; they are not a complete prompt-injection or data-loss prevention boundary |
| S6 | Incomplete / verify per component | Key-management and audit-viewer capabilities are not a complete managed security service |
| S7 | Incomplete / verify per component | Signed-update and rollback controls require end-to-end release and recovery validation |
| S8 | Not formally verified | Formal-methods scaffolding is not equivalent to machine-checked proofs of the cryptographic implementation |
| S9 | Not independently audited | Test and pentest tooling does not replace an external penetration test or security certification |

## Known Issues

No coordinated vulnerability disclosures are recorded in this policy. This does not mean the codebase has no known security weaknesses. The audit findings below remain open until fixes and regression tests are merged.

### Test Suite Edge Cases

- Ed25519: Fixed to full RFC 8032 compliance (all 4 test vectors pass). The base-point T-ordering bug (init_base_point computed B.T before parity negation) has been resolved.
- Argon2: 1/4 timing test fails on certain hardware
- These are pre-existing S0 limitations and do not represent security vulnerabilities

### S2 Obfuscation — Audit Findings (2026-08-16)

See `docs/security/obf_audit.md` for the full audit report. Key findings:
- **Critical:** S2 uses `rand()` (CWE-338) for obfuscation decisions — predictable PRNG state can make choices reproducible
- **High:** Whitebox AES table lookups are timing-variable (CWE-208)
- **Medium:** `bogus_cf_redirect` has no restore path; its reversible XOR transformation is not cryptographic protection
- These findings are not resolved merely by documenting them. S2 must not be treated as a defense against a capable attacker until the issues are fixed and independently validated. See `docs/security/obf_audit.md`.

## Hall of Fame

No security researchers have been credited yet. Be the first.

## Disclosure Policy

- BDFL decides disclosure timeline
- Coordinated disclosure preferred
- No exploit bounty at this time
