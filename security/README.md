# Security — S0–S9 Multi-Layer Defense

The `security/` directory implements **10 layers of defense (S0–S9)** plus supplementary subsystems.
Written in C11 with MASM x64 hot-path routines and Python bindings. 21,809+ lines of C code,
31 Python test suites, 300+ tests.

---

## Security Layers

### S0 — Build Integrity
Ensures the software supply chain from source to binary is verifiable and tamper-proof.
- **SBOM** generation and verification
- **Reproducible builds** with deterministic compilation
- **TUF** (The Update Framework) for metadata signing

### S1 — Memory Hardening
Runtime memory protection against corruption, leakage, and exploitation.
- **Guard pages** around heap allocations
- **Stack canaries** for buffer overflow detection
- **ASLR** (Address Space Layout Randomization)
- **Locked memory** to prevent swapping of secrets
- **W^X** policy enforcement
- Files: `memory/memory_hardening.c`, `memory/secure_allocator.c`, `memory/memory_leak_detector.c`

### S2 — Code Obfuscation
C++-based obfuscation pipeline protecting intellectual property.
- **CFG flattening** — Control-flow graph obfuscation
- **Instruction substitution** — Pattern-based replacement
- **Opaque predicates** — Computed jump targets
- **String encryption** — Runtime decryption of literals
- **VM obfuscation** — Virtual machine-based protection
- Files: `obfuscation/`, `cpp/`

### S3 — Runtime Monitoring
Real-time integrity verification and anomaly detection.
- **Integrity monitoring** — File and memory hash verification
- **Container breakout detection** — Namespace escape detection
- **ML anomaly detection** — Behavioral ML for threat identification
- Files: `monitor/integrity_monitor.c`, `monitor/monitor_advanced.c`, `monitor/container_breakout.c`

### S4 — Network Security
Comprehensive network defense with multi-ring firewall architecture.
- **TLS 1.3** with mTLS, cert pinning, ALPN
- **Noise protocol** (NK/XX/IK handshakes)
- **QUIC** transport support
- **DDoS mitigation** — Rate limiting, connection tracking, SYN flood protection
- **Port knocking** — Sequence-based authentication
- **4-Ring Firewall**: Transport (TLS), Network (CIDR/rate), Application (injection filter), Middleware
- Files: `network/transport_security.c`, `network/ddos_mitigation.c`, `network/identity_management.c`, `firewall/`

### S5 — AI Safety
Protects against AI-specific attacks and ensures responsible output.
- **RLHF safety** — Reward model integration
- **Differential privacy** — Gradient noise for DP-SGD
- **Prompt injection defense** — Semantic filtering
- **Output verification** — Toxicity, bias, and factual accuracy
- **Watermarking** — Model output watermarking
- Files: `ai/rlhf_safety.c`, `ai/differential_privacy.c`, `ai/prompt_filter.c`, `ai/output_verifier.c`

### S6 — AI Sanitizer
Advanced defense against semantic and encoded attacks on ML systems.
- **Semantic injection detection** — Intent classification
- **Encoded attack decoder** — Base64/hex/unicode deobfuscation
- **Token anomaly scoring** — Distribution shift detection
- **Model inversion defense** — Gradient masking
- Files: `ai/s5_extensions.c`, `ai/`

### S7 — Supply Chain Security
End-to-end software update integrity.
- **TUF multi-role keys** — Root/targets/snapshot/timestamp delegation
- **bsdiff deltas** — Binary patch generation and application
- **A/B partitions** — Atomic update with rollback
- **TPM PCR** — Hardware-attested boot measurements
- **Canary rollout** — Staged deployment with health checks
- Files: `updates/signed_update.c`, `updates/container_security.c`, `updates/s7_extensions.c`

### S8 — Formal Verification
Mathematical proof and model checking of critical security properties.
- **TLA+ parser** — Spec language parsing
- **LTL model checking** — Linear temporal logic verification
- **Symbolic execution** — Path exploration
- **Lean 4 export** — Theorem prover integration
- Files: `formal/model_checking.c`, `formal/s8_extensions.c`

### S9 — Penetration Testing
Continuous automated security assessment.
- **CVE scanner** — Dependency vulnerability matching
- **Network fuzzer** — Protocol-aware fuzzing
- **API security scanner** — REST/gRPC endpoint testing
- **Supply chain audit** — Dependency tree analysis
- **Compliance auto-checker** — Regulatory mapping (SOC2/ISO/HIPAA)
- Files: `pentest/network_fuzzer.c`, `pentest/self_audit.c`, `pentest/s9_extensions.c`

---

## Post-Quantum Cryptography

Classical and post-quantum cryptographic primitives in `crypto/c/` with MASM x86-64 acceleration:

| Primitive | Type | Standard |
|-----------|------|----------|
| **Kyber** (ML-KEM) | Key Encapsulation | FIPS 203 (draft) |
| **Dilithium** (ML-DSA) | Digital Signature | FIPS 204 (draft) |
| **SPHINCS+** | Hash-based Signature | FIPS 205 (draft) |
| **X25519** | Key Exchange | RFC 7748 |
| **Ed25519** | Digital Signature | RFC 8032 |
| **AES-256-GCM** | Symmetric Encryption | FIPS 197 |
| **ChaCha20-Poly1305** | AEAD | RFC 8439 |
| **SHA-256/384/512** | Hash | FIPS 180-4 |
| **SHA3-256/512** | Hash | FIPS 202 |
| **BLAKE3** | Hash | — |
| **Argon2id** | Password Hashing | RFC 9106 |
| **HKDF** | Key Derivation | RFC 5869 |
| **PBKDF2** | Password Derivation | RFC 8018 |

MASM-accelerated: AES-NI, SHA-NI, ChaCha20 AVX2, Poly1305 SSE, Ed25519 scalar multiplication,
constant-time operations, secure memory wipe.

---

## Additional Security Subsystems

| Directory | Purpose |
|-----------|---------|
| `blockchain/` | Distributed ledger verification |
| `compliance/` | Regulatory compliance (SOC2, ISO 27001, HIPAA, GDPR) |
| `zero_trust/` | Zero-trust architecture implementation |
| `hardware/` | Hardware security module integration |
| `chaos/` | Chaos engineering for security resilience |
| `threat_intel/` | Threat intelligence feed processing |
| `incident_response/` | Automated incident response playbooks |
| `automation/` | Security automation and orchestration |
| `verification/` | Formal verification tooling |
| `fuzzing/` | Fuzz testing harnesses |
| `bench/` | Security benchmarks |
| `config/` | Security configuration files |
| `ui/` | User interface (key vault, audit logger) |
| `rust/`, `go/`, `cs/` | Polyglot security modules |

## Build Target

| Target | Description |
|--------|-------------|
| `neural_security_c` | C security library (all S0-S9 + crypto) |
| `neural_security_cpp` | C++ obfuscation library (S2) |

## MASM x64 Accelerated Routines

Located in `crypto/asm/x86_64/`:
- `aes_ni.asm` — AES-NI round encryption
- `sha_ni.asm` — SHA-NI hash computation
- `chacha20_avx2.asm` — ChaCha20 AVX2 stream cipher
- `poly1305_sse.asm` — Poly1305 SSE MAC
- `ed25519_scalarmult.asm` — Ed25519 scalar multiplication
- `montgomery_mul.asm` — Montgomery modular multiplication
- `barrett_reduce.asm` — Barrett reduction
- `keccak_f1600_avx2.asm` — Keccak-f1600 permutation
- `constant_time_cmp.asm` — Constant-time comparison
- `secure_wipe.asm` — Secure memory zeroing
- `cache_constant_lookup.asm` — Cache-timing resistant lookup
- `ct_swap_ops.asm` — Constant-time swap
- `sc_cmov.asm` — Conditional move
- `gf256_mul.asm` — GF(256) multiplication
- `fe25519_ops.asm` — Field element operations


## Native Firewall Policy Engine

The native policy engine is available through
`neural_core/security/sneppx_firewall.h` and is compiled into
`neural_security_c`.

- IPv4 and IPv6 source/destination CIDR matching
- IP protocol, ingress/egress direction, and destination-port range matching
- Priority-based rule selection; deny wins equal-priority ties
- Default-deny policy unless explicitly changed
- Bounded rule table (256 rules), thread-safe evaluation, and packet/byte counters
- Explicit rule add/remove operations and result identifiers

The engine evaluates packet metadata supplied by a network adapter. It does not
install Windows Filtering Platform, Linux netfilter/eBPF, or macOS Network
Extension hooks, and it does not capture packets itself. Those platform adapters
must be implemented and deployed separately before it can filter host traffic.


## Python Firewall Deployment

The Python firewall can enforce a default-deny route policy with exact paths and
trailing-prefix patterns such as `/health` and `/v1/*`. Configure these under
`firewall.application.allowed_paths` and set `block_all_by_default: true`.
An empty route list intentionally denies every HTTP path. Body buffering, header
count, and per-client concurrency limits are enforced before inference handlers
receive a request.

TLS configuration accepts explicit TLS 1.2 or TLS 1.3 minimums. Older TLS
versions and malformed certificate fingerprints are rejected during startup.
