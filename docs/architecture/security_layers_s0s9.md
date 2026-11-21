# Security Layers Deep Dive (S0–S9)

SNEPPX-Algo applies a **defense-in-depth** model: ten stacked, independently
auditable security layers. Each layer has a distinct *mission* and a small,
formally-named surface of `SNEPPX_*` C functions exposed through Python
bindings.

```
 S0  Cryptographic Core         (sign, encrypt, hash, PQ-KEM)
 S1  Secure Memory              (guard pages, canaries, mlock, leak detect)
 S2  Obfuscation Engine         (CF flattening, string encryption, VM)
 S3  Behavioral Monitor         (integrity, breakout, anomaly)
 S4  Network Security           (DDoS, TLS, identity, certs)
 S5  AI Sanitizer               (prompt filter, DP, RLHF safety)
 S6  Key Vault / Security UI    (encrypted store, audit log)
 S7  Secure Updates             (signed bundles, rollback protection)
 S8  Formal Verification        (model checking, symbolic execution)
 S9  Penetration Testing        (fuzzing, self-audit, red team)
```

The layers are **additive**: S0 protects secrets at rest, S1 protects secrets
in memory, S2 protects secrets in code, S3/S4 protect the running process and
its network I/O, S5/S6 protect the AI payload and operator credentials, and
S7–S9 protect the supply chain and verify the whole stack.

---

## S0 — Cryptographic Core

**Mission:** Authenticity, confidentiality, and entropy for everything else.

| Primitive | Algorithm | Source |
|-----------|-----------|--------|
| Signatures | Ed25519 (RFC 8032) | `security/crypto/c/ed25519/` |
| Signatures | **Dilithium-2/3/5** (FIPS 204 ML-DSA) | `security/crypto/c/dilithium/` |
| Signatures | **SPHINCS+-128/192/256** (FIPS 205 SLH-DSA) | `security/crypto/c/sphincs+/` |
| KEM | **Kyber-512/768/1024** (FIPS 203 ML-KEM) | `security/crypto/c/kyber/` |
| KEM | X25519 (RFC 7748) | `security/crypto/c/x25519/` |
| AEAD | ChaCha20-Poly1305 (RFC 8439) | `security/crypto/c/chacha/` |
| AEAD | AES-GCM + AES-NI | `security/crypto/c/aes/` + `asm/` |
| Hash | SHA-256/384/512, SHA3-224/256/384/512 | `security/crypto/c/sha/` |
| Hash | BLAKE3, SipHash | `security/crypto/c/blake3/` |
| KDF | Argon2id (RFC 9106), HKDF, PBKDF2 | `security/crypto/c/kdf/` |
| RNG | OS CPRNG (CNG / getrandom) | `security/crypto/c/rng/` |

All crypto lives in `security/crypto/c/` with x86-64 MASM acceleration in
`security/crypto/asm/x86_64/` (AES-NI, SHA-NI, AVX2, SSE2). The S0 layer is
**post-quantum ready** — Kyber/Dilithium/SPHINCS+ are built and unit-tested
alongside the classical suite.

```python
from SneppX_ALG import Ed25519, Dilithium, KyberKEM, sha256

sig  = Dilithium(level=3).sign(b"payload")
assert Dilithium(level=3).verify(b"payload", sig)
ct, ss = KyberKEM(k=768).encapsulate()
pt    = KyberKEM(k=768).decapsulate(ct)
assert ss == pt
```

---

## S1 — Memory Hardening

**Mission:** Keep secrets out of RAM, swap, and crash dumps.

- **Guard pages** — inaccessible pages bracketing every heap allocation.
- **Stack canaries** — random per-function canary, verified before `ret`.
- **ASLR** — `SNEPPX_secure_malloc` randomizes mmap base addresses.
- **Locked memory** — `mlock`/`VirtualLock` pins secret buffers (no swap).
- **Secure wipe** — `SNEPPX_secure_free` uses a compiler-barrier `memset`.
- **Constant-time compare** — `CRYPTO_memcmp` for all secret comparisons.
- **Leak detector** — `MemoryLeakDetector` tracks allocations, reports orphans.

```python
from SneppX_ALG import SecureAllocator, StackCanary, MemoryHardening, MemoryLeakDetector

hard = MemoryHardening()
buf  = SecureAllocator(4096).alloc()          # guard pages + locked
StackCanary().install()                        # per-function canary
MemoryLeakDetector().scan()                  # report unfreed blocks
```

## S2 — Obfuscation Engine

**Mission:** Raise the cost of static reverse engineering.

- Control-flow flattening (switch dispatch replaces natural CFG)
- String encryption with rotating keys
- Instruction substitution (NAND-equivalent sequences)
- Opaque predicates (always-true/false branches)
- Code virtualization (bytecode + encrypted dispatch table)
- Anti-debug, anti-dump, multi-VM diversity, instruction scheduling randomization

```python
from SneppX_ALG import (
    ObfuscationPipeline, CfgFlattening, StringEncryption,
    InstructionSubstitution, OpaquePredicate,
)

pipe = ObfuscationPipeline([
    StringEncryption(key=0xDEADBEEF),
    CfgFlattening(),
    InstructionSubstitution(),
    OpaquePredicate(density=0.15),
])
pipe.apply(source="src/", dest="src.obf/")
```

## S3 — Behavioral Monitor

**Mission:** Detect tampering and anomalous runtime behavior.

- **IntegrityMonitor** — CRC32 verification of code/data regions.
- **ContainerBreakoutDetector** — cgroup/namespace/mount inspection.
- **AdvancedMonitor** — frequency/timing anomaly analysis via `rdtsc`.

```python
from SneppX_ALG import IntegrityMonitor, ContainerBreakoutDetector, AdvancedMonitor

mon = IntegrityMonitor(); mon.watch_region("neural_core")
cbo = ContainerBreakoutDetector(); cbo.check()
adv = AdvancedMonitor(); adv.baseline_and_watch(threshold=3.0)
```

## S4 — Network Security

**Mission:** Harden the transport and identity plane.

- **DDoS mitigation** — SYN-flood detector, token-bucket rate limiters, IP
  reputation tables, connection tracking.
- **Transport security** — TLS handshake padding, traffic-analysis resistance.
- **Identity management** — certificate pinning, peer fingerprint verification.

```python
from SneppX_ALG import SYNFloodDetector, TokenBucket, IPReputationManager

det   = SYNFloodDetector(window=10, threshold=100)
rl    = TokenBucket(rate=10, burst=50)
rep   = IPReputationManager()
```

## S5 — AI Sanitizer

**Mission:** Defend the model itself from adversarial AI threats.

- **Prompt injection** detection (regex + embedded-pattern scan).
- **Differential privacy** (Laplace/Gaussian mechanisms, RDP accountant).
- **Data poisoning** defense (gradient outlier + loss spike monitoring).
- **RLHF safety** — reward-model validation, preference alignment,
  harmful-output filtering (`S5RLHFSafety`).
- **Output verifier** — constraint checking against allow/deny lists.

```python
from SneppX_ALG import GaussianMech, RDPAccountant, S5RLHFSafety, S5PromptFilter

acct = RDPAccountant(); mech = GaussianMech(eps=1.0, delta=1e-5, accountant=acct)
safety = S5RLHFSafety(allowed_topics=["tech", "science"])
prompt_ok = S5PromptFilter(max_len=1024).check("user query")
```

## S6 — Security UI / Key Vault

**Mission:** Credential and audit lifecycle.

- **AuditLogger** — structured JSON entries with severity, timestamp, source,
  chain-hash verification.
- **KeyVault** — in-memory AES-encrypted key store with PIN-protected access
  and key-rotation.

```python
from SneppX_ALG import KeyVault, AuditLogger, audit_log, audit_login

vault = KeyVault(master_pin="…")
vault.store("signing-key", key_bytes)
audit_log("model.train", actor="operator", severity="INFO")
```

## S7 — Secure Updates

**Mission:** Authenticated, rollback-safe software distribution.

- **SignedUpdateManager** — Ed25519-signed update bundles with version rolling.
- **ContainerSecurityManager** — OCI layer verification, manifest integrity,
  image signing, SBOM generation (`generate_sbom`), CVE scanning (`scan_image`,
  `scan_requirements`).

```python
from SneppX_ALG import SignedUpdateManager, ContainerSecurityManager, scan_image

mgr  = SignedUpdateManager(repo="s3://releases")
mgr.install_update("v1.2.0")          # verifies signature + monotonicity
csm  = ContainerSecurityManager()
report = scan_image("sneppx/serve:latest")
```

## S8 — Formal Verification

**Mission:** Mathematical guarantees for the most critical paths.

- **ModelChecker** — bounded state-space exploration (`LTLProperty`).
- **Invariant verification** — pre/post-condition checking on memory safety.
- **Symbolic execution** — path-constraint generation for NPE bytecode
  (`StateGraph`).
- **Formal safety** — `formal_verify.SymCC`-style verification hooks.

## S9 — Penetration Testing

**Mission:** Continuous red-teaming of the stack itself.

- **NetworkFuzzer** — protocol-aware engine with mutation strategies.
- **SelfAuditor** — internal consistency checks across all layers.
- **Report generation** — structured findings export.

```python
from SneppX_ALG import NetworkFuzzer, SelfAuditor

fuzz = NetworkFuzzer(target="127.0.0.1:8000", budget=1000)
fuzz.run(); report = fuzz.report()
SelfAuditor().run_full()
```

---

## Threat Model

**Assumptions:**
- Hardware is trusted (no CPU side-channels).
- OS/hypervisor is trusted (no kernel-level compromise).
- The network is **untrusted**. Other AI models are **untrusted** and may be adversarial.
- Supply chain is trusted (verified commits, signed releases).

| Threat | Defense (layer) |
|--------|-----------------|
| Signature forgery | Ed25519 / Dilithium (S0) |
| Data breach at rest | ChaCha20-Poly1305 / AES-GCM (S0) |
| Side-channel timing | Constant-time ops (S0/S1) |
| Memory scraping | `mlock` + guard pages (S1) |
| Swap forensics | Locked memory (S1) |
| Static RE | CF flattening + string encryption (S2) |
| Runtime tampering | IntegrityMonitor + canary (S1/S3) |
| Network abuse | DDoS mitigation + TLS (S4) |
| Prompt injection | S5 AI sanitizer |
| Gradient leakage | ARC obfuscation + DP (S5) |
| Model inversion | ARC output verifier (S5) |
| Credential theft | Key vault + audit log (S6) |
| Downgrade attack | Signed updates + monotonic counter (S7) |
| Logic bugs | Formal verification (S8) |
| Unknown vulns | Fuzz + self-audit (S9) |

**Reporting:** security issues → `algoSNEPPX@gmail.com` (see `SECURITY.md`).

## Verifying the layers locally

```powershell
# Build the C security core
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target neural_security_c

# Run the security test suites (safe — no network, no LLM)
cd build && ctest -C Release -R "kyber|dilithium|sphincs|aes|chacha|sha3|argon" --output-on-failure
```

The Python security bindings (`SneppX_ALG`) are thin wrappers over these C
libraries; they require the C backend to be linked (`_HAS_C_BACKEND is True`).
