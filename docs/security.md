# Security Architecture

## Overview

SNEPPX-Algo organizes its security work into ten phases (S0-S9), covering cryptography, memory hardening, obfuscation, runtime monitoring, network security, AI sanitization, key management, secure updates, formal verification, and penetration-testing infrastructure. Implementation maturity varies by component: some areas are experimental, partial, or scaffolded. This document describes the intended architecture and current implementation evidence; it is not a security certification or a claim that every control is production-ready.

Each phase builds on the previous. The architecture is designed so that security is not a bolt-on layer but a property of the entire system.

```
S0 ── S1 ── S2 ── S3 ── S4 ── S5 ── S6 ── S7 ── S8 ── S9
│     │     │     │     │     │     │     │     │     │
Crypto Memory Obfusc Monitor Network AI    UI    Update Formal Pentest
Core  Secure Engine  Engine  Sec   San    Sec   Sec    Verif  Report
      Mem
```

## S0 — Cryptographic Core (experimental)

**Status**: Experimental / partial
**Source**: `security/crypto/c/` and `include/neural_core/security/`

All crypto source files follow the four-layer commenting standard in [COMMENTING.md](COMMENTING.md): Layer 1 file headers, Layer 2 concept blocks, Layer 3 inline "why" comments, and Layer 4 Doxygen API docs on every public `SNEPPX_*` function.

### Primitives

| Primitive | Standard | Use Case | Verified |
|-----------|----------|----------|----------|
| Ed25519   | RFC 8032 | Signatures | 304/306 test vectors pass |
| X25519    | RFC 7748 | Key exchange | Full DH exchange |
| ChaCha20-Poly1305 | RFC 8439 | Authenticated encryption | 100+ test vectors |
| SHA-3     | FIPS 202 | Hashing (224/256/384/512) | NIST test vectors |
| SHA-256   | FIPS 180-4 | General-purpose hashing | NIST test vectors |
| BLAKE3    | Reference | Fast hashing | Reference test vectors |
| Argon2id  | RFC 9106 | Secure KDF | Test vectors + timing defense |
| Secure Random | OS CPRG | Entropy | Windows CNG / Linux getrandom |
| Kyber-512/768/1024 | FIPS 203 (ML-KEM) | PQ key encapsulation | Partial ACVP checks; see report |
| Dilithium-2/3/5 | FIPS 204 (ML-DSA) | PQ digital signatures | Experimental; valid-vector interoperability incomplete |
| SPHINCS+-128/192/256 | FIPS 205 (SLH-DSA) | Stateless PQ signatures | Experimental; valid-vector interoperability incomplete |

> **PQC validation limitations.** The repository includes official NIST ACVP
> sample vectors and a harness, but this does not establish FIPS validation or
> full standards interoperability. SHA3-256 vectors match; ML-KEM checks include
> key-generation comparisons and encapsulation/decapsulation tests. The current
> ML-DSA and SLH-DSA C implementations reject invalid vectors but cannot yet
> verify all valid finalized FIPS signatures because their signature formats
> remain pre-finalization. See [`docs/pqc_kat.md`](pqc_kat.md) for the detailed
> per-algorithm interop report. PQC remains experimental and should not be used
> as the sole protection for production secrets.

### API

```c
#include "cryptographic_primitives_bundle.h"

// Ed25519 signing
uint8_t pk[32], sk[64];
SNEPPX_ed25519_keypair(pk, sk);
SNEPPX_ed25519_sign(sig, msg, msglen, sk, pk);

// ChaCha20-Poly1305 encryption
SNEPPX_chacha20_poly1305_encrypt(ct, &ctlen, pt, ptlen, key, nonce, aad, aadlen);

// Argon2id key derivation
SNEPPX_s0_argon2id_hash(hash, hashlen, pwd, pwdlen, salt, saltlen, t_cost, m_cost);

// Kyber key encapsulation (CCA-secure via Fujisaki-Okamoto transform)
SNEPPX_kyber_keypair(pk, sk, variant);
SNEPPX_kyber_encaps(ct, ss, pk, variant);
SNEPPX_kyber_decaps(ss, ct, sk, variant);

// Dilithium signatures
SNEPPX_dilithium_keypair(pk, sk, variant);
SNEPPX_dilithium_sign(sig, &siglen, msg, msglen, sk, variant);
SNEPPX_dilithium_verify(sig, siglen, msg, msglen, pk, variant);

// SPHINCS+ signatures (stateless, hash-based)
SNEPPX_sphincsplus_keypair(pk, sk, variant);
SNEPPX_sphincsplus_sign(sig, &siglen, msg, msglen, sk, variant);
SNEPPX_sphincsplus_verify(sig, siglen, msg, msglen, pk, variant);
```

### Known Issues

1. **Ed25519 verification**: historical documentation records 304/306 vectors passing. Re-run the current RFC 8032 vector suite and reconcile the remaining failures before claiming full compliance.
2. **Argon2 timing**: historical tests reported timing variation on some hardware. Noise injection is not a substitute for constant-time design or side-channel analysis.
3. **PQ primitives**: the ML-KEM checks are partial; ML-DSA and SLH-DSA valid-signature interoperability remains incomplete for finalized FIPS formats.

## S1 — Secure Memory (partial)

**Status**: Partial; independent audit required
**Source**: `security/memory/` and `include/neural_core/security/`

### Features

| Feature | Description |
|---------|-------------|
| Guard Pages | Allocate with PROT_NONE guard on both sides in debug mode |
| Canaries | Stack overflow detection on memory regions |
| ASLR | Heap randomization via VirtualAlloc (Windows) / mmap (Linux) |
| Locked Memory | mlock / VirtualLock to prevent swap to disk |
| Secure Wipe | Compiler-barrier-protected zeroing of sensitive data |
| Constant-Time Compare | Timing-attack-resistant memory comparison |
| Memory Leak Detector | Tracks allocations and reports unfreed blocks on shutdown |

### API

```c
#include "protected_memory_manager.h"
#include "memory_leak_detector.h"

// Allocate locked memory (cannot be swapped to disk)
void* ptr = SNEPPX_s1_alloc_locked(4096);

// Securely free
SNEPPX_s1_free_locked(ptr, 4096);

// Constant-time comparison
int32_t match = SNEPPX_s1_memcmp_consttime(a, b, n);

// Memory leak detection
SNEPPX_memory_leak_detector_init(max_tracked_allocations);
void* p = malloc(256);
SNEPPX_memory_leak_detector_track(p, 256, "cache buffer");
SNEPPX_memory_leak_detector_untrack(p);
SNEPPX_memory_leak_detector_report(); // prints leaks to stderr
```

### Platform Requirements

- **Linux**: Requires `CAP_IPC_LOCK` capability for memory locking
- **Windows**: Automatic via `VirtualLock`
- **macOS**: Not fully supported (limited mlock)

## S2 — Obfuscation Engine (known high-risk findings)

> **Current remediation:** `obfuscation_advanced.cpp` no longer uses
> `rand()` / `srand()` for junk-pattern selection; it uses the S0
> `SNEPPX_random_bytes` interface. The unsafe bogus control-flow rewrite
> APIs now fail closed and do not modify the supplied code buffer. This
> disables that feature rather than making it functional. Whitebox AES
> timing/alignment concerns and IAT concurrency findings remain open.

**Status**: Known high-risk findings remain open
**Source**: `security/obfuscation/` and `security/cpp/`

### Implemented

- Control flow flattening: converts natural control flow to switch-based dispatch
- String encryption: XOR-based compile-time string obfuscation with rotating keys
- Instruction substitution: replace ADD/SUB/MUL/AND/OR/XOR with NAND-gate equivalent sequences
- Opaque predicates: always-true/false branches to confuse static analysis
- Code virtualization: basic-block-level bytecode compilation with encrypted handler dispatch table
- Anti-debug: ptrace detection, NtGlobalFlag check, timing anomaly detection, breakpoint scanning
- Binary substitution: opcode-level replacement with prefix/suffix byte insertion
- Junk code insertion: NOP slides, dead-store MOVs, identity XORs
- Constant unfolding: integer constants expressed as (a + b) with random splits
- IAT protection: hash-based import resolution, integrity scanning
- SEH/VEH obfuscation: exception-based control flow
- TLS callback obfuscation: runtime function pointer encoding
- Anti-dump: PE/ELF header XOR encryption with CRC integrity verification
- Multi-VM diversity: multiple bytecode handlers with slot switching
- Instruction scheduling randomization: basic-block-level instruction reordering

## S3 — Behavioral Monitor (partial)

**Status**: Partial
**Source**: `security/monitor/`

### Implemented

- Integrity monitoring: CRC32-based memory region verification
- Container breakout detection: monitors for escape attempts via cgroup, namespace, and mount inspection
- Frequency analysis: detect unusual API call patterns
- Timing analysis: detect side-channel probing via rdtsc and execution time measurement
- Anomaly detection: statistical baseline comparison for runtime behavior

## S4 — Network Security (experimental)

**Status**: Experimental / partial
**Source**: `security/network/`

### Implemented

- DDoS mitigation: SYN flood detection, rate limiting, connection tracking, IP blacklisting
- Transport security: TLS-compatible handshake padding, traffic analysis resistance
- Identity management: certificate pinning, peer fingerprint verification
- Certificate validation: chain-of-trust verification with expiry checking

### API

```c
#include "ddos_mitigation.h"
#include "transport_security.h"

// DDoS mitigation
SNEPPXDDoSState ddos;
SNEPPX_ddos_init(&ddos, 1000, 100);  // 1000 pkt/s window, 100 burst limit
if (SNEPPX_ddos_is_attack(&ddos, src_ip, dst_ip, now_us)) {
    SNEPPX_ddos_block_ip(&ddos, src_ip);
}

// Transport security
SNEPPXTransportSec ts;
SNEPPX_transport_sec_init(&ts);
SNEPPX_transport_sec_handshake(&ts, cert, cert_len, key);
```

## S5 — AI Sanitizer (partial)

**Status**: Partial
**Source**: `security/ai/`

### Implemented

- Prompt injection detection: regex + embedded pattern scanning for jailbreak attempts
- Differential privacy: Laplace mechanism with configurable epsilon for training data protection
- Data poisoning defense: gradient outlier detection, abnormal loss spike monitoring
- RLHF safety: reward model validation, preference alignment checking, harmful output filtering
- Output verifier: model output constraint checking against allow/deny lists

### API

```c
#include "differential_privacy.h"
#include "rlhf_safety.h"
#include "data_poisoning_defense.h"

// Differential privacy (Laplace mechanism)
double noisy_value = SNEPPX_differential_privacy_laplace(original_value, sensitivity, epsilon);

// RLHF safety check
SNEPPXRLHFSafety rlhf;
SNEPPX_rlhf_safety_init(&rlhf);
if (SNEPPX_rlhf_safety_check_output(&rlhf, model_output, output_len) == 0) {
    // output is safe
}

// Poisoning detection
SNEPPXPoisonDetect pd;
SNEPPX_poison_detect_init(&pd, threshold_gradient_norm);
SNEPPX_poison_detect_feed_gradient(&pd, grad, grad_len);
int is_poisoned = SNEPPX_poison_detect_is_anomalous(&pd);
```

## S6 — Security UI (incomplete)

**Status**: Incomplete
**Source**: `security/ui/`

### Implemented

- Audit logging: structured JSON log entries with severity levels, timestamps, and source tracking
- Key vault: in-memory encrypted key store with PIN-protected access
- Dashboard: (see `docs/api/c.md` for C API or `bindings/python/` for Python bindings)

## S7 — Secure Updates (incomplete)

**Status**: Incomplete
**Source**: `security/updates/`

### Implemented

- Container security: OCI-compatible layer verification, manifest integrity checking, image signing
- Signed update bundles: Ed25519-signed update payloads with version rolling
- Rollback protection: monotonic version counter preventing downgrade attacks
- Staged rollout: gradual update distribution with health-check gating

### API

```c
#include "container_security.h"
#include "signed_update.h"

// Container image verification
SNEPPXContainerSec cs;
SNEPPX_container_sec_init(&cs);
if (SNEPPX_container_sec_verify_layer(&cs, layer_data, layer_len, expected_hash)) {
    // layer integrity verified
}

// Signed update
SNEPPXSignedUpdate su;
SNEPPX_signed_update_init(&su, signing_sk);
SNEPPX_signed_update_create(&su, update_data, update_len, &bundle, &bundle_len);
```

## S8 — Formal Verification (not fully verified)

**Status**: Not formally verified
**Source**: `security/formal/`

### Implemented

- Model checking: bounded state-space exploration for critical algorithm paths
- Invariant verification: pre/post-condition checking on memory safety and data flow
- Symbolic execution: path constraint generation for NPE bytecode verification
- Container breakout detection (formal): state-machine-based detection rule engine

### API

```c
#include "model_checking.h"
#include "container_breakout.h"

// Model checking
SNEPPXModelCheck mc;
SNEPPX_model_check_init(&mc, max_states, max_transitions);
SNEPPX_model_check_add_invariant(&mc, "ptr != NULL", INV_PTR_NOT_NULL);
int ok = SNEPPX_model_check_verify(&mc, program, program_len);

// Container breakout detection
SNEPPXBreakoutDetect bd;
SNEPPX_breakout_detect_init(&bd);
SNEPPX_breakout_detect_watch_syscall(&bd, "mount", BREAKOUT_FLAG_NAMESPACE_ESCAPE);
```

## S9 — Penetration Testing (tooling only; no independent audit)

**Status**: Not independently audited
**Source**: `security/pentest/`

### Implemented

- Network fuzzer: protocol-aware fuzzing engine with mutation strategies (bit flip, boundary, dictionary)
- Self-audit: comprehensive internal consistency checks across all security layers
- Security report generation: structured output of audit findings
- Capture-the-flag utilities: challenge scaffolding for internal red-team exercises

### API

```c
#include "network_fuzzer.h"

// Network fuzzing
SNEPPXFuzzer fz;
SNEPPX_fuzzer_init(&fz, SNEPPX_FUZZ_TCP);
SNEPPX_fuzzer_set_target(&fz, "192.168.1.1", 443);
SNEPPX_fuzzer_set_mutation(&fz, SNEPPX_FUZZ_MUTATE_BITFLIP, 0.05);
SNEPPX_fuzzer_run(&fz, duration_sec, &report);
```

## Threat Model

### Assumptions

- The hardware is trusted (no side-channel attacks on CPU)
- The OS is trusted (no kernel-level compromise)
- The network is untrusted
- Other AI models are untrusted and potentially adversarial
- Supply chain is trusted (verified commits, signed releases)

### Defenses

| Threat | Defense | Status |
|--------|---------|--------|
| Signature forgery | Ed25519 | ✅ |
| Data breach at rest | ChaCha20-Poly1305 | ✅ |
| Side-channel timing | Constant-time ops | ✅ |
| Memory scraping | Locked memory + guard pages | ✅ |
| Swap forensic | mlock/VirtualLock | ✅ |
| Reverse engineering | CF flattening + string encryption | ⚠️ Partial |
| Runtime tampering | Behavioral monitor | ⚠️ Partial |
| Adversarial input | ARC input guard | ✅ |
| Gradient leakage | ARC gradient obfuscation | ✅ |
| Model inversion | ARC output verifier | ✅ |

## Reporting Vulnerabilities

See [SECURITY.md](https://github.com/SneppX/sneppx-alg/blob/main/../SECURITY.md)

## Security Best Practices

1. **Always verify signatures**: Use `SNEPPX_ed25519_verify` on any external data
2. **Lock sensitive data**: Use `SNEPPX_s1_alloc_locked` for keys and secrets
3. **Wipe after use**: Call `SNEPPX_s1_free_locked` or `SNEPPX_secure_zero` on sensitive buffers
4. **Use constant-time comparisons**: Never use `memcmp` for secret comparison
5. **Use AEAD**: Always use `SNEPPX_chacha20_poly1305_encrypt` with associated data
6. **Don't roll your own**: Use the provided crypto primitives; do not implement your own
