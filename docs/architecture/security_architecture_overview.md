# Security Architecture Overview

## Defense-in-Depth Layers

The SNEPPX-ALG security architecture implements a defense-in-depth strategy:

### S0 - Cryptographic Core
Provides all cryptographic primitives: signatures (Ed25519, Dilithium 2/3/5, SPHINCS+), encryption (ChaCha20-Poly1305), hashing (SHA-3, BLAKE3), KDF (Argon2id), and PQ key encapsulation (Kyber-512/768/1024). All primitives implement constant-time operations.

### S1 - Secure Memory
Guard pages, stack canaries, ASLR, locked memory, secure wipe, constant-time comparison, and memory leak detection. Protected memory for sensitive data (keys, gradients, model weights).

### S2 - Obfuscation Engine
Control flow flattening, string encryption, instruction substitution, code virtualization, anti-debug, opaque predicates, anti-dump, multi-VM diversity, IAT protection.

### S3 - Behavioral Monitor
Runtime integrity monitoring (CRC32), container breakout detection, frequency and timing analysis, statistical anomaly detection.

### S4 - Network Security
DDoS mitigation (SYN flood detection, rate limiting), transport security, identity management, certificate pinning.

### S5 - AI Sanitizer
Prompt injection detection, differential privacy (Laplace mechanism), data poisoning defense, RLHF safety, harmful output filtering.

### S6 - Security UI / Key Vault
Audit logging (structured JSON), encrypted key store with PIN-protected access.

### S7 - Secure Updates
Container image verification (OCI), signed update bundles (Ed25519), rollback protection, staged rollout.

### S8 - Formal Verification
Model checking, invariant verification, symbolic execution for NPE bytecode, container breakout state-machine rules.

### S9 - Penetration Testing
Network fuzzing, self-audit (internal consistency checks), security report generation, CTF utilities.

## Trust Model

- **Hardware**: Trusted (no CPU side-channel attacks)
- **OS**: Trusted (no kernel-level compromise)
- **Network**: Untrusted (MTLS, Noise protocol, QUIC)
- **Other AI models**: Untrusted and potentially adversarial
- **Supply chain**: Trusted (verified commits, signed releases)

## Key Security Properties

1. **Confidentiality**: Model weights encrypted at rest (ChaCha20-Poly1305), locked in memory (mlock), never swapped to disk
2. **Integrity**: Code verified via signatures, memory guarded by canaries, runtime monitored by behavioral analysis
3. **Availability**: DDoS mitigation, graceful degradation, resource limits
4. **Privacy**: Differential privacy during training, on-device inference, no telemetry
5. **Non-repudiation**: Ed25519 signatures on all contributions, audit logging

## Simulation & Visualization

The security architecture can be simulated and visualized using:

```bash
# Run attack simulations
python tests/security/test_threat_simulations.py

# Generate security report
python scripts/security_report.py --output report.json
```

## See Also

- `docs/security.md` - Full security architecture document
- `docs/security_layers.txt` - Deep dive into S0-S9
- `docs/threat_modeling/threat_modeling_overview.md` - Threat modeling methodology
- `docs/incident_response/ir_framework_overview.md` - Incident response procedures
- `docs/compliance/compliance_overview.md` - Compliance framework coverage
