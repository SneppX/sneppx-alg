# Compliance Overview

## Standards Coverage

SNEPPX-ALG aligns with the following compliance frameworks:

| Standard | Scope | Status |
|----------|-------|--------|
| **SOC 2** | Security, availability, confidentiality | Planned (v1.0) |
| **ISO 27001** | Information security management | Planned (v1.0) |
| **GDPR** | Data protection, right to explanation | Designed-in (S5 AI sanitizer) |
| **HIPAA** | Health data privacy | Planned (v2.0) |
| **FIPS 140-3** | Cryptographic module validation | S0 primitives designed for FM validation (pending formal cert) |

## Data Protection

- **At rest**: All sensitive data encrypted with ChaCha20-Poly1305 (S0)
- **In transit**: Transport security with handshake padding (S4)
- **In use**: Differential privacy during training (S5), secure memory lockdown (S1)
- **After use**: Secure wipe with compiler barrier (S1)

## Architecture

- No central data repository: Federated Memory (FM) with trust-weighted sync
- Local training: models train on-device, only gradient shares leave
- Differential privacy: Laplace mechanism with configurable epsilon (S5)
- Audit logging: structured JSON entries for all security events (S6)
