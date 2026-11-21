# Fuzzing Corpus

This directory contains seed corpora for fuzz testing SneppX-ALG components.

## Subdirectories

- `crypto/` - Seed inputs for cryptographic operations (keys, ciphertexts, signatures)
- `network/` - Seed inputs for network protocol parsers (TLS handshake, DNS, HTTP)
- `neural/` - Seed inputs for neural engine operations (tensors, model parameters)
- `serde/` - Seed inputs for serialization/deserialization (JSON, CBOR, protobuf)
- `security/` - Seed inputs for security checks (XSS, SQL injection, shell injection)
- `config/` - Seed inputs for configuration parsers

## Adding Corpora

Each corpus file should be:
- Small (< 1 KB for seeds)
- Valid input for the target function
- Named to indicate what code path it exercises

Use `-merge=1` with libFuzzer to merge new seeds from discovered crashes.
