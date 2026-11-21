# Threat Modeling Overview

This directory contains threat modeling artifacts for the SNEPPX-ALG platform.

## Methodology

SNEPPX-ALG uses the following threat modeling methodologies:

1. **STRIDE** (Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, Elevation of Privilege)
2. **PASTA** (Process for Attack Simulation and Threat Analysis)
3. **Attack Trees** - Hierarchical representation of attacker goals and sub-goals
4. **DREAD** (Damage, Reproducibility, Exploitability, Affected Users, Discoverability)
5. **LINDUN** (Linkability, Identifiability, Non-repudiation, Detectability, Disclosure of information, Unawareness, Non-compliance)

## Process

1. Define system scope and boundaries
2. Identify assets and trust boundaries
3. Enumerate threats (STRIDE per component)
4. Rank threats (DREAD)
5. Identify mitigations
6. Validate with attack tree analysis
7. Document residual risks

## Scope

- **HSS**: Timing side-channels via state transition matrix access patterns
- **SER**: Expert weight leakage via routing patterns
- **ARC**: Adversarial robustness, gradient leakage, model inversion
- **NPE**: Bytecode tampering, register side-channels, JIT poisoning
- **FM**: Gradient inversion, differential privacy budget exhaustion
- **S0-S9**: Full security layer threat analysis

## Artifacts

- `attack_tree.h` - C header defining attack tree data structures
- `stride_model_core.h` - C header defining STRIDE model data structures
- `threat_modeling_overview.md` - This document (methodology overview)
- STRIDE per-component spreadsheets (in `docs/`)
- Attack trees per-component (in `docs/`)
- DREAD risk matrices (in `docs/`)

## Risk Rating (DREAD)

| Rating | Range | Response |
|--------|-------|----------|
| CRITICAL | 9-10 | Immediate fix required |
| HIGH | 7-8 | Fix within 30 days |
| MEDIUM | 4-6 | Fix within 90 days |
| LOW | 1-3 | Accept or fix next release |
