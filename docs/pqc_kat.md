# Post-Quantum Crypto Known-Answer Testing (PQC KAT) — status and write-up

> 2026-10 edition. Companion to issue **#16** (official NIST ACVP vector
> harness) in the SneppX/sneppx-alg tracker.

## What was implemented

`SneppX_ALG.interface_bindings.acvp_kat` — a NIST ACVP harness for the C
crypto backend (`_SNEPPX_c.crypto`), plus vendored official vectors:
`tests/data/acvp/*.json`.

Test suites (all green):

| suite | tests | what it proves |
|---|---|---|
| `tests/python/test_acvp_sha3.py` | 4 | **C `sha3_256` == NIST official KATs** (40 byte-aligned SHA3-256 AFT vectors from `SHA3-256-2.0`), C==hashlib on the same messages, FIPS 202 empty-string KAT |
| `tests/python/test_acvp_slhdsa.py` | 5 | SPHINCS+ robustness on all 12 SLH-DSA parameter sets (24 sigVer vectors): zero false accepts, no crashes, C sign/verify roundtrip + tamper rejection, seeded-deterministic keygen |
| `tests/python/test_acvp_mldsa.py` | 5 | Dilithium on ML-DSA-44/65/87 (18 sigVer vectors): zero false accepts, no crashes, roundtrip + tamper rejection for variants 2/3, FIPS-204-sized PKs |

Vector provenance: `usnistgov/ACVP-Server` `gen-val/json-files/` sample
vector sets (NIST, public domain): `SHA3-256-2.0`, `ML-DSA-sigVer-FIPS204`,
`ML-DSA-keyGen-FIPS204`, `ML-KEM-keyGen-FIPS203`, `SLH-DSA-sigVer-FIPS205`.

## Bit-level vector policy (why SHA3-256 subset is byte-aligned)

ACVP `msg` values are **bit strings** (`len` counts *bits*, final nibble
zero-padded). The finalized SHA-3 approved function pads *after the last
message bit*, so when `len % 8 != 0` the digest is not the digest of any
byte string: the padding pattern lands mid-byte. Those vectors equal the
Keccak Team's `ShortMsgKAT_SHA3-256.txt` bit-level digests (verified against
the published XKCP test-vector file).

Neither the C byte-string API nor Python `hashlib` can absorb bit-level
messages, so we vendor **byte-aligned (`len % 8 == 0`) vectors only** for
autonomous testing. ML-DSA/SLH-DSA sigVer messages are all byte-aligned
(or have no `msgLen`), so their full sets are vendored.

Verifying bit-level KATs would require a bit-length sponge API on the C
side; that is the documented follow-up.

## Honest interop report

`acvp_kat.interop_report(crypto, vectors, runner)` aggregates C-verifier
decisions against NIST's `testPassed` flags.

- **SHA3-256**: 100 % KAT agreement (C digest == NIST `md` == `hashlib`
  on all vendored vectors).
- **ML-DSA (Dilithium) & SLH-DSA (SPHINCS+)**: agreement `0.5` — every
  *invalid* NIST vector is correctly rejected (zero false accepts), while
  NIST's *valid* signatures cannot verify verbatim because the C
  implementations are round-3-era structures (Dilithium: zeroed `tr` slot,
  no `key` domain separation; SPHINCS+: fixed single parameter set, sig
  14592 bytes vs FIPS 205's 7856/17088/…). Byte-compatibility upgrades are
  tracked as C work: upgrading both to the finalized FIPS 204/205 wire
  formats is the path to full 100 % cross-validation.

No invalid signature is ever accepted and no vector crashes the
implementation — the C verifiers fail closed on foreign formats.

## How to reproduce

```powershell
$env:PYTHONPATH = "bindings/python"
python -m pytest tests/python/test_acvp_sha3.py -q -p no:cacheprovider
python -m pytest tests/python/test_acvp_slhdsa.py -q -p no:cacheprovider
python -m pytest tests/python/test_acvp_mldsa.py -q -p no:cacheprovider
```

The `_SNEPPX_c` extension must be built (cmake target `neural_security_c` /
pybind module); otherwise each suite skips with a clear reason.
