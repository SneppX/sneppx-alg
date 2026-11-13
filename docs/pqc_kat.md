# Post-Quantum Crypto Known-Answer Testing (PQC KAT) — status and write-up

 > 2026-11-13 update. Companion to issue **#16** (official NIST ACVP vector
> harness) in the SneppX/sneppx-alg tracker. The deterministic RNG used by
> seeded key-generation tests is now compiled only into explicit test builds.

## What was implemented

`SneppX_ALG.interface_bindings.acvp_kat` — a NIST ACVP harness for the C
crypto backend (`_SNEPPX_c.crypto`), plus vendored official vectors:
`tests/data/acvp/*.json`.

Test suites (availability depends on build configuration):

| suite | tests | what it proves |
|---|---|---|
| `tests/python/test_acvp_sha3.py` | 4 | **C `sha3_256` == NIST official KATs** (40 byte-aligned SHA3-256 AFT vectors from `SHA3-256-2.0`), C==hashlib on the same messages, FIPS 202 empty-string KAT |
| `tests/python/test_acvp_mlkem.py` | conditional | Deterministic keyGen and encaps/decaps tests require a test-only extension built with `SNEPPX_ENABLE_TEST_RNG=ON`; production builds skip this module. No FIPS validation is claimed. |
| `tests/python/test_acvp_slhdsa.py` | conditional | Signature-verification vectors and C roundtrip/tamper tests; seeded determinism requires a test-only build. KeyGen vectors are explicitly skipped because they provide no reproducible entropy input accepted by the current C API. |
| `tests/python/test_acvp_mldsa.py` | conditional | Signature-verification and roundtrip/tamper tests; deterministic keyGen vectors require a test-only extension built with `SNEPPX_ENABLE_TEST_RNG=ON`. |

Vector provenance: `usnistgov/ACVP-Server` `gen-val/json-files/` sample
vector sets (NIST, public domain): `SHA3-256-2.0`, `ML-KEM-keyGen-FIPS203`,
`ML-KEM-encapsDecaps-FIPS203`, `ML-DSA-sigVer-FIPS204`, `ML-DSA-keyGen-FIPS204`,
`SLH-DSA-sigVer-FIPS205`, `SLH-DSA-keyGen-FIPS205`.

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
- **ML-KEM (Kyber)**: deterministic keygen and encaps/decaps KATs require
  the test-only RNG hook. That hook is not compiled into production builds.
  Keygen agreement is measured byte-for-byte against NIST `ek`/`dk`; the
  implementation remains experimental and is not FIPS validated.
- **ML-DSA & SLH-DSA**: invalid foreign signatures are rejected, but valid
  finalized FIPS signatures are not fully interoperable with the current C
  wire formats. These implementations remain experimental.

Keygen coverage:
- **ML-KEM keyGen**: `run_mlkem_keygen()` seeds the C test RNG with each
  vector's `d` value and compares `ek`/`dk` byte-for-byte against FIPS 203
  expected values.
- **ML-DSA keyGen**: `run_mldsa_keygen()` seeds the C test RNG with each
  vector's `seed` value and compares `pk`/`sk` against FIPS 204 expected
  values.
- **SLH-DSA keyGen**: explicitly marked unsupported by the harness because
  the vendored sample set does not provide reproducible entropy inputs that
  can be passed to the current key-generation API. Those vectors are counted
  as skipped, not as successful interoperability.

The harness reports unsupported suites as skipped. It is a regression and
interoperability tool, not a FIPS validation certificate. No invalid signature
is accepted by the tested signature-verification paths.

## How to reproduce

```powershell
$env:PYTHONPATH = "bindings/python"
python -m pytest tests/python/test_acvp_sha3.py -q -p no:cacheprovider
python -m pytest tests/python/test_acvp_mlkem.py -q -p no:cacheprovider
python -m pytest tests/python/test_acvp_slhdsa.py -q -p no:cacheprovider
python -m pytest tests/python/test_acvp_mldsa.py -q -p no:cacheprovider
```

The `_SNEPPX_c` extension must be built. Production builds omit deterministic
RNG controls. To run seeded key-generation tests, configure a separate test
build with `-DSNEPPX_ENABLE_TEST_RNG=ON` and build the Python extension. Never
ship or distribute that test build. Signature-verification tests that do not
need deterministic key generation can run against a production extension.
