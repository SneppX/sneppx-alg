# S7 Secure-Update Hardening — Implementation Plan

**Goal:** Close the S7 dev-build fallback and make signed-update verification fail-closed (Ed25519 over BLAKE3 against a configured trusted public key). No trusted key + no valid signature = reject. Gate `apply()` on a passing verification.

**Why:** AGENTS.md / README flag S7 as partial: "a constant-time unsigned hash-equality fallback remains for dev builds." `docs/security_layers.md:122` describes S7 as Ed25519-signed payloads with rollback protection — the current code does not enforce that when a trusted key is absent.

## Affected files

- `security/updates/signed_update.c` — `SNEPPX_update_verifier_check` (the fallback), `SNEPPX_update_verifier_apply` (does not call check)
- `include/neural_core/security/signed_update.h` — structs/constants (reference only; likely no change)
- `bindings/python/SneppX_ALG/interface_bindings/secure_updates.py` — Python `SignedUpdate` (HMAC today, no trusted key, `apply` not gated)
- `tests/python/test_secure_updates.py` — extend/replace HMAC tests with Ed25519-over-BLAKE3 + fail-closed tests
- `docs/security_layers.md` — optional note that S7 dev fallback removed (keep minimal)

## Current behavior (verified by reading code)

C (`signed_update.c:96-135`):
- `SNEPPX_update_verifier_check`: when `has_trusted_key` is set, verifies Ed25519 signature over BLAKE3(update) hash vs `trusted_pubkey` — secure path.
- When NO trusted key: falls through to `SNEPPX_ed25519_verify` against a zeroed `trusted_pubkey` (always fails), then to `SNEPPX_ct_equal(computed_hash, update->update_hash)` — the dev-build hash-equality fallback. An attacker controls both sides, so this is forgeable.
- `SNEPPX_update_verifier_apply` (line 146): never calls `check`; only does rollback check, then bumps version. An unverified update can be applied.

Python (`secure_updates.py`):
- `hash_package` = SHA-256 (not BLAKE3).
- `sign_update`/`verify_signature` = HMAC-SHA256 with a non-constant-time `all(a==b)` compare.
- No trusted-public-key concept; `apply`/`rollback` just flip status strings.

## Design decisions (resolved)

1. **Single secure verification path only.** `check` returns 1 (ok) iff a valid Ed25519 signature over BLAKE3(update) verifies against a configured `trusted_pubkey`. Remove the `ct_equal` hash-equality fallback entirely. When `!has_trusted_key`, return 0 (fail closed). No dev exception.
2. **`apply` requires prior verification.** `apply` calls `check` internally and returns -1 if verification fails or verification is disabled. Rollback protection still applies.
3. **Python mirrors C.** `SignedUpdate` gains `set_trusted_public_key(bytes)`. `sign_update`/`verify_signature` use Ed25519 over BLAKE3 via the C bindings (`Ed25519`, `Blake3`) when the C backend is loaded; otherwise fall back to a pure-Python constant-time Ed25519 is NOT available — so in no-C-backend mode, `sign_update`/`verify_signature` raise unless a trusted key was pre-registered (fail-closed). `hash_package` uses BLAKE3 via C binding when available, else `hashlib.blake2s` (note: not BLAKE3, but collision-resistant; documented).
4. **`apply`/`rollback` in Python gate on verification:** `apply(update_id)` returns False unless the stored signature verifies against the registered trusted public key.
5. **Keep existing public method names** (`sign_update`, `verify_signature`, `hash_package`, `apply`, `rollback`, `register_update`, `status`) so callers don't break; change semantics to Ed25519 + fail-closed.

## Edge cases / scenarios to cover in tests

- Valid Ed25519 signature over BLAKE3 hash + trusted key → `check`/verify returns ok; `apply` succeeds.
- Tampered payload (flip a byte) with valid signature → verify fails; `apply` rejected.
- No trusted key configured → verify fails / `apply` rejected even if signature present (fail closed).
- Wrong public key → verify fails.
- Rollback protection: update with version <= current rejected by `check`/`apply`.
- Python: `verify_signature` uses constant-time comparison (no early-exit `all()`).
- Python: `sign_update` produces a signature that `verify_signature` accepts with the matching public key and rejects with a different key.

## Ordered tasks

1. **C: remove hash-equality fallback in `check`.** In `security/updates/signed_update.c`, delete lines ~132-134 (the `SNEPPX_ct_equal` block). Make the no-trusted-key branch return 0. Keep the trusted-key Ed25519 path as the only success path.
2. **C: gate `apply` on verification.** In `SNEPPX_update_verifier_apply`, call `SNEPPX_update_verifier_check(uv, update)` first; return -1 if it returns 0 or `verification_enabled` is 0. Keep rollback check.
3. **Python: BLAKE3 + Ed25519 signing/verification.** In `secure_updates.py`, reimplement `hash_package`/`sign_update`/`verify_signature` using `Blake3` and `Ed25519` bindings when `_HAS_C`, with fail-closed behavior otherwise. Add `set_trusted_public_key`. Use `hmac.compare_digest` for any byte comparison.
4. **Python: gate `apply`/`rollback`.** `apply` returns False unless signature verifies against registered trusted key. `register_update` stores the signature to verify on apply.
5. **Tests: extend `tests/python/test_secure_updates.py`.** Replace/augment HMAC tests with Ed25519-over-BLAKE3 tests + fail-closed cases + rollback cases.
6. **Validate.** Run Python test: `PYTHONPATH=bindings/python python -m pytest tests/python/test_secure_updates.py -q -p no:cacheprovider`. Run C target if present: `ctest -R signed_update --output-on-failure` (optional; C harness may not exist yet).

## Validation (gate)

- Python test file above must pass.
- C build must still compile (no new symbols removed that other code references — `SNEPPX_update_verifier_check`/`_apply` signatures unchanged).
- Do NOT run full test suite (hangs); per-file only per AGENTS.md.

## Open items (implementation agent to confirm)

- Exact method names on `Ed25519`/`Blake3` Python bindings (`sign`/`verify`/`digest`) — confirm against `crypto_sign.py` / `crypto_hash.py`; plan uses generic names.
- Whether a C unit-test target for `signed_update.c` exists; if not, Python test via ctypes/_SNEPPX_c is the gate.
- `docs/security_layers.md` update is optional/minimal (out of scope if it adds churn).
