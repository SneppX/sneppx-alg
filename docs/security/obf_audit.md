# S2 Obfuscation Layer Security Audit

**Original audit date:** 2026-08-16  
**Remediation review:** 2026-11-13  
**Scope:** `security/obfuscation/obfuscation_advanced.cpp`, `obfuscation_advanced.h`  
**Classification:** S2 — Obfuscation (Code transformation, string encryption, anti-debug)  
**Auditor:** Automated review of S2 layer

---

## Remediation update (2026-11-13)

The findings below are the historical audit record. The current source has
changed in several areas:

| Finding | Current status | Evidence |
|---|---|---|
| Predictable `rand()` / `srand()` choices | **Remediated in `obfuscation_advanced.cpp`** | The file now uses `SNEPPX_random_bytes` for junk-pattern selection and contains no `rand()` / `srand()` calls. This is not a repository-wide RNG audit. |
| Junk-code insertion bounds | **Hardened** | The current implementation validates the pattern count and uses subtraction-based capacity checks before moving bytes. Regression coverage is in `tests/security/cpp/test_obf_hardening.cpp`. |
| Bogus control-flow redirect | **Disabled, not fixed** | Add/redirect APIs now fail closed; redirect does not modify the caller's code buffer. A safe reversible implementation does not exist yet. |
| Whitebox AES placeholder | **Disabled, not fixed** | Init returns -1 and encryption zeroes output. A standards-compliant constant-time AES implementation is not provided. |
| IAT protection concurrency | **Open** | The shared-entry synchronization issue requires a separate design and test. |
| TLS callback pointer mangling | **Disabled, not fixed** | The previous implementation modified function pointers without a safe restore path; the API now returns -1 without mutation. |
| Anti-dump image mutation | **Disabled, not fixed** | The previous POSIX path treated a `dlopen()` handle as an image base and modified live executable memory; all anti-dump APIs now fail closed. |

The S2 implementation remains experimental and must not be treated as a
security boundary. A fix to one source file does not establish the security
of other obfuscation, string-encryption, or runtime-monitoring modules.

## Summary of Original Findings

| Severity | Count | Description |
|----------|-------|-------------|
| Critical | 3 | Insecure RNG for obfuscation decisions, buffer overflow in junk code insertion, type-punning alignment |
| High | 2 | Non-constant-time memory access in whitebox AES, IAT protection race condition |
| Medium | 2 | Bogus CF trampoline overwrites real code unconditionally, array index obfuscation is reversible |
| Low | 1 | Dead code in array obfuscation stride multiplication |

---

## 1. `SNEPPX_junk_code_insert` — historical bounds review (hardened)

```cpp
size_t junk_len=0;
for (int i=0;i<16;i++) if (jcg->junk_code[idx][i]) junk_len=i+1;
```

**CWE-787 (Out-of-bounds Write):** The `junk_code` storage is `uint8_t junk_code[64][16]` (from header). The loop finds `junk_len` by scanning for non-zero bytes — but **does not stop at `SNEPPX_FAKE_BLOCK_SIZE` boundary**. If `junk_code[idx]` contains all 16 bytes non-zero, `junk_len=16`. Then:

```cpp
if (*code_len+junk_len>max_len) return -1;
```

This correctly checks against `max_len`. However, `memcpy(code+pos, jcg->junk_code[idx], junk_len)` at line 138 writes `junk_len` bytes starting at `pos`. If `pos + junk_len > *code_len + junk_len` (i.e., `pos > *code_len`), this writes beyond the original code buffer. The `position` parameter (line 130: `int position`) is cast to `int pos` at line 136 with bounds checking, so this specific path is safe only if `position >= 0`. If `position < 0`, it defaults to `code_len` — OK. But the `memmove` at line 137 `code+pos+junk_len` could overlap if `pos+2*junk_len > code_len`... actually this is fine for memmove.

**Lower severity in practice** — the bounds check at line 135 catches the actual overflow. Marking as **Medium** (CWE-787 near-miss).

---

## 2. Use of `rand()` for obfuscation decisions — **CWE-338 Insecure RNG** (remediated in this file)

```cpp
SNEPPX_bogus_seed();  // line 217: srand(seed)
int idx=rand()%jcg->junk_count;  // line 132
size_t len=(size_t)(rand()%12+2);  // line 231
uint8_t v=(uint8_t)(rand()&0xFF);  // line 233
int fake_idx=rand()%SNEPPX_bogus_int.count;  // line 285
```

**CWE-338 (Use of Cryptographically Weak PRNG):** The entire S2 obfuscation layer uses `rand()` + `srand()` seeded with `std::chrono::steady_clock`. An attacker who can predict or brute-force the seed (1-second granularity via `time(NULL)` or ~1μs from steady_clock with ~2^32 range depending on implementation) can reproduce:
- Which junk pattern is inserted (defeats anti-disassembly)
- Which fake block is used for trampoline (defeats control flow obfuscation)
- All pattern bytes for bogus CF blocks

This makes the obfuscation **deterministic and reproducible**, completely defeating the purpose of S2. The seed is only set once (guarded by `seeded` flag), so all `rand()` calls throughout execution share the same PRNG state — a single prediction leaks all future obfuscation choices.

**Recommendation:** Replace `rand()` with `SNEPPX_S1_secure_random()` or a dedicated ChaCha20 DRBG from the S0 crypto layer. Never use `rand()` in security-sensitive code.

---

## 3. `SNEPPX_whitebox_aes_init` — **CWE-1041 Type-Punning / Alignment** (disabled)

```cpp
wb->te0[i]=s<<24|s<<16|s<<8|s;  // line 362
```

**CWE-1041 (Type-Punning Undefined Behavior):** The code casts `uint8_t in[16]` and `uint8_t out[16]` to `uint32_t*` at lines 377, 379, etc. On architectures requiring aligned access (e.g., ARMv6, some RISC-V), unaligned `uint32_t*` reads/writes cause SIGBUS. On x86 it works by accident. Use `memcpy` or `std::bit_cast` instead.

---

## 4. `SNEPPX_bogus_cf_redirect` — **CWE-717 Incorrect Code Transformation** (disabled)

```cpp
bcf->real_entry=(uintptr_t)code;  // saved
memcpy(code,trampoline,copy_len);  // OVERWRITES real code start
for (size_t i=flen;i<code_len;i++) {
    code[i]^=(uint8_t)((i*0x5A)^(fake_idx+1));  // XOR-encrypts body
}
```

**CWE-717 (Incorrect Code Transformation):** The function saves `real_entry` pointer but **immediately overwrites the start of the real code** with a trampoline (jump to fake block → jump back). The XOR loop encrypts the **rest** of the code. However:
1. There is **no corresponding "restore" function** — the encrypted code (`code[i] ^= ...`) can only be decrypted if `fake_idx` is known, which is stored only in `SNEPPX_bogus_int.count` (a global). If `bogus_int.count` changes between encrypt and decrypt, the code is **permanently corrupted**.
2. `flen` (fake block length) ranges from 2-12 bytes. The trampoline overwrite is `copy_len = min(tp, code_len)` where `tp` includes the fake block + 5-byte JMP. If `code_len < 5`, the trampoline is truncated and the JMP is incomplete → **corrupted executable code**.

**Recommendation:** Store decryption metadata alongside the `real_entry` pointer. Add `SNEPPX_bogus_cf_restore()`.

---

## 5. `SNEPPX_whitebox_aes_encrypt` — **CWE-208 Timing Side-Channel** (disabled)

```cpp
for (int r=1;r<=10;r++) {
    for (int i=0;i<4;i++) tk[i]=s[i];
```

The whitebox AES implementation accesses `te0`, `te1`, `te2`, `te3` tables indexed by secret-dependent data (the state S). **Table lookups are inherently timing-variable** on any system with a cache hierarchy. The implementation provides **no cache-safe masking or constant-time guarantees**.

This contradicts the S2 threat model which claims "constant-time" operation. While whitebox cryptography inherently leaks via side channels (that's the design tradeoff), the S0-S9 model documents S1 as "constant-time by default" and S2 as "obfuscation" — but the whitebox AES here doesn't document its side-channel exposure.

**Recommendation:** Document the timing-exposure contract. Add a note that `SNEPPX_whitebox_aes_*` is obfuscation-only and must not be used for genuine secrecy.

---

## 6. `SNEPPX_iat_protect_scan` (lines 328-335) — **CWE-362 Race Condition**

```cpp
int hooked=0;
for (int i=0;i<iat->count;i++) {
    if (iat->entries[i].original!=iat->entries[i].current) hooked++;
}
```

**CWE-362 (Concurrent Execution using Shared Resource):** The IAT entries `original` and `current` are `void*` pointers compared with `!=`. If another thread concurrently modifies `current` (e.g., via hot-patching), the read is non-atomic and may observe a torn pointer on 64-bit systems (though pointer writes are typically atomic on x86-64). More importantly, the hook detection is **not constant-time** — an attacker can measure how many entries are hooked via timing, leaking whether a hook is installed.

**Recommendation:** Use `SNEPPX_ct_equal` for pointer comparisons or document the non-constant-time nature.

---

## 7. `SNEPPX_array_obfuscate_indices` (lines 187-197) — **CWE-327 Reversible Obfuscation**

```cpp
obfuscated_indices[j]^=0xDEADBEEF;  // line 194
```

The "obfuscation" is XOR with a **hardcoded constant** (`0xDEADBEEF`). This is trivially reversible — an attacker who reverse-engineers the binary recovers all obfuscated indices immediately. The stride multiplication loop (lines 190-193) adds no entropy; it's a deterministic transformation.

**Recommendation:** Use a per-process PRNG state seeded from the S0 CSPRNG, not a hardcoded constant.

---

## 8. Dead code in `SNEPPX_array_obfuscate_indices` (lines 190-196) — **CWE-563**

```cpp
for (int i=ndim-1;i>=0;i--) {
    for (int j=0;j<n_indices;j++) obfuscated_indices[j]=(obfuscated_indices[j]%dims[i])*stride;
    stride*=dims[i];
}
```

**CWE-563 (Dead Store):** `obfuscated_indices[j]` is computed at line 191 but then immediately overwritten at line 195 (`obfuscated_indices[j]+=linearized[j]*dims[ndim-1-i]`). The stride multiplication loop is dead computation — it does work that is immediately discarded by the next line's `+=`. Only the final XOR at line 194 has any effect on the output.

---

## S0-S9 Model Alignment

| Layer | Documented | Implemented | Gap |
|-------|-----------|-------------|-----|
| S2 | Code transformation, string encryption, anti-debug | ✓ `obfuscation_advanced.cpp` (binary subst, junk code, opaque pred, CFG flatten, IAT protect, bogus CF, whitebox AES) | String encryption is in separate `string_encrypt.cpp` — verify it's not using the same `rand()` |
| S7 | Secure updates, rollback protection | `security/updates/` | N/A — outside S2 scope |
| S8 | Formal verification | `security/formal/` | N/A — outside S2 scope |
| S9 | Pentest, self-audit | `security/pentest/` | N/A — outside S2 scope |

The S2 layer documentation in `docs/security_layers.md` correctly lists the implemented components. However, it does **not** document the known timing and RNG weaknesses identified above.

---

## Files Reviewed

- `security/obfuscation/obfuscation_advanced.cpp` (772 lines) — primary S2 implementation
- `include/neural_core/security/obfuscation_advanced.h` — public types and API declarations
- `docs/security/threat_modeling_overview.md` — S2 threat model scope
