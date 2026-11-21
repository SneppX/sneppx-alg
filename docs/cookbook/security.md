# Cookbook — Security

## 1. Scan source for vulnerabilities

**Intent:** Run `sneppx-analyze` on C/C++/CUDA code.

```bash
sneppx-analyze scan algorithms/hss/core/ --format c,h,cu,cpp --fail-on medium
```

Flags mirror `sneppx-analyze` (static binary analysis tool): `--format`,
`--fail-on <level>`, `--output report.json`, `--sarif` (IDE import). Targets
buffer overflows, NULL derefs, integer overflows, uninitialized reads, and
crypto-misuse patterns.

**Notes:** :material-alert-decagram: Runs purely static AST analysis — no
executables. CPU-safe.

## 2. Ed25519 / Dilithium signing

**Intent:** Post-quantum authentication of payloads.

```python
from SneppX_ALG import Ed25519, Dilithium           # S0 bindings

msg = b"model checkpoint v1.2.0"
sig = Dilithium(level=3).sign(msg)          # PQ (NIST FIPS 204)
assert Dilithium(level=3).verify(msg, sig)

sig2 = Ed25519().sign(msg)                  # classical (RFC 8032)
assert Ed25519().verify(msg, sig2)
```

**Notes:** :material-alert-decagram: C backend required for the full crypto
suite. Python bindings are thin wrappers over `security/crypto/c/`.

## 3. Kyber key encapsulation

**Intent:** PQ key exchange for secure serving.

```python
from SneppX_ALG import KyberKEM

kem = KyberKEM(k=768)                       # Kyber-768 (ML-KEM)
ct, session_key = kem.encapsulate()         # sender
shared = kem.decapsulate(ct)                # receiver
assert shared == session_key
```

## 4. Secure memory allocation

**Intent:** Lock sensitive buffers out of swap.

```python
from SneppX_ALG import SecureAllocator, StackCanary, MemoryLeakDetector

buf = SecureAllocator(8192).alloc()         # guard pages + mlock
StackCanary().install()                     # per-function stack canary
leaks = MemoryLeakDetector().scan()         # returns orphan list
```

**Notes:** These wrap `SNEPPX_secure_malloc`/`SNEPPX_secure_free` from
`security/memory/`. CPU-safe.

## 5. Audit logging + key vault

**Intent:** Record actions and protect signing keys.

```python
from SneppX_ALG import KeyVault, AuditLogger, audit_log

vault = KeyVault(master_pin="…")
vault.store("release-sign-key", key_material)
audit_log("checkpoint.publish", actor="operator", severity="INFO", extra={"tag": "v1.2.0"})

logger = AuditLogger(); logger.flush()      # verify chain hash
```

**Notes:** `audit_log` emits JSON with an append-only chain hash;
`verify_audit_chain()` (in the audit module) checks integrity. The key vault
is encrypted at rest and PIN-protected.

## 6. Differential privacy for gradients

**Intent:** DP-SGD with an RDP accountant.

```python
from SneppX_ALG import GaussianMech, RDPAccountant, DPSGD

acct  = RDPAccountant()
dp    = GaussianMech(eps=1.0, delta=1e-5, accountant=acct)
opt   = DPSGD(model.parameters(), base_optimizer=AdamW, l2_norm_clip=1.0, noise_multiplier=1.1, alphas=[2,4,8], accountant=acct)
# per-step: opt.zero_grad; (loss * batch).backward(); opt.step() (clips+noises)
```

**Notes:** `acct.get_epsilon(delta=1e-5)` returns the spent budget.
