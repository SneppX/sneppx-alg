# Tutorial — Security Scanning

**Notebook:** [`security_scanning.ipynb`](https://github.com/SneppX/sneppx-alg/blob/main/notebooks/security_scanning.ipynb)
([download](https://github.com/SneppX/sneppx-alg/blob/main/notebooks/security_scanning.ipynb))

## What you'll build

Run the **`sneppx-analyze`** static vulnerability scanner on SNEPPX's own C
source, inspect the report, and exercise the **S0** post-quantum crypto
bindings (Dilithium signing, Kyber KEM) from Python.

## Setup

```powershell
# sneppx-analyze is a globally installed tool (sneppx-toolkit[all]):
python -m pip install "sneppx-toolkit[all]"
```

```python
import json
from SneppX_ALG import Ed25519, Dilithium, KyberKEM, sha256, SecureAllocator
HAS_C = __import__("SneppX_ALG")._HAS_C_BACKEND
```

## 1. Scan source with sneppx-analyze

```python
import subprocess

result = subprocess.run(
    ["sneppx-analyze", "scan", "algorithms/hss/core/",
     "--format", "c,h", "--json"],
    capture_output=True, text=True,
)
report = json.loads(result.stdout) if result.stdout else {"findings": []}
print("findings:", len(report.get("findings", [])))
for f in report.get("findings", [])[:5]:
    print(f"  [{f['severity']}] {f['rule']} @ {f['file']}:{f['line']}")
```

Flags: `--format c,h,cu,cpp`, `--fail-on <info|low|medium|high|critical>`,
`--sarif report.sarif` (GitHub-Code-scanning import), `--output out.json`.

> The scanner is **static only** — it never executes the analyzed code, so it
> is safe on untrusted sources.

## 2. Dilithium post-quantum signing (S0)

```python
if HAS_C:
    msg = b"SNEPPX-Algo release v1.2.0"
    dil = Dilithium(level=3)          # FIPS 204 ML-DSA-87
    sig = dil.sign(msg)
    assert dil.verify(msg, sig), "signature invalid"
    print("Dilithium-3 OK, sig len:", len(sig))
else:
    print("C backend required for S0 crypto (build neural_security_c)")
```

## 3. Kyber key encapsulation (S0)

```python
if HAS_C:
    kem  = KyberKEM(k=768)            # ML-KEM-768
    ct, sk = kem.encapsulate()
    ss = kem.decapsulate(ct)
    assert ss == sk, "KEM mismatch"
    print("Kyber-768 KEM OK")
```

## 4. Hash + secure memory attestation

```python
digest = sha256(b"model-weights.bin")
print("sha256:", digest.hex())

if HAS_C:
    buf = SecureAllocator(8192).alloc()   # guard pages + mlock
    StackCanary().install()
leaks = MemoryLeakDetector().scan()
print("leaks:", leaks)
```

## 5. S7 signed-update attestation (CLI)

```bash
sneppx-analyze verify model.sneppx --expected-signer "release@snepx"
```

Verifies the Ed25519 manifest signature embedded by `CheckpointWriter`.

## Key takeaways

- `sneppx-analyze` covers buffer overflows, NULL derefs, integer overflows,
  uninitialized reads, and crypto-misuse patterns.
- S0 crypto (Dilithium/Kyber) requires the compiled `neural_security_c` target.
- S1 memory hardening (`SecureAllocator`, guard pages) is always on for the C
  path; the Python API raises if the backend is missing.
- Never commit secrets — the scanner flags `SNEPPX_secure_free` violations
  (buffers not wiped before free).

## Next steps

- Read [Security Layers](../architecture/security_layers_s0s9.md) for the full
  S0–S9 model.
- See [Security Cookbook](../cookbook/security.md) for PQ key management recipes.
