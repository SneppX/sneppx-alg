# Audit — `bindings/python/SneppX_ALG` (Python bindings)

**Scope:** `bindings/python/SneppX_ALG` only (225 `.py` files incl. 146 in `interface_bindings/` + subpackages `agents, callbacks, datasets, dispatch, graph, linalg, metrics, observability, rag, regularization, sampling, signal, stats, structured_output, speculative`). No C/C++, docs, or tests audited.

**Method (all findings verified, none inferred):**
- AST parse of all 225 files → 4,896 function definitions, 73,387 lines; classified every function body (empty / `pass` / `...` / `raise NotImplementedError` / constant-return / log-only).
- Ripgrep sweeps: `NotImplementedError`, `stub|placeholder|fake|dummy|simulated|synthetic|mock|scaffold`, `TODO|FIXME|XXX|HACK` (**zero** TODO/FIXME hits), `reference_scaffold|c_reference_no_weights|not supported|experimental|refusing`.
- Runtime import of **all 225** modules (`PYTHONPATH=bindings/python`).
- Static + runtime `__all__` resolution for every module.
- Cross-module top-level-name collision analysis (last-import-wins resolution checked at runtime).
- Reverse-import graph for orphan detection.
- Behavioural probes (no model inference, no GPU): `graph_optimizer.trace`, `TransportSecurity.verify_certificate`, `AdvancedMonitor.detect_anomaly`, BLAKE3/SHA vectors.
- `--help` smoke test of all 7 `interface_bindings` CLI `main()`s + all 16 `ctm_tools.py.*` entry targets.
- **Not run:** any `transformers`/`llama_cpp`/`from_pretrained`/model inference/GPU code. `cryptography` and `scipy` are **not installed**; `numpy 2.4.6`, `onnx 1.5.2`, `fastapi` absent.

---

## Summary table

| Category | Count | Notes |
|---|---|---|
| Python files in scope | 225 | 73,387 lines, 4,896 functions |
| Total stub-pattern findings (AST) | **127** across 38 files | `PASS_ONLY` 49, `CONST_RETURN` 44, `RAISE_NIE` 25, `ELLIPSIS_ONLY` 4, docstring-only 2, log/pass-only 3 |
| …of which **legitimate** ABC/Protocol/base-class design | **~100** | abstract hooks properly overridden by concrete subclasses |
| …of which **genuine** stubs / fakes | **~27** | listed in §2 and §3 |
| Modules failing import | **2** | both only because `fastapi` is not installed |
| Modules importing cleanly | 223 / 225 (99.1%) | |
| `__all__` names that don't resolve | **13** in 2 modules | `vision_transformers` (11), `distillation` (1) + 2 false positives |
| Orphan modules (never imported by any other file) | **15** | 5 are legitimate CLI entry points + 2 are test modules → **8** genuinely orphaned library modules |
| Cross-module name collisions | **157** | several are full duplicate implementations |
| `_HAS_C` native-library flags | **66** across ~40 modules | all `False` in this environment |
| Console scripts declared | 23 | **23/23 resolve**; **0** missing entry points |
| TODO / FIXME / XXX / HACK | **0** | (self-documented debts use prose "placeholder"/"stub" instead) |

**Headline:** the package is in better shape than its name-level duplication suggests. Abstract-method usage is disciplined, crypto **fails closed rather than fabricating**, and every declared entry point resolves. The real defects cluster in (a) a handful of silent-success stubs, (b) one fail-open certificate check, (c) 13 dead `__all__` names, and (d) heavy duplicated-implementation shadowing.

---

## 2. Stubs — genuine (not ABC design)

| # | Location | What is missing |
|---|---|---|
| 1 | `interface_bindings/graph_optimizer.py:96-106` | `trace(func, *args)` body is `pass` → **returns `None`**. Docstring advertises `graph = trace(f, input_a, input_b)`. Verified: `trace(lambda a,b:a*b,1,2)` → `None`. The public tracing API is non-functional. |
| 2 | `interface_bindings/graph_optimizer.py:53-57` | `dead_code_elimination()` is `pass`; body is only a comment saying the pass "just verifies". Called by `optimize()` (L87). |
| 3 | `interface_bindings/graph_optimizer.py:59-62` | `memory_planning()` is `pass` — and is **never called** by `optimize()` (L84-90), so it is dead code too. |
| 4 | `interface_bindings/onnx_export.py:994-1008` | `TensorRTExporter.export()` and `.build_engine()` are `pass` → return `None`, no engine, no error. |
| 5 | `interface_bindings/onnx_export.py:1011-1020` | `export_torch()` and `import_torch()` are `pass` → return `None`. Declared PyTorch interop is absent. |
| 6 | `interface_bindings/serving_engine.py:286-287` | `max_batch_size_for(_)` returns hardcoded `0` (module-level function, no guarding). |
| 7 | `interface_bindings/serving_engine.py:977-978` | `Handler.log_message` is `pass` — intentional request-log suppression (acceptable, but noted: no HTTP access log). |
| 8 | `interface_bindings/inference_server.py:856-860` | `AsyncStreamer.put_text()` / `.end()` are `pass`; only token IDs are buffered. Any text-mode streaming silently drops text callbacks. |
| 9 | `interface_bindings/nccl.py:107-111` | `_NCCLComm.send()` / `.recv()` have **docstring-only bodies** → return `None`. Verified `EMPTY_DOCSTRING_ONLY`. |
| 10 | `interface_bindings/nccl.py:163-165` | `ProcessGroup.barrier()` is `pass`. |
| 11 | `interface_bindings/net_bindings.py:385-388` | `read()`/`write()` return constant `0` (gRPC "service stub", L393). |
| 12 | `interface_bindings/net_bindings.py:451,473,509` | `comm_destroy`, `broadcast`, `barrier` return constant `0`. |
| 13 | `interface_bindings/dispatch/infra_dispatch.py:57-58` | `socket_recv()` returns `b""` unconditionally — indistinguishable from clean peer EOF. |
| 14 | `interface_bindings/dispatch/infra_dispatch.py:63-68` | `rdma_connect()` returns `0` (success) without connecting; `rdma_disconnect()` is `pass`. |
| 15 | `interface_bindings/driver_bindings.py` (≈15 sites) | Whole CUDA-driver shim returns **success codes**: `launch_kernel` → `0` (L203, L313), `tc_gemm` → `0` (L215), `event_record`/`event_synchronize`/`stream_synchronize` → `0` (L154-160, L269-281), `event_elapsed_ms` → `0.0` (L160), `execute()` → `0` (L383), `destroy_context`/`stream_destroy`/`event_destroy`/`executable_destroy` `pass`. Gated by `is_available()` → `False` (L108, L237, L331), so it is not reachable by accident, but the shim reports kernel success for kernels it never ran. |
| 16 | `interface_bindings/model_implementations.py:1702` | `get_model_flops()` returns constant `0.0` — FLOP accounting is a fake metric. |
| 17 | `interface_bindings/experiment_tracker.py:110-154` | `Tracker` base: `start_run`, `log_metric`, `log_param`, `log_artifact`, `set_tag`, `set_status`, `end_run` all `pass`. Documented as "no-ops by default" — but `LogTracker` (L157) overrides only `start_run`/`log_metrics`/`end_run`, so `LogTracker.log_metric` and `.log_param` silently discard. |
| 18 | `interface_bindings/experiment_tracker.py:326` | `set_status` is `pass` in a concrete tracker. |
| 19 | `interface_bindings/serving_engine.py:592` | `self.metrics.record(0, n, error=False)` — literal `# placeholder latency`: every request records 0 ms latency. |
| 20 | `interface_bindings/checkpoint.py:152` | `# Write header placeholder` in the checkpoint writer. |
| 21 | `interface_bindings/optim_advanced.py:668` | `"adam": lambda: None,  # placeholder - use optim.Adam` — a registry entry that returns `None` for `adam`. |
| 22 | `interface_bindings/vision_transformers.py:939-942, 949-951, 956-959` | **11 names in `__all__` are never defined** (see §5). |
| 23 | `interface_bindings/distillation.py:646` | `distillation_loss` in `__all__` but never defined (see §5). |

### Stubs that are *correct* ABC/Protocol design (explicitly cleared — do not "fix")

`autograd.py:145,149` (`Function.forward/backward`, subclass contract) · `nn.py:143` (`Module.forward`), `nn.py:1047` (`ModuleDict.forward`, documented) · `optim.py:86` (`Optimizer.step`) · `callbacks/base.py:16-25` (no-op lifecycle hooks) · `data_loader.py:31,38` + `train.py:270,273` (`Dataset`/`DataLoader` abstracts) · `eval_harness.py:31,34` (`Task`, concrete tasks below) · `edge_runtime.py:190,193` (`EdgeBackend`, concrete at L291) · `adversarial.py:77` (`Attack`, `FGSM` at L95) · `hparams.py:28,31,180` (`Param`/`Sampler`, concrete `UniformFloat` L35, `GridSampler` L187) · `hpo.py:245` (`_BaseSampler`, `_RandomSampler` L252) · `pruning.py:231` (`SparsityScheduler`, `PolynomialSparsityScheduler` L241) · `keras_api.py:403` (`_OptimizerBase`, `_SGD` L407) · `pytorch_api.py:243,426,520,523` · `structured_output/constrain.py:35,39,42` (`Constrainer`, `ChoiceConstrainer` L46) · `audit_logger.py:267-285` (`AuditBackend` ABC, `FileAuditBackend` L288) · `cuda_device.py:195`, `cuda_kernels.py:70,93` (`synchronize` no-ops on CPU fallback) · `logger`/`progress`/`lr_scheduler` callbacks · `speculative/decoding.py:12-22` (4 × `...` are `typing.Protocol` method declarations — correct idiom, **not** stubs) · `train.py:162` (`LRScheduler.step` base) · `lora.py:489,492` (`__enter__`/`__exit__`) · `tokenizer`/`graph`/`linalg`/`stats`/`signal`/`metrics`/`rag`/`agents`/`observability`/`datasets`/`regularization`/`sampling` subpackages: **no genuine stubs found**.

---

## 3. Fake / simulated implementations

| Location | Finding |
|---|---|
| **`s4_network.py:111-112`** | **`TransportSecurity.verify_certificate(cert_pem, expected_cn) -> True` unconditionally.** Both arguments ignored; `self._has_c` (L94) is never consulted. Runtime-verified: `verify_certificate("", "totally-wrong.example")` → `True`; `verify_certificate("NOT A CERT", "x")` → `True`. **Fail-open TLS certificate validation.** |
| **`runtime_monitor.py:102-104`** | `AdvancedMonitor.detect_anomaly(metrics)` → constant `0.0`, static, args ignored. Runtime-verified: `{}` → `0.0`, `{"cpu":999.0,"err":1e9}` → `0.0`. Anomaly detection can never fire. |
| **`onnx_export.py:865-877`** | `export_onnx(...)` writes an ONNX file containing the declared inputs plus one **dummy output** `add_output(output_names[0], [1, 1000])`; the model is never traversed (`# This is a simplified example`, L868). Produces a structurally valid but semantically empty artifact. |
| `graph_optimizer.py:114-115` | `visualize_graph` — comment literally `# Simple visualization stub: print the nodes`; prints instead of rendering. |
| `dispatch/dispatch_cuda.py:1,42` | Module docstring: "delegates to compiled CUDA kernels or **simulated** fallback"; "Otherwise a simulated CUDA" path. |
| `cuda_device.py:1` | Docstring: "**simulated** CUDA device management for Python backend". |
| `hw_backends.py:14` | `cpu_fp16` — "NumPy executor casting to float16 (**simulated** fp16 math)". |
| `hw_backends.py:229,278,287,418` | Four `available` properties return constant `True` regardless of actual backend presence. |
| `edge_runtime.py:86`, `edge_runtime.py:291` | `supports_int8` → `True`; `Int8Backend.supports()` → `True` for every node (claims universal op coverage). |
| `amp.py:6` | Docstring references "the (**simulated**) backward". |
| `trainer_v2.py:266` | `# backward (simulated: populate grads via finite difference of loss)` — gradients approximated by finite differences. |
| `trainer_v2.py:332` | `return float(np.asarray(_p.data).mean()) * 0.0  # placeholder keep finite` — returns 0.0. |
| `trainer_v2.py:417` | `self.profiler.record_step(0.0)  # placeholder (timing done by caller)`. |
| `distillation.py:434` | Docstring: "The **simulated** tensor backend produces non-differentiable loss tensors". |
| `train_cli.py:46` | `"""Build a **synthetic** data loader for testing."""` — CLI trains on synthetic random data. |
| `model_zoo.py:1275` | `data = b"\x00" * 4  # placeholder` in a weight-loading path. |
| `net_bindings.py:393` | `"""gRPC service **stub** for distributed coordination."""` |
| `speculative/decoding.py:8,47` | `decode()` imports `residual_distribution, speculative_accept` but **never uses them** (grep confirms the import line is the only occurrence) — so it is greedy draft-matching, not the speculative-sampling algorithm the module name promises. Class docstring: "Greedy-approximate **skeleton**". |
| `nccl.py:146-165` | `all_reduce`/`broadcast` return the input tensor **unchanged** for `world_size > 1`; `barrier()` no-op. Honestly documented (L147-152, "not a reduction"), but a caller cannot distinguish it from a real reduction. |
| `security_memory`-class constants | `secure_memory.py:99` `w_xor_x()` → `False`; `secure_memory.py:120` `seccomp_filter()` → `False`. Return "unsupported" rather than success — acceptable fail-closed signalling, but neither ever succeeds. |

**Crypto is NOT fake — verified genuine.** `crypto_hash.py` + `_blake3_pure.py` are real implementations, confirmed against reference vectors: `sha256(b"")`, `sha3_256(b"")`, `shake_128(b"",16)` all match `hashlib`; `blake3_hash(b"")` = `af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262` (official BLAKE3 empty-string vector, exact match). **No constant-byte crypto found anywhere.**

*Minor API defect:* `crypto_hash.Blake3` (L49) exposes no `update()` method — `Blake3().update(b"abc")` raises `AttributeError`. Only constructor/one-shot digest is usable.

---

## 4. Failing-closed features (loud) vs silent no-ops

### 4a. Fail closed — raises, cannot silently mislead ✅

| Location | Behaviour |
|---|---|
| `crypto_kem.py:18-29` | `_require_kyber_backend()` **always** raises `RuntimeError("ML-KEM is not available through the Python bindings; refusing to fabricate keys, ciphertexts or shared secrets")`. Called unconditionally at `KyberKEM.keypair` (L58) — i.e. **all of `KyberKEM` is a permanent hard failure**, even if the C lib were present. Explicit comment records the past bug: random bytes made "both peers derive unrelated secrets while reporting success". |
| `crypto_symmetric.py:47,58,86,97` | "AES-GCM requires the 'cryptography' backend; refusing to emit unencrypted data" / "refusing to skip tag verification". |
| `crypto_sign.py:86-87,101,149-150,156,193-194,201` | "refusing to generate placeholder keys". ML-DSA/SLH-DSA need the native extension; `verify()` returns `False` (deny) when absent (L161, L207). |
| `crypto_kem.py:76-103` | X25519: "refusing to fabricate keys / shared secrets". |
| `crypto_kdf.py:137` | Argon2: "refusing to silently [fall back]". |
| `tensor.py:295,320` | `RuntimeError("native C tensor engine is not available")`. |
| `serve_cli.py:351-352` | `RuntimeError("--checkpoint is not supported by serve_cli yet. The model would otherwise …")` — **checkpoint loading is unimplemented in the server CLI**. |
| `fsdp.py:105`, `distributed.py:67,217` | "Multi-rank FSDP is not supported", "refusing to start a training", "Multi-process launch is not supported yet". |
| `checkpoint.py:298,302,310` | "refusing to load" on unsigned/tampered checkpoint. |
| `inference_server.py:376` | "Security middleware is not initialized; refusing unauthenticated access". |
| `serving_client.py:44` | "refusing to send a serving API key over non-local HTTP". |
| `graph_compiler.py:583` | Unsupported op → emitted as `/* unsupported op %r skipped */` (silently skipped in codegen, but the comment is at least emitted). |

**The crypto security posture is correct and deliberate** — the "refusing to fabricate" language is honest engineering, not a gap. Note that in a pure-Python install (`cryptography` absent) **all of Ed25519, AES-GCM, ChaCha20-Poly1305, X25519, Argon2 and Kyber are unavailable**, and ML-DSA/SLH-DSA need the native extension. Net: *no working public-key crypto ships in the pure-Python wheel.*

### 4b. Silent no-op — returns success/neutral value with no error ⚠️

| Location | Silent behaviour |
|---|---|
| `s4_network.py:111` | cert always valid (fail-**open**, worst case in the audit). |
| `runtime_monitor.py:103` | anomaly score always `0.0`. |
| `graph_optimizer.py:96` | `trace()` → `None`. |
| `graph_optimizer.py:53,59` | optimisation passes do nothing. |
| `nccl.py:101-111,146-165` | all_reduce/broadcast identity, send/recv/barrier no-op. |
| `net_bindings.py:385-509` | read/write/broadcast/barrier → `0`. |
| `infra_dispatch.py:57-58` | `socket_recv` → `b""`. |
| `driver_bindings.py` | kernel launches report success. |
| `inference_server.py:856-860` | streamed text dropped. |
| `serving_engine.py:286,592` | batch size `0`; latency always `0`. |
| `onnx_export.py:865-877` | exports a dummy graph. |
| `experiment_tracker.py:110-154` | metrics/params/artifacts silently discarded. |

---

## 5. Missing entry points & declared-but-missing names

### 5a. Entry points — **0 missing**

`pyproject.toml` `[project.scripts]` declares **23** console scripts. All were resolved and exercised:

- **7 → `SneppX_ALG.interface_bindings.*:main`** — all files define `main`, all respond to `--help`:

| Script | Target | `main` at |
|---|---|---|
| `sneppx-train` | `train_cli:main` | `train_cli.py:82` ✅ |
| `sneppx-serve` | `serve_cli:main` | `serve_cli.py:62` ✅ |
| `sneppx-experiment` | `experiment_cli:main` | `experiment_cli.py:10` ✅ |
| `sneppx-eval` | `eval_cli:main` | `eval_cli.py:7` ✅ |
| `sneppx-quantize` | `quantize_cli:main` | `quantize_cli.py:8` ✅ |
| `sneppx-rlhf` | `rlhf_cli:main` | `rlhf_cli.py:7` ✅ |
| `sneppx-onnx` | `onnx_cli:main` | `onnx_cli.py:22` ✅ (note: delegates to third-party `onnx.cli`, `onnx 1.5.2` present) |

- **16 → `ctm_tools.py.*:main`** (note the unusual `ctm_tools/py/` directory, i.e. module `ctm_tools.py`): all 16 import and define `main` ✅.

*Caveat:* the brief's premise of "8 CLI entry points mapping to `interface_bindings`" is off by one — there are **7** such scripts plus 16 `ctm_tools` ones. Nothing is missing.

### 5b. `__all__` names that do not resolve — **13 dead names**

Runtime-confirmed via `hasattr` after import (not just static analysis):

| Module | Dead names |
|---|---|
| `vision_transformers.py:935-964` | **11**: `ViTModel`, `ViTForMaskedLM`, `ViTForImageClassification`, `ViTConfig`, `deit_tiny_patch16_224`, `deit_small_patch16_224`, `deit_base_patch16_224`, `swin_tiny_patch4_window7_224`, `swin_small_patch4_window7_224`, `swin_base_patch4_window7_224`, `swin_large_patch4_window7_224` |
| `distillation.py:631-646` | **1**: `distillation_loss` (file defines `kd_loss` L16, `self_distillation_loss` L239, `multi_teacher_distillation_loss` L184 — but not this) |

Impact: `from SneppX_ALG.interface_bindings.vision_transformers import *` **raises `AttributeError`**. The parent `interface_bindings/__init__.py` imports explicitly, so `import SneppX_ALG` is unaffected — the breakage is local to the two modules and to `__all__`-driven tooling/docs.

Confirmed **present** in `vision_transformers.py`: `PatchEmbedding`, `ViTBlock:60`, `ViTMoEBlock:257`, `VisionTransformer`, `vit_{tiny,small,base,large,huge}_patch*:343-391`, `DeiTDistilled:408`, `SwinTransformer`, `SwinBlock`, `SwinPatchEmbed`, `SwinPatchMerging`, `MAE`, `mae_*`, `create_vision_model`. So the `deit_*`/`swin_*` factories exist only as keys inside `create_vision_model`'s dict — the module-level factory functions were never written.

Two static-analysis hits were **false positives**, cleared at runtime: `vizmon.py:354` (`_HAS_C`) and `dispatch/__init__.py:69` (`_HAS_C`) both resolve fine.

### 5c. `_neural_engine_bridge` fallback is an empty module

`__init__.py:10-18`: if neither `_arix_c` nor `_SNEPPX_c` imports, it substitutes `types.ModuleType("_neural_engine_bridge_fallback")` **with zero attributes**. Verified at runtime: `sx._neural_engine_bridge` has **0** public attributes; `sx._neural_engine_bridge.SNEPPXDtype` → `AttributeError`. Yet `__init__.py:5-7` documents exactly that call as the usage example. No native `.pyd`/`.so` exists in the package directory.

### 5d. `sx.functional` is missing despite being documented

`AGENTS.md` states "`sx.functional` name is **RESERVED** … use `F` alias". Runtime: `SneppX_ALG.functional` → `AttributeError: module 'SneppX_ALG' has no attribute 'functional'`, and `hasattr(interface_bindings, "functional")` is `False`. `SneppX_ALG/interface_bindings/functional.py` exists and imports fine standalone (exports `grad, jacfwd, jacrev, jvp, vjp`), but it is never wired into either `__init__`. So the reserved name is unreachable from the public namespace.

---

## 6. Orphan & duplicate modules

### 6a. Orphans — 15 detected, **8 genuinely orphaned library modules** (+5 legit CLI entry points, +2 test modules)

Never imported by any other module in the package (reverse-import graph over `from .X import` / `import ...X`):

| Module | Size | Verdict |
|---|---|---|
| `train_cli.py`, `eval_cli.py`, `quantize_cli.py`, `rlhf_cli.py`, `onnx_cli.py` | 143/51/92/79/36 lines | **Not orphans** — reached via console-script entry points (§5a). |
| `interface_bindings/tests/test_integration.py`, `tests/test_model_config.py` | 348/310 lines | Test modules, not part of the library graph (expected). |
| **`init.py`** | **190 lines, 14 defs** | **Dead duplicate.** `interface_bindings.init` is never loaded (`'…init' in sys.modules` → `False`); `nn_init` is the live one. Its whole namespace (`zeros_`, `ones_`, `xavier_*`, `kaiming_*`, `calculate_gain`, …) collides with `nn_init.py` and loses. |
| **`functional.py`** | **154 lines, 7 defs** | **Dead duplicate** of `autograd.py`'s `jvp`/`vjp`/`grad` — and unreachable as `sx.functional` (§5d). |
| **`nccl.py`** | 343 lines, 15 defs | Never imported by package code; duplicates `distributed.py`. Imports with a warning `[SNEPPX NCCL] Warning: NCCL library not found, using CPU fallback`. |
| **`acvp_kat.py`** | **480 lines, 27 defs** | Never imported. CAVP/KAT test-vector runner (standalone harness). |
| **`c_tensor_bridge.py`** | 197 lines, 9 defs | Never imported — NumPy↔C tensor conversion helper. |
| **`artifact_integrity.py`** | 88 lines, 7 defs | Never imported. |
| **`secrets_redact.py`** | 88 lines, 3 defs | Never imported. |
| **`rng.py`** | 29 lines, 2 defs | Never imported. |

### 6b. Duplicates — 157 colliding top-level names

Resolution is **last-import-wins inside `interface_bindings/__init__.py`**, so the loser is silently unreachable from the package namespace. Runtime-verified winners:

| Symbol | Contenders | Winner |
|---|---|---|
| `run_benchmarks` | `benchmarking.py:741`, `benchmarks.py:623` | **benchmarks.py** |
| `BenchmarkConfig/Result/Suite/Timer` | `benchmark.py:44-98`, `benchmarking.py:23-114`, `benchmarks.py:25-116` | **benchmarks.py** |
| `Trainer` | `train.py:328`, `trainer_v3.py:224`, `pytorch_api.py:779` | **train.py** |
| `TrainConfig` | `train.py:9`, `trainer_v3.py:38` | train.py |
| `ModelConfig` | `model.py:10`, `model_zoo.py:20` | **model.py** |
| `SearchSpace`, `Trial` | `hparams.py`, `hpo.py` | hpo (one of two competing HPO stacks) |
| `zeros_/ones_/*_init`, `calculate_gain` | `init.py`, `nn_init.py` | **nn_init.py** (`init.py` dead) |
| `grad`, `jvp`, `vjp` | `autograd.py`, `functional.py`, `jit.py` | `grad` → **jit.py**; `jvp`/`vjp` unresolved at package level |
| ~70 ops (`conv2d`, `linear`, `relu`, `batch_norm`, `max_pool2d`, `pad`, …) | `advanced_ops.py`, `nn_functional.py` | **mixed** — `conv2d` from `advanced_ops`, but `linear`/`relu`/`batch_norm` from `nn_functional` |
| All loss/activation/layer classes (`CrossEntropyLoss`, `MSELoss`, `BCELoss`, `Conv1d/2d/3d`, `LayerNorm`, `RMSNorm`, `Dropout`, …) | `autograd_ops.py` **and** `nn.py` (and `pytorch_api.py`/`keras_api.py`/`train.py` for several) | `nn.py` |
| Snapshot/metrics: `Adam`, `AdamW`, `SGD`, `Optimizer`, `LRScheduler`, `StepLR`, `CosineAnnealingLR`, `ReduceLROnPlateau`, `ExponentialLR` | `optim.py`/`schedulers.py` **and** `train.py`/`pytorch_api.py` | `optim.py`/`schedulers.py` |
| `DataLoader`, `Dataset`, `TensorDataset` | `data.py`, `data_loader.py`, `pytorch_api.py`, `train.py` | last-imported wins |
| `Tensor` | `tensor.py` **and** `pytorch_api.py` | `tensor.py` |
| Distributed: `init_process_group`, `all_reduce`, `barrier`, `get_rank`, `get_world_size`, `destroy_process_group`, `ProcessGroup`, `_NCCLBackend`, `DistributedDataParallel`, `DistributedSampler` | `distributed.py` **and** `nccl.py` | `distributed.py` |
| Firewall/security: `FirewallResult` (×3 files), `ConnectionTracker` (×2), `DDoSMitigation` (×2), `PromptFilter`/`OutputVerifier` (×2), `AuditLogger` (×2), `ConstantTime` (×2) | `firewall_*`, `ddos_mitigation`, `ai_safety_ext`, `security_middleware`, `audit_logger`, `key_vault`, `asm_bridge`, `crypto_util` | varies |
| `ModelVersion` | `mlflow_like.py`, `serving_engine.py` | varies |
| `from_pretrained` | `hf_integration.py`, `model_zoo.py` | varies |
| `get_model_config` | `llama_models.py`, `model_implementations.py`, `model_zoo.py` | varies |
| `get_optimizer`, `_as_array` | `optim_advanced.py`, `optim_extra.py` | varies |
| `MemoryTracker` | `benchmarking.py:88`, `benchmarks.py:90`, `profiler.py` | varies |

**Do they conflict?** For the benchmark trio, **yes** — three independent `BenchmarkSuite`/`BenchmarkConfig`/`BenchmarkResult`/`BenchmarkTimer` implementations with ~500-760 lines each, and two different `run_benchmarks` signatures (`benchmarking.run_benchmarks` vs `benchmarks.run_benchmarks(output_dir=None)`). `benchmark.py` additionally has `run_benchmark_cli` (L465) which is never reached. For `Trainer`, four classes named `Trainer`/`UltraTrainer`; `trainer_v3.Trainer` and `pytorch_api.Trainer` are unreachable by public name. This is genuine API surface confusion, not just tidiness.

**Experiment stack is 3-deep and partially dissociated:** `experiment.py` (exports `Experiment`), `experiment_tracker.py` (exports `ExperimentTracker`/`Tracker`/`LogTracker`), `experiment_cli.py` (CLI). `hparams.py:14` imports `ExperimentTracker` — so the tracker is live — while `experiment.py` provides a separate `Experiment` concept.

---

## 7. Environment-induced degradation (structural, not a code bug)

**66 `_HAS_C`/`_LIB`/`_C` flags across ~40 modules evaluate to `False`/`None`** here because the `neural_security_c` shared library is absent. Affected: `ai_safety_ext, algo_arc, algo_fm, algo_hss, algo_npe, algo_ser, asm_bridge, c_arch, c_attention, c_logger, c_memory, c_simd_gemm, c_tensor_expr, c_thread, crypto_hash, crypto_kdf, crypto_kem, crypto_sign, crypto_symmetric, crypto_util, ddos_mitigation, dispatch, dispatch_cpu, kernel_dispatch, driver_bindings, formal_verify, key_vault, net_bindings, pen_testing, runtime_monitor, s4_network, s5_safety, secure_memory, secure_updates, vizmon`.

The `c_loader.load_library()` probe is well-built (env-var override, candidate names, search dirs) and degrades gracefully — but this means the *entire natively-accelerated and native-crypto surface is inert* on a stock pure-Python install, and several modules keep advertising capability (`hw_backends.available` → `True`, `edge_runtime.supports` → `True`) without checking.

**Hard third-party dependencies not in the declared extras:** `fastapi` is imported at module top-level by `api_gateway.py` and `inference_server.py` → both fail `import` when absent (the only 2/225 failures). `cryptography` (absent) gates all symmetric/PK crypto; `scipy` (absent) is used at `tensor.py:1067`; `onnxruntime` (`hw_backends.py:119,313`), `safetensors` (`model_zoo.py:1545`), `argon2` (`crypto_kdf.py:137`), `psutil` (`benchmarking.py`/`benchmarks.py`, guarded) are optional-but-unguarded-by-metadata. `onnx` 1.5.2 is present but is *not* in `pyproject.toml` dependencies despite `sneppx-onnx` delegating to it.

---

## 8. Top 20 highest-impact gaps (ranked)

| # | Gap | Location | Why it matters |
|---|---|---|---|
| 1 | **Certificate verification always `True`** | `s4_network.py:111-112` | Fail-open TLS. Runtime-verified. Any caller relying on it accepts any cert/CN. Security-critical. |
| 2 | **`trace()` returns `None`** | `graph_optimizer.py:96-106` | Documented public API (build a compute DAG) silently produces nothing. Runtime-verified. |
| 3 | **Anomaly detection constant `0.0`** | `runtime_monitor.py:102-104` | Monitoring can never alert; `raise_alert` machinery is dead weight. Runtime-verified. |
| 4 | **`onnx_export.export_onnx` writes a dummy graph** | `onnx_export.py:865-877` | Emits an apparently-valid ONNX file that ignores the model (`[1,1000]` dummy output). Silent wrong artifact. |
| 5 | **11 dead `__all__` names in `vision_transformers`** | `vision_transformers.py:939-942,949-951,956-959` | `from …vision_transformers import *` raises `AttributeError`; ViT/DeiT/Swin public classes+fatories don't exist. Runtime-verified. |
| 6 | **`distillation_loss` in `__all__` but undefined** | `distillation.py:646` | Same star-import failure for the distillation module. Runtime-verified. |
| 7 | **`sx.functional` unreachable** | `__init__.py:20-54`, `functional.py` | AGENTS.md documents it as reserved; it raises `AttributeError`. Entire `functional.py` (jvp/vjp/grad) + dead duplicate of `autograd.py`. |
| 8 | **`_neural_engine_bridge` fallback empty** | `__init__.py:10-18` | Root usage example `ax.SNEPPXDtype.FLOAT32` fails; 0 attributes. No native lib shipped. |
| 9 | **`KyberKEM` permanently raises** | `crypto_kem.py:18-29,58` | Fails closed (good) but **no** KEM path works at all, even with the C lib — `_require_kyber_backend` is unconditional. |
| 10 | **No working public-key crypto in pure-Python** | `crypto_sign.py:86-201`, `crypto_symmetric.py:47-97`, `crypto_kem.py:76-103`, `crypto_kdf.py:137` | `cryptography` absent → Ed25519/AES-GCM/ChaCha20/X25519/Argon2 all refuse. Correctly loud, but a capability hole. |
| 11 | **`serve_cli --checkpoint` unsupported** | `serve_cli.py:351-352` | Serving CLI cannot load a checkpoint — the flag exists but raises. |
| 12 | **`max_batch_size_for()` → `0`** | `serving_engine.py:286-287` | Continuous batching sizing stub; latency always recorded as `0` (`:592`). |
| 13 | **Streamed text silently dropped** | `inference_server.py:856-860` | `AsyncStreamer.put_text/end` are `pass`; text-mode SSE emits nothing for text callbacks. |
| 14 | **CUDA driver shim reports kernel success** | `driver_bindings.py:203,215,313,383,…` | `launch_kernel`/`tc_gemm` return `0` without running; safe only because `is_available()` is `False`. |
| 15 | **`socket_recv` → `b""`** | `dispatch/infra_dispatch.py:57-58` | Indistinguishable from peer EOF; `rdma_connect` → `0` (success) without connecting (L63-64). |
| 16 | **gRPC distributed service is a stub** | `net_bindings.py:385-388,451,473,509` | `read`/`write`/`broadcast`/`barrier` all return `0`; cross-node coordination is inert. |
| 17 | **`nccl` collectives are identity/no-op** | `nccl.py:101-111,146-165` | `all_reduce`/`broadcast` return input unchanged for `world_size>1`; send/recv/barrier no-op. Documented honestly, but results look valid. |
| 18 | **4 competing `Trainer`s + 3 `BenchmarkSuite`s** | `train.py:328`, `trainer_v3.py:224`, `pytorch_api.py:779`, `trainer_v2.py:159`; `benchmark*.py` | 157 colliding names; losers silently unreachable. `trainer_v2` gradients are finite-difference "simulated" (`:266,332`). |
| 19 | **`init.py` + `functional.py` dead duplicates** | `init.py` (190 L), `functional.py` (154 L) | Both never imported; `init.py` loses every name to `nn_init.py`. ~344 lines of dead code. |
| 20 | **`fastapi` hard-required, undeclared** | `api_gateway.py`, `inference_server.py` | Only 2/225 import failures; whole inference-server + API-gateway surface unavailable without it, and it's not in `pyproject.toml` deps/extras. |

**Runner-up gaps:** `experiment_tracker` no-op base silently discarding metrics (`:110-154`); `optim_advanced` `adam` registry entry returning `None` (`:668`); `get_model_flops` → `0.0` (`model_implementations.py:1702`); `Blake3` missing `update()` (`crypto_hash.py:49`); `hw_backends.available`/`edge_runtime.supports` claiming `True` unconditionally; 8 orphaned library modules totalling **1,569 lines** (`init.py` 190, `functional.py` 154, `nccl.py` 343, `acvp_kat.py` 480, `c_tensor_bridge.py` 197, `artifact_integrity.py` 88, `secrets_redact.py` 88, `rng.py` 29).

---

### Scope caveats

- Everything above is **static reading + import/`__all__`/behavioural probes** in a NumPy-only environment. No GPU, no CUDA, no model inference, no `transformers`/`llama_cpp` import, no `from_pretrained`.
- Behavioural claims marked "runtime-verified" were executed on this host (`trace`→`None`; `verify_certificate`→`True`; `detect_anomaly`→`0.0`; BLAKE3/SHA vectors matched).
- `cryptography` and `scipy` are absent, so crypto/extras behaviour was assessed from source plus guarded-import paths, and the "refusal" branches are exactly what executes here.
- Patterns were matched per `.py` file; generated `__pycache__` excluded.
