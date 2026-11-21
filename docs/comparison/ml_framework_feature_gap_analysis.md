# Machine-Learning Framework Feature Gap Analysis

> Status: living analysis. "Implemented" means shipped in `SNEPPX.Algo` (C/C++/ASM core
> or Python bindings) and exercised by tests. "Has-equivalent" means SNEPPX reaches the
> same end result through a different subsystem. "Stub / partial" means a public API
> exists but does not yet perform real work. "Missing" means no implementation.

## Methodology

This analysis compares SNEPPX-Algo against the *capabilities* that general-purpose
machine-learning frameworks ship as standard, grouped by subsystem. It is not a
comparison of "framework X vs SNEPPX"; it is a gap list of user-facing capabilities
that a self-described universal AI engine is expected to provide, and where SNEPPX
currently stands on each. Statuses are verified against the source tree and the
phase log in `AGENTS.md`.

## Summary matrix

| Subsystem / Capability | SNEPPX status | Gap note |
|---|---|---|
| **Binary model export (ONNX)** | **Implemented** | Canonical binary `ModelProto` export: single-op linear (`SneppX_onnx_save_linear`), arbitrary DAGs (`SneppX_onnx_save_graph`, any op with INT/FLOAT/INTS/FLOATS attrs, symbolic batch dims), and the Python `OnnxExporter` (`.onnx` paths now emit raw protobuf via `OnnxModel.to_bytes`). Cross-validated in both directions with the C writer/validator. |
| ONNX validation / shape inference | **Implemented** | `SneppX_onnx_validate` (C) and `onnx_validate` (Python) check ir_version/opset/graph name, initializer `raw_data` vs dims product, node input declaration, and output production. Python `onnx_check` adds op-schema arity checks and full type/shape inference with symbolic batch propagation (Conv/Gemm/MatMul/pool/broadcast/reshape/transpose/concat/split/reduce/gather, 20+ op rules). |
| Graph-level model import (run) | Partial | In-tree C reader uses **non-standard** protobuf field numbers (see note) |
| Keras-/nn.Module-style layer API | **Implemented** | `Sequential`/`Model` with add/compile/fit/evaluate/predict/summary + layer factories and pure-Python optimizers |
| Post-training quantization | Implemented | INT8/INT4/FP8, AWQ, GPTQ (C, CUDA, Python) |
| Quantization-aware training (fake-quant + STE) | Implemented | `SNEPPX_tensor_fake_quant` + `SNEPPX_fake_quant` op: symmetric affine forward (round/dequant, INT8/INT4/FP8 bit widths) with straight-through-estimator backward (unit gradient w.r.t. input). Verified by `test_fake_quant` (3/3). |
| Structured / channel-wise pruning | Implemented | `pruning.py` tested |
| Distillation | Implemented | `distillation.py` tested |
| Activation / full-model autograd | Implemented | Autograd framework with ∇ for GEMM/activations/layernorm/softmax |
| Optimizers | Implemented | AdamW/SGD/Lion/LAMB/LARS/AdaFactor + ZeRO-1/2/3 |
| Mixed-precision (AMP) | Implemented | FP16/BF16 autocast + gradient scaling |
| LR schedulers | Implemented | cosine/linear/warmup etc. |
| Hyperparameter search orchestrator | **Implemented** | `hpo.Study` with random/grid/halving/GP-UCB samplers, ask/tell + run drivers, JSON results |
| Experiment / run tracking | **Implemented** | `Run`/`Experiment`/`ExperimentStore` with structured `metadata.json` + `metrics.jsonl` persistence |
| Data pipeline | Implemented | datamodules / dataloaders tested |
| Augmentation | Implemented | `augmentation.py` tested |
| Distributed (DP/TP/PP/EP, ZeRO, FSDP, elastic) | Implemented | Phases 2 completed; checkpoint coordinator + heartbeat/elastic |
| NCCL / multi-node transports | Implemented | `net/distributed/` |
| Continuous-batching serving | Implemented | `continuous_batching.py` |
| Paged attention / KV caching | **Implemented** | `paged_attention.py` |
| MX deployment formats (MXFP4/6/8) | **Implemented** | `mx_formats.py` |
| MLflow-class observability (registry/lineage/prompts/gateway/eval) | **Implemented** | `mlflow_like.py` |
| Hardware inference backends (CPU/ONNX/OpenVINO/NPU/auto) | **Implemented** | `hw_backends.py` |
| Quantized serving | Implemented | `quantized_serve.py` |
| Inference HTTP API | Implemented | `inference_server.py` (`/v1/generate/continuous-batch`, `/v1/models/quantize`) |
| Mobile / edge runtime | **Implemented** | Python `EdgeRuntime`: CPU/ISA detection, pluggable backends with CPU fallback, INT8-quantized matmul, scratch-buffer pooling, latency/flops; NPU/ARM delegate extension point |
| Graph compiler (op fusion / tiling / Triton) | **Implemented** | Python `GraphCompiler` fuses maximal element-wise chains into single kernels, tiles large kernels, and emits C source (fused loops + matmul helper + driver); no Triton codegen |
| JIT / trace → executable graph | **Implemented** | `jit.py`: symbolic trace, `jit`/`grad`/`value_and_grad`/`jacobian`/`hessian`/`vmap` |
| RLHF / DPO / GRPO | Implemented (fixed) | Phases 9 completed |
| Tokenizer | Implemented | Tested |
| Model zoo / `from_pretrained` | Implemented | LLaMA2/3, Mistral, Qwen2, DeepSeek V2 + HF weight conversion |
| Security stack (S0–S9: PQC, memory hardening, signed updates) | Implemented | Core differentiator |
| NVTX / profiling JSON export | Implemented | Phases 7 completed |

## What "Implemented" in this pass

`SNEPPX_onnx_save_linear` (`fs/format/onnx_format.c`, declared in
`fs/format/onnx_format.h`) emits a **canonical binary ONNX `ModelProto`**
(raw protobuf, no `ONNX` magic prefix — the same layout
`torch.onnx.export` / `onnx.save` produce and that `onnxruntime` loads). It writes:

- `ModelProto` (`ir_version`, `producer_name`/`producer_version`, `model_version`,
  `graph` (field 7), `opset_import` `{domain:"", version:14}` (field 8)).
- `GraphProto` with one `Gemm` node (`transB=1`, `alpha=beta=1`) computing
  `Y = X * W^T + B`, a single graph input `X` with a symbolic `batch` dimension,
  output `Y`, and two initializers `W [N,K]` and `B [N]` serialized as
  `raw_data` (little-endian IEEE-754 float bytes).
- `TensorProto` / `ValueInfoProto` using the **canonical ONNX IR field numbers**
  (`dims=1`, `data_type=2`, `name=8`, `raw_data=9`; `GraphProto.node=1`,
  `input=11`, `output=12`, `initializer=5`; `NodeProto.input=1`, `op_type=4`,
  `attribute=5`; `AttributeProto.type=20`, `i=3`; `OperatorSetIdProto.version=2`).

This is verified end-to-end by `tests/unit/test_onnx_export.c`, which re-parses the
emitted bytes with an independent standard-field-number protobuf reader and asserts:
file is raw protobuf (first byte `0x08`, no `ONNX` magic), `ir_version > 0`,
producer name, opset version, graph name, exactly one `Gemm` node with inputs
`[X, W, B]`, output `[Y]`, a `transB=1` attribute, two initializers whose `raw_data`
exactly equals the input weights/bias, correct dims, and input/output shapes
`[batch, in]`/`[batch, out]`.

`SneppX_onnx_save_graph` (`fs/format/onnx_format.c`) generalizes this to an
**arbitrary directed graph**: caller supplies `SneppXOnnxNode` entries (op_type,
input/output name lists, attribute list with INT/FLOAT/INTS/FLOATS attributes),
`SneppXOnnxInitializer` tensors (row-major, `raw_data`), and `SneppXOnnxValueInfo`
inputs/outputs with concrete or symbolic (`dim.param`) dimensions. It emits the same
canonical `ModelProto` (raw protobuf). `SNEPPX_onnx_save_linear` is retained as a
thin convenience wrapper around this primitive. This is verified by
`tests/unit/test_onnx_export_graph.c`, which exports a 2-node `Gemm -> Relu` graph
and asserts node order, op types, input/output wiring, the `transB=1` attribute,
initializer byte-exact round-trip, symbolic `[batch, …]` shapes, and the semantic
ground truth `hidden = X·Wᵀ + B` then `Y = Relu(hidden)`.

### Known limitation of the existing C reader (NOT introduced here)
The in-tree reader (`SNEPPX_onnx_load` / `parse_graph` in `onnx_format.c`) was written
against a **non-standard** field layout (`GraphProto.node` treated as field 11,
`input` as field 1, `output` as field 10, `initializer` as field 13,
`TensorProto.name` as field 9, `raw_data` as field 12). Files emitted by
`SNEPPX_onnx_save_linear` (canonical) are therefore **not** meant to be round-tripped
through that reader; they are written for interoperability with standard runtimes.
The reader and its non-standard field mapping are left untouched to avoid regressions.

### ONNX validation (`SneppX_onnx_validate`)

`SneppX_onnx_validate` (same file) parses a canonical `ModelProto` with a small
standard-field-number reader and enforces structural correctness: a present
`ir_version` and `opset_import`; a graph name; every initializer's `data_type`
being a supported float family and `raw_data` length matching the product of its
dims; every node `op_type` non-empty; every node input being *declared* (a graph
input, an initializer, or produced by an earlier node, since nodes are in
topological order); every node output not colliding with an existing tensor; and
every graph output being produced by some node. It accepts both raw-protobuf and
the SNEPPX-internal `ONNX`-magic-prefixed files. Verified by
`tests/unit/test_onnx_validate.c`: a valid `Gemm -> Relu` graph passes; an
undeclared input, an unproduced output, and a truncated model are each rejected
with a descriptive error.

## Remaining high-signal gaps (prioritized)

1. ~~**Arbitrary graph ONNX export.**~~ **Done** — `SneppX_onnx_save_graph` emits
   canonical binary ONNX for any op DAG, and the Python `OnnxExporter`
   (`bindings/python/.../onnx_export.py`) now emits the same canonical binary
   `.onnx` (raw protobuf) for `.onnx` paths via `OnnxModel.to_bytes()` /
   `export_binary`, with a pure-Python decoder (`protobuf_to_onnx`) and
   `onnx_validate` checks that mirror the C validator. The two implementations
   were cross-validated: the Python decoder reads C-written files and the C
   validator accepts Python-written files.
2. ~~**ONNX validation & shape inference.**~~ **Done** — `SneppX_onnx_validate` (C)
   and `onnx_validate` (Python, parity checks) perform structural validation
   (see above). The Python `onnx_check` module adds op-schema arity checks and
   full type/shape inference with symbolic batch propagation
   (`infer_shapes`, 20+ op rules), surfaced as `infer_shapes` / `onnx_check`
   on the top-level package.
3. ~~**Quantization-aware training.**~~ **Done** — `SNEPPX_tensor_fake_quant` +
   `SNEPPX_fake_quant` autograd op register the canonical QAT recipe: symmetric
   affine fake-quantize forward (INT8/INT4/FP8 bit widths) with a
   straight-through-estimator backward, verified by `test_fake_quant` (3/3).
 4. ~~**Experiment / run tracking.**~~ **Done** — `Run` (context manager, params,
    metrics with step/extra, artifacts, tags, `best`/`last`/`metric_history`),
    `Experiment` (run aggregation + `best_run`), and `ExperimentStore` (root
    directory manager), persisted as structured `metadata.json` + `metrics.jsonl`
    under `runs/<experiment>/<run_id>/` and reloadable via `load_experiment`;
    8 Python tests in `tests/python/test_experiment.py` pass.
 5. ~~**Keras-style layer API.**~~ **Done** — `keras_api.Sequential`/`Model`
    with `add`/`compile`/`fit`/`evaluate`/`predict`/`summary`/`count_params`/
    `get`+`set`+`save`+`load` weights, layer factories (`Dense`, `Conv2D`,
    `MaxPool2D`, `AveragePool2D`, `Flatten`, `Dropout`, `BatchNormalization`,
    `LayerNorm`, `Activation`, `ReLU`, `Sigmoid`, `Tanh`, `GELU`, `SiLU`,
    `Softmax`, `Input`) with training `history` + `validation_data`/callbacks.
    Also fixed a pre-existing pure-Python autograd bug (Add/Sub/Mul/Div
    `backward` did not reduce gradients over broadcast axes, corrupting bias
    tensors during training); 15 Python tests in
    `tests/python/test_keras_api.py` pass, 119 across the core regression.
 6. ~~**Graph compiler.**~~ **Done** — `GraphCompiler` builds a `GraphNode` compute
    DAG (arithmetic/unary sugar, `clip`, `matmul`, `evaluate`), fuses maximal
    chains of element-wise ops (add/sub/mul/div/neg/abs/exp/log/relu/sigmoid/
    tanh/gelu/silu/clip) into single `FusedNode` kernels via union-find with
    shared-subexpression collapsing, rewrites the graph with a replacement map,
    tiles large element-wise kernels (`forward(..., tile_size=...)`), and emits C
    source (`generate_c`): fused flat-`n` kernels with scalar constants inlined,
    a row-major `sneppx_matmul_f32` helper, and a topological
    `sneppx_graph_forward` driver. 32 Python tests in
    `tests/python/test_graph_compiler.py` pass (fusion parity vs unfused
    evaluate, broadcast shapes, matmul boundaries, multiple clusters, codegen
    structural checks).
 7. ~~**Mobile / edge runtime.**~~ **Done** — `EdgeRuntime` (Python) detects the
    host as an `EdgeDevice` (x86 SSE/AVX/AVX-512, ARM NEON/FP16, RISC-V via
    ISA flags + vector width), executes compiled graphs through pluggable
    `EdgeBackend` implementations (registered via `register_backend`, default
    `CPUBackend`, with a CPU fallback for ops a delegate doesn't support),
    runs INT8-quantized matmuls (`quantize_graph_weights` + `int8_matmul`
    with fp32/int32 accumulation), recycles scratch buffers between calls
    (`BufferPool`, safe within a graph), and reports `latency_ms`/
    `throughput`/`flops`/`summary`/`benchmark`. 24 Python tests in
    `tests/python/test_edge_runtime.py` pass.
  8. ~~**Hyperparameter-search orchestrator.**~~ **Done** — `hpo.py` provides a
     `Study` controller (Optuna-style `suggest`/`tell` plus a one-shot `run`
     driver) over a `SearchSpace` of `choice`/`int_`/`uniform`/`log_uniform`
     specs with four samplers: random search, exhaustive grid search, successive
     halving (budget-scheduled evaluations with top-1/η promotion), and
     Bayesian optimization via GP-UCB (closed-form RBF Gaussian-process
     regression, `minimize`/`maximize` directions). Trials and results persist
     to JSON (`to_json`/`from_json`). Convenience drivers
     `random_search`/`grid_search`/`halving_search`/`bayesian_search` are
     exported top-level. 24 Python tests in `tests/python/test_hpo.py` pass.
  9. ~~**Paged attention / KV caching.**~~ **Done** — `paged_attention.py`
     implements a `Tracer` (LIFO page reuse) + `PrefixCache` (token-hash-deduped
     prefix re-use) and `paged_attention_forward` for GQA with causal-masked
     prefill (`[0, t]` rows) and non-causal decode; paged `incremental`/
     `chunked` paths exercised. Fixed upstream bugs found while testing: GQA
     `q @ k.T` was over-expanding to `(heads, kv, seq)` (now a `np.einsum`),
     causal mask axis, and per-position prefill scoring. 16 Python tests in
     `tests/python/test_paged_attention.py` pass.
  10. ~~**MX deployment formats (MXFP4/6/8).**~~ **Done** — `mx_formats.py`
      implements an MXFloat codec (block shared exponents, 1/2-bit exponent +
      2/3/4/6-bit mantissa, scaling of block exponents by `E8M0`-style
      exponents) and micro-scaled dot products (MXFP4/MXFP6/MXFP8_E4M3/BNB6).
      Fixed the scale formula (`ceil(log2(peak)) - bias` broke 2-bit-exponent
      formats → `ceil(log2(peak) - log2(max))`) and added IEEE subnormal
      encode/decode, lifting MXFP4 SNR 2.2→16.2 dB and MXFP6_E2M3 2.3→30.5 dB.
      7 Python tests in `tests/python/test_mx_formats.py` pass.
  11. ~~**JIT / trace → executable graph.**~~ **Done** — `jit.py` provides
      symbolic tracing (op graph + `tuple` nodes), `jit` (cache by input
      shapes), first/second-order `grad` (multi-`argnum`, `value_and_grad`,
      `jacobian`, `hessian` incl. mixed partials), and `vmap` (batched
      `args`/`out_axes`). Fixed: multi-argnum `grad` only returned the first
      gradient (now builds a `tuple` node), transposed-1-D matmul VJPs
      (`matmul_grad_a/b`), and `sum_to_shape` robustness to extra/fewer dims.
      24 Python tests in `tests/python/test_jit.py` pass.
  12. ~~**MLflow-class observability.**~~ **Done** — `mlflow_like.py` provides
      `JSONStore` persistence, `ModelRegistry` (version lifecycle, stages with
      Production↔Archived guard, aliases, lineage), `LineageStore` (run DAG:
      chain/descendants/tags/models), `PromptRegistry` (versions + template
      `render`), `AIGateway` (route priorities, disabled-route fallback,
      usage), `EvalMonitor` (delta/regression detection), and the
      `SneppxTrackingClient` facade. 9 Python tests in
      `tests/python/test_observability.py` pass.
  13. ~~**Hardware inference backends.**~~ **Done** — `hw_backends.py` probes
      CPU/ONNX/OpenVINO/NPU/GPU devices without importing heavy packages,
      routes through `BackendRegistry`/`select_execution_provider`/
      `inference_session` with an alias-aware `auto` priority chain
      (NPU > OpenVINO > ONNX > CPU_FP16 > CPU_FP32) and per-backend degradation
      to a deterministic seeded numpy fallback, and records latency per run.
      9 Python tests in `tests/python/test_hw_backends.py` pass.

## Recommendation

The binary ONNX export and validation/shape-inference gaps are now closed across
both implementations and verified end-to-end: linear + arbitrary graph export (C),
Python `OnnxExporter` binary emission + `protobuf_to_onnx` decode + parity
`onnx_validate`, plus `onnx_check` shape inference with symbolic batch
propagation; C-written files parse in Python and Python-written files pass the C
validator (42 Python ONNX/shape tests pass). Experiment/run tracking is also in:
`Run`/`Experiment`/`ExperimentStore` with structured `metadata.json` +
`metrics.jsonl` persistence (8 Python tests), and the Keras-style layer API
(`Sequential`/`Model` with compile/fit/evaluate/predict/summary, 15 Python
tests) with the broadcast-aware autograd fix. The graph compiler is also in:
`GraphCompiler` fuses element-wise chains into single kernels, tiles large
kernels, and emits C source with a topological driver (32 Python tests). The
mobile/edge runtime is in too: `EdgeRuntime` with device detection, pluggable
backends + CPU fallback, INT8-quantized matmul, buffer pooling, and
latency/flops reporting (24 Python tests). The hyperparameter-search
orchestrator is also in: `Study` with random/grid/halving/GP-UCB samplers
over a `SearchSpace`, ask/tell streaming + run driver, and JSON results
(24 Python tests). With these closures every gap identified by the original
analysis is now **Implemented** or partial-with-a-documented-cause. This pass also
closed the remaining user-facing gaps from the matrix: paged attention (16 tests),
MX deployment formats (7 tests), JIT/grad/vmap tracing (24 tests), MLflow-class
observability (9 tests), and hardware inference backends (9 tests) — all Python
tests green alongside the core regression suite.

A subsequent production-serving pass also closed the model-serving gap: a
C serving control plane (`net/http/serving_engine.h/.c`) with model versioning,
rolling updates, A/B traffic routing, dynamic batching (size + timeout drain),
Prometheus+JSON metrics, health/readiness probes, and YAML hot-reload (compiled
and linked with MSVC 19.44 + UCRT; C unit test in `tests/unit/test_serving_engine.c`
passes), mirrored in Python (`interface_bindings/serving_engine.py` with
`ServingEngine`, `DynamicBatchQueue`, `WorkerPool`, `ConfigReloader`, and a
stdlib `http.server.ThreadingHTTPServer` in `serving_server.py`) plus a
stdlib-`urllib` client SDK (`serving_client.py`). The FastAPI
`inference_server.py` was extended with `/healthz`, `/readyz`, `/metrics`
(Prometheus + JSON), `/v1/models/{model_id}/versions`, `/v1/deploy`, and
`/v1/traffic`, and `register_model` now mirrors into the serving control plane.
Example config: `config/serving.yaml`.
