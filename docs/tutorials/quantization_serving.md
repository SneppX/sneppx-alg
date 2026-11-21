# Tutorial — Quantization & Serving

**Notebook:** [`quantization_serving.ipynb`](https://github.com/SneppX/sneppx-alg/blob/main/notebooks/quantization_serving.ipynb)
([download](https://github.com/SneppX/sneppx-alg/blob/main/notebooks/quantization_serving.ipynb))

## What you'll build

Quantize the weights of a small Transformer to **INT4** (AWQ) and **FP8**,
measure the error/SNR, estimate the on-disk size, and launch the
`sneppx-serve` FastAPI server with an authenticated, quantized model.

## Setup

```powershell
$env:PYTHONPATH = "bindings/python"
# Serving deps (optional):
python -m pip install "sneppx-alg[serve]"
```

```python
import numpy as np
from SneppX_ALG import (
    Transformer, Tensor, AdamW,
    QuantMode, quantize_int8_sym, dequantize_int8_sym,
    quantize_int4_sym, dequantize_int4_sym,
    quantize_int8_channel, dequantize_int8_channel,
    awq_quantize, awq_scale_weights, quantize_error,
)
from SneppX_ALG.interface_bindings.quantized_serve import (
    QuantizedModelConfig, quantize_model_weights,
    dequantize_weights, estimate_model_size_mb,
)
HAS_C = __import__("SneppX_ALG")._HAS_C_BACKEND
```

## 1. A small model + its raw weights

```python
model = Transformer(vocab_size=500, dim=128, num_heads=4, num_layers=2, ffn_dim=256, max_seq_len=64)
params = {name: p.data.copy() for name, p in model.named_parameters()}
print("FP32 size MB:", sum(v.nbytes for v in params.values()) / 1e6)
```

## 2. Per-tensor INT8 sanity check

```python
w = Tensor.from_numpy(params["lm_head.weight"].astype(np.float32))
qw, scale = quantize_int8_sym(w)
wq = dequantize_int8_sym(qw, scale)
print("INT8 SNR (dB):", quantize_error(w, wq, metric="snr"))
```

## 3. Per-channel INT8 (lower error)

```python
qw, scales = quantize_int8_channel(w, dim=-1)
wq = dequantize_int8_channel(qw, scales)
print("per-channel INT8 SNR:", quantize_error(w, wq, metric="snr"))
```

## 4. INT4 AWQ (activation-aware)

```python
act_scales = Tensor.randn((w.shape[1],))           # per-input-channel act scale
qw, scales = awq_quantize(w, act_scales, group_size=64)
# (decode path: per-group dequant then matmul — see QuantizedLinear)
```

## 5. Quantize the whole model for serving

```python
cfg = QuantizedModelConfig(
    quant_mode=QuantMode.INT4_SYM,
    skip_layers=["lm_head", "embedding.weight"],   # keep embeddings FP32
)
quantized = quantize_model_weights(params, cfg)
print("INT4 size MB:", estimate_model_size_mb(quantized))
```

## 6. Round-trip (quantize → dequantize → compare)

```python
restored = dequantize_weights(quantized)
max_err = max(
    np.abs(restored[k].astype(np.float32) - params[k].astype(np.float32)).max()
    for k in ["layers.0.attn.q_proj.weight"]
)
print("max abs error:", max_err)
```

## 7. Serve with `sneppx-serve`

The quantized weights can be saved and loaded by the FastAPI server:

```python
import json, numpy as np
# Save quantized weights as a .npz or safetensors; write a config.
np.savez("/tmp/q_model.npz", **{k: (v.weight.data if hasattr(v, "weight") else v)
                                for k, v in quantized.items()})
cfg_out = {"quant_mode": QuantMode.INT4_SYM, "skip_layers": ["lm_head"]}
json.dump(cfg_out, open("/tmp/q_config.json", "w"))
print("saved /tmp/q_model.npz + /tmp/q_config.json")
```

Then from the shell:

```powershell
sneppx-serve --port 8000 --host 127.0.0.1 `
    --auth-mode api-key --api-keys "dev-key" `
    --model-config /tmp/q_config.json --checkpoint /tmp/q_model.npz `
    --disable-output-verify
```

```bash
curl -X POST http://127.0.0.1:8000/v1/generate \
  -H "Authorization: Bearer dev-key" -H "Content-Type: application/json" \
  -d '{"prompt":"hi","max_new_tokens":8,"do_sample":false}'
```

## Key takeaways

- `QuantizedLinear.from_float` is the fastest drop-in replacement for a single
  `Linear` layer.
- Whole-model quantization via `quantize_model_weights` skips embeddings by
  default — never quantize the LM head if you want stable logits.
- `quantize_error(metric="snr")` gives a single scalar for quality; aim for
  SNR > 20 dB for INT4.
- `sneppx-serve` applies the **S4/S5** firewall and prompt/output filters by
  default — pass `--disable-*` only in trusted environments.

## Next steps

- Quantize a real HF model — see
  [Model Conversion](model_conversion.md) → `convert_hf_to_sneppx`.
- Profile the quantized forward — see
  [Profiling & Benchmarks](profiling_benchmarks.md).
