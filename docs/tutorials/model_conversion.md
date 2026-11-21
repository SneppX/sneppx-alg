# Tutorial — Model Conversion (HF → SNEPPX)

**Notebook:** [`model_conversion.ipynb`](https://github.com/SneppX/sneppx-alg/blob/main/notebooks/model_conversion.ipynb)
([download](https://github.com/SneppX/sneppx-alg/blob/main/notebooks/model_conversion.ipynb))

## What you'll build

Convert a HuggingFace-style model directory (`.safetensors` + `config.json`)
into a **SNEPPX checkpoint** (`.sneppx`) using `convert_hf_to_sneppx`, inspect
the weight-name remapping, and load it back with `CheckpointReader`.

## Setup

```powershell
$env:PYTHONPATH = "bindings/python"
# You need an HF model dir on disk (safetensors + config.json).
# Example: download via `huggingface-cli download` or use a local copy.
```

```python
import os, json, glob
from SneppX_ALG import (
    from_pretrained, get_model_config, list_available_models,
    read_safetensors, convert_hf_to_sneppx, HF_WEIGHT_MAP,
)
from SneppX_ALG import CheckpointReader, validate_checkpoint
HAS_C = __import__("SneppX_ALG")._HAS_C_BACKEND
```

## 1. Pick a supported family

```python
print(list_available_models())     # ['llama2:7B', 'llama2:13B', 'llama3:8B', 'mistral:7B', 'qwen2:7B', ...]

cfg = get_model_config("mistral", "7B")   # config + param estimate
print(cfg["hidden_size"], cfg["num_hidden_layers"], cfg["num_attention_heads"])
```

## 2. Inspect the weight name map

```python
print("HF -> SneppX name remaps (non-layer):")
for hf_name, sx_name in HF_WEIGHT_MAP["mistral"].items():
    print(f"  {hf_name:40s} -> {sx_name}")
```

Layer weights follow the pattern:

```
model.layers.{i}.{submodule}.weight   ->   layers.{i}.{remapped}.weight
model.embed_tokens.weight            ->   embedding.weight
model.norm.weight                    ->   norm.weight
lm_head.weight                       ->   lm_head.weight
```

## 3. Read a safetensors file directly

```python
meta, tensors = read_safetensors("models/mistral-7b/model.safetensors")
print("metadata:", meta)
print("tensors found:", len(tensors))
sample = next(iter(tensors))
print(sample, "bytes:", len(tensors[sample]))
```

## 4. Convert to a .sneppx checkpoint

```python
n = convert_hf_to_sneppx(
    hf_dir="models/mistral-7b",
    family="mistral",
    output_path="/ckpts/mistral-7b.sneppx",
    verbose=True,
)
print(f"converted {n} weights")
```

`convert_hf_to_sneppx` writes via `CheckpointWriter` (binary format with
header + tensor records + metadata), so the result is **S7-signed-update ready**.
Missing weights are written as zero-byte placeholders and logged as
`WARNING`.

## 5. Validate + read back

```python
ok, errs = validate_checkpoint("/ckpts/mistral-7b.sneppx")
print("valid:", ok, "errors:", errs)

reader = CheckpointReader("/ckpts/mistral-7b.sneppx")
meta_out = reader.metadata()
print(meta_out["family"], meta_out["num_converted"])

# Read a specific tensor (returns numpy)
w = reader.read_tensor("layers.0.attn.q_proj.weight")
print(w.shape, w.dtype)
```

## 6. From a config dict → nn.Transformer

```python
from SneppX_ALG import build_transformer_from_config

cfg = get_model_config("llama2", "7B")
model = build_transformer_from_config(cfg)    # nn.Module (NumPy path)
print("params:", sum(p.numel for p in model.parameters()))
```

## Key takeaways

- `convert_hf_to_sneppx` is the **only** supported path for real weights.
  `from_pretrained` returns a config dict, not weights.
- `HF_WEIGHT_MAP` covers `llama2`, `llama3`, `mistral`, `qwen2`, `deepseek_v2`
  (LLaMA-3 reuses LLaMA-2 mappings).
- `read_safetensors` is a standalone TLV parser — no `safetensors` package
  required.
- `.sneppx` checkpoints are S7-signed-update ready; verify with
  `sneppx-analyze verify` before serving.
- CPU-safe (conversion is NumPy); loading weights into `Transformer.forward`
  for generation needs the C backend.

## Next steps

- Quantize the converted checkpoint — see
  [Quantization & Serving](quantization_serving.md).
- Inspect with the `CheckpointReader` in
  [Checkpointing](../cookbook/checkpointing.md).
