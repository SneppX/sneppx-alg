# Cookbook — Conversion

## 1. Inspect an HF config → SNEPPX param estimate

**Intent:** See how big a model is before converting weights.

```python
from SneppX_ALG import from_pretrained, get_model_config

cfg = get_model_config("llama2", "7B")
info = from_pretrained("llama-2-7b")
print(info["param_str"], info["estimated_memory_gb"], "GB (FP32)")
# -> 7.12B 7.12 GB
```

**Notes:** `from_pretrained` maps the HF model ID to a `(family, size)` and
returns a config + param-count dict (no network access). CPU-safe.

## 2. Convert HuggingFace safetensors → SNEPPX checkpoint

**Intent:** Produce a `.sneppx` checkpoint from an HF repo dir.

```python
from SneppX_ALG import convert_hf_to_sneppx

n = convert_hf_to_sneppx(
    hf_dir="models/llama-2-7b/",          # dir with *.safetensors + config.json
    family="llama2",
    output_path="/ckpts/llama2-7b.sneppx",
    verbose=True,
)
print(f"converted {n} weight tensors")
```

**Notes:** Uses `HF_WEIGHT_MAP` to remap names (`model.layers.{i}.self_attn.q_proj.weight`
→ `layers.{i}.attn.q_proj.weight`). Missing weights are written as zero-byte
placeholders (logged as `WARNING`). :material-alert-decagram: Writes via
`CheckpointWriter` (binary format, S7-ready for signing).

## 3. Read safetensors manually

**Intent:** Inspect a single `.safetensors` file without the `safetensors`
package.

```python
from SneppX_ALG import read_safetensors, HF_WEIGHT_MAP

meta, tensors = read_safetensors("models/llama-2-7b/model.safetensors")
print(meta)                  # __metadata__ dict + per-tensor offsets
print(len(tensors), "tensors")
# tensors[name] -> raw bytes; convert with:
import numpy as np
def to_np(raw_bytes, shape, dtype=np.float32):
    return np.frombuffer(raw_bytes, dtype=dtype).reshape(shape)
```

**Notes:** `read_safetensors` is a standalone TLV reader (8-byte little-endian
header length, JSON header, then raw tensor bodies). CPU-safe, no deps.

## 4. Round-trip through ONNX

**Intent:** Export a SNEPPX `nn.Module` to ONNX for other runtimes.

```python
from SneppX_ALG import Transformer
from SneppX_ALG.interface_bindings.onnx_export import OnnxExporter

model = Transformer(vocab_size=1000, dim=128, num_heads=4, num_layers=2, ffn_dim=512, max_seq_len=64)
exp = OnnxExporter()
exp.export(model, "model.onnx", input_names=["ids"], output_names=["logits"])
ok, errs = exp.validate("model.onnx")
print(ok, errs)
```

**Notes:** The ONNX toolkit is also available standalone as
`SneppX_ALG.onnx` (numpy executor). `OnnxImporter` loads models back into
SNEPPX tensors for inspection. CPU-safe.
