# Cookbook — Quantization

## 1. Symmetric INT8 (per-tensor)

**Intent:** Fast 8-bit weight-only compression.

```python
from SneppX_ALG import Tensor, quantize_int8_sym, dequantize_int8_sym, quantize_error, QuantMode

w   = Tensor.randn((512, 512))
qw, scale = quantize_int8_sym(w)            # (int8 tensor, float scale)
wq  = dequantize_int8_sym(qw, scale)
print("SNR:", quantize_error(w, wq, metric="snr"))
```

**Notes:** `QuantMode.INT8_SYM` is the constant equivalent. CPU-safe.

## 2. Per-channel INT8 (per-row)

**Intent:** Lower-error quantization for weight matrices.

```python
from SneppX_ALG import Tensor, quantize_int8_channel, dequantize_int8_channel

w   = Tensor.randn((64, 256))
qw, scales = quantize_int8_channel(w, dim=-1)     # scales: (64,)
wq = dequantize_int8_channel(qw, scales)
```

**Notes:** `dim` is the axis that gets its own scale. CPU-safe.

## 3. Asymmetric INT8 (with zero-point)

**Intent:** When activations have a non-zero mean.

```python
from SneppX_ALG import Tensor, quantize_int8_asym, dequantize_int8_asym

w = Tensor.randn((64, 256))
qw, scale, zp = quantize_int8_asym(w)             # uint8 + zp
wq = dequantize_int8_asym(qw, scale, zp)
```

## 4. INT4 packed (2 values per byte)

**Intent:** 2× compression over INT8; needs the *original* element count to decode.

```python
from SneppX_ALG import Tensor, quantize_int4_sym, dequantize_int4_sym

w   = Tensor.randn((128, 128))
flat = w.data.flatten().size                      # original element count
qw, scale = quantize_int4_sym(w)
wq = dequantize_int4_sym(qw, scale, n=flat)
```

**Notes:** Remember to pass the **unpacked** `n`. CPU-safe.

## 5. FP8 E4M3 / E5M2

**Intent:** Modern GPU-friendly 8-bit float for activation quantization.

```python
from SneppX_ALG import Tensor, quantize_fp8_e4m3, dequantize_fp8_e4m3, quantize_fp8_e5m2

x  = Tensor.randn((4, 64))
q  = quantize_fp8_e4m3(x)
xr = dequantize_fp8_e4m3(q)
q2 = quantize_fp8_e5m2(x)
```

**Notes:** Per-element Python loop in the NumPy path — vectorize for big
tensors or use the CUDA kernel (`quantize_cuda.cu`) with `SNEPPX_BUILD_CUDA=ON`.

## 6. AWQ (activation-aware weight quantization)

**Intent:** Quantize weights with minimal accuracy drop.

```python
from SneppX_ALG import Tensor, awq_scale_weights, awq_quantize

w   = Tensor.randn((4096, 4096))
act = Tensor.randn((4096,))          # per-column activation scales
ws  = awq_scale_weights(w, act, group_size=128)   # scaled weights
qw, scales = awq_quantize(w, act, group_size=128)  # INT8 per group
```

## 7. GPTQ (Hessian-based one-shot quantization)

**Intent:** Second-order error compensation in a single pass.

```python
from SneppX_ALG import Tensor, gptq_compute_hessian, gptq_quantize

acts = Tensor.randn((128, 4096))      # calibration activations
H    = gptq_compute_hessian(acts, reg=1e-5)
qw, scales, zeros = gptq_quantize(
    Tensor.randn((4096, 4096)), hessian=H, group_size=128, bits=4, sym=True
)
```

**Notes:** `gptq_quantize` mutates a copy of the weights in column order using
the Cholesky inverse of the Hessian. CPU-safe (slow NumPy).

## 8. Quantize an MLP layer in-place (QuantizedLinear)

**Intent:** Drop-in 8-bit replacement for `Linear`.

```python
from SneppX_ALG import Linear, QuantizedLinear, QuantMode
from SneppX_ALG import Tensor

lin = Linear(64, 128)
ql  = QuantizedLinear.from_float(lin, mode=QuantMode.INT8_SYM)
x   = Tensor.randn((4, 64))
y   = ql(x)                  # W8A16: dequantize then matmul
```

**Notes:** `QuantizedLinear.forward` dequantizes weights on the fly — ideal for
memory-constrained CPU serving. For GPU, enable the CUDA path.

## 9. Quantize a whole model for serving

**Intent:** Reduce model size on disk + in RAM.

```python
from SneppX_ALG import Transformer
from SneppX_ALG.interface_bindings.quantized_serve import quantize_model_weights, QuantizedModelConfig, estimate_model_size_mb
import numpy as np

model = Transformer(vocab_size=1000, dim=128, num_heads=4, num_layers=2, ffn_dim=512, max_seq_len=64)
params = {k: v.data.copy() for k, v in model.named_parameters()}

cfg = QuantizedModelConfig(quant_mode=4, skip_layers=["lm_head"])  # 4 = FP8_E4M3
q   = quantize_model_weights(params, cfg)
print("size MB:", estimate_model_size_mb(q))
```

**Notes:** `skip_layers` skips large embeddings to preserve quality. FP8 path
(`quant_mode = QuantMode.FP8_E4M3`) is fastest on Hopper CPUs with AVX-512.

## 10. MX (microscaling) formats

**Intent:** FP4/EPX formats for next-gen GPUs.

```python
from SneppX_ALG import quantize_mx, dequantize_mx, MX_FORMATS

x  = Tensor.randn((4, 64))
q  = quantize_mx(x, MX_FORMATS["mxfp4"])
xr = dequantize_mx(q)
```

## 11. Quantize via CLI (`sneppx-quantize`)

**Intent:** Command-line quantization for offline checkpoint compression.

```powershell
# Quantize model weights to INT8 symmetric
sneppx-quantize --model-dir ./checkpoints/llama2-7b --mode int8 --out ./quantized/llama2-7b-int8

# Quantize to INT4 with group size 128
sneppx-quantize --model-dir ./checkpoints/llama2-7b --mode int4 --group-size 128 --out ./quantized/llama2-7b-int4

# Quantize to FP8 E4M3 for faster CPU inference
sneppx-quantize --model-dir ./checkpoints/llama2-7b --mode fp8 --out ./quantized/llama2-7b-fp8
```

