# Multi-Head Attention Forward Pass

**Introduced:** v1.0.0  
**Scope:** LLaMA 2, LLaMA 3, Mistral, Qwen 2, DeepSeek V2

## Overview

Implements a full multi-head attention (MHA) forward pass with multi-head projection, RoPE application, scaled dot-product attention (Flash Attention compatible), and output projection. Supports GQA (grouped-query attention), sliding window attention, and MLA (DeepSeek V2).

## Pipeline

```
x → [Q/K/V projections] → [RoPE] → [Flash/SDPA] → [O projection] → output
```

### 1. Q/K/V projections
- Weight matrices: Wq, Wk, Wv
- Bias: optional (enabled by default)
- GQA: K and V have `num_kv_heads` heads, Q has `num_heads` heads
- MLA (DeepSeek V2): absorbed low-rank KV projections

### 2. RoPE (Rotary Position Embeddings)
- Applies rotation to Q and K tensors
- Base theta: architecture-specific (10000 for LLaMA 2, 500000 for LLaMA 3)
- Scaled RoPE: LLaMA 3 variant with frequency scaling
- NTK-aware scaling: used after `extend_context()`

### 3. Attention computation
- Standard scaled dot-product: `softmax(Q·K^T / sqrt(d_k)) · V`
- Flash Attention: memory-efficient kernel when `use_flash_attn` is set
- Sliding window: attention restricted to `sliding_window` tokens

### 4. Output projection
- Wo weight matrix projects concatenated heads back to hidden_size

## Key parameters

| Parameter          | LLaMA 3 8B | Mistral 7B | Qwen 2 72B | DeepSeek V2 |
|-------------------|------------|------------|------------|-------------|
| hidden_size       | 4096       | 4096       | 8192       | 5120        |
| num_heads          | 32         | 32         | 64         | 40          |
| num_kv_heads       | 8          | 8          | 8          | 40          |
| head_dim           | 128        | 128        | 128        | 128         |
| intermediate_size  | 14336      | 14336      | 24576      | 12288       |

## Python API

```python
import sneppx

cfg = sneppx.LLMConfig.from_name("llama3", "8B")
output = cfg.forward_mha(
    hidden_states,  # [batch, seq_len, hidden_size]
    attention_mask, # [batch, 1, seq_len, seq_len]
    position_ids    # [batch, seq_len]
)
# output shape: [batch, seq_len, hidden_size]
```

## Verification

See `tests/test_mha_forward.c` for comprehensive tests validating:
- Output shape correctness for all architectures
- GQA head count ratios
- Flash Attention vs SDPA numerical equivalence
