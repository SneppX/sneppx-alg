# 128K Context Extension

**Introduced:** v1.0.0  
**Scope:** LLaMA 2, LLaMA 3, Mistral, Qwen 2, DeepSeek V2

## Overview

Extends a model's maximum context window to 128K tokens using NTK-aware RoPE scaling and YaRN (Yet another RoPE extensioN) interpolation. No fine-tuning is required — the extension is applied at the configuration level via the `SNEPPX_llm_config_extend_context()` API.

## How it works

1. **max_position_embeddings** is set to `target_len` (e.g. 131072).
2. **rope_theta** is scaled using the NTK-aware formula:
   ```
   theta' = theta * scaling_factor^(head_dim / (head_dim - 2))
   ```
   where `scaling_factor = target_len / base_len`.
3. **rope_scaling_factor** is set on architectures that support explicit scaling (Qwen 2).
4. Attention types may be adjusted (e.g. sliding window disabled for extended contexts).

## API

```c
int SNEPPX_llm_config_extend_context(SNEPPXLLMConfig* cfg, size_t target_len);
```

| Parameter   | Description                          |
|-------------|--------------------------------------|
| cfg         | Config to modify (in-place)          |
| target_len  | Desired max sequence length (128K)   |

Returns 0 on success, nonzero on error.

## Architecture-specific behavior

| Architecture | Rope scaling method          | Notes                              |
|-------------|------------------------------|------------------------------------|
| LLaMA 2     | NTK-aware theta scaling      | RoPE theta adjusted linearly       |
| LLaMA 3     | NTK-aware + scaled_rope      | Higher base theta (500000)         |
| Mistral     | NTK-aware theta scaling      | Sliding window adjusted            |
| Qwen 2      | NTK-aware + rope_scaling     | Uses explicit rope_scaling_factor   |
| DeepSeek V2 | NTK-aware theta scaling      | MLA-compatible scaling             |

## Python usage

```python
import sneppx

cfg = sneppx.LLMConfig.from_name("llama3", "8B")
cfg.extend_context(131072)  # 128K context
print(cfg.to_json())
```

## Verification

Test coverage is in `tests/test_context_extension.c` which validates:
- All 5 architectures successfully extend to 128K
- max_position_embeddings == target_len after extension
- rope_theta is modified (not equal to original)
- Error handling for invalid target lengths
