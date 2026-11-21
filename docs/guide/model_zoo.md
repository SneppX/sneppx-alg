# Model Zoo

**Introduced:** v1.0.0

## Overview

The Model Zoo provides preset configurations and weight management for 5 supported model families. Configurations are defined in `include/neural_core/kernel/model_zoo.h` and implemented in `kernel/model_zoo.c`.

## Supported architectures

| Family        | Sizes          | Key features                              |
|---------------|----------------|-------------------------------------------|
| LLaMA 2       | 7B, 13B, 70B  | RoPE, GQA (70B), SwiGLU                   |
| LLaMA 3       | 8B, 70B        | Scaled RoPE, higher theta, GQA            |
| Mistral       | 7B             | Sliding window (4096), GQA                |
| Qwen 2        | 7B, 72B        | Rope scaling (YARN/NTK), DCA              |
| DeepSeek V2   | Lite, Full     | MLA (Multi-Head Latent Attention), MoE    |

## Using the model zoo

### C API

```c
SNEPPXLLMConfig cfg;
SNEPPX_llm_config_from_name("llama3", "8B", &cfg);
// cfg now holds LLaMA 3 8B configuration
```

### Python API

```python
import sneppx

# Create config from name
cfg = sneppx.LLMConfig.from_name("deepseek_v2", "Full")
print(cfg.hidden_size)  # 5120

# Serialize to JSON
json_str = cfg.to_json()
print(json_str)

# Load from JSON
cfg2 = sneppx.LLMConfig.from_json(json_str)
```

## Configuration fields

Each architecture has a dedicated config struct. Common fields across all architectures:

| Field                  | Type   | Description                        |
|------------------------|--------|------------------------------------|
| hidden_size            | size_t | Embedding dimension                |
| intermediate_size      | size_t | FFN intermediate dimension         |
| num_hidden_layers      | size_t | Number of transformer blocks       |
| num_attention_heads    | size_t | Number of query heads              |
| num_key_value_heads    | size_t | KV heads for GQA                   |
| vocab_size             | size_t | Vocabulary size                    |
| max_position_embeddings| size_t | Base maximum sequence length       |
| rms_norm_eps           | float  | RMSNorm epsilon                    |
| rope_theta             | float  | RoPE theta base frequency          |
| head_dim               | int    | Per-head dimension                 |

## Weight management

```c
// Get weight name prefix
const char* prefix = SNEPPX_llm_weight_prefix(SNEPPX_MODEL_LLAMA_3);
// prefix == "model.layers"

// Get total number of weight tensors
int count = SNEPPX_llm_num_weight_tensors(SNEPPX_MODEL_LLAMA_3, 32);
```

## Verification

See `tests/test_model_zoo.c` for tests covering:
- All named preset sizes resolve successfully
- JSON round-trip (serialize → parse → verify fields)
- All weight name prefixes match expected values
- Context extension integration
