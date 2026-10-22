# Model Zoo Browser

SNEPPX-Alg provides architecture presets for popular LLM families.

## Supported families

| Family | Sizes | Status |
|--------|-------|--------|
| LLaMA 2 | 7B, 13B, 70B | Config preset |
| LLaMA 3 | 8B, 70B | Config preset |
| Mistral | 7B | Config preset |
| Qwen 2 | 0.5B, 1.5B, 7B, 72B | Config preset |
| DeepSeek V2 | 16B, 236B | Config preset |

## Usage

```python
from SneppX_ALG.interface_bindings.model_zoo import from_pretrained

model = from_pretrained("meta-llama/Llama-3-8b")
print(model.config)
```

## from_pretrained

Resolves a model ID to a family + size, builds the architecture skeleton, and returns a config.

```python
# Illustrative
from SneppX_ALG.interface_bindings.model_zoo import from_pretrained
model = from_pretrained("mistralai/Mistral-7B-v0.1")
```

## save_pretrained

Save model weights to disk:

```python
model.save_pretrained("./my-model")
# Writes model.npz (or model.safetensors if safetensors installed)
```

## Weight loading status

End-to-end pretrained-weight download/conversion is in progress. The safetensors / PyTorch `.bin` parsers in `hf_integration.py` are real but not yet wired into the loader.