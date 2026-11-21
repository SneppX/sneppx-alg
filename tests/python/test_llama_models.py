"""Tests for LLaMA/Mistral/Qwen/DeepSeek Model Implementations."""

from SneppX_ALG.interface_bindings.llama_models import (
    LlamaRMSNorm, LlamaRotaryEmbedding, LlamaAttention, LlamaMLP,
    LlamaDecoderLayer, LlamaModel, LlamaForCausalLM,
    MistralForCausalLM, Qwen2ForCausalLM, DeepSeekV2ForCausalLM,
    get_model_config, create_model,
)
from SneppX_ALG.interface_bindings.tensor import Tensor
import numpy as np


def test_get_model_config_llama():
    cfg = get_model_config("llama-7b")
    assert cfg is not None
    assert "hidden_dim" in cfg or "dim" in str(cfg)


def test_get_model_config_mistral():
    cfg = get_model_config("mistral-7b")
    assert cfg is not None


def test_create_llama_model():
    config = {"vocab_size": 128, "hidden_dim": 64, "num_layers": 2,
              "num_heads": 4, "head_dim": 16, "max_seq_len": 64}
    model = LlamaForCausalLM(config)
    assert model is not None


def test_create_mistral_model():
    config = {"vocab_size": 128, "hidden_dim": 64, "num_layers": 2,
              "num_heads": 4, "num_kv_heads": 2, "head_dim": 16, "max_seq_len": 64}
    model = MistralForCausalLM(config)
    assert model is not None


def test_create_qwen2_model():
    config = {"vocab_size": 128, "hidden_dim": 64, "num_layers": 2,
              "num_heads": 4, "head_dim": 16, "max_seq_len": 64}
    model = Qwen2ForCausalLM(config)
    assert model is not None


def test_llama_rms_norm():
    norm = LlamaRMSNorm(dim=16)
    x = Tensor.randn((2, 8, 16))
    out = norm(x)
    assert out.shape == (2, 8, 16)


def test_llama_rotary_embedding():
    rope = LlamaRotaryEmbedding(dim=16, max_seq_len=64)
    cos, sin = rope(seq_len=8)
    assert cos is not None


def test_llama_mlp():
    mlp = LlamaMLP(hidden_dim=32, intermediate_dim=64)
    x = Tensor.randn((2, 4, 32))
    out = mlp(x)
    assert out.shape == (2, 4, 32)


def test_llama_attention():
    attn = LlamaAttention(hidden_dim=32, num_heads=4, head_dim=8)
    x = Tensor.randn((2, 4, 32))
    out = attn(x)
    assert out.shape == (2, 4, 32)


def test_llama_decoder_layer():
    layer = LlamaDecoderLayer(hidden_dim=32, num_heads=4, head_dim=8,
                              intermediate_dim=64)
    x = Tensor.randn((2, 4, 32))
    out = layer(x)
    assert out.shape == (2, 4, 32)


def test_llama_model_forward():
    config = {"vocab_size": 128, "hidden_dim": 32, "num_layers": 2,
              "num_heads": 4, "head_dim": 8, "max_seq_len": 32,
              "intermediate_dim": 64}
    model = LlamaModel(config)
    x = Tensor.from_numpy(np.random.randint(0, 128, size=(2, 8)).astype(np.int64))
    out = model(x)
    assert out is not None


def test_llama_for_causal_lm():
    config = {"vocab_size": 128, "hidden_dim": 32, "num_layers": 2,
              "num_heads": 4, "head_dim": 8, "max_seq_len": 32,
              "intermediate_dim": 64}
    model = LlamaForCausalLM(config)
    x = Tensor.from_numpy(np.random.randint(0, 128, size=(2, 8)).astype(np.int64))
    out = model(x)
    assert out is not None


def test_mistral_for_causal_lm():
    config = {"vocab_size": 128, "hidden_dim": 32, "num_layers": 2,
              "num_heads": 4, "num_kv_heads": 2, "head_dim": 8,
              "max_seq_len": 32, "intermediate_dim": 64}
    model = MistralForCausalLM(config)
    x = Tensor.from_numpy(np.random.randint(0, 128, size=(2, 8)).astype(np.int64))
    out = model(x)
    assert out is not None


if __name__ == "__main__":
    import sys
    locals_ = {k: v for k, v in locals().items() if k.startswith("test_")}
    passed = 0
    failed = 0
    for name, fn in sorted(locals_.items()):
        try:
            fn()
            print(f"  PASS {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            failed += 1
    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed")
    sys.exit(failed)
