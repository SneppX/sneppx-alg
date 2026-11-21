"""Tests for BERT/GPT/T5 Model Implementations."""

from SneppX_ALG.interface_bindings.model_implementations import (
    BertConfig, BertEmbeddings, BertSelfAttention, BertLayer,
    BertEncoder, BertPooler, BertModel, BertForMaskedLM,
    GPTConfig, GPTAttention, GPTMLP, GPTBlock, GPTModel, GPTLMHeadModel,
    T5Config, T5LayerNorm, T5DenseReluDense, T5LayerSelfAttention,
    T5Block, T5Stack, T5Model, T5ForConditionalGeneration,
    create_bert_model, create_gpt_model, create_t5_model,
    get_model_config, get_model_size_mb, get_model_flops,
)
from SneppX_ALG.interface_bindings.tensor import Tensor
import numpy as np


def test_bert_config_defaults():
    cfg = BertConfig()
    assert cfg.hidden_size == 768


def test_bert_config_custom():
    cfg = BertConfig(hidden_size=128, num_attention_heads=4, num_hidden_layers=2)
    assert cfg.hidden_size == 128
    assert cfg.num_attention_heads == 4


def test_bert_embeddings():
    emb = BertEmbeddings(vocab_size=100, hidden_size=32, max_position_embeddings=64)
    x = Tensor.from_numpy(np.random.randint(0, 100, size=(2, 8)).astype(np.int64))
    out = emb(x)
    assert out.shape == (2, 8, 32)


def test_bert_self_attention():
    attn = BertSelfAttention(hidden_size=32, num_attention_heads=4)
    x = Tensor.randn((2, 8, 32))
    out = attn(x)
    assert out.shape == (2, 8, 32)


def test_bert_layer():
    layer = BertLayer(hidden_size=32, num_attention_heads=4, intermediate_size=64)
    x = Tensor.randn((2, 8, 32))
    out = layer(x)
    assert out.shape == (2, 8, 32)


def test_bert_encoder():
    encoder = BertEncoder(hidden_size=32, num_attention_heads=4,
                          num_hidden_layers=2, intermediate_size=64)
    x = Tensor.randn((2, 8, 32))
    out = encoder(x)
    assert out.shape == (2, 8, 32)


def test_bert_pooler():
    pooler = BertPooler(hidden_size=32)
    x = Tensor.randn((2, 8, 32))
    out = pooler(x)
    assert out.shape == (2, 32)


def test_bert_model():
    config = BertConfig(vocab_size=100, hidden_size=32, num_attention_heads=4,
                        num_hidden_layers=2, intermediate_size=64, max_position_embeddings=64)
    model = BertModel(config)
    x = Tensor.from_numpy(np.random.randint(0, 100, size=(2, 8)).astype(np.int64))
    out = model(x)
    assert out is not None


def test_bert_for_masked_lm():
    config = BertConfig(vocab_size=100, hidden_size=32, num_attention_heads=4,
                        num_hidden_layers=2, intermediate_size=64, max_position_embeddings=64)
    model = BertForMaskedLM(config)
    x = Tensor.from_numpy(np.random.randint(0, 100, size=(2, 8)).astype(np.int64))
    out = model(x)
    assert out is not None


def test_gpt_config_defaults():
    cfg = GPTConfig()
    assert cfg.n_embd == 768


def test_gpt_config_custom():
    cfg = GPTConfig(vocab_size=100, n_embd=32, n_layer=2, n_head=4)
    assert cfg.n_embd == 32
    assert cfg.n_layer == 2


def test_gpt_attention():
    attn = GPTAttention(n_embd=32, n_head=4, block_size=64)
    x = Tensor.randn((2, 8, 32))
    out = attn(x)
    assert out.shape == (2, 8, 32)


def test_gpt_mlp():
    mlp = GPTMLP(n_embd=32)
    x = Tensor.randn((2, 8, 32))
    out = mlp(x)
    assert out.shape == (2, 8, 32)


def test_gpt_block():
    block = GPTBlock(n_embd=32, n_head=4, block_size=64)
    x = Tensor.randn((2, 8, 32))
    out = block(x)
    assert out.shape == (2, 8, 32)


def test_gpt_model():
    config = GPTConfig(vocab_size=100, n_embd=32, n_layer=2, n_head=4, block_size=64)
    model = GPTModel(config)
    x = Tensor.from_numpy(np.random.randint(0, 100, size=(2, 8)).astype(np.int64))
    out = model(x)
    assert out is not None


def test_gpt_lm_head_model():
    config = GPTConfig(vocab_size=100, n_embd=32, n_layer=2, n_head=4, block_size=64)
    model = GPTLMHeadModel(config)
    x = Tensor.from_numpy(np.random.randint(0, 100, size=(2, 8)).astype(np.int64))
    out = model(x)
    assert out is not None


def test_t5_config_defaults():
    cfg = T5Config()
    assert cfg.d_model == 512


def test_t5_config_custom():
    cfg = T5Config(d_model=32, d_ff=64, num_layers=2, num_heads=4)
    assert cfg.d_model == 32


def test_t5_layer_norm():
    ln = T5LayerNorm(hidden_size=32)
    x = Tensor.randn((2, 8, 32))
    out = ln(x)
    assert out.shape == (2, 8, 32)


def test_t5_dense_relu_dense():
    dense = T5DenseReluDense(d_model=32, d_ff=64)
    x = Tensor.randn((2, 8, 32))
    out = dense(x)
    assert out.shape == (2, 8, 32)


def test_t5_block():
    block = T5Block(d_model=32, d_ff=64, num_heads=4, has_relative_attention_bias=False)
    x = Tensor.randn((2, 8, 32))
    out = block(x)
    assert out is not None


def test_create_bert_model():
    config = BertConfig(vocab_size=100, hidden_size=32, num_attention_heads=4,
                        num_hidden_layers=2, intermediate_size=64)
    model = create_bert_model(config)
    assert model is not None


def test_create_gpt_model():
    config = GPTConfig(vocab_size=100, n_embd=32, n_layer=2, n_head=4, block_size=64)
    model = create_gpt_model(config)
    assert model is not None


def test_create_t5_model():
    config = T5Config(d_model=32, d_ff=64, num_layers=2, num_heads=4)
    model = create_t5_model(config)
    assert model is not None


def test_get_model_config():
    cfg = get_model_config("bert-base-uncased")
    assert cfg is not None


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
