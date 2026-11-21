"""Tests for nn.py — neural network layers and modules."""

import numpy as np
from SneppX_ALG.interface_bindings import (
    Tensor,
    Module,
    Linear,
    Embedding,
    Dropout,
    LayerNorm,
    RMSNorm,
    GELU,
    SiLU,
    ReLU,
    Sigmoid,
    Tanh,
    Sequential,
    MultiheadAttention,
    TransformerBlock,
    Transformer,
)


def test_module_base():
    m = Module()
    assert m._training == True
    assert m._name == "Module"
    m.eval()
    assert m._training == False
    m.train()
    assert m._training == True
    print("  test_module_base PASS")


def test_linear_forward():
    linear = Linear(4, 3)
    x = Tensor.randn((2, 4))
    out = linear(x)
    assert out.shape == (2, 3), f"Expected (2,3), got {out.shape}"
    print("  test_linear_forward PASS")


def test_linear_no_bias():
    linear = Linear(4, 3, bias=False)
    assert linear.bias is None
    x = Tensor.randn((2, 4))
    out = linear(x)
    assert out.shape == (2, 3)
    print("  test_linear_no_bias PASS")


def test_linear_parameters():
    linear = Linear(4, 3)
    params = linear.parameters()
    names = [n for n, _ in linear.named_parameters()]
    assert len(params) == 2  # weight + bias
    assert "weight" in names
    assert "bias" in names
    print("  test_linear_parameters PASS")


def test_embedding():
    emb = Embedding(10, 8)
    x = Tensor([[1, 2, 3]])
    out = emb(x)
    assert out.shape == (1, 3, 8), f"Expected (1,3,8), got {out.shape}"
    print("  test_embedding PASS")


def test_dropout_train():
    dp = Dropout(0.5)
    dp.train()
    x = Tensor.ones((100,))
    out = dp(x)
    zero_frac = (out.data == 0).mean()
    assert 0.2 < zero_frac < 0.8, f"Dropout ratio unexpected: {zero_frac}"
    print("  test_dropout_train PASS")


def test_dropout_eval():
    dp = Dropout(0.5)
    dp.eval()
    x = Tensor.ones((100,))
    out = dp(x)
    assert np.allclose(out.data, 1.0)
    print("  test_dropout_eval PASS")


def test_layernorm():
    ln = LayerNorm(4)
    x = Tensor.randn((2, 4))
    out = ln(x)
    assert out.shape == (2, 4)
    mean = out.data.mean(axis=-1)
    std = out.data.std(axis=-1)
    assert np.allclose(mean, 0, atol=1e-4), f"mean={mean}"
    assert np.allclose(std, 1, atol=1e-3), f"std={std}"
    print("  test_layernorm PASS")


def test_rmsnorm():
    rn = RMSNorm(4)
    x = Tensor.randn((2, 4))
    out = rn(x)
    assert out.shape == (2, 4)
    rms = np.sqrt((out.data**2).mean(axis=-1))
    assert np.allclose(rms, 1, atol=1e-4), f"rms={rms}"
    print("  test_rmsnorm PASS")


def test_activations():
    x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert np.allclose(ReLU()(x).data, [0, 0, 0, 1, 2])
    assert np.allclose(Sigmoid()(x).data, 1 / (1 + np.exp(-x.data)))
    assert np.allclose(Tanh()(x).data, np.tanh(x.data))
    assert np.allclose(
        GELU()(x).data,
        0.5 * x.data * (1 + np.tanh(0.79788456 * (x.data + 0.044715 * x.data**3))),
    )
    assert np.allclose(SiLU()(x).data, x.data * (1 / (1 + np.exp(-x.data))))
    print("  test_activations PASS")


def test_sequential():
    seq = Sequential(
        Linear(4, 8),
        ReLU(),
        Linear(8, 2),
    )
    x = Tensor.randn((3, 4))
    out = seq(x)
    assert out.shape == (3, 2)
    assert len(seq.parameters()) == 4  # 2 weight + 2 bias
    print("  test_sequential PASS")


def test_multihead_attention():
    mha = MultiheadAttention(8, 4)
    x = Tensor.randn((2, 6, 8))
    out = mha(x)
    assert out.shape == (2, 6, 8), f"Expected (2,6,8), got {out.shape}"
    print("  test_multihead_attention PASS")


def test_transformer_block():
    block = TransformerBlock(8, 4, 32)
    x = Tensor.randn((2, 6, 8))
    out = block(x)
    assert out.shape == (2, 6, 8)
    print("  test_transformer_block PASS")


def test_transformer_output_shape():
    model = Transformer(vocab_size=100, dim=16, num_heads=4, num_layers=2, ffn_dim=64)
    tokens = Tensor([[1, 2, 3, 4, 5]])
    logits = model(tokens)
    assert logits.shape == (1, 5, 100), f"Expected (1,5,100), got {logits.shape}"
    print("  test_transformer_output_shape PASS")


def test_transformer_parameters():
    model = Transformer(vocab_size=100, dim=16, num_heads=2, num_layers=2, ffn_dim=32)
    params = model.parameters()
    assert len(params) > 0
    print(f"  test_transformer_parameters: {len(params)} params PASS")


if __name__ == "__main__":
    test_module_base()
    test_linear_forward()
    test_linear_no_bias()
    test_linear_parameters()
    test_embedding()
    test_dropout_train()
    test_dropout_eval()
    test_layernorm()
    test_rmsnorm()
    test_activations()
    test_sequential()
    test_multihead_attention()
    test_transformer_block()
    test_transformer_output_shape()
    test_transformer_parameters()
    print("\nAll nn tests passed!")
