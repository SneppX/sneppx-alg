"""Tests for Vision Transformer Implementations."""

from SneppX_ALG.interface_bindings.vision_transformers import (
    PatchEmbedding, ViTBlock, VisionTransformer, VisionTransformerMoE,
    SwinPatchEmbed, SwinPatchMerging, SwinBlock, WindowAttention,
    SwinTransformer, SwinStage, MAE,
    vit_tiny_patch16_224, vit_base_patch16_224,
    mae_base, create_vision_model,
)
from SneppX_ALG.interface_bindings.tensor import Tensor
import numpy as np


def test_patch_embedding():
    embed = PatchEmbedding(img_size=32, patch_size=8, in_channels=3, embed_dim=64)
    x = Tensor.randn((2, 3, 32, 32))
    out = embed(x)
    assert out.shape == (2, 16, 64)


def test_vit_block():
    block = ViTBlock(hidden_dim=64, num_heads=4, mlp_dim=128)
    x = Tensor.randn((2, 16, 64))
    out = block(x)
    assert out.shape == (2, 16, 64)


def test_vision_transformer():
    model = VisionTransformer(img_size=32, patch_size=8, in_channels=3,
                               embed_dim=64, num_layers=2, num_heads=4,
                               mlp_dim=128, num_classes=10)
    x = Tensor.randn((2, 3, 32, 32))
    out = model(x)
    assert out.shape == (2, 10)


def test_vit_tiny():
    model = vit_tiny_patch16_224(num_classes=10)
    x = Tensor.randn((2, 3, 224, 224))
    out = model(x)
    assert out is not None


def test_swin_patch_embed():
    embed = SwinPatchEmbed(img_size=32, patch_size=4, in_channels=3, embed_dim=32)
    x = Tensor.randn((2, 3, 32, 32))
    out = embed(x)
    assert out is not None


def test_swin_patch_merging():
    merging = SwinPatchMerging(input_dim=32, output_dim=64)
    x = Tensor.randn((2, 64, 32))
    out = merging(x)
    assert out is not None


def test_window_attention():
    attn = WindowAttention(dim=32, num_heads=4, window_size=4)
    x = Tensor.randn((2, 16, 32))
    out = attn(x)
    assert out.shape == (2, 16, 32)


def test_swin_block():
    block = SwinBlock(dim=32, num_heads=4, window_size=4)
    x = Tensor.randn((2, 64, 32))
    out = block(x)
    assert out is not None


def test_swin_stage():
    stage = SwinStage(dim=32, num_heads=4, window_size=4, depth=2, input_resolution=(8, 8))
    x = Tensor.randn((2, 64, 32))
    out = stage(x)
    assert out is not None


def test_mae_init():
    mae = MAE(img_size=32, patch_size=8, embed_dim=64, num_layers=2,
              num_heads=4, decoder_embed_dim=32, decoder_num_layers=1,
              decoder_num_heads=2)
    assert mae is not None


def test_mae_forward():
    mae = MAE(img_size=32, patch_size=8, embed_dim=64, num_layers=2,
              num_heads=4, decoder_embed_dim=32, decoder_num_layers=1,
              decoder_num_heads=2, masking_ratio=0.5)
    x = Tensor.randn((2, 3, 32, 32))
    loss, pred, mask = mae(x)
    assert loss is not None


def test_create_vision_model():
    model = create_vision_model("vit_tiny", num_classes=10)
    assert model is not None


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
