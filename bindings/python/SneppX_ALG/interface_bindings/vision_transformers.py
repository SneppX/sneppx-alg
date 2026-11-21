"""Vision Transformer (ViT) and related architectures."""

from typing import Optional, List, Tuple, Dict, Any, Union
import math
import numpy as np

from .tensor import Tensor
from .advanced_ops import (
    linear,
    gelu,
    relu,
    dropout,
    softmax,
    multi_head_attention,
    adaptive_avg_pool2d,
    flatten,
    cat,
    topk,
)
from .nn import (
    Module,
    Linear,
    LayerNorm,
    Dropout,
    Sequential,
    Conv2d,
    MultiheadAttention,
)


class PatchEmbedding(Module):
    """Image to Patch Embedding."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)  # [B, embed_dim, H/patch, W/patch]
        x = flatten(x, start_dim=2).transpose(1, 2)  # [B, num_patches, embed_dim]
        x = self.norm(x)
        return x


class ViTBlock(Module):
    """Vision Transformer Block."""

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        mlp_dim: int = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        dim: int = None,
    ):
        super().__init__()
        if dim is not None:
            hidden_dim = dim
        self.dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim if mlp_dim is not None else int(hidden_dim * mlp_ratio)
        self.norm1 = LayerNorm(hidden_dim)
        self.attn = MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout)
        self.drop_path = Dropout(dropout)
        self.norm2 = LayerNorm(hidden_dim)
        self.mlp = Sequential(
            [
                Linear(hidden_dim, self.mlp_dim),
                gelu,
                Dropout(dropout),
                Linear(self.mlp_dim, hidden_dim),
                Dropout(dropout),
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        x = self.attn(x, x, x)
        x = self.drop_path(x)
        x = x + residual

        # MLP with residual
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x


class VisionTransformer(Module):
    """Vision Transformer (ViT)."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_layers: int = None,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        mlp_dim: int = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        embed_dropout: float = 0.0,
        representation_size: Optional[int] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dim
        if num_layers is not None:
            depth = num_layers
        if mlp_dim is not None:
            mlp_ratio = mlp_dim / embed_dim
        self.embed_dim = embed_dim
        self.num_tokens = 1  # cls token

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Class token
        self.cls_token = Tensor.from_numpy(
            np.random.normal(0, 0.02, (1, 1, embed_dim)).astype(np.float32)
        )

        # Positional embedding
        self.pos_embed = Tensor.from_numpy(
            np.random.normal(0, 0.02, (1, num_patches + 1, embed_dim)).astype(
                np.float32
            )
        )

        self.pos_drop = Dropout(embed_dropout)

        # Transformer blocks
        self.blocks = Sequential(
            [
                ViTBlock(
                    hidden_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=int(embed_dim * mlp_ratio),
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(depth)
            ]
        )

        self.norm = LayerNorm(embed_dim)

        # Classification head
        if representation_size is not None:
            self.pre_logits = Linear(embed_dim, representation_size)
            self.head = Linear(representation_size, num_classes)
        else:
            self.pre_logits = None
            self.head = Linear(embed_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]

        # Add class token
        cls_token = Tensor.from_numpy(np.repeat(self.cls_token.data, B, axis=0))
        x = cat([cls_token, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Classification token
        x = x[:, 0]

        if self.pre_logits is not None:
            x = self.pre_logits(x)
        x = self.head(x)
        return x


class VisionTransformerMoE(Module):
    """Vision Transformer with Mixture of Experts."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = Tensor.from_numpy(
            np.random.normal(0, 0.02, (1, 1, embed_dim)).astype(np.float32)
        )
        self.pos_embed = Tensor.from_numpy(
            np.random.normal(0, 0.02, (1, num_patches + 1, embed_dim)).astype(
                np.float32
            )
        )

        self.blocks = Sequential(
            [
                ViTMoEBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=4.0,
                    num_experts=num_experts,
                    top_k=top_k,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.norm = LayerNorm(embed_dim)
        self.head = Linear(embed_dim, num_classes)


class ViTMoEBlock(Module):
    """ViT Block with MoE MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm2 = LayerNorm(dim)
        self.mlp = MoEMLP(dim, int(dim * mlp_ratio), num_experts, top_k, dropout)
        self.dropout = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class MoEMLP(Module):
    """Mixture of Experts MLP."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.gate = Linear(dim, num_experts)
        self.experts = [
            Sequential(
                [
                    Linear(dim, hidden_dim),
                    gelu,
                    Dropout(dropout),
                    Linear(hidden_dim, dim),
                    Dropout(dropout),
                ]
            )
            for _ in range(num_experts)
        ]

    def forward(self, x: Tensor) -> Tensor:
        B, L, D = x.shape

        # Gate weights
        gate_logits = self.gate(x)  # [B, L, num_experts]
        gate_weights = gate_logits.softmax(dim=-1)

        # Top-k routing
        topk_weights, topk_indices = topk(gate_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-12)

        # Dispatch to experts
        out = np.zeros((B, L, x.shape[-1]), dtype=np.float32)
        for i in range(self.top_k):
            expert_idx = topk_indices[:, :, i]
            weights = topk_weights[:, :, i : i + 1]

            for e in range(self.num_experts):
                mask = expert_idx == e
                if not np.any(mask):
                    continue
                expert_out = self.experts[e](x[mask])
                out[mask] += expert_out * weights[mask]

        return Tensor.from_numpy(out)


# =========================================================================
# Vision Transformer Variants
# =========================================================================


def vit_tiny_patch16_224(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        num_classes=num_classes,
        **kwargs,
    )


def vit_small_patch16_224(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_classes=num_classes,
        **kwargs,
    )


def vit_base_patch16_224(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_classes=num_classes,
        **kwargs,
    )


def vit_large_patch16_224(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        num_classes=num_classes,
        **kwargs,
    )


def vit_huge_patch14_224(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    return VisionTransformer(
        img_size=224,
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        num_classes=num_classes,
        **kwargs,
    )


# =========================================================================
# DeiT (Data-efficient Image Transformers)
# =========================================================================


class DeiTDistilled(VisionTransformer):
    """DeiT with distillation token."""

    def __init__(self, *args, teacher: Optional[Module] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = Tensor.from_numpy(
            np.random.normal(0, 0.02, (1, 1, self.embed_dim)).astype(np.float32)
        )
        self.dist_head = Linear(self.embed_dim, kwargs.get("num_classes", 1000))
        self.teacher = teacher

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_token = Tensor.from_numpy(np.repeat(self.cls_token.data, B, axis=0))
        dist_token = Tensor.from_numpy(np.repeat(self.dist_token.data, B, axis=0))
        x = cat([cls_token, dist_token, x], dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_output = self.head(x[:, 0])
        dist_output = self.dist_head(x[:, 1])

        return cls_output, dist_output


# =========================================================================
# Swin Transformer
# =========================================================================


class SwinPatchEmbed(Module):
    """Swin Patch Embedding."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 96,
    ):
        super().__init__()
        self.proj = Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        x = flatten(x, start_dim=2).transpose(1, 2)
        x = self.norm(x)
        return x


class SwinPatchMerging(Module):
    """Patch Merging Layer."""

    def __init__(self, dim: int = None, out_dim: Optional[int] = None,
                 input_dim: int = None, output_dim: int = None):
        super().__init__()
        if input_dim is not None:
            dim = input_dim
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else (
            output_dim if output_dim is not None else 2 * dim
        )
        self.reduction = Linear(4 * dim, self.out_dim, bias=False)
        self.norm = LayerNorm(4 * dim)

    def forward(self, x: Tensor, H: int = None, W: int = None) -> Tuple[Tensor, int, int]:
        B, L, C = x.shape
        if H is None:
            H = int(math.sqrt(L))
            W = H
        assert L == H * W

        x = x.reshape(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = cat([x0, x1, x2, x3], dim=-1)
        x = x.reshape(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)

        return x, H // 2, W // 2


class SwinBlock(Module):
    """Swin Transformer Block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        shift_size: int = 0,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.drop_path = Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = Sequential(
            [
                Linear(dim, int(dim * 4)),
                gelu,
                Dropout(dropout),
                Linear(int(dim * 4), dim),
                Dropout(dropout),
            ]
        )

    def forward(self, x: Tensor, H: int = None, W: int = None) -> Tensor:
        # Window attention with shifted windows
        # Simplified implementation
        if H is None:
            L = x.shape[1]
            H = int(math.sqrt(L))
            W = H
        shortcut = x
        x = self.norm1(x)
        x = self.attn.forward(x, H, W)
        x = shortcut + x

        x = x + self.mlp(self.norm2(x))
        return x


class WindowAttention(Module):
    """Window-based multi-head attention."""

    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = Linear(dim, dim * 3, bias=True)
        self.proj = Linear(dim, dim)

    def forward(self, x: Tensor, H: int = None, W: int = None) -> Tensor:
        # Window partition
        B, L, C = x.shape
        if H is None:
            H = int(math.sqrt(L))
            W = H
        assert L == H * W

        # Simplified: just use global attention
        qkv = (
            self.qkv(x)
            .reshape(B, L, 3, self.num_heads, self.dim // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        x = self.proj(x)
        return x


class SwinTransformer(Module):
    """Swin Transformer."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (2, 2, 6, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_classes: int = 1000,
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_embed = SwinPatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.pos_drop = Dropout(0.0)

        # Build layers
        self.layers = []
        self.downsamples = []
        dim = embed_dim
        for i in range(self.num_layers):
            layer = SwinStage(
                dim=dim,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=7,
                mlp_ratio=4.0,
                dropout=dropout,
            )
            self.layers.append(layer)

            if i < self.num_layers - 1:
                self.downsamples.append(SwinPatchMerging(dim))
                dim *= 2

        self.norm = LayerNorm(dim)
        self.avgpool = adaptive_avg_pool2d((1, 1))
        self.flatten = flatten(1)
        self.head = Linear(dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for i, layer in enumerate(self.layers):
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
            x = layer(x)

        x = self.norm(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.head(x)
        return x


class SwinStage(Module):
    """Swin Transformer Stage."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        input_resolution: Tuple[int, int] = None,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.blocks = Sequential(
            [
                SwinBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: Tensor, H: int = None, W: int = None) -> Tensor:
        if H is None and self.input_resolution is not None:
            H, W = self.input_resolution
        for block in self.blocks._modules.values():
            x = block(x, H, W)
        return x


# =========================================================================
# MAE (Masked Autoencoder)
# =========================================================================


class MAE(Module):
    """Masked Autoencoder for Vision."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        encoder_depth: int = 12,
        decoder_depth: int = 8,
        num_layers: int = None,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.75,
        decoder_embed_dim: int = 512,
        decoder_num_layers: int = None,
        decoder_num_heads: int = 8,
        masking_ratio: float = None,
    ):
        super().__init__()
        if num_layers is not None:
            encoder_depth = num_layers
        if decoder_num_layers is not None:
            decoder_depth = decoder_num_layers
        if masking_ratio is not None:
            self.mask_ratio = masking_ratio
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = Tensor.from_numpy(
            np.random.normal(0, 0.02, (1, 1, embed_dim)).astype(np.float32)
        )
        self.pos_embed = Tensor.from_numpy(
            np.random.normal(0, 0.02, (1, num_patches + 1, embed_dim)).astype(
                np.float32
            )
        )

        self.encoder = Sequential(
            [
                ViTBlock(
                    hidden_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=int(embed_dim * mlp_ratio),
                )
                for _ in range(encoder_depth)
            ]
        )
        self.norm = LayerNorm(embed_dim)

        # Decoder
        self.decoder_embed = Linear(embed_dim, decoder_embed_dim)
        self.mask_token = Tensor.from_numpy(
            np.random.normal(0, 0.02, (1, 1, decoder_embed_dim)).astype(np.float32)
        )
        self.decoder_pos_embed = Tensor.from_numpy(
            np.random.normal(0, 0.02, (1, num_patches + 1, decoder_embed_dim)).astype(
                np.float32
            )
        )

        self.decoder_blocks = Sequential(
            [
                ViTBlock(
                    hidden_dim=decoder_embed_dim,
                    num_heads=decoder_num_heads,
                    mlp_dim=int(decoder_embed_dim * mlp_ratio),
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = LayerNorm(decoder_embed_dim)
        self.decoder_pred = Linear(decoder_embed_dim, patch_size**2 * 3)

    def random_masking(
        self, x: Tensor, mask_ratio: float
    ) -> Tuple[Tensor, Tensor, Tensor]:
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))

        noise = np.random.rand(x.shape[0], N)
        ids_shuffle = np.argsort(noise, axis=1)
        ids_restore = np.argsort(ids_shuffle, axis=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = x[np.arange(x.shape[0])[:, None], ids_keep]

        mask = np.ones((x.shape[0], N), dtype=bool)
        mask[np.arange(x.shape[0])[:, None], ids_keep] = False

        return x_masked, Tensor.from_numpy(mask), Tensor.from_numpy(ids_restore)

    def forward(self, imgs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.patch_embed(imgs)
        x = x + self.pos_embed[:, 1:, :]

        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)

        cls_token = Tensor.from_numpy(np.repeat(self.cls_token.data, x.shape[0], axis=0))
        x = cat([cls_token, x], dim=1)
        x = x + self.pos_embed[:, : x.shape[1], :]

        x = self.encoder(x)
        x = self.norm(x)

        x = self.decoder_embed(x)
        x = x + self.decoder_pos_embed[:, : x.shape[1], :]

        # Decoder
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)

        pred = self.decoder_pred(x)
        pred = pred[:, 1:, :]  # Remove cls token

        loss = Tensor.from_numpy(
            np.array([np.mean((pred.data) ** 2)], dtype=np.float32)
        )

        return loss, pred, mask


def mae_base(**kwargs) -> MAE:
    return MAE(embed_dim=768, encoder_depth=12, decoder_depth=8, num_heads=12, **kwargs)


def mae_large(**kwargs) -> MAE:
    return MAE(
        embed_dim=1024,
        encoder_depth=24,
        decoder_depth=8,
        num_heads=16,
        decoder_embed_dim=512,
        **kwargs,
    )


def mae_huge(**kwargs) -> MAE:
    return MAE(
        embed_dim=1280,
        encoder_depth=32,
        decoder_depth=16,
        num_heads=16,
        decoder_embed_dim=512,
        **kwargs,
    )


# =========================================================================
# Model Factory
# =========================================================================


def create_vision_model(
    model_name: str, num_classes: int = 1000, pretrained: bool = False, **kwargs
) -> Module:
    """Factory function to create vision models."""
    models = {
        # ViT
        "vit_tiny": lambda: vit_tiny_patch16_224(num_classes),
        "vit_small": lambda: vit_small_patch16_224(num_classes),
        "vit_base": lambda: vit_base_patch16_224(num_classes),
        "vit_large": lambda: vit_large_patch16_224(num_classes),
        "vit_huge": lambda: vit_huge_patch14_224(num_classes),
        # DeiT
        "deit_tiny": lambda: DeiTDistilled(
            img_size=224,
            patch_size=16,
            embed_dim=192,
            depth=12,
            num_heads=3,
            num_classes=num_classes,
        ),
        "deit_small": lambda: DeiTDistilled(
            img_size=224,
            patch_size=16,
            embed_dim=384,
            depth=12,
            num_heads=6,
            num_classes=num_classes,
        ),
        "deit_base": lambda: DeiTDistilled(
            img_size=224,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            num_classes=num_classes,
        ),
        # Swin
        "swin_tiny": lambda: SwinTransformer(
            embed_dim=96,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            num_classes=num_classes,
        ),
        "swin_small": lambda: SwinTransformer(
            embed_dim=96,
            depths=(2, 2, 18, 2),
            num_heads=(3, 6, 12, 24),
            num_classes=num_classes,
        ),
        "swin_base": lambda: SwinTransformer(
            embed_dim=128,
            depths=(2, 2, 18, 2),
            num_heads=(4, 8, 16, 32),
            num_classes=num_classes,
        ),
        "swin_large": lambda: SwinTransformer(
            embed_dim=192,
            depths=(2, 2, 18, 2),
            num_heads=(6, 12, 24, 48),
            num_classes=num_classes,
        ),
        # MAE
        "mae_base": lambda: mae_base(),
        "mae_large": lambda: mae_large(),
        "mae_huge": lambda: mae_huge(),
    }

    if model_name not in models:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(models.keys())}"
        )

    model = models[model_name]()
    return model


# Export
__all__ = [
    "PatchEmbedding",
    "ViTBlock",
    "VisionTransformer",
    "ViTModel",
    "ViTForMaskedLM",
    "ViTForImageClassification",
    "ViTConfig",
    "vit_tiny_patch16_224",
    "vit_small_patch16_224",
    "vit_base_patch16_224",
    "vit_large_patch16_224",
    "vit_huge_patch14_224",
    "DeiTDistilled",
    "deit_tiny_patch16_224",
    "deit_small_patch16_224",
    "deit_base_patch16_224",
    "SwinTransformer",
    "SwinBlock",
    "SwinPatchEmbed",
    "SwinPatchMerging",
    "swin_tiny_patch4_window7_224",
    "swin_small_patch4_window7_224",
    "swin_base_patch4_window7_224",
    "swin_large_patch4_window7_224",
    "MAE",
    "mae_base",
    "mae_large",
    "mae_huge",
    "create_vision_model",
]
