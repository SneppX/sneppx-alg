import json
import os
import struct
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple, Callable, Any
from enum import IntEnum

from .tensor import Tensor, _HAS_C_BACKEND


# =========================================================================
# ModelConfig - Unified config dataclass (Phase 1)
# =========================================================================


@dataclass
class ModelConfig:
    """Unified model configuration matching C ModelConfig schema.

    Supports JSON serialization, validation, and conversion to/from C config.
    """

    # Identity
    name: str = ""
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    license: str = ""
    repository: str = ""
    homepage: str = ""

    # Architecture
    architecture: str = "transformer"
    model_type: str = "causal-lm"  # e.g., "causal-lm", "seq2seq", "vision"
    framework: str = "sneppx"

    # Model details
    num_parameters: int = 0
    num_layers: int = 0
    num_hidden_layers: int = 0
    hidden_size: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    intermediate_size: int = 0
    vocab_size: int = 0
    max_seq_len: int = 0

    # Normalization
    layer_norm_eps: float = 1e-5
    rms_norm_eps: float = 1e-6
    use_rms_norm: bool = True

    # Activations
    hidden_act: str = "silu"
    ffn_act: str = "silu"
    gated_ffn: bool = True

    # Attention
    attention_type: str = "causal"
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    use_flash_attention: bool = True
    sliding_window: int = 0

    # Position encoding
    pos_encoding: str = "rope"
    rope_theta: float = 10000.0
    rope_scaling: int = 0

    # Initialization
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False

    # Quantization
    quantize: bool = False
    quant_bits: int = 0
    quant_group_size: int = 128

    # Distributed
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    sequence_parallel: bool = False

    # Training
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    max_steps: int = 0
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True

    # MoE
    num_experts: int = 0
    num_experts_per_token: int = 0
    router_aux_loss_coef: float = 0.0

    # Custom fields (key-value pairs)
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding empty/zero values."""
        return {
            k: v
            for k, v in asdict(self).items()
            if v not in ("", 0, 0.0, False, None, {})
        }

    def to_json(self, pretty: bool = True) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2 if pretty else None)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create from dictionary, ignoring unknown fields."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, json_str: str) -> "ModelConfig":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def save(self, path: str) -> None:
        """Save to JSON file."""
        with open(path, "w") as f:
            f.write(self.to_json(pretty=True))

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        """Load from JSON file."""
        with open(path) as f:
            return cls.from_json(f.read())

    def validate(self) -> List[str]:
        """Validate config, return list of errors (empty if valid)."""
        errors = []
        if not self.name:
            errors.append("name is required")
        if self.hidden_size <= 0:
            errors.append("hidden_size must be positive")
        if self.num_layers <= 0 and self.num_hidden_layers <= 0:
            errors.append("num_layers or num_hidden_layers must be positive")
        if self.num_attention_heads <= 0:
            errors.append("num_attention_heads must be positive")
        elif self.hidden_size % self.num_attention_heads != 0:
            errors.append("hidden_size must be divisible by num_attention_heads")
        if self.vocab_size <= 0:
            errors.append("vocab_size must be positive")
        if self.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        return errors

    def to_c_config(self) -> Dict[str, Any]:
        """Convert to dict suitable for C ModelConfig."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "license": self.license,
            "architecture": self.architecture,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_attention_heads,
            "num_kv_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "max_seq_len": self.max_seq_len,
            "layer_norm_eps": self.layer_norm_eps,
            "rms_norm_eps": self.rms_norm_eps,
            "use_rms_norm": self.use_rms_norm,
            "hidden_act": self.hidden_act,
            "ffn_act": self.ffn_act,
            "gated_ffn": self.gated_ffn,
            "attention_type": self.attention_type,
            "attention_dropout": self.attention_dropout,
            "hidden_dropout": self.hidden_dropout,
            "use_flash_attention": self.use_flash_attention,
            "sliding_window": self.sliding_window,
            "pos_encoding": self.pos_encoding,
            "rope_theta": self.rope_theta,
            "rope_scaling": self.rope_scaling,
            "initializer_range": self.initializer_range,
            "tie_word_embeddings": self.tie_word_embeddings,
            "quantize": self.quantize,
            "quant_bits": self.quant_bits,
            "quant_group_size": self.quant_group_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "sequence_parallel": self.sequence_parallel,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "mixed_precision": self.mixed_precision,
            "num_experts": self.num_experts,
            "num_experts_per_token": self.num_experts_per_token,
            "router_aux_loss_coef": self.router_aux_loss_coef,
        }

    @classmethod
    def from_c_config(cls, c_config: Dict[str, Any]) -> "ModelConfig":
        """Create from C ModelConfig dict."""
        return cls(
            name=c_config.get("name", ""),
            version=c_config.get("version", "0.1.0"),
            description=c_config.get("description", ""),
            author=c_config.get("author", ""),
            license=c_config.get("license", ""),
            architecture=c_config.get("architecture", "transformer"),
            vocab_size=c_config.get("vocab_size", 32000),
            hidden_size=c_config.get("hidden_size", 4096),
            num_layers=c_config.get("num_layers", 32),
            num_attention_heads=c_config.get("num_heads", 32),
            num_key_value_heads=c_config.get(
                "num_kv_heads", c_config.get("num_heads", 32)
            ),
            intermediate_size=c_config.get("intermediate_size", 11008),
            max_position_embeddings=c_config.get("max_position_embeddings", 2048),
            max_seq_len=c_config.get("max_seq_len", 2048),
            layer_norm_eps=c_config.get("layer_norm_eps", 1e-5),
            rms_norm_eps=c_config.get("rms_norm_eps", 1e-6),
            use_rms_norm=c_config.get("use_rms_norm", True),
            hidden_act=c_config.get("hidden_act", "silu"),
            ffn_act=c_config.get("ffn_act", "silu"),
            gated_ffn=c_config.get("gated_ffn", True),
            attention_type=c_config.get("attention_type", "causal"),
            attention_dropout=c_config.get("attention_dropout", 0.0),
            hidden_dropout=c_config.get("hidden_dropout", 0.0),
            use_flash_attention=c_config.get("use_flash_attention", True),
            sliding_window=c_config.get("sliding_window", 0),
            pos_encoding=c_config.get("pos_encoding", "rope"),
            rope_theta=c_config.get("rope_theta", 10000.0),
            rope_scaling=c_config.get("rope_scaling", 0),
            initializer_range=c_config.get("initializer_range", 0.02),
            tie_word_embeddings=c_config.get("tie_word_embeddings", False),
            quantize=c_config.get("quantize", False),
            quant_bits=c_config.get("quant_bits", 0),
            quant_group_size=c_config.get("quant_group_size", 128),
            tensor_parallel_size=c_config.get("tensor_parallel_size", 1),
            pipeline_parallel_size=c_config.get("pipeline_parallel_size", 1),
            sequence_parallel=c_config.get("sequence_parallel", False),
            learning_rate=c_config.get("learning_rate", 2e-4),
            weight_decay=c_config.get("weight_decay", 0.01),
            max_grad_norm=c_config.get("max_grad_norm", 1.0),
            warmup_steps=c_config.get("warmup_steps", 0),
            max_steps=c_config.get("max_steps", 0),
            gradient_accumulation_steps=c_config.get("gradient_accumulation_steps", 1),
            mixed_precision=c_config.get("mixed_precision", True),
            num_experts=c_config.get("num_experts", 0),
            num_experts_per_token=c_config.get("num_experts_per_token", 0),
            router_aux_loss_coef=c_config.get("router_aux_loss_coef", 0.0),
        )


# Existing code continues below...
class ModelFamily(IntEnum):
    LLAMA_2 = 0
    LLAMA_3 = 1
    MISTRAL = 2
    QWEN_2 = 3
    DEEPSEEK_V2 = 4
    UNKNOWN = 255


@dataclass
class LlamaConfig:
    family: ModelFamily = ModelFamily.LLAMA_2
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    vocab_size: int = 32000
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    use_scaled_rope: bool = False
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"
    head_dim: int = 128
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    num_experts: int = 0
    num_experts_per_tok: int = 0
    router_aux_loss_coef: float = 0.0
    custom: Dict[str, Any] = field(default_factory=dict)

    # Additional fields for ModelConfig compatibility
    name: str = "llama"
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    license: str = ""
    repository: str = ""
    homepage: str = ""
    architecture: str = "transformer"
    model_type: str = "causal-lm"
    framework: str = "sneppx"
    num_parameters: int = 0
    max_seq_len: int = 4096
    layer_norm_eps: float = 1e-5
    rms_norm_eps: float = 1e-6
    use_rms_norm: bool = True
    ffn_act: str = "silu"
    gated_ffn: bool = True
    attention_type: str = "causal"
    use_flash_attention: bool = True
    sliding_window: int = 0
    pos_encoding: str = "rope"
    rope_scaling: int = 0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False
    quantize: bool = False
    quant_bits: int = 0
    quant_group_size: int = 128
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    sequence_parallel: bool = False
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    max_steps: int = 0
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    num_experts_per_token: int = 0
    router_aux_loss_coef: float = 0.0
    custom: Dict[str, Any] = field(default_factory=dict)

    # Property aliases for backward compatibility
    @property
    def num_head_dim(self) -> int:
        return self.head_dim

    @property
    def num_kv_heads(self) -> int:
        return self.num_key_value_heads

    @property
    def num_layers(self) -> int:
        return self.num_hidden_layers

    def to_model_config(self) -> "ModelConfig":
        """Convert to unified ModelConfig."""
        return ModelConfig(
            name=self.name,
            version=self.version,
            description=self.description,
            author=self.author,
            license=self.license,
            repository=self.repository,
            architecture=self.architecture,
            model_type=self.model_type,
            framework=self.framework,
            num_parameters=self.num_parameters,
            num_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            vocab_size=self.vocab_size,
            max_seq_len=self.max_position_embeddings,
            layer_norm_eps=self.rms_norm_eps,
            rms_norm_eps=self.rms_norm_eps,
            use_rms_norm=True,
            hidden_act=self.hidden_act,
            ffn_act=self.ffn_act,
            gated_ffn=True,
            attention_type=self.attention_type,
            attention_dropout=self.attention_dropout,
            hidden_dropout=self.hidden_dropout,
            use_flash_attention=True,
            sliding_window=self.sliding_window,
            pos_encoding="rope",
            rope_theta=self.rope_theta,
            rope_scaling=0,
            initializer_range=0.02,
            tie_word_embeddings=self.tie_word_embeddings,
            quantize=False,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            sequence_parallel=False,
            learning_rate=2e-4,
            weight_decay=0.01,
            max_grad_norm=1.0,
            num_experts=self.num_experts,
            num_experts_per_token=self.num_experts_per_tok,
            router_aux_loss_coef=0.0,
        )

    @classmethod
    def from_model_config(cls, config: "ModelConfig") -> "LlamaConfig":
        """Create LlamaConfig from unified ModelConfig."""
        return cls(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_seq_len,
            rms_norm_eps=config.layer_norm_eps,
            rope_theta=config.rope_theta,
            use_scaled_rope=config.rope_scaling > 0,
            tie_word_embeddings=config.tie_word_embeddings,
            hidden_act=config.hidden_act,
            attention_dropout=config.attention_dropout,
            hidden_dropout=config.hidden_dropout,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_token,
        )


@dataclass
class MistralConfig:
    family: ModelFamily = ModelFamily.MISTRAL
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    vocab_size: int = 32000
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    sliding_window: int = 4096
    head_dim: int = 128
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    # Additional fields for ModelConfig compatibility
    name: str = "mistral"
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    license: str = ""
    architecture: str = "transformer"
    num_parameters: int = 0
    max_seq_len: int = 32768
    layer_norm_eps: float = 1e-5
    rms_norm_eps: float = 1e-6
    use_rms_norm: bool = True
    ffn_act: str = "silu"
    gated_ffn: bool = True
    attention_type: str = "causal"
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    use_flash_attention: bool = True
    sliding_window: int = 4096
    pos_encoding: str = "rope"
    rope_theta: float = 10000.0
    rope_scaling: int = 0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False
    quantize: bool = False
    quant_bits: int = 0
    quant_group_size: int = 128
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    sequence_parallel: bool = False
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    max_steps: int = 0
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    num_experts: int = 0
    num_experts_per_token: int = 0
    router_aux_loss_coef: float = 0.0
    custom: Dict[str, Any] = field(default_factory=dict)

    # Property aliases for backward compatibility
    @property
    def num_head_dim(self) -> int:
        return self.head_dim

    @property
    def num_kv_heads(self) -> int:
        return self.num_key_value_heads

    @property
    def num_layers(self) -> int:
        return self.num_hidden_layers

    def to_model_config(self) -> "ModelConfig":
        """Convert to unified ModelConfig."""
        return ModelConfig(
            name=self.name,
            version=self.version,
            description="",
            author="",
            license="",
            architecture=self.architecture,
            model_type="causal-lm",
            framework="sneppx",
            num_parameters=self.num_parameters,
            num_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            vocab_size=self.vocab_size,
            max_seq_len=self.max_position_embeddings,
            layer_norm_eps=self.rms_norm_eps,
            rms_norm_eps=self.rms_norm_eps,
            use_rms_norm=True,
            hidden_act="silu",
            ffn_act="silu",
            gated_ffn=True,
            attention_type="causal",
            attention_dropout=self.attention_dropout,
            hidden_dropout=self.hidden_dropout,
            use_flash_attention=True,
            sliding_window=self.sliding_window,
            pos_encoding="rope",
            rope_theta=self.rope_theta,
            rope_scaling=0,
            initializer_range=0.02,
            tie_word_embeddings=False,
            quantize=False,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            sequence_parallel=False,
            learning_rate=2e-4,
            weight_decay=0.01,
            max_grad_norm=1.0,
            num_experts=0,
            num_experts_per_token=0,
            router_aux_loss_coef=0.0,
        )

    @classmethod
    def from_model_config(cls, config: "ModelConfig") -> "MistralConfig":
        """Create MistralConfig from unified ModelConfig."""
        return cls(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_seq_len,
            rms_norm_eps=config.layer_norm_eps,
            rope_theta=config.rope_theta,
            sliding_window=config.sliding_window,
        )


@dataclass
class Qwen2Config:
    family: ModelFamily = ModelFamily.QWEN_2
    hidden_size: int = 3584
    intermediate_size: int = 18944
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    vocab_size: int = 152064
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    rope_scaling_factor: float = 1.0
    use_rope_scaling: bool = False
    head_dim: int = 128
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    # Additional fields for ModelConfig compatibility
    name: str = "qwen2"
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    license: str = ""
    architecture: str = "transformer"
    num_parameters: int = 0
    layer_norm_eps: float = 1e-6
    rms_norm_eps: float = 1e-6
    use_rms_norm: bool = True
    ffn_act: str = "silu"
    gated_ffn: bool = True
    attention_type: str = "causal"
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    use_flash_attention: bool = True
    sliding_window: int = 0
    pos_encoding: str = "rope"
    rope_theta: float = 1000000.0
    rope_scaling: int = 0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False
    quantize: bool = False
    quant_bits: int = 0
    quant_group_size: int = 128
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    sequence_parallel: bool = False
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    max_steps: int = 0
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    num_experts: int = 0
    num_experts_per_token: int = 0
    router_aux_loss_coef: float = 0.0
    custom: Dict[str, Any] = field(default_factory=dict)

    # Property aliases for backward compatibility
    @property
    def num_head_dim(self) -> int:
        return self.head_dim

    @property
    def num_kv_heads(self) -> int:
        return self.num_key_value_heads

    @property
    def num_layers(self) -> int:
        return self.num_hidden_layers

    @property
    def max_seq_len(self) -> int:
        return self.max_position_embeddings

    def to_model_config(self) -> "ModelConfig":
        """Convert to unified ModelConfig."""
        return ModelConfig(
            name=self.name,
            version=self.version,
            description="",
            author="",
            license="",
            architecture=self.architecture,
            model_type="causal-lm",
            framework="sneppx",
            num_parameters=self.num_parameters,
            num_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            vocab_size=self.vocab_size,
            max_seq_len=self.max_position_embeddings,
            layer_norm_eps=self.rms_norm_eps,
            rms_norm_eps=self.rms_norm_eps,
            use_rms_norm=True,
            hidden_act="silu",
            ffn_act="silu",
            gated_ffn=True,
            attention_type="causal",
            attention_dropout=self.attention_dropout,
            hidden_dropout=self.hidden_dropout,
            use_flash_attention=True,
            sliding_window=self.sliding_window,
            pos_encoding="rope",
            rope_theta=self.rope_theta,
            rope_scaling=1 if self.use_rope_scaling else 0,
            initializer_range=0.02,
            tie_word_embeddings=False,
            quantize=False,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            sequence_parallel=False,
            learning_rate=2e-4,
            weight_decay=0.01,
            max_grad_norm=1.0,
            num_experts=self.num_experts,
            num_experts_per_token=self.num_experts_per_token,
            router_aux_loss_coef=0.0,
        )

    @classmethod
    def from_model_config(cls, config: "ModelConfig") -> "Qwen2Config":
        """Create Qwen2Config from unified ModelConfig."""
        return cls(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_seq_len,
            rms_norm_eps=config.layer_norm_eps,
            rope_theta=config.rope_theta,
        )


@dataclass
class DeepSeekV2Config:
    family: ModelFamily = ModelFamily.DEEPSEEK_V2
    hidden_size: int = 2048
    intermediate_size: int = 10944
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    vocab_size: int = 102400
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    head_dim: int = 128
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    # Additional fields for ModelConfig compatibility
    name: str = "deepseek_v2"
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    license: str = ""
    architecture: str = "transformer"
    num_parameters: int = 0
    layer_norm_eps: float = 1e-6
    rms_norm_eps: float = 1e-6
    use_rms_norm: bool = True
    ffn_act: str = "silu"
    gated_ffn: bool = True
    attention_type: str = "causal"
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    use_flash_attention: bool = True
    sliding_window: int = 0
    pos_encoding: str = "rope"
    rope_theta: float = 10000.0
    rope_scaling: int = 0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False
    quantize: bool = False
    quant_bits: int = 0
    quant_group_size: int = 128
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    sequence_parallel: bool = False
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    max_steps: int = 0
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    num_experts: int = 0
    num_experts_per_token: int = 0
    router_aux_loss_coef: float = 0.0
    custom: Dict[str, Any] = field(default_factory=dict)

    # Property aliases for backward compatibility
    @property
    def num_head_dim(self) -> int:
        return self.head_dim

    @property
    def num_kv_heads(self) -> int:
        return self.num_key_value_heads

    @property
    def num_layers(self) -> int:
        return self.num_hidden_layers

    @property
    def max_seq_len(self) -> int:
        return self.max_position_embeddings

    def to_model_config(self) -> "ModelConfig":
        """Convert to unified ModelConfig."""
        return ModelConfig(
            name=self.name,
            version=self.version,
            description="",
            author="",
            license="",
            architecture=self.architecture,
            model_type="causal-lm",
            framework="sneppx",
            num_parameters=self.num_parameters,
            num_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            vocab_size=self.vocab_size,
            max_seq_len=self.max_position_embeddings,
            layer_norm_eps=self.rms_norm_eps,
            rms_norm_eps=self.rms_norm_eps,
            use_rms_norm=True,
            hidden_act="silu",
            ffn_act="silu",
            gated_ffn=True,
            attention_type="causal",
            attention_dropout=self.attention_dropout,
            hidden_dropout=self.hidden_dropout,
            use_flash_attention=True,
            sliding_window=self.sliding_window,
            pos_encoding="rope",
            rope_theta=self.rope_theta,
            rope_scaling=0,
            initializer_range=0.02,
            tie_word_embeddings=False,
            quantize=False,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            sequence_parallel=False,
            learning_rate=2e-4,
            weight_decay=0.01,
            max_grad_norm=1.0,
            num_experts=self.num_experts,
            num_experts_per_token=self.num_experts_per_tok,
            router_aux_loss_coef=0.0,
        )

    @classmethod
    def from_model_config(cls, config: "ModelConfig") -> "DeepSeekV2Config":
        """Create DeepSeekV2Config from unified ModelConfig."""
        return cls(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_seq_len,
            rms_norm_eps=config.layer_norm_eps,
            rope_theta=config.rope_theta,
        )


# Registry of known model architectures
_MODEL_REGISTRY: Dict[str, Dict[str, dict]] = {
    "llama2": {
        "7B": {
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5,
        },
        "13B": {
            "hidden_size": 5120,
            "intermediate_size": 13824,
            "num_hidden_layers": 40,
            "num_attention_heads": 40,
            "num_key_value_heads": 40,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5,
        },
        "70B": {
            "hidden_size": 8192,
            "intermediate_size": 28672,
            "num_hidden_layers": 80,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5,
        },
    },
    "llama3": {
        "8B": {
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 128256,
            "max_position_embeddings": 8192,
            "use_scaled_rope": True,
            "rope_theta": 500000.0,
        },
        "70B": {
            "hidden_size": 8192,
            "intermediate_size": 28672,
            "num_hidden_layers": 80,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "vocab_size": 128256,
            "max_position_embeddings": 8192,
            "use_scaled_rope": True,
            "rope_theta": 500000.0,
        },
    },
    "mistral": {
        "7B": {
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 32000,
            "max_position_embeddings": 32768,
            "sliding_window": 4096,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5,
        },
    },
    "qwen2": {
        "7B": {
            "hidden_size": 3584,
            "intermediate_size": 18944,
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "vocab_size": 152064,
            "max_position_embeddings": 32768,
            "rope_theta": 1000000.0,
        },
        "72B": {
            "hidden_size": 8192,
            "intermediate_size": 29568,
            "num_hidden_layers": 80,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "vocab_size": 152064,
            "max_position_embeddings": 32768,
            "rope_theta": 1000000.0,
        },
    },
    "deepseek_v2": {
        "lite": {
            "hidden_size": 2048,
            "intermediate_size": 10944,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "vocab_size": 102400,
            "max_position_embeddings": 4096,
            "kv_lora_rank": 512,
            "q_lora_rank": 1536,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-6,
        },
        "full": {
            "hidden_size": 5120,
            "intermediate_size": 12288,
            "num_hidden_layers": 60,
            "num_attention_heads": 64,
            "num_key_value_heads": 64,
            "vocab_size": 102400,
            "max_position_embeddings": 4096,
            "kv_lora_rank": 512,
            "q_lora_rank": 1536,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-6,
        },
    },
}


def get_model_config(family: str, size: str) -> dict:
    """Get a model config dict by family and size name."""
    family = family.lower().replace("-", "_")
    if family == "llama":
        family = "llama2"
    if family not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model family '{family}'. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )
    if size not in _MODEL_REGISTRY[family]:
        raise ValueError(
            f"Unknown size '{size}' for {family}. "
            f"Available: {list(_MODEL_REGISTRY[family].keys())}"
        )
    cfg = dict(_MODEL_REGISTRY[family][size])
    cfg["family"] = family
    cfg["size"] = size
    return cfg


def get_model_config_obj(family: str, size: str) -> "ModelConfig":
    """Get a ModelConfig object by family and size name.

    Creates a unified ModelConfig from the registry data.
    """
    config_dict = get_model_config(family, size)
    config_dict["name"] = f"{family}-{size}"
    config_dict["version"] = "0.1.0"
    config_dict["description"] = f"{family.upper()} {size} model"

    # Map registry keys to ModelConfig fields
    if "num_hidden_layers" in config_dict:
        config_dict["num_layers"] = config_dict.pop("num_hidden_layers")
    if "max_position_embeddings" in config_dict:
        config_dict["max_seq_len"] = config_dict.pop("max_position_embeddings")
    if "num_attention_heads" in config_dict:
        config_dict["num_attention_heads"] = config_dict["num_attention_heads"]
    if "num_key_value_heads" in config_dict:
        config_dict["num_key_value_heads"] = config_dict["num_key_value_heads"]

    return ModelConfig.from_dict(config_dict)


def config_from_json(path: str) -> dict:
    """Load model config from a JSON file."""
    with open(path) as f:
        return json.load(f)


def extend_context(
    config: dict,
    target_len: int = 131072,
    scaling: str = "ntk",
) -> dict:
    """Extend a model config to support longer context via RoPE scaling.

    Args:
        config: Model config dict (from get_model_config).
        target_len: Desired maximum sequence length (default 131072 for 128K).
        scaling: Scaling method - "ntk" (NTK-aware), "linear", or "yarn".

    Returns:
        Updated config dict with extended max_position_embeddings and scaling.
    """
    config = dict(config)
    orig_len = config.get("max_position_embeddings", 4096)
    if target_len <= orig_len:
        return config

    scale = target_len / orig_len
    rope_theta = config.get("rope_theta", 10000.0)

    if scaling == "ntk":
        rope_theta *= scale
    elif scaling == "linear":
        pass  # linear scaling uses theta unchanged
    elif scaling == "yarn":
        rope_theta *= scale**0.8  # YaRN: partial NTK interpolation

    config["max_position_embeddings"] = target_len
    config["rope_theta"] = rope_theta
    config["rope_scaling"] = 2 if scaling == "ntk" else 1
    config["rope_scaling_factor"] = scale
    config["use_rope_scaling"] = 1
    return config


def list_available_models() -> List[str]:
    """List all known model family:size combinations."""
    models = []
    for family, sizes in _MODEL_REGISTRY.items():
        for size in sizes:
            models.append(f"{family}:{size}")
    return models


# =========================================================================
# HF Weight Name Mapping
# =========================================================================

HF_WEIGHT_MAP: Dict[str, Dict[str, str]] = {
    "llama2": {
        "model.embed_tokens.weight": "embedding.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "lm_head.weight",
        "self_attn.q_proj.weight": "attn.q_proj.weight",
        "self_attn.k_proj.weight": "attn.k_proj.weight",
        "self_attn.v_proj.weight": "attn.v_proj.weight",
        "self_attn.o_proj.weight": "attn.o_proj.weight",
        "mlp.gate_proj.weight": "mlp.gate_proj.weight",
        "mlp.up_proj.weight": "mlp.up_proj.weight",
        "mlp.down_proj.weight": "mlp.down_proj.weight",
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "mlp_norm.weight",
    },
    "mistral": {
        "model.embed_tokens.weight": "embedding.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "lm_head.weight",
        "self_attn.q_proj.weight": "attn.q_proj.weight",
        "self_attn.k_proj.weight": "attn.k_proj.weight",
        "self_attn.v_proj.weight": "attn.v_proj.weight",
        "self_attn.o_proj.weight": "attn.o_proj.weight",
        "mlp.gate_proj.weight": "mlp.gate_proj.weight",
        "mlp.up_proj.weight": "mlp.up_proj.weight",
        "mlp.down_proj.weight": "mlp.down_proj.weight",
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "mlp_norm.weight",
    },
    "qwen2": {
        "model.embed_tokens.weight": "embedding.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "lm_head.weight",
        "self_attn.q_proj.weight": "attn.q_proj.weight",
        "self_attn.k_proj.weight": "attn.k_proj.weight",
        "self_attn.v_proj.weight": "attn.v_proj.weight",
        "self_attn.o_proj.weight": "attn.o_proj.weight",
        "self_attn.q_proj.bias": "attn.q_proj.bias",
        "self_attn.k_proj.bias": "attn.k_proj.bias",
        "self_attn.v_proj.bias": "attn.v_proj.bias",
        "mlp.gate_proj.weight": "mlp.gate_proj.weight",
        "mlp.up_proj.weight": "mlp.up_proj.weight",
        "mlp.down_proj.weight": "mlp.down_proj.weight",
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "mlp_norm.weight",
    },
    "deepseek_v2": {
        "model.embed_tokens.weight": "embedding.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "lm_head.weight",
        "self_attn.q_proj.weight": "attn.q_proj.weight",
        "self_attn.kv_b_proj.weight": "attn.kv_b_proj.weight",
        "self_attn.o_proj.weight": "attn.o_proj.weight",
        "mlp.gate_proj.weight": "mlp.gate_proj.weight",
        "mlp.up_proj.weight": "mlp.up_proj.weight",
        "mlp.down_proj.weight": "mlp.down_proj.weight",
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "mlp_norm.weight",
    },
}

# LLaMA 3 renamed q_proj/k_proj/v_proj -> qkv_proj in some configs
HF_WEIGHT_MAP["llama3"] = dict(HF_WEIGHT_MAP["llama2"])


def _remap_hf_weight_name(hf_name: str, family: str) -> Optional[str]:
    """Map an HF weight name to SneppX internal format using the layer pattern.

    HF format: model.layers.{i}.{module}.weight
    SneppX format: layers.{i}.{module}.weight
    """
    if family not in HF_WEIGHT_MAP:
        return None

    mapping = HF_WEIGHT_MAP[family]

    # Non-layer weights
    if hf_name in mapping:
        return mapping[hf_name]

    # Layer weights: model.layers.{i}.{submodule}.{param}
    parts = hf_name.split(".")
    if len(parts) >= 4 and parts[0] == "model" and parts[1] == "layers":
        try:
            layer_idx = int(parts[2])
        except ValueError:
            return None
        suffix = ".".join(parts[3:])
        if suffix in mapping:
            remapped = f"layers.{layer_idx}.{mapping[suffix]}"
            return remapped
    return None


def _generate_sneppx_weight_names(family: str, num_layers: int) -> List[str]:
    """Generate all expected SneppX weight names for a given model."""
    names = ["embedding.weight", "norm.weight", "lm_head.weight"]
    for i in range(num_layers):
        names.append(f"layers.{i}.attn_norm.weight")
        names.append(f"layers.{i}.attn.q_proj.weight")
        names.append(f"layers.{i}.attn.k_proj.weight")
        names.append(f"layers.{i}.attn.v_proj.weight")
        names.append(f"layers.{i}.attn.o_proj.weight")
        names.append(f"layers.{i}.mlp_norm.weight")
        names.append(f"layers.{i}.mlp.gate_proj.weight")
        names.append(f"layers.{i}.mlp.up_proj.weight")
        names.append(f"layers.{i}.mlp.down_proj.weight")
    return names


# =========================================================================
# Safetensors Reader
# =========================================================================


def read_safetensors(path: str) -> Tuple[dict, Dict[str, bytes]]:
    """Read a safetensors file, returns (metadata, {name: tensor_bytes})."""
    with open(path, "rb") as f:
        header_len_bytes = f.read(8)
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        header_bytes = f.read(header_len)
        header = json.loads(header_bytes)
        tensors = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            f.seek(8 + header_len + info["data_offsets"][0])
            tensors[name] = f.read(info["data_offsets"][1] - info["data_offsets"][0])
        return header.get("__metadata__", {}), tensors


# =========================================================================
# Weight Converter
# =========================================================================


def convert_hf_to_sneppx(
    hf_dir: str,
    family: str,
    output_path: str,
    num_layers: Optional[int] = None,
    verbose: bool = True,
) -> int:
    """Convert HuggingFace safetensors weights to SneppX checkpoint format.

    Args:
        hf_dir: Directory containing .safetensors files and config.json
        family: Model family ('llama2', 'llama3', 'mistral', 'qwen2', 'deepseek_v2')
        output_path: Where to write the .sneppx checkpoint
        num_layers: Override number of layers (auto-detected if None)

    Returns:
        Number of weights converted
    """
    from .checkpoint import CheckpointWriter

    # Find all safetensors files
    import glob

    st_files = sorted(glob.glob(os.path.join(hf_dir, "*.safetensors")))
    if not st_files:
        raise FileNotFoundError(f"No .safetensors files found in {hf_dir}")

    # Auto-detect config
    config_path = os.path.join(hf_dir, "config.json")
    hf_config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            hf_config = json.load(f)

    if num_layers is None:
        num_layers = hf_config.get("num_hidden_layers", 0)
        if num_layers == 0:
            raise ValueError("num_layers not specified and not found in config.json")

    # Read all tensors
    if verbose:
        print(
            f"[SNEPPX Convert] Reading {len(st_files)} safetensors files from {hf_dir}"
        )
    all_tensors = {}
    for st_path in st_files:
        meta, tensors = read_safetensors(st_path)
        all_tensors.update(tensors)

    if verbose:
        print(f"[SNEPPX Convert] Found {len(all_tensors)} weight tensors")

    # Map to SneppX names
    converted = {}
    mapped_count = 0
    for hf_name, data in all_tensors.items():
        sneppx_name = _remap_hf_weight_name(hf_name, family)
        if sneppx_name:
            converted[sneppx_name] = data
            mapped_count += 1
            if verbose:
                print(f"  {hf_name} -> {sneppx_name} ({len(data)} bytes)")
        else:
            if verbose:
                print(f"  {hf_name} -> (skipped, no mapping)")

    # Write checkpoint
    w = CheckpointWriter(output_path)
    for name in _generate_sneppx_weight_names(family, num_layers):
        data = converted.get(name)
        if data is None:
            if verbose:
                print(f"  WARNING: missing weight '{name}', writing zeros")
            data = b"\x00" * 4  # placeholder
        w.write_tensor(data, shape=(len(data),), dtype=0)
    meta = {
        "family": family,
        "num_layers": num_layers,
        "num_converted": mapped_count,
        "model_config": json.dumps(hf_config),
    }
    w.write_metadata(meta)
    w.close()

    if verbose:
        print(f"[SNEPPX Convert] Saved {mapped_count} weights to {output_path}")
    return mapped_count


# =========================================================================
# Model Builder
# =========================================================================


def build_model_from_config(
    config: dict,
    batch_size: int = 1,
    seq_len: int = 128,
) -> "Tensor":
    """Build a model from a config dict and return example output shape info.

    In a real implementation this would construct Transformer layers, but the
    current NN module (nn.py) already has TransformerBlock and Transformer
    classes. This function serves as the config-to-architecture mapper.

    Returns the estimated parameter count.
    """
    hidden_size = config.get("hidden_size", 4096)
    intermediate_size = config.get("intermediate_size", 11008)
    num_layers = config.get("num_hidden_layers", 32)
    num_heads = config.get("num_attention_heads", 32)
    kv_heads = config.get("num_key_value_heads", num_heads)
    vocab_size = config.get("vocab_size", 32000)
    max_seq_len = config.get("max_position_embeddings", 4096)
    head_dim = hidden_size // num_heads

    # Parameter count estimation
    # Embedding
    params = vocab_size * hidden_size
    # Per layer
    attn_q = hidden_size * (num_heads * head_dim)
    attn_k = hidden_size * (kv_heads * head_dim)
    attn_v = hidden_size * (kv_heads * head_dim)
    attn_o = (num_heads * head_dim) * hidden_size
    mlp_gate = hidden_size * intermediate_size
    mlp_up = hidden_size * intermediate_size
    mlp_down = intermediate_size * hidden_size
    norms = hidden_size * 2  # attn_norm + mlp_norm
    per_layer = attn_q + attn_k + attn_v + attn_o + mlp_gate + mlp_up + mlp_down + norms
    params += num_layers * per_layer
    # Final norm + lm_head
    params += hidden_size + vocab_size * hidden_size

    result = {
        "config": config,
        "total_params": params,
        "param_str": f"{params / 1e9:.2f}B",
        "estimated_memory_gb": params * 4 / (1024**3),  # FP32
        "layers": num_layers,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "max_seq_len": max_seq_len,
        "has_gqa": kv_heads != num_heads,
    }

    # Check for DeepSeek MLA
    if "kv_lora_rank" in config:
        result["has_mla"] = True
        result["kv_lora_rank"] = config["kv_lora_rank"]
        result["q_lora_rank"] = config.get("q_lora_rank", 0)

    return result


def build_transformer_from_config(
    config: dict,
) -> "Module":
    """Build a Transformer nn.Module from a config dict."""
    from .nn import Transformer, TransformerBlock, MultiheadAttention

    hidden_size = config.get("hidden_size", 4096)
    intermediate_size = config.get("intermediate_size", 11008)
    num_layers = config.get("num_hidden_layers", 32)
    num_heads = config.get("num_attention_heads", 32)
    kv_heads = config.get("num_key_value_heads", num_heads)
    vocab_size = config.get("vocab_size", 32000)
    max_seq_len = config.get("max_position_embeddings", 4096)
    dropout = config.get("hidden_dropout", 0.0)

    blocks = []
    for i in range(num_layers):
        block = TransformerBlock(
            dim=hidden_size,
            num_heads=num_heads,
            ffn_dim=intermediate_size,
            dropout=dropout,
        )
        blocks.append(block)

    model = Transformer(
        vocab_size=vocab_size,
        dim=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_dim=intermediate_size,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )
    return model


# =========================================================================
# from_pretrained() API
# =========================================================================


def from_pretrained(
    model_id: str,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    verbose: bool = True,
) -> dict:
    """Load a pretrained model configuration.

    This is a simplified version that returns a config dict. In production,
    this would download weights from HuggingFace and build the full model.

    Args:
        model_id: HF-style model ID (e.g., 'meta-llama/Llama-2-7b-hf')
        cache_dir: Cache directory for downloaded files
        force_download: Re-download even if cached
        verbose: Print progress

    Returns:
        Model config dict with architecture info
    """
    # Map model_id to (family, size)
    model_id_lower = model_id.lower()
    mapping = [
        ("llama-2-7b", "llama2", "7B"),
        ("llama-2-13b", "llama2", "13B"),
        ("llama-2-70b", "llama2", "70B"),
        ("llama-3-8b", "llama3", "8B"),
        ("llama-3-70b", "llama3", "70B"),
        ("mistral-7b", "mistral", "7B"),
        ("qwen2-7b", "qwen2", "7B"),
        ("qwen2-72b", "qwen2", "72B"),
        ("deepseek-v2-lite", "deepseek_v2", "lite"),
        ("deepseek-v2", "deepseek_v2", "full"),
    ]
    family = None
    size = None
    for pattern, f, s in mapping:
        if pattern in model_id_lower:
            family, size = f, s
            break

    if family is None:
        raise ValueError(
            f"Unknown model_id '{model_id}'. Supported: {[p[0] for p in mapping]}"
        )

    config = get_model_config(family, size)
    if verbose:
        print(f"[SNEPPX from_pretrained] {model_id} -> family={family}, size={size}")
        print(
            f"  hidden_size={config['hidden_size']}, "
            f"layers={config['num_hidden_layers']}, "
            f"heads={config['num_attention_heads']}, "
            f"kv_heads={config.get('num_key_value_heads', 'N/A')}"
        )

    result = build_model_from_config(config)
    result["model_id"] = model_id
    result["family"] = family
    result["size"] = size
    return result


# =========================================================================
# Model Hub - High-level pretrained model loading/saving
# =========================================================================


class ModelHub:
    """High-level interface for loading and saving pretrained models.

    Example:
        >>> hub = ModelHub()
        >>> model = hub.from_pretrained("llama-2-7b")
        >>> hub.save_pretrained(model, "./my-model")
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(
            cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "sneppx")
        )
        os.makedirs(self.cache_dir, exist_ok=True)

    def from_pretrained(
        self,
        model_id: str,
        revision: str = "main",
        local_files_only: bool = False,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Load a pretrained model by ID.

        Args:
            model_id: Model identifier (e.g., "llama-2-7b", "mistral-7b")
            revision: Model revision/tag (not yet implemented)
            local_files_only: If True, only use local cached files
            verbose: Print progress messages

        Returns:
            Dictionary with keys: model_id, family, size, config, architecture, weights
        """
        return from_pretrained(model_id, verbose=True)

    def save_pretrained(
        self,
        model: Dict[str, Any],
        save_directory: str,
        safe_serialization: bool = True,
        max_shard_size: str = "5GB",
    ) -> None:
        """Save a model to disk in SneppX format.

        Args:
            model: Model dictionary from from_pretrained
            save_directory: Directory to save the model
            safe_serialization: Use safetensors format
            max_shard_size: Maximum shard size for large models
        """
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config = model.get("config", {})
        config_path = save_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Save weights (stub - would save actual weights in real implementation)
        weights = model.get("weights", {})
        if weights:
            weights_path = save_dir / "model.safetensors"
            # safetensors.torch.save_file(weights, weights_path)
            pass

        # Save model card
        model_card = model.get("model_card", "")
        if not model_card:
            model_card = self._generate_model_card(model)
        (save_dir / "README.md").write_text(model_card)

        print(f"Model saved to {save_dir}")

    def _generate_model_card(self, model: Dict[str, Any]) -> str:
        """Generate a basic model card."""
        model_id = model.get("model_id", "unknown")
        family = model.get("family", "unknown")
        size = model.get("size", "unknown")
        return f"""# {model_id}

{size} {family.upper()} model from SneppX.

## Model Details

- **Family**: {family}
- **Size**: {size}
- **Architecture**: Transformer
- **Framework**: SneppX

## Usage

```python
from sneppx_alg import ModelHub
hub = ModelHub()
model = hub.from_pretrained("{model_id.lower().replace("-", "-")}")
```
"""
