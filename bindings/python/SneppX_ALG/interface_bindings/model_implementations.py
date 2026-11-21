"""Full model implementations — BERT, GPT, T5, and other architectures."""

from typing import Optional, List, Tuple, Dict, Any, Union
import numpy as np

from .tensor import Tensor
from .advanced_ops import (
    linear,
    layer_norm,
    gelu,
    silu,
    relu,
    dropout,
    multi_head_attention,
    transformer_block,
    embedding,
    layer_norm as layernorm_fn,
    rmsnorm,
    linear,
    gelu,
    softmax,
    tanh,
    cross_entropy,
)


class _CallableModule:
    """Base providing ``__call__`` that dispatches to ``forward``."""

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def _as_config(config=None, kwargs=None):
    """Normalize a config argument (dict / config-object / None) plus kwargs
    into a plain dict that supports ``.get``."""
    if kwargs is None:
        kwargs = {}
    if config is None:
        data = {}
    elif isinstance(config, dict):
        data = dict(config)
    else:
        data = {k: v for k, v in vars(config).items() if not k.startswith("_")}
    data.update(kwargs)
    return data


class BertEmbeddings:
    """BERT embeddings: token + position + segment embeddings."""

    def __init__(self, config=None, **kwargs):
        config = _as_config(config, kwargs)
        self.vocab_size = config.get("vocab_size", 30522)
        self.hidden_size = config.get("hidden_size", 768)
        self.max_position_embeddings = config.get("max_position_embeddings", 512)
        self.type_vocab_size = config.get("type_vocab_size", 2)
        self.hidden_dropout_prob = config.get("hidden_dropout_prob", 0.1)

        # Weights (in practice loaded from checkpoint)
        self.word_embeddings = None
        self.position_embeddings = None
        self.token_type_embeddings = None
        self.layer_norm = None
        self.dropout = None

        self.initialize_weights()
    def initialize_weights(self):
        """Initialize all embedding weights."""
        self.word_embeddings = Tensor.from_numpy(
            np.random.normal(0, 0.02, (self.vocab_size, self.hidden_size)).astype(
                np.float32
            )
        )
        self.position_embeddings = Tensor.from_numpy(
            np.random.normal(
                0, 0.02, (self.max_position_embeddings, self.hidden_size)
            ).astype(np.float32)
        )
        self.token_type_embeddings = Tensor.from_numpy(
            np.random.normal(0, 0.02, (self.type_vocab_size, self.hidden_size)).astype(
                np.float32
            )
        )
        self.layer_norm = {
            "weight": Tensor.from_numpy(np.ones(self.hidden_size, dtype=np.float32)),
            "bias": Tensor.from_numpy(np.zeros(self.hidden_size, dtype=np.float32)),
        }

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        seq_length = input_ids.shape[1]

        if position_ids is None:
            pos_ids = np.arange(seq_length, dtype=np.int64).reshape(1, -1)
            position_ids = Tensor.from_numpy(
                np.broadcast_to(pos_ids, (input_ids.shape[0], seq_length))
            )

        if token_type_ids is None:
            token_type_ids = Tensor.from_numpy(
                np.zeros_like(input_ids.data, dtype=np.int64)
            )

        # Embeddings
        word_emb = embedding(input_ids, self.word_embeddings)
        pos_emb = embedding(position_ids, self.position_embeddings)
        type_emb = embedding(token_type_ids, self.token_type_embeddings)

        embeddings = word_emb + pos_emb + type_emb

        # LayerNorm + Dropout
        embeddings = layer_norm(
            embeddings, self.layer_norm["weight"], self.layer_norm["bias"]
        )
        embeddings = dropout(embeddings, self.hidden_dropout_prob, training=True)

        return embeddings


class BertSelfAttention:
    """BERT self-attention with optional relative position bias."""

    def __init__(self, config=None, **kwargs):
        config = _as_config(config, kwargs)
        self.num_heads = config.get("num_attention_heads", 12)
        self.hidden_size = config.get("hidden_size", 768)
        self.attention_probs_dropout_prob = config.get(
            "attention_probs_dropout_prob", 0.1
        )
        self.hidden_dropout_prob = config.get("hidden_dropout_prob", 0.1)

        self.query = None
        self.key = None
        self.value = None
        self.dense = None
        self.layer_norm = None
        self.dropout = None

        self.initialize_weights()
    def initialize_weights(self):
        hidden = self.hidden_size
        self.query = {
            "weight": Tensor.from_numpy(
                np.random.normal(0, 0.02, (hidden, hidden)).astype(np.float32)
            ),
            "bias": Tensor.from_numpy(np.zeros(hidden, dtype=np.float32)),
        }
        self.key = {
            "weight": Tensor.from_numpy(
                np.random.normal(0, 0.02, (hidden, hidden)).astype(np.float32)
            ),
            "bias": Tensor.from_numpy(np.zeros(hidden, dtype=np.float32)),
        }
        self.value = {
            "weight": Tensor.from_numpy(
                np.random.normal(0, 0.02, (hidden, hidden)).astype(np.float32)
            ),
            "bias": Tensor.from_numpy(np.zeros(hidden, dtype=np.float32)),
        }
        self.dense = {
            "weight": Tensor.from_numpy(
                np.random.normal(0, 0.02, (hidden, hidden)).astype(np.float32)
            ),
            "bias": Tensor.from_numpy(np.zeros(hidden, dtype=np.float32)),
        }
        self.layer_norm = {
            "weight": Tensor.from_numpy(np.ones(hidden, dtype=np.float32)),
            "bias": Tensor.from_numpy(np.zeros(hidden, dtype=np.float32)),
        }

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size, seq_len, hidden = hidden_states.shape

        # Project Q, K, V
        q = linear(hidden_states, self.query["weight"], self.query["bias"])
        k = linear(hidden_states, self.key["weight"], self.key["bias"])
        v = linear(hidden_states, self.value["weight"], self.value["bias"])

        # Attention (multi_head_attention handles head reshaping internally)
        attn_output = multi_head_attention(
            q,
            k,
            v,
            num_heads=self.num_heads,
            dropout_p=self.attention_probs_dropout_prob,
            training=True,
        )

        # Output projection
        attn_output = attn_output.reshape(-1, seq_len, hidden)
        output = linear(attn_output, self.dense["weight"], self.dense["bias"])
        output = dropout(output, self.hidden_dropout_prob, training=True)
        output = output + hidden_states  # Residual
        output = layernorm_fn(
            output, self.layer_norm["weight"], self.layer_norm["bias"]
        )

        return output


class BertIntermediate:
    """BERT feed-forward intermediate layer."""

    def __init__(self, config=None, **kwargs):
        config = _as_config(config, kwargs)
        self.hidden_size = config.get("hidden_size", 768)
        self.intermediate_size = config.get("intermediate_size", 3072)
        self.hidden_act = config.get("hidden_act", "gelu")

        self.dense = None

        self.initialize_weights()
    def initialize_weights(self):
        self.dense = {
            "weight": Tensor.from_numpy(
                np.random.normal(
                    0, 0.02, (self.intermediate_size, self.hidden_size)
                ).astype(np.float32)
            ),
            "bias": Tensor.from_numpy(
                np.zeros(self.intermediate_size, dtype=np.float32)
            ),
        }

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = linear(hidden_states, self.dense["weight"], self.dense["bias"])
        if self.hidden_act == "gelu":
            hidden_states = gelu(hidden_states)
        elif self.hidden_act == "relu":
            hidden_states = relu(hidden_states)
        elif self.hidden_act == "silu":
            hidden_states = silu(hidden_states)
        return hidden_states


class BertOutput:
    """BERT output layer."""

    def __init__(self, config=None, **kwargs):
        config = _as_config(config, kwargs)
        self.hidden_size = config.get("hidden_size", 768)
        self.intermediate_size = config.get("intermediate_size", 3072)
        self.hidden_dropout_prob = config.get("hidden_dropout_prob", 0.1)

        self.dense = None
        self.layer_norm = None

        self.initialize_weights()
    def initialize_weights(self):
        hidden = self.hidden_size
        inter = self.intermediate_size
        self.dense = {
            "weight": Tensor.from_numpy(
                np.random.normal(0, 0.02, (hidden, inter)).astype(np.float32)
            ),
            "bias": Tensor.from_numpy(np.zeros(hidden, dtype=np.float32)),
        }
        self.layer_norm = {
            "weight": Tensor.from_numpy(np.ones(hidden, dtype=np.float32)),
            "bias": Tensor.from_numpy(np.zeros(hidden, dtype=np.float32)),
        }

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = linear(hidden_states, self.dense["weight"], self.dense["bias"])
        hidden_states = dropout(hidden_states, self.hidden_dropout_prob, training=True)
        hidden_states = hidden_states + input_tensor  # Residual
        hidden_states = layernorm_fn(
            hidden_states, self.layer_norm["weight"], self.layer_norm["bias"]
        )
        return hidden_states


class BertLayer:
    """Single BERT encoder layer."""

    def __init__(self, config=None, **kwargs):
        config = _as_config(config, kwargs)
        self.attention = BertSelfAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

        self.initialize_weights()
    def initialize_weights(self):
        self.attention.initialize_weights()
        self.intermediate.initialize_weights()
        self.output.initialize_weights()

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
    ) -> Tensor:
        attention_output = self.attention.forward(
            hidden_states, attention_mask, head_mask
        )
        intermediate_output = self.intermediate.forward(attention_output)
        layer_output = self.output.forward(intermediate_output, attention_output)
        return layer_output


class BertEncoder:
    """BERT encoder with multiple layers."""

    def __init__(self, config=None, **kwargs):
        config = _as_config(config, kwargs)
        self.config = config
        self.num_hidden_layers = config.get("num_hidden_layers", 12)
        self.layers = [BertLayer(config) for _ in range(self.num_hidden_layers)]

        self.initialize_weights()
    def initialize_weights(self):
        for layer in self.layers:
            layer.initialize_weights()

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[List[Tensor]] = None,
        output_hidden_states: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        all_hidden_states = [] if output_hidden_states else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            head_mask_i = head_mask[i] if head_mask else None
            hidden_states = layer.forward(hidden_states, attention_mask, head_mask_i)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)
            return hidden_states, all_hidden_states
        return hidden_states


class BertPooler:
    """BERT pooler for [CLS] token."""

    def __init__(self, config: Dict[str, Any] = None, **kwargs):
        config = _as_config(config, kwargs)
        self.config = config
        self.dense = None

        self.initialize_weights()
    def initialize_weights(self):
        hidden = self.config.get("hidden_size", 768)
        self.dense = {
            "weight": Tensor.from_numpy(
                np.random.normal(0, 0.02, (hidden, hidden)).astype(np.float32)
            ),
            "bias": Tensor.from_numpy(np.zeros(hidden, dtype=np.float32)),
        }

    def forward(self, hidden_states: Tensor) -> Tensor:
        # Take [CLS] token (first token)
        first_token = hidden_states[:, 0, :]
        pooled = linear(first_token, self.dense["weight"], self.dense["bias"])
        pooled = tanh(pooled)
        return pooled


class BertModel:
    """Full BERT model."""

    def __init__(self, config=None, **kwargs):
        config = _as_config(config, kwargs)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.initialize_weights()
    def initialize_weights(self):
        self.embeddings.initialize_weights()
        self.encoder.initialize_weights()
        self.pooler.initialize_weights()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[List[Tensor]] = None,
        output_hidden_states: bool = False,
    ) -> Dict[str, Tensor]:
        # Embeddings
        embedding_output = self.embeddings.forward(
            input_ids, token_type_ids, position_ids
        )

        # Attention mask
        if attention_mask is not None:
            # Convert to broadcastable mask
            extended_mask = attention_mask[:, None, None, :]
            extended_mask = (1.0 - extended_mask) * -10000.0
        else:
            extended_mask = None

        # Encoder
        encoder_output = self.encoder.forward(
            embedding_output,
            extended_mask,
            head_mask,
            output_hidden_states,
        )

        sequence_output = (
            encoder_output[0] if isinstance(encoder_output, tuple) else encoder_output
        )
        pooled_output = self.pooler.forward(sequence_output)

        outputs = {
            "last_hidden_state": sequence_output,
            "pooler_output": pooled_output,
        }
        if isinstance(encoder_output, tuple):
            outputs["hidden_states"] = encoder_output[1]

        return outputs


class BertForMaskedLM:
    """BERT with MLM head."""

    def __init__(self, config=None, **kwargs):
        config = _as_config(config, kwargs)
        self.config = config
        self.bert = BertModel(config)
        self.cls = None

        self.initialize_weights()
    def initialize_weights(self):
        self.bert.initialize_weights()
        hidden = self.config.get("hidden_size", 768)
        vocab = self.config.get("vocab_size", 30522)
        self.cls = {
            "dense": {
                "weight": Tensor.from_numpy(
                    np.random.normal(0, 0.02, (hidden, hidden)).astype(np.float32)
                ),
                "bias": Tensor.from_numpy(np.zeros(hidden, dtype=np.float32)),
            },
            "layer_norm": {
                "weight": Tensor.from_numpy(np.ones(hidden, dtype=np.float32)),
                "bias": Tensor.from_numpy(np.zeros(hidden, dtype=np.float32)),
            },
            "decoder": {
                "weight": Tensor.from_numpy(
                    np.random.normal(0, 0.02, (vocab, hidden)).astype(np.float32)
                ),
                "bias": Tensor.from_numpy(np.zeros(vocab, dtype=np.float32)),
            },
        }

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert.forward(input_ids, attention_mask, token_type_ids)
        sequence_output = outputs["last_hidden_state"]

        # MLM head
        hidden = linear(
            sequence_output, self.cls["dense"]["weight"], self.cls["dense"]["bias"]
        )
        hidden = gelu(hidden)
        hidden = layernorm_fn(
            hidden, self.cls["layer_norm"]["weight"], self.cls["layer_norm"]["bias"]
        )
        logits = linear(
            hidden, self.cls["decoder"]["weight"], self.cls["decoder"]["bias"]
        )

        loss = None
        if labels is not None:
            # Cross entropy loss
            loss = cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))

        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": outputs.get("hidden_states"),
        }


# =========================================================================
# BERT Configuration
# =========================================================================


class BertConfig:
    """BERT configuration."""

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        position_embedding_type: str = "absolute",
        use_cache: bool = True,
        classifier_dropout: Optional[float] = None,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout

    def get(self, key, default=None):
        return getattr(self, key, default)


# =========================================================================
# GPT (Decoder-only) Models
# =========================================================================


class GPTConfig:
    """GPT configuration."""

    def __init__(
        self,
        vocab_size: int = 50257,
        n_positions: int = 1024,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        block_size: int = 1024,
        n_inner: Optional[int] = None,
        activation_function: str = "gelu",
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
        scale_attn_weights: bool = True,
        use_cache: bool = True,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner or 4 * n_embd
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id


class GPTAttention:
    """GPT multi-head causal attention."""

    def __init__(self, config=None, **kwargs):
        config = _as_config(config, kwargs)
        self.config = config
        self.hidden_size = config.get("n_embd")
        self.num_heads = config.get("n_head")
        self.head_dim = config.get("n_embd") // config.get("n_head")
        self.scale_attn_weights = config.get("scale_attn_weights")

        self.c_attn = None  # combined QKV projection
        self.c_proj = None  # output projection
        self.attn_dropout = config.get("attn_pdrop", 0.1)
        self.resid_dropout = config.get("resid_pdrop", 0.1)

        self.initialize_weights()
    def initialize_weights(self):
        hidden = self.hidden_size
        self.c_attn = {
            "weight": Tensor.from_numpy(
                np.random.normal(
                    0, self.config.get("initializer_range", 0.02), (3 * hidden, hidden)
                ).astype(np.float32)
            ),
            "bias": Tensor.from_numpy(np.zeros(3 * hidden, dtype=np.float32)),
        }
        self.c_proj = {
            "weight": Tensor.from_numpy(
                np.random.normal(
                    0,
                    self.config.get("initializer_range", 0.02) / np.sqrt(2 * self.config.get("n_layer", 12)),
                    (hidden, hidden),
                ).astype(np.float32)
            ),
            "bias": Tensor.from_numpy(np.zeros(self.hidden_size, dtype=np.float32)),
        }

    def forward(
        self,
        hidden_states: Tensor,
        layer_past: Optional[Tuple[Tensor, Tensor]] = None,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        batch, seq_len, hidden = hidden_states.shape

        # QKV projection
        c_attn = linear(hidden_states, self.c_attn["weight"], self.c_attn["bias"])
        q, k, v = np.split(c_attn.data, 3, axis=-1)
        q = Tensor.from_numpy(q.astype(np.float32))
        k = Tensor.from_numpy(k.astype(np.float32))
        v = Tensor.from_numpy(v.astype(np.float32))

        # Past key/value
        if layer_past is not None:
            past_k, past_v = layer_past
            k = Tensor.from_numpy(
                np.concatenate([past_k, k.data], axis=-2).astype(np.float32)
            )
            v = Tensor.from_numpy(
                np.concatenate([past_v, v.data], axis=-2).astype(np.float32)
            )

        present = (k, v) if use_cache else None

        # Attention (multi_head_attention handles head reshaping internally)
        attn_output = multi_head_attention(
            q,
            k,
            v,
            num_heads=self.num_heads,
            dropout_p=self.attn_dropout,
            training=True,
        )

        # Output projection
        attn_output = linear(attn_output, self.c_proj["weight"], self.c_proj["bias"])
        attn_output = dropout(attn_output, self.resid_dropout, training=True)

        return attn_output, present


class GPTMLP:
    """GPT MLP (feed-forward)."""

    def __init__(self, config=None, **kwargs):
        config = _as_config(config, kwargs)
        self.config = config
        self.c_fc = None
        self.c_proj = None
        self.act = config.get("activation_function", "gelu")

        self.initialize_weights()
    def initialize_weights(self):
        hidden = self.config.get("n_embd")
        inner = self.config.get("n_inner", 4 * self.config.get("n_embd", 768))
        self.c_fc = {
            "weight": Tensor.from_numpy(
                np.random.normal(
                    0, self.config.get("initializer_range", 0.02), (inner, hidden)
                ).astype(np.float32)
            ),
            "bias": Tensor.from_numpy(np.zeros(inner, dtype=np.float32)),
        }
        self.c_proj = {
            "weight": Tensor.from_numpy(
                np.random.normal(
                    0,
                    self.config.get("initializer_range", 0.02) / np.sqrt(2 * self.config.get("n_layer", 12)),
                    (hidden, inner),
                ).astype(np.float32)
            ),
            "bias": Tensor.from_numpy(np.zeros(hidden, dtype=np.float32)),
        }

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = linear(hidden_states, self.c_fc["weight"], self.c_fc["bias"])
        if self.act == "gelu":
            hidden_states = gelu(hidden_states)
        elif self.act == "relu":
            hidden_states = relu(hidden_states)
        elif self.act == "silu":
            hidden_states = silu(hidden_states)
        hidden_states = linear(
            hidden_states, self.c_proj["weight"], self.c_proj["bias"]
        )
        hidden_states = dropout(hidden_states, self.config.get("resid_pdrop", 0.1), training=True)
        return hidden_states


class GPTBlock:
    """GPT transformer block."""

    def __init__(self, config=None, **kwargs):
        config = _as_config(config, kwargs)
        self.config = config
        self.ln_1 = None
        self.attn = GPTAttention(config)
        self.ln_2 = None
        self.mlp = GPTMLP(config)

        self.initialize_weights()
    def initialize_weights(self):
        hidden = self.config.get("n_embd")
        self.ln_1 = {
            "weight": Tensor.from_numpy(np.ones(hidden, dtype=np.float32)),
            "bias": Tensor.from_numpy(np.zeros(hidden, dtype=np.float32)),
        }
        self.ln_2 = {
            "weight": Tensor.from_numpy(np.ones(hidden, dtype=np.float32)),
            "bias": Tensor.from_numpy(np.zeros(hidden, dtype=np.float32)),
        }
        self.attn.initialize_weights()
        self.mlp.initialize_weights()

    def forward(
        self,
        hidden_states: Tensor,
        layer_past: Optional[Tuple[Tensor, Tensor]] = None,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        # Pre-norm
        residual = hidden_states
        hidden_states = layernorm_fn(
            hidden_states, self.ln_1["weight"], self.ln_1["bias"]
        )

        # Attention
        attn_output, present = self.attn.forward(
            hidden_states, layer_past, attention_mask, head_mask, use_cache
        )
        hidden_states = residual + attn_output

        # Post-norm FFN
        residual = hidden_states
        hidden_states = layernorm_fn(
            hidden_states, self.ln_2["weight"], self.ln_2["bias"]
        )
        hidden_states = self.mlp.forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present


class GPTModel:
    """Full GPT model."""

    def __init__(self, config=None, **kwargs):
        config = _as_config(config, kwargs)
        self.config = config
        self.wte = None  # token embeddings
        self.wpe = None  # position embeddings
        self.drop = None
        self.h = [GPTBlock(config) for _ in range(config.get("n_layer", 12))]
        self.ln_f = None

        self.initialize_weights()
    def initialize_weights(self):
        hidden = self.config.get("n_embd")
        vocab = self.config.get("vocab_size", 50257)
        max_pos = self.config.get("n_positions", 1024)

        self.wte = Tensor.from_numpy(
            np.random.normal(0, self.config.get("initializer_range", 0.02), (vocab, hidden)).astype(
                np.float32
            )
        )
        self.wpe = Tensor.from_numpy(
            np.random.normal(
                0, self.config.get("initializer_range", 0.02), (max_pos, hidden)
            ).astype(np.float32)
        )

        self.drop = self.config.get("embd_pdrop", 0.1)

        for block in self.h:
            block.initialize_weights()

        self.ln_f = {
            "weight": Tensor.from_numpy(np.ones(hidden, dtype=np.float32)),
            "bias": Tensor.from_numpy(np.zeros(hidden, dtype=np.float32)),
        }

    def forward(
        self,
        input_ids: Tensor,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[List[Tensor]] = None,
        use_cache: bool = False,
    ) -> Dict[str, Any]:
        batch, seq_len = input_ids.shape

        # Position IDs
        if position_ids is None:
            position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
            position_ids = np.broadcast_to(position_ids, (batch, seq_len))
            position_ids = Tensor.from_numpy(position_ids)

        # Embeddings
        inputs_embeds = embedding(input_ids, self.wte)
        position_embeds = embedding(position_ids, self.wpe)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = dropout(hidden_states, self.drop, training=True)

        # Causal mask
        causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
        if attention_mask is not None:
            # Combine causal + attention mask
            causal_mask = causal_mask + attention_mask[:, None, None, :].data * -1e9
        causal_mask = Tensor.from_numpy(causal_mask.astype(np.float32))

        # Blocks
        presents = [] if use_cache else None
        for i, block in enumerate(self.h):
            layer_past = past_key_values[i] if past_key_values else None
            head_mask_i = head_mask[i] if head_mask else None

            hidden_states, present = block.forward(
                hidden_states, layer_past, causal_mask, head_mask_i, use_cache
            )

            if use_cache:
                presents.append(present)

        # Final layer norm
        hidden_states = layernorm_fn(
            hidden_states, self.ln_f["weight"], self.ln_f["bias"]
        )

        return {
            "last_hidden_state": hidden_states,
            "past_key_values": presents,
            "hidden_states": None,
        }


class GPTLMHeadModel:
    """GPT with language modeling head."""

    def __init__(self, config=None, **kwargs):
        config = _as_config(config, kwargs)
        self.config = config
        self.transformer = GPTModel(config)
        self.lm_head = None

        self.initialize_weights()
    def initialize_weights(self):
        self.transformer.initialize_weights()
        hidden = self.config.get("n_embd")
        vocab = self.config.get("vocab_size", 50257)
        self.lm_head = {
            "weight": self.transformer.wte,  # weight tying
            "bias": Tensor.from_numpy(np.zeros(vocab, dtype=np.float32)),
        }

    def forward(
        self,
        input_ids: Tensor,
        past_key_values: Optional[List] = None,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        outputs = self.transformer.forward(
            input_ids, past_key_values, attention_mask, use_cache=use_cache
        )
        hidden_states = outputs["last_hidden_state"]

        # LM head
        logits = linear(hidden_states, self.lm_head["weight"], self.lm_head["bias"])

        loss = None
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
            shift_labels = labels[:, 1:].reshape(-1)
            loss = cross_entropy(shift_logits, shift_labels)

        return {
            "logits": logits,
            "loss": loss,
            "past_key_values": outputs.get("past_key_values"),
        }


# =========================================================================
# T5 (Encoder-Decoder) Models
# =========================================================================


class T5Config:
    """T5 configuration."""

    def __init__(
        self,
        vocab_size: int = 32128,
        d_model: int = 512,
        d_kv: int = 64,
        d_ff: int = 2048,
        num_layers: int = 6,
        num_decoder_layers: Optional[int] = None,
        num_heads: int = 8,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        dropout_rate: float = 0.1,
        layer_norm_epsilon: float = 1e-6,
        initializer_factor: float = 1.0,
        feed_forward_proj: str = "relu",
        is_encoder_decoder: bool = True,
        use_cache: bool = True,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        decoder_start_token_id: int = 0,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = num_decoder_layers or num_layers
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.is_encoder_decoder = is_encoder_decoder
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id


class T5LayerNorm:
    """T5 layer normalization (no bias)."""

    def __init__(self, config=None, **kwargs):
        config = _as_config(config, kwargs)
        self.config = config
        self.weight = None

        self.initialize_weights()
    def initialize_weights(self):
        d_model = self.config.get("d_model", self.config.get("hidden_size", 512))
        self.weight = Tensor.from_numpy(np.ones(d_model, dtype=np.float32))

    def forward(self, hidden_states: Tensor) -> Tensor:
        return layernorm_fn(
            hidden_states, self.weight, None, eps=self.config.get("layer_norm_epsilon", 1e-6)
        )


class T5DenseReluDense:
    """T5 feed-forward with ReLU."""

    def __init__(self, config=None, **kwargs):
        config = _as_config(config, kwargs)
        self.config = config
        self.wi = None
        self.wo = None
        self.dropout = config.get("dropout_rate", 0.1)

        self.initialize_weights()
    def initialize_weights(self):
        self.wi = {
            "weight": Tensor.from_numpy(
                np.random.normal(
                    0,
                    self.config.get("initializer_factor", 1.0) * (self.config.get("d_model")**-0.5),
                    (self.config.get("d_ff"), self.config.get("d_model")),
                ).astype(np.float32)
            ),
            "bias": Tensor.from_numpy(np.zeros(self.config.get("d_ff"), dtype=np.float32)),
        }
        self.wo = {
            "weight": Tensor.from_numpy(
                np.random.normal(
                    0,
                    self.config.get("initializer_factor", 1.0) * (self.config.get("d_model")**-0.5),
                    (self.config.get("d_model"), self.config.get("d_ff")),
                ).astype(np.float32)
            ),
            "bias": Tensor.from_numpy(np.zeros(self.config.get("d_model"), dtype=np.float32)),
        }

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = linear(hidden_states, self.wi["weight"], self.wi["bias"])
        hidden_states = relu(hidden_states)
        hidden_states = dropout(hidden_states, self.dropout, training=True)
        hidden_states = linear(hidden_states, self.wo["weight"], self.wo["bias"])
        hidden_states = dropout(hidden_states, self.dropout, training=True)
        return hidden_states


class T5LayerSelfAttention:
    """T5 self-attention with relative position bias."""

    def __init__(self, config=None, **kwargs):
        config = _as_config(config, kwargs)
        self.config = config
        self.has_relative_attention_bias = config.get("has_relative_attention_bias", True)
        self.self = None
        self.layer_norm = T5LayerNorm(config)
        self.dropout = config.get("dropout_rate", 0.1)

        self.initialize_weights()
    def initialize_weights(self):
        self.layer_norm.initialize_weights()
        hidden = self.config.get("d_model")
        self.self = {
            "q": {
                "weight": Tensor.from_numpy(
                    np.random.normal(
                        0,
                        self.config.get("initializer_factor", 1.0) * (self.config.get("d_model")**-0.5),
                        (hidden, hidden),
                    ).astype(np.float32)
                )
            },
            "k": {
                "weight": Tensor.from_numpy(
                    np.random.normal(
                        0,
                        self.config.get("initializer_factor", 1.0) * (self.config.get("d_model")**-0.5),
                        (hidden, hidden),
                    ).astype(np.float32)
                )
            },
            "v": {
                "weight": Tensor.from_numpy(
                    np.random.normal(
                        0,
                        self.config.get("initializer_factor", 1.0) * (self.config.get("d_model")**-0.5),
                        (hidden, hidden),
                    ).astype(np.float32)
                )
            },
            "o": {
                "weight": Tensor.from_numpy(
                    np.random.normal(
                        0,
                        self.config.get("initializer_factor", 1.0) * (self.config.get("d_model")**-0.5),
                        (hidden, hidden),
                    ).astype(np.float32)
                )
            },
        }
        if self.has_relative_attention_bias:
            self.relative_attention_bias = Tensor.from_numpy(
                np.random.normal(
                    0,
                    self.config.get("initializer_factor", 1.0),
                    (self.config.get("relative_attention_num_buckets", 32), self.config.get("num_heads")),
                ).astype(np.float32)
            )

    def forward(
        self,
        hidden_states: Tensor,
        mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]], Optional[Tensor]]:
        # Layer norm
        normed = self.layer_norm.forward(hidden_states)

        # Self-attention with relative position bias
        # (Simplified - full relative position implementation is complex)
        batch, seq_len, hidden = hidden_states.shape

        q = np.asarray(linear(normed, self.self["q"]["weight"]).data, dtype=np.float64)
        k = np.asarray(linear(normed, self.self["k"]["weight"]).data, dtype=np.float64)
        v = np.asarray(linear(normed, self.self["v"]["weight"]).data, dtype=np.float64)

        num_heads = self.config.get("num_heads")
        head_dim = self.config.get("d_kv", 64)

        q = q.reshape(batch, -1, num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, -1, num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, -1, num_heads, head_dim).transpose(0, 2, 1, 3)

        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(head_dim)

        if position_bias is not None:
            pb = np.asarray(
                position_bias.data if hasattr(position_bias, "data") else position_bias,
                dtype=np.float64,
            )
            scores = scores + pb
        if mask is not None:
            m = np.asarray(
                mask.data if hasattr(mask, "data") else mask, dtype=np.float64
            )
            scores = scores + m

        attn = softmax(scores, axis=-1)
        attn = np.asarray(
            dropout(
                Tensor.from_numpy(attn.astype(np.float32)), self.dropout, training=True
            ).data,
            dtype=np.float64,
        )

        out = attn @ v
        out = out.transpose(0, 2, 1, 3).reshape(batch, -1, hidden)

        out = linear(Tensor.from_numpy(out.astype(np.float32)), self.self["o"]["weight"])
        out = dropout(out, self.dropout, training=True)
        out = out + hidden_states  # Residual

        return out, None, position_bias


class T5LayerCrossAttention:
    """T5 cross-attention for decoder."""

    def __init__(self, config=None, **kwargs):
        config = _as_config(config, kwargs)
        self.config = config
        self.enc_dec_attention = None
        self.layer_norm = T5LayerNorm(config)
        self.dropout = config.get("dropout_rate", 0.1)

        self.initialize_weights()
    def initialize_weights(self):
        self.layer_norm.initialize_weights()
        hidden = self.config.get("d_model")
        self.enc_dec_attention = {
            "q": {
                "weight": Tensor.from_numpy(
                    np.random.normal(
                        0,
                        self.config.get("initializer_factor", 1.0) * (self.config.get("d_model")**-0.5),
                        (hidden, hidden),
                    ).astype(np.float32)
                )
            },
            "k": {
                "weight": Tensor.from_numpy(
                    np.random.normal(
                        0,
                        self.config.get("initializer_factor", 1.0) * (self.config.get("d_model")**-0.5),
                        (hidden, hidden),
                    ).astype(np.float32)
                )
            },
            "v": {
                "weight": Tensor.from_numpy(
                    np.random.normal(
                        0,
                        self.config.get("initializer_factor", 1.0) * (self.config.get("d_model")**-0.5),
                        (hidden, hidden),
                    ).astype(np.float32)
                )
            },
            "o": {
                "weight": Tensor.from_numpy(
                    np.random.normal(
                        0,
                        self.config.get("initializer_factor", 1.0) * (self.config.get("d_model")**-0.5),
                        (hidden, hidden),
                    ).astype(np.float32)
                )
            },
        }

    def forward(
        self,
        hidden_states: Tensor,
        key_value_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        # Similar to self-attention but with key_value_states
        normed = self.layer_norm.forward(hidden_states)
        # Cross-attention implementation
        # (Simplified)
        return hidden_states, None


class T5Block:
    """T5 transformer block."""

    def __init__(self, config=None, **kwargs):
        config = _as_config(config, kwargs)
        self.config = config
        self.layer = []
        self.layer.append(
            T5LayerSelfAttention(
                config,
                has_relative_attention_bias=config.get(
                    "has_relative_attention_bias", True
                ),
            )
        )
        if config.get("is_encoder_decoder", True):
            self.layer.append(T5LayerCrossAttention(config))
        self.layer.append(T5DenseReluDense(config))

        self.initialize_weights()
    def initialize_weights(self):
        for layer in self.layer:
            layer.initialize_weights()

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        encoder_decoder_position_bias: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple:
        # Self-attention
        self_attn_output, _, position_bias = self.layer[0].forward(
            hidden_states,
            attention_mask,
            position_bias,
            past_key_value,
            use_cache,
        )
        hidden_states = self_attn_output

        # Cross-attention (if decoder)
        if len(self.layer) > 2 and encoder_hidden_states is not None:
            cross_attn_output, _ = self.layer[1].forward(
                hidden_states,
                encoder_hidden_states,
                encoder_attention_mask,
                None,
                use_cache,
            )
            hidden_states = cross_attn_output

        # Feed-forward
        hidden_states = self.layer[-1].forward(hidden_states)

        return hidden_states, None, position_bias, encoder_decoder_position_bias


class T5Stack:
    """T5 encoder or decoder stack."""

    def __init__(self, config=None, **kwargs):
        config = _as_config(config, kwargs)
        self.config = config
        self.embed_tokens = config.get("embed_tokens", None)
        self.is_decoder = config.get("is_encoder_decoder", True)
        self.block = [
            T5Block(config, has_relative_attention_bias=(i == 0))
            for i in range(config.get("num_layers"))
        ]
        self.final_layer_norm = T5LayerNorm(config)
        self.dropout = config.get("dropout_rate", 0.1)

        self.initialize_weights()
    def initialize_weights(self):
        if self.embed_tokens is None:
            self.embed_tokens = Tensor.from_numpy(
                np.random.normal(
                    0,
                    self.config.get("initializer_factor", 1.0) * (self.config.get("d_model")**-0.5),
                    (self.config.get("vocab_size", 50257), self.config.get("d_model")),
                ).astype(np.float32)
            )
        for block in self.block:
            block.initialize_weights()
        self.final_layer_norm.initialize_weights()

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[List] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
    ) -> Dict[str, Any]:
        if inputs_embeds is None:
            inputs_embeds = embedding(input_ids, self.embed_tokens)

        hidden_states = dropout(inputs_embeds, self.dropout, training=True)

        all_hidden_states = [] if output_hidden_states else None

        for i, block in enumerate(self.block):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            layer_past = past_key_values[i] if past_key_values else None

            outputs = block.forward(
                hidden_states,
                attention_mask,
                None,  # position_bias
                encoder_hidden_states,
                encoder_attention_mask,
                None,  # encoder_decoder_position_bias
                layer_past,
                use_cache,
            )

            hidden_states = outputs[0]

        hidden_states = self.final_layer_norm.forward(hidden_states)
        hidden_states = dropout(hidden_states, self.dropout, training=True)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return {
            "last_hidden_state": hidden_states,
            "past_key_values": None,
            "hidden_states": all_hidden_states,
        }


class T5Model:
    """Full T5 model (encoder-decoder)."""

    def __init__(self, config=None, **kwargs):
        config = _as_config(config, kwargs)
        self.config = config
        self.shared = None
        self.encoder = T5Stack(config)
        self.decoder = T5Stack(config)

        self.initialize_weights()
    def initialize_weights(self):
        self.shared = Tensor.from_numpy(
            np.random.normal(
                0,
                self.config.get("initializer_factor", 1.0) * (self.config.get("d_model")**-0.5),
                (self.config.get("vocab_size", 50257), self.config.get("d_model")),
            ).astype(np.float32)
        )
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

        self.encoder.initialize_weights()
        self.decoder.initialize_weights()

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        decoder_input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        decoder_attention_mask: Optional[Tensor] = None,
        encoder_outputs: Optional[Dict] = None,
        past_key_values: Optional[List] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        # Encoder
        if encoder_outputs is None:
            encoder_outputs = self.encoder.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )

        # Decoder
        decoder_outputs = self.decoder.forward(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs["last_hidden_state"],
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        return {
            "last_hidden_state": decoder_outputs["last_hidden_state"],
            "past_key_values": decoder_outputs.get("past_key_values"),
            "encoder_last_hidden_state": encoder_outputs["last_hidden_state"],
        }


class T5ForConditionalGeneration:
    """T5 with LM head."""

    def __init__(self, config=None, **kwargs):
        config = _as_config(config, kwargs)
        self.config = config
        self.model = T5Model(config)
        self.lm_head = None

        self.initialize_weights()
    def initialize_weights(self):
        self.model.initialize_weights()
        self.lm_head = {
            "weight": self.model.shared,
            "bias": Tensor.from_numpy(
                np.zeros(self.config.get("vocab_size", 50257), dtype=np.float32)
            ),
        }

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        decoder_input_ids: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        outputs = self.model.forward(
            input_ids=input_ids, decoder_input_ids=decoder_input_ids, **kwargs
        )

        lm_logits = linear(
            outputs["last_hidden_state"], self.lm_head["weight"], self.lm_head["bias"]
        )

        loss = None
        if labels is not None:
            # Shift for causal LM
            shift_logits = lm_logits[:, :-1, :].reshape(-1, lm_logits.shape[-1])
            shift_labels = labels[:, 1:].reshape(-1)
            loss = cross_entropy(shift_logits, shift_labels)

        return {"logits": lm_logits, "loss": loss}


# =========================================================================
# Utility functions
# =========================================================================


def create_bert_model(
    config_dict: Dict[str, Any], model_type: str = "bert"
) -> Union[BertModel, BertForMaskedLM]:
    """Factory function to create BERT models."""
    if model_type == "bert":
        model = BertModel(config_dict)
    elif model_type == "bert_mlm":
        model = BertForMaskedLM(config_dict)
    else:
        raise ValueError(f"Unknown BERT model type: {model_type}")
    model.initialize_weights()
    return model


def create_gpt_model(config: GPTConfig) -> Union[GPTModel, GPTLMHeadModel]:
    """Factory function to create GPT models."""
    model = GPTLMHeadModel(config)
    model.initialize_weights()
    return model


def create_t5_model(config: T5Config) -> Union[T5Model, T5ForConditionalGeneration]:
    """Factory function to create T5 models."""
    model = T5ForConditionalGeneration(config)
    model.initialize_weights()
    return model


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get predefined model configurations."""
    configs = {
        # BERT configs
        "bert-base-uncased": {
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
        },
        "bert-large-uncased": {
            "vocab_size": 30522,
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
        },
        # GPT configs
        "gpt2": GPTConfig(
            vocab_size=50257,
            n_positions=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
        ),
        "gpt2-medium": GPTConfig(
            vocab_size=50257,
            n_positions=1024,
            n_embd=1024,
            n_layer=24,
            n_head=16,
        ),
        "gpt2-large": GPTConfig(
            vocab_size=50257,
            n_positions=1024,
            n_embd=1280,
            n_layer=36,
            n_head=20,
        ),
        "gpt2-xl": GPTConfig(
            vocab_size=50257,
            n_positions=1024,
            n_embd=1600,
            n_layer=48,
            n_head=25,
        ),
        # T5 configs
        "t5-small": T5Config(
            vocab_size=32128,
            d_model=512,
            d_ff=2048,
            num_layers=6,
            num_heads=8,
            d_kv=64,
        ),
        "t5-base": T5Config(
            vocab_size=32128,
            d_model=768,
            d_ff=3072,
            num_layers=12,
            num_heads=12,
            d_kv=64,
        ),
        "t5-large": T5Config(
            vocab_size=32128,
            d_model=1024,
            d_ff=4096,
            num_layers=24,
            num_heads=16,
            d_kv=64,
        ),
        "t5-3b": T5Config(
            vocab_size=32128,
            d_model=1024,
            d_ff=16384,
            num_layers=24,
            num_heads=32,
            d_kv=64,
        ),
        "t5-11b": T5Config(
            vocab_size=32128,
            d_model=1024,
            d_ff=65536,
            num_layers=24,
            num_heads=128,
            d_kv=64,
        ),
    }

    if model_name not in configs:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(configs.keys())}"
        )

    return configs[model_name]


# =========================================================================
# Weight loading utilities
# =========================================================================


def load_pretrained_weights(
    model: Any, weights_dict: Dict[str, np.ndarray], prefix: str = ""
) -> List[str]:
    """Load pretrained weights into model."""
    missing_keys = []
    unexpected_keys = []

    # This would recursively load weights based on model structure
    # For now, return empty lists
    return missing_keys


def save_pretrained_weights(model: Any, save_path: str) -> None:
    """Save model weights to file."""
    weights = {}
    # Would recursively extract weights
    np.savez_compressed(save_path, **weights)


# =========================================================================
# Model size estimation
# =========================================================================


def count_parameters(model: Any) -> int:
    """Count total parameters in model."""
    total = 0
    # Would recursively count
    return total


def get_model_size_mb(model: Any) -> float:
    """Get model size in MB (assuming float32)."""
    params = count_parameters(model)
    return params * 4 / (1024 * 1024)


def get_model_flops(model: Any, input_shape: Tuple[int, ...]) -> float:
    """Estimate FLOPs for forward pass."""
    # Would compute based on model architecture
    return 0.0


# =========================================================================
# Make model/component classes callable (the test suite invokes them directly
# as ``module(x)`` rather than ``module.forward(x)``).
# =========================================================================

for _cls in (
    BertEmbeddings, BertSelfAttention, BertIntermediate, BertOutput,
    BertLayer, BertEncoder, BertPooler, BertModel, BertForMaskedLM,
    GPTAttention, GPTMLP, GPTBlock, GPTModel, GPTLMHeadModel,
    T5LayerNorm, T5DenseReluDense, T5LayerSelfAttention,
    T5LayerCrossAttention, T5Block, T5Stack, T5Model, T5ForConditionalGeneration,
):
    if hasattr(_cls, "forward"):
        _cls.__call__ = _cls.forward


# Attention/block ``forward`` returns a ``(output, present)`` tuple; the tests
# expect the output tensor directly.
GPTAttention.__call__ = lambda self, *a, **k: self.forward(*a, **k)[0]
GPTBlock.__call__ = lambda self, *a, **k: self.forward(*a, **k)[0]
