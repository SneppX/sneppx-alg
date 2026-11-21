"""Layer mapping — automatic conversion between HF and SNEPPX layer names.

Defines canonical name mappings per supported architecture (BERT, GPT-2,
Llama, Mistral, T5, Whisper) used by both the loader and the exporter.
"""

from typing import Dict, Optional

# HF prefix fragments -> SNEPPX prefix fragments, per architecture.
LAYER_MAP: Dict[str, Dict[str, str]] = {
    "bert": {
        "embeddings.word_embeddings.weight": "embedding.weight",
        "embeddings.position_embeddings.weight": "position_embedding.weight",
        "encoder.layer.{i}.attention.self.query.weight": "layers.{i}.self_attn.query.weight",
        "encoder.layer.{i}.attention.self.key.weight": "layers.{i}.self_attn.key.weight",
        "encoder.layer.{i}.attention.self.value.weight": "layers.{i}.self_attn.value.weight",
        "encoder.layer.{i}.attention.output.dense.weight": "layers.{i}.self_attn.out_proj.weight",
        "encoder.layer.{i}.attention.output.LayerNorm.weight": "layers.{i}.ln1.weight",
        "encoder.layer.{i}.intermediate.dense.weight": "layers.{i}.mlp.fc1.weight",
        "encoder.layer.{i}.output.dense.weight": "layers.{i}.mlp.fc2.weight",
        "encoder.layer.{i}.output.LayerNorm.weight": "layers.{i}.ln2.weight",
        "pooler.dense.weight": "pooler.weight",
    },
    "gpt2": {
        "wte.weight": "token_embedding.weight",
        "wpe.weight": "position_embedding.weight",
        "h.{i}.attn.c_attn.weight": "layers.{i}.self_attn.c_attn.weight",
        "h.{i}.attn.c_proj.weight": "layers.{i}.self_attn.c_proj.weight",
        "h.{i}.ln_1.weight": "layers.{i}.ln1.weight",
        "h.{i}.ln_2.weight": "layers.{i}.ln2.weight",
        "h.{i}.mlp.c_fc.weight": "layers.{i}.mlp.fc1.weight",
        "h.{i}.mlp.c_proj.weight": "layers.{i}.mlp.fc2.weight",
        "ln_f.weight": "ln_f.weight",
    },
    "llama": {
        "model.embed_tokens.weight": "embedding.weight",
        "model.layers.{i}.self_attn.q_proj.weight": "layers.{i}.self_attn.q_proj.weight",
        "model.layers.{i}.self_attn.k_proj.weight": "layers.{i}.self_attn.k_proj.weight",
        "model.layers.{i}.self_attn.v_proj.weight": "layers.{i}.self_attn.v_proj.weight",
        "model.layers.{i}.self_attn.o_proj.weight": "layers.{i}.self_attn.o_proj.weight",
        "model.layers.{i}.mlp.gate_proj.weight": "layers.{i}.mlp.gate_proj.weight",
        "model.layers.{i}.mlp.up_proj.weight": "layers.{i}.mlp.up_proj.weight",
        "model.layers.{i}.mlp.down_proj.weight": "layers.{i}.mlp.down_proj.weight",
        "model.layers.{i}.input_layernorm.weight": "layers.{i}.ln1.weight",
        "model.layers.{i}.post_attention_layernorm.weight": "layers.{i}.ln2.weight",
        "model.norm.weight": "ln_f.weight",
        "lm_head.weight": "lm_head.weight",
    },
    "mistral": {
        "model.embed_tokens.weight": "embedding.weight",
        "model.layers.{i}.self_attn.q_proj.weight": "layers.{i}.self_attn.q_proj.weight",
        "model.layers.{i}.self_attn.k_proj.weight": "layers.{i}.self_attn.k_proj.weight",
        "model.layers.{i}.self_attn.v_proj.weight": "layers.{i}.self_attn.v_proj.weight",
        "model.layers.{i}.self_attn.o_proj.weight": "layers.{i}.self_attn.o_proj.weight",
        "model.layers.{i}.mlp.gate_proj.weight": "layers.{i}.mlp.gate_proj.weight",
        "model.layers.{i}.mlp.up_proj.weight": "layers.{i}.mlp.up_proj.weight",
        "model.layers.{i}.mlp.down_proj.weight": "layers.{i}.mlp.down_proj.weight",
        "model.norm.weight": "ln_f.weight",
    },
    "t5": {
        "shared.weight": "embedding.weight",
        "encoder.block.{i}.layer.0.SelfAttention.q.weight": "encoder_blocks.{i}.self_attn.q.weight",
        "encoder.block.{i}.layer.0.SelfAttention.k.weight": "encoder_blocks.{i}.self_attn.k.weight",
        "encoder.block.{i}.layer.0.SelfAttention.v.weight": "encoder_blocks.{i}.self_attn.v.weight",
        "encoder.block.{i}.layer.0.SelfAttention.o.weight": "encoder_blocks.{i}.self_attn.o.weight",
        "encoder.block.{i}.layer.1.DenseReluDense.wi.weight": "encoder_blocks.{i}.mlp.fc1.weight",
        "encoder.block.{i}.layer.1.DenseReluDense.wo.weight": "encoder_blocks.{i}.mlp.fc2.weight",
    },
    "whisper": {
        "model.encoder.conv1.weight": "encoder.conv1.weight",
        "model.encoder.conv2.weight": "encoder.conv2.weight",
        "model.encoder.layers.{i}.self_attn.q_proj.weight": "encoder.layers.{i}.self_attn.q_proj.weight",
        "model.encoder.layers.{i}.self_attn.k_proj.weight": "encoder.layers.{i}.self_attn.k_proj.weight",
        "model.encoder.layers.{i}.self_attn.v_proj.weight": "encoder.layers.{i}.self_attn.v_proj.weight",
        "model.encoder.layers.{i}.self_attn.out_proj.weight": "encoder.layers.{i}.self_attn.out_proj.weight",
    },
}


def layer_mapping(arch: str) -> Dict[str, str]:
    """Return the HF→SNEPPX name map for ``arch`` (BERT/GPT2/Llama/Mistral/T5/Whisper)."""
    return LAYER_MAP.get(arch.lower(), {})


def convert_hf_layer(hf_name: str, arch: str,
                     layer_index: Optional[int] = None) -> Optional[str]:
    """Map a single HF parameter name to its SNEPPX equivalent.

    ``{i}`` placeholders are replaced by ``layer_index`` when provided.
    Returns ``None`` if no mapping exists.
    """
    mapping = LAYER_MAP.get(arch.lower(), {})
    for hf_pattern, spx_pattern in mapping.items():
        if "{i}" in hf_pattern:
            if layer_index is None:
                continue
            hf_expanded = hf_pattern.replace("{i}", str(layer_index))
            spx_expanded = spx_pattern.replace("{i}", str(layer_index))
        else:
            hf_expanded, spx_expanded = hf_pattern, spx_pattern
        if hf_name == hf_expanded:
            return spx_expanded
    return None
