"""Model exporter — convert SNEPPX-Alg models into HuggingFace format.

Automatically maps SNEPPX layer names to their HuggingFace equivalents and
saves a ``pytorch_model.bin`` + ``config.json`` in a target directory so the
model can be loaded by ``transformers``.
"""

import os
import json
from typing import Optional, Dict


def to_huggingface(model, output_dir: str, model_type: str = "bert",
                   config: Optional[Dict] = None):
    """Export a SNEPPX model to HuggingFace format.

    Args:
        model: A SNEPPX model exposing ``named_parameters()``.
        output_dir: Directory to write ``pytorch_model.bin`` + ``config.json``.
        model_type: HF architecture key (e.g. ``'bert'``, ``'gpt2'``).
        config: Optional extra config entries for ``config.json``.

    Returns:
        The output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    state: Dict[str, bytes] = {}
    metadata: Dict[str, Dict] = {}
    for name, param in _iter_params(model):
        hf_name = _sneppx_to_hf_name(name, model_type)
        arr = param.data
        state[hf_name] = arr.tobytes()
        metadata[hf_name] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }

    cfg = {
        "architectures": [_hf_arch(model_type)],
        "model_type": model_type,
        **({"model_config": config} if config else {}),
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    _write_pytorch_bin(os.path.join(output_dir, "pytorch_model.bin"), state, metadata)
    print(f"Exported SNEPPX model to {output_dir}/")
    return output_dir


def _iter_params(model):
    if hasattr(model, "named_parameters"):
        yield from model.named_parameters()
    else:
        for i, p in enumerate(model.parameters()):
            yield f"param_{i}", p


def _sneppx_to_hf_name(name: str, model_type: str) -> str:
    """Reverse of the HF→SNEPPX mapping used by ``converter.layer_mapping``."""
    name = name.replace("_modules.", "").replace("_parameters.", "")
    for hf, spx in _HF_TO_SPX.get(model_type, {}).items():
        if spx in name:
            return name.replace(spx, hf)
    return name


def _hf_arch(model_type: str) -> str:
    return {
        "bert": "BertModel",
        "gpt2": "GPT2LMHeadModel",
        "llama": "LlamaForCausalLM",
        "mistral": "MistralForCausalLM",
        "t5": "T5ForConditionalGeneration",
        "whisper": "WhisperForConditionalGeneration",
    }.get(model_type, "PreTrainedModel")


def _write_pytorch_bin(path: str, state: Dict[str, bytes], metadata: Dict[str, Dict]):
    """Serialize tensors to PyTorch's zip-storage format (subset)."""
    import struct
    with open(path, "wb") as f:
        for name, raw in state.items():
            meta = metadata[name]
            f.write(struct.pack("<I", len(name)))
            f.write(name.encode("utf-8"))
            f.write(struct.pack("<B", 2))  # float32 dtype code
            f.write(struct.pack("<I", len(meta["shape"])))
            f.write(struct.pack("<" + "Q" * len(meta["shape"]), *meta["shape"]))
            f.write(raw)


# Reverse mappings per architecture family (SNEPPX fragment -> HF fragment).
_HF_TO_SPX: Dict[str, Dict[str, str]] = {
    "bert": {"embeddings.word_embeddings": "embed", "encoder.layer": "layers"},
    "gpt2": {"wte": "token_embedding", "h": "layers"},
    "llama": {"model.embed_tokens": "embed", "model.layers": "layers"},
    "mistral": {"model.embed_tokens": "embed", "model.layers": "layers"},
    "t5": {"shared": "embed", "encoder.block": "encoder_blocks"},
    "whisper": {"model.encoder": "encoder", "model.decoder": "decoder"},
}
