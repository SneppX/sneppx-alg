"""SNEPPX-Alg &harr; HuggingFace Transformers integration.

Bi-directional model import/export, pipeline mirror, dataset loading, and
tokenizer compatibility. Minimal dependencies — only ``numpy`` is required;
``huggingface_hub`` / ``transformers`` are used when available.
"""

from .loader import from_huggingface
from .exporter import to_huggingface
from .converter import convert_hf_layer, layer_mapping
from .pipeline import pipeline
from .datasets import load_hf_dataset

__all__ = [
    "from_huggingface",
    "to_huggingface",
    "convert_hf_layer",
    "layer_mapping",
    "pipeline",
    "load_hf_dataset",
]
