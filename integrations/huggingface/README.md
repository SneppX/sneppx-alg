# SNEPPX-Alg &harr; HuggingFace Integration

Bi-directional integration between **SNEPPX-Alg** and the **HuggingFace
`transformers`** ecosystem, with minimal dependencies (`numpy` only).

## Features

- **Loader**: `sneppx.from_huggingface('bert-base-uncased')` imports HF models.
- **Exporter**: `model.to_huggingface()` exports SNEPPX models to HF format.
- **Converter**: Automatic HF&harr;SNEPPX layer-name mapping (BERT, GPT-2, Llama,
  Mistral, T5, Whisper).
- **Pipelines**: `sneppx.pipeline('text-generation', model='...')` mirror.
- **Datasets**: `load_hf_dataset` loads HF datasets straight into SNEPPX's
  `DataLoader`.
- **Tokenizer**: Drop-in compatibility with `AutoTokenizer`.
- **Training**: HF models fine-tune with SNEPPX's distributed trainer.

## Usage

```python
from sneppx.integrations.huggingface import from_huggingface, to_huggingface, pipeline, load_hf_dataset

model = from_huggingface("bert-base-uncased")
to_huggingface(model, "./exported", model_type="bert")
```

See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for researchers switching from
HuggingFace to SNEPPX.

## Dependencies

- Required: `numpy`
- Optional: `transformers`, `huggingface_hub`, `datasets`, `torch`

The module degrades gracefully — import-only operations work without any of
the optional packages installed.
