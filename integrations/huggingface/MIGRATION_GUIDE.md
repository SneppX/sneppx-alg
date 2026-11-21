# Migration Guide: HuggingFace &rarr; SNEPPX-Alg

This guide helps researchers switch from the HuggingFace `transformers`
ecosystem to **SNEPPX-Alg** without rewriting their models or pipelines.

---

## 1. Loading a HuggingFace Model

### Before (transformers)

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

### After (SNEPPX)

```python
from sneppx.integrations.huggingface import from_huggingface

model = from_huggingface("bert-base-uncased")
```

The loader returns a SNEPPX-compatible wrapper exposing `named_parameters()`
and `forward()`, so weights can be consumed by SNEPPX's optimizer and trainer.

## 2. Exporting a SNEPPX Model to HF

```python
from sneppx.integrations.huggingface import to_huggingface

to_huggingface(my_sneppx_model, "./exported", model_type="bert")
# writes ./exported/pytorch_model.bin + ./exported/config.json
```

## 3. Pipelines

### Before

```python
from transformers import pipeline

gen = pipeline("text-generation", model="gpt2")
print(gen("Once upon a time")[0]["generated_text"])
```

### After

```python
from sneppx.integrations.huggingface import pipeline

gen = pipeline("text-generation", model="gpt2")
print(gen("Once upon a time")[0]["generated_text"])
```

## 4. Datasets

### Before

```python
from datasets import load_dataset
from torch.utils.data import DataLoader

ds = load_dataset("imdb", split="train")
loader = DataLoader(ds, batch_size=8, shuffle=True)
```

### After

```python
from sneppx.integrations.huggingface import load_hf_dataset
from SneppX_ALG import DataLoader

ds = load_hf_dataset("imdb", split="train",
                     preprocess=lambda ex: (ex["text"], ex["label"]))
loader = DataLoader(ds, batch_size=8, shuffle=True)
```

## 5. Supported Architectures

| Architecture | Loader | Exporter | Notes |
|--------------|--------|----------|-------|
| BERT | &check; | &check; | encoder-only |
| GPT-2 | &check; | &check; | decoder-only |
| Llama | &check; | &check; | decoder-only |
| Mistral | &check; | &check; | decoder-only |
| T5 | &check; | &check; | encoder-decoder |
| Whisper | &check; | &check; | speech |

## 6. Tokenizer Compatibility

Use the HuggingFace `AutoTokenizer` (Rust/C++ based) directly with SNEPPX:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer("Hello world", return_tensors="np")
```

## 7. Fine-Tuning with SNEPPX's Distributed Trainer

SNEPPX models loaded via `from_huggingface` expose `named_parameters()`, so
they can be plugged into SNEPPX's `Trainer` / `DistributedDataParallel`:

```python
from SneppX_ALG import Trainer, AdamW, DistributedDataParallel
from sneppx.integrations.huggingface import from_huggingface

model = from_huggingface("bert-base-uncased")
ddp = DistributedDataParallel(model)
optimizer = AdamW(model.parameters(), lr=2e-5)

trainer = Trainer(model, loss_fn, optimizer)
trainer.fit(train_loader, epochs=3)
```

## 8. What Changes

| Concept | HuggingFace | SNEPPX-Alg |
|---------|-------------|------------|
| Model loading | `AutoModel.from_pretrained` | `from_huggingface` |
| Model export | `model.save_pretrained` | `to_huggingface` |
| Pipelines | `transformers.pipeline` | `sneppx.pipeline` |
| Datasets | `datasets.load_dataset` | `load_hf_dataset` |
| Tokenizers | `AutoTokenizer` | Same (drop-in) |
| Trainer | `Trainer` / `training_args` | `Trainer` + `TrainConfig` |
