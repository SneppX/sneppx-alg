# Cookbook — Generation

## 1. Greedy generation (deterministic)

**Intent:** Produce the single most-likely next-token sequence.

```python
from SneppX_ALG.interface_bindings.generation import generate, GenerationConfig
from SneppX_ALG import Transformer

model = Transformer(vocab_size=1000, dim=128, num_heads=4, num_layers=2, ffn_dim=512, max_seq_len=128)
cfg = GenerationConfig(max_new_tokens=32, do_sample=False)   # greedy when temp→0
out = generate(model, [1, 2, 3, 4], generation_config=cfg)
print(out["output_ids"])           # shape (1, seq+32)
```

**Notes:** `generate` treats `num_beams>1` as beam search, `do_sample and
temp>0` as sampling, otherwise greedy. `model.forward` must return logits of
shape `(batch, seq, vocab)`; see recipe #4 for a wrapped LM head.

## 2. Sampling with top-k / top-p (nucleus)

**Intent:** Stochastically sample diverse completions.

```python
from SneppX_ALG.interface_bindings.generation import generate, GenerationConfig

cfg = GenerationConfig(
    max_new_tokens=64,
    do_sample=True,
    temperature=0.8,
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.1,
    stop_strings=["\n\n"],
)
out = generate(model, input_ids, generation_config=cfg)
```

**Notes:** `repetition_penalty` > 1 penalizes already-generated tokens.
`stop_strings` halts early on a substring (tokenized internally). CPU-safe
(pure NumPy sampling loop).

## 3. Beam search

**Intent:** Best-first decoding for higher-quality output.

```python
from SneppX_ALG.interface_bindings.generation import generate, GenerationConfig

cfg = GenerationConfig(
    max_new_tokens=48,
    num_beams=4,
    length_penalty=0.7,
    early_stopping=True,
)
out = generate(model, input_ids, generation_config=cfg)
```

**Notes:** Currently supports `batch_size=1`. `early_stopping=True` ends a
beam as soon as `num_beams` completed sequences are found.

## 4. Stream tokens to the terminal

**Intent:** Print tokens one-by-one as they are generated.

```python
from SneppX_ALG.interface_bindings.generation import generate, GenerationConfig, TextStreamer
from SneppX_ALG import Tokenizer

tok = Tokenizer(vocab_size=1000)
streamer = TextStreamer(tokenizer=tok, skip_prompt=True)

cfg = GenerationConfig(max_new_tokens=80, do_sample=True, temperature=0.7, top_p=0.9)
generate(model, prompt_ids, generation_config=cfg, streamer=streamer)
# tokens appear immediately as they are decoded
```

**Notes:** `TextStreamer` writes to `print` by default; pass `print_fn` to
redirect to a GUI or socket. CPU-safe.

## 5. Batch generation (padded prompts)

**Intent:** Decode many prompts at once.

```python
from SneppX_ALG.interface_bindings.generation import batch_generate, GenerationConfig

prompts = [[1,2,3], [4,5,6,7,8], [9,10]]        # variable-length token lists
cfg = GenerationConfig(max_new_tokens=32, temperature=0.7)
out = batch_generate(model, prompts, generation_config=cfg)
print(out["output_ids"].shape)     # (3, max_len + 32)
```

**Notes:** `batch_generate` pads to the longest prompt with `pad_token_id`,
builds an attention mask, and delegates to `generate`. CPU-safe.
