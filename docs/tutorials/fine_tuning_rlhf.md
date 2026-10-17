# Tutorial — RLHF Fine-Tuning with LoRA + DPO

**Notebook:** [`fine_tuning_rlhf.ipynb`](https://github.com/SneppX/sneppx-alg/blob/main/notebooks/fine_tuning_rlhf.ipynb)
([download](https://github.com/SneppX/sneppx-alg/blob/main/notebooks/fine_tuning_rlhf.ipynb))

## What you'll build

A parameter-efficient fine-tuning recipe that (1) attaches **LoRA** adapters
to a small model, (2) trains with **DPO** (Direct Preference Optimization) via
`SneppX_ALG.DPOTrainer`, and (3) exercises the **S5 AI safety** filter on the
generated output.

## Setup

```powershell
$env:PYTHONPATH = "bindings/python"
```

```python
import numpy as np
from SneppX_ALG import (
    Transformer, Tokenizer, AdamW, Tensor,
    DPOTrainer, LoRAConfig, LoRALinear,
    S5RLHFSafety, S5OutputVerifier,
)
HAS_C = __import__("SneppX_ALG")._HAS_C_BACKEND
```

## 1. Attach LoRA adapters

LoRA freezes the base weights and injects low-rank `A·B` updates into attention
projections.

```python
base = Transformer(vocab_size=300, dim=128, num_heads=4, num_layers=4, ffn_dim=256, max_seq_len=64)

lora_cfg = LoRAConfig(r=8, alpha=32, dropout=0.1)
# Wrap a Linear in a LoRA-augmented layer
base.lm_head = LoRALinear(base.lm_head, r=lora_cfg.r, alpha=lora_cfg.alpha)
print("trainable params:", sum(p.numel for n, p in base.named_parameters() if "lora_" in n))
```

## 2. DPOTrainer

DPO optimizes the policy to prefer *chosen* over *rejected* completions.

```python
trainer = DPOTrainer(
    policy=base,
    ref_policy=None,            # uses policy as its own reference on first call
    beta=0.1,
    optimizer=AdamW(filter(lambda p: p.requires_grad, base.parameters()), lr=5e-4),
)

# Toy preference data: (prompt, chosen_tail, rejected_tail) token-id lists
prefs = [
    ([1, 2, 3], [10, 11, 12], [99, 98]),
    ([4, 5, 6], [20, 21],    [88, 87, 86]),
]
for prompt, chosen, rejected in prefs:
    if not HAS_C:
        print("C backend required for DPO backward — skipping")
        break
    loss = trainer.dpo_loss(
        prompt_input_ids=prompt,
        chosen_input_ids=chosen,
        rejected_input_ids=rejected,
    )
    trainer.optimizer.zero_grad()
    loss.backward()
    trainer.optimizer.step()
    print("dpo loss:", float(loss.data))
```

> The DPO loss is `loss = -log_σ(β·(logπ(y_w|x) − logπ(y_l|x) − (β/r)·logπ_ref/ratio))`.
> `DPOTrainer._forward_logps` computes log-probs with proper softmax.

## 3. GRPO alternative

```python
from SneppX_ALG import GRPOTrainer

grpo = GRPOTrainer(
    policy=base,
    optimizer=AdamW(base.parameters(), lr=3e-4),
    num_generations=4,
    beta=0.01,
)
# grpo_loss = ratio-based PPO surrogate; see trainer_v3
```

## 4. Safety-check the output

Run the generated continuation through the **S5** safety layers:

```python
prompt = "Explain how to..."
verifier = S5OutputVerifier()
safe = S5RLHFSafety(allowed_topics=["science", "technology"])
# After generation, verify + filter:
#   verifier.check(output_text)   -> bool
#   safe.is_safe(output_text)     -> bool
```

## Key takeaways

- LoRA keeps the full model frozen and trains only injected low-rank weights —
  cheap to checkpoint/export.
- DPO needs `backward()` (C backend). GRPO is the PPO-style alternative.
- Always run `S5OutputVerifier` / `S5RLHFSafety` before serving RLHF-tuned
  output (S5 layer).
- Checkpoints are small (only adapter weights); merge via `apply_lora`.

## Next steps

- Quantize the fine-tuned model — see
  [Quantization + Serving](quantization_serving.md).
- Serve with `sneppx-serve --rlhf-safety` to apply S5 filters at the API.
