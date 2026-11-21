# Cookbook — Training

## 1. Minimal training loop (manual)

**Intent:** Full control over forward/backward/step.

```python
from SneppX_ALG import Transformer, AdamW, CrossEntropyLoss, Tensor
import numpy as np

model  = Transformer(vocab_size=1000, dim=128, num_heads=4, num_layers=2, ffn_dim=512, max_seq_len=128)
opt    = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
lossfn = CrossEntropyLoss()

for step in range(500):
    ids = Tensor(np.random.randint(0, 1000, (4, 16)))   # (batch, seq)
    tgt = Tensor(np.random.randint(0, 1000, (4, 16)))
    opt.zero_grad()
    logits = model(ids)                                  # (4, 16, 1000)
    loss = lossfn(logits.reshape((-1, 1000)), tgt.reshape((-1,)))
    loss.backward()
    opt.step()
    if step % 50 == 0:
        print(step, loss.item())
```

:material-alert-decagram: **C backend required** for `backward()`/`opt.step()`.

## 2. Use the Trainer.fit convenience loop

**Intent:** Let the framework drive epochs + checkpointing.

```python
from SneppX_ALG import Trainer, TrainConfig, Transformer
from SneppX_ALG.interface_bindings.data_loader import DataLoader
from SneppX_ALG import TensorDataset

cfg = TrainConfig()
cfg.num_epochs  = 10
cfg.batch_size  = 32
cfg.learning_rate = 1e-3
cfg.save_interval = 5

model   = Transformer(vocab_size=800, dim=128, num_heads=4, num_layers=2, ffn_dim=512, max_seq_len=64)
trainer = Trainer(model, cfg)

x = Tensor.randn((256, 64))
y = Tensor.randn((256, 64))           # toy regression
loader = DataLoader(TensorDataset(x, y), batch_size=cfg.batch_size, shuffle=True)
trainer.fit(loader)                  # saves checkpoint_epoch_5.ckpt, etc.
```

**Notes:** `train_step`/`evaluate` require the C backend; `fit` calls them
per epoch. Checkpoints are written next to the CWD.

## 3. Cosine LR schedule with warmup

**Intent:** Standard pretrain schedule.

```python
from SneppX_ALG import AdamW, LinearWarmupCosineDecay

opt   = AdamW(model.parameters(), lr=3e-4)
sched = LinearWarmupCosineDecay(opt, warmup_steps=100, total_steps=10000, min_lr=1e-5)
for step in range(10000):
    sched.step()
    opt.zero_grad(); loss.backward(); opt.step()
```

**Notes:** Many schedulers available: `StepLR`, `ExponentialLR`,
`ReduceLROnPlateau`, `OneCycleLR`, `TriStageLR`, `SequentialLR`.

## 4. Gradient clipping

**Intent:** Stabilize training.

```python
from SneppX_ALG.interface_bindings.trainer_v3 import clip_grad_norm_

for x, y in loader:
    loss = model(x)
    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
```

**Notes:** `clip_grad_norm_` lives in `trainer_v3`; the per-step gradient
buffers are managed by the C optimizer.

## 5. Mixed-precision (AMP)

**Intent:** bf16/fp16 training where supported.

```python
from SneppX_ALG import autocast, GradScaler

scaler = GradScaler()
for x, y in loader:
    with autocast():                        # bf16 on CUDA, no-op on CPU
        out = model(x)
        loss = lossfn(out, y)
    scaler.scale(loss).backward()
    scaler.step(opt); scaler.update()
```

**Notes:** CPU path is a passthrough; CUDA path casts to bf16. Needs
`SNEPPX_BUILD_CUDA=ON`.

## 6. Gradient checkpointing (memory savings)

**Intent:** Trade compute for memory on deep models.

```python
from SneppX_ALG import checkpoint, GradientCheckpointer
from SneppX_ALG import Sequential, TransformerBlock

blocks = Sequential(*[TransformerBlock(...) for _ in range(24)])

# checkpoint a sub-sequence of blocks
def run_blocks(x):
    return blocks(x)
out = checkpoint(run_blocks, x, use_reentrant=False)

# Or wrap a whole model
gp = GradientCheckpointer(model, checkpoint_every=4)
out = gp.forward(x)
```

:material-alert-decagram: **C backend + CUDA** recommended; pure-NumPy
checkpointing works but offers no memory savings.
