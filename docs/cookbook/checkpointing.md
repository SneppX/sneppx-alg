# Cookbook — Checkpointing

## 1. Save and load model state (simple)

**Intent:** Basic checkpoint persistence.

```python
from SneppX_ALG import Transformer

model = Transformer(vocab_size=1000, dim=128, num_heads=4, num_layers=2, ffn_dim=512, max_seq_len=64)
model.save_checkpoint("/tmp/model.ckpt")       # pickle of param data + config
# later, in a fresh process:
model.load_checkpoint("/tmp/model.ckpt")
```

**Notes:** The format stores `{param_data, config}` as a pickle of numpy
arrays. :material-alert-decagram: For real weight I/O use
`CheckpointWriter`/`CheckpointReader` (binary format, S7-signed). CPU-safe.

## 2. Use the async checkpoint coordinator

**Intent:** Non-blocking, fault-tolerant distributed saves.

```python
from SneppX_ALG import CheckpointCoordinator, HeartbeatMonitor, FaultToleranceManager

coord = CheckpointCoordinator(
    checkpoint_root="/ckpts",
    save_every_n_steps=1000,
    max_to_keep=5,
    async_save=True,                # background I/O thread
)
coord.save(model, step=1000)        # returns immediately

hb  = HeartbeatMonitor(peers=["10.0.0.2:29500", "10.0.0.3:29500"])
ft  = FaultToleranceManager(heartbeat=hb, max_restarts=3)
ft.wait_for_quorum()                # blocks until >50% peers healthy
```

**Notes:** `async_save` overlaps D2D→D2H copies with computation
(`double-buffered`). `HeartbeatMonitor` uses UDP; `FaultToleranceManager`
wraps restart + versioned checkpoint rollback. :material-alert-decagram: C
backend.

## 3. Elastic training with automatic restart

**Intent:** Survive node failures without manual re-launch.

```python
from SneppX_ALG import ElasticTrainer

trainer = ElasticTrainer(
    train_fn=train_loop,
    checkpoint_dir="/ckpts",
    max_restarts=5,
    restart_delay=10,
    join_timeout=600,
)
trainer.run(num_nodes=2, num_gpus=4)
# On failure: ranks re-shard, load latest checkpoint, resume
```

**Notes:** `ElasticTrainer` (from `checkpoint.py`) implements the join/leave/
failure/reconfigure lifecycle (`elastic.c`). Requires a shared checkpoint
store (NFS / S3). :material-alert-decagram: C backend + NCCL for multi-rank.

## 4. Validate a checkpoint

**Intent:** Guard against silently corrupted weights.

```python
from SneppX_ALG import validate_checkpoint

ok, errors = validate_checkpoint("/ckpts/model.ckpt")
print(ok, errors)    # True []
```

**Notes:** `validate_checkpoint` checks the magic header, CRC, and tensor
shape table. Used by S7 (`SignedUpdateManager.install_update`).
