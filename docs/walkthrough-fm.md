# Code Walkthrough — FM (Fractal Memory Orchestrator)

FM is the fifth and final stage of the SNEPPX-Alg pipeline. It manages
distributed, federated memory: a network of nodes, each with its own memory
bank, that periodically synchronize their knowledge. FM also compresses
gradients, adds privacy noise, and protects against catastrophic forgetting.

- Public header: `include/neural_core/architecture/fractal_memory_orchestrator.h`
- Implementation: `algorithms/fm/core/`
- CUDA extensions: `algorithms/fm/cuda/`

## Configuration

```c
SNEPPXFMConfig cfg = SNEPPX_fm_config_default();
cfg.num_nodes              = 8;      // participating nodes/ranks
cfg.memory_dim             = 512;    // width of each memory entry
cfg.memory_capacity        = 10000;  // entries per bank
cfg.sync_interval          = 100;    // steps between syncs
cfg.sync_method            = SNEPPX_SYNC_ALL_REDUCE; // ALL_REDUCE|GOSSIP|TOPOLOGY
cfg.compression_ratio      = 0.25f;  // fraction of gradients transmitted
cfg.privacy_epsilon        = 1.0f;   // DP noise budget
cfg.catastrophic_forgetting_protection = 1;
cfg.ewm_alpha              = 0.05f;  // exponential moving average decay
```

## Memory bank

`SNEPPXFMMemoryBank` is an associative key-value store over tensors:

- `SNEPPX_fm_memory_bank_create(memory_dim, capacity)`
- `SNEPPX_fm_memory_bank_write(bank, key, value)` — upsert by key
- `SNEPPX_fm_memory_bank_read(bank, key)` — lookup by key
- `SNEPPX_fm_memory_bank_forget(bank, forget_rate)` — age out stale entries
  using `access_counts`/`timestamps`

## Nodes and controller

A node (`SNEPPXFMNode`) pairs a memory bank with a gradient accumulator,
liveness state, last-sync timestamp, and a `trust_score`. The controller
(`SNEPPXFMController`) owns all nodes plus a shared sync state (`global_memory`,
`sync_round`, `node_contributions`, `conflict_log`).

```c
SNEPPXFMController* ctrl = SNEPPX_fm_controller_create(&cfg);
SNEPPXTensor* out = NULL;
int rc = SNEPPX_fm_forward(ctrl, node_id, input, &out);
```

FM has no trainable parameters of its own — `SNEPPX_fm_get_params` returns 0
and `SNEPPX_fm_build_train_graph` is a pass-through (`input_var` → `output_var`).

## Synchronization

`algorithms/fm/core/sync.c` implements three sync strategies:

| Method | Function | Pattern |
|--------|----------|---------|
| All-reduce | `SNEPPX_fm_sync_all_reduce` | ring/butterfly all-reduce of memory deltas |
| NCCL | `SNEPPX_fm_sync_nccl` | pluggable callback (`SNEPPXFMSyncCallback`) for a real NCCL process group |
| Gossip | `SNEPPX_fm_sync_gossip` | randomized pairwise exchange |
| Topology | `SNEPPX_fm_sync_topology` | fixed neighbor graph sync |

Gradient exchange helpers: `SNEPPX_fm_send_gradients` /
`SNEPPX_fm_receive_gradients`.

## Compression, privacy, forgetting

- **Compression**: `SNEPPX_fm_compress_gradients(gradients, ratio)` top-k
  sparsifies gradients.
- **Error feedback (EF-SGD)**: `SNEPPX_fm_error_feedback_create(dim, ratio)` +
  `SNEPPX_fm_compress_with_error(ef, gradient)` accumulate the compression
  error so the top-k selection stays accurate across steps.
- **Privacy**: `SNEPPX_fm_add_privacy_noise(data, epsilon)` adds calibrated
  Gaussian noise for (ε)-differential privacy.
- **Forgetting**: `SNEPPX_fm_ewm_update(bank, alpha)` applies an exponential
  moving average to bank entries so old knowledge decays gradually instead of
  being overwritten.
- **Adaptive sync**: `SNEPPX_fm_compute_change_rate(bank, new_values)` +
  `SNEPPX_fm_adaptive_sync_interval(ctrl, base_interval)` shrink the sync
  interval when memory changes rapidly and lengthen it when it is stable.

## Minimal example

```c
#include "fractal_memory_orchestrator.h"

SNEPPXFMConfig cfg = SNEPPX_fm_config_default();
cfg.num_nodes = 4;
cfg.memory_dim = 8; cfg.memory_capacity = 16;

SNEPPXFMController* ctrl = SNEPPX_fm_controller_create(&cfg);
SNEPPXTensor* in = /* build (1, 8) tensor */;
SNEPPXTensor* out = NULL;
SNEPPX_fm_forward(ctrl, 0, in, &out);
SNEPPX_fm_sync_all_reduce(ctrl);   // merge all banks
SNEPPX_fm_controller_destroy(ctrl);
```

## Public API summary

- Lifecycle: `SNEPPX_fm_config_default`, `SNEPPX_fm_memory_bank_create/destroy`,
  `SNEPPX_fm_node_create/destroy`, `SNEPPX_fm_controller_create/destroy`.
- Bank I/O: `SNEPPX_fm_memory_bank_write/read/forget`.
- Sync: `SNEPPX_fm_sync_all_reduce`, `SNEPPX_fm_sync_nccl`,
  `SNEPPX_fm_sync_gossip`, `SNEPPX_fm_sync_topology`,
  `SNEPPX_fm_send_gradients`, `SNEPPX_fm_receive_gradients`.
- Advanced: `SNEPPX_fm_compress_gradients`, `SNEPPX_fm_error_feedback_create/
  destroy`, `SNEPPX_fm_compress_with_error`, `SNEPPX_fm_add_privacy_noise`,
  `SNEPPX_fm_ewm_update`, `SNEPPX_fm_compute_change_rate`,
  `SNEPPX_fm_adaptive_sync_interval`.
- Inference: `SNEPPX_fm_forward`.
- Training: `SNEPPX_fm_get_params`, `SNEPPX_fm_build_train_graph`.
