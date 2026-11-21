# Code Walkthrough — SER (Sparse Expert Routing)

SER is the second stage of the SNEPPX-Alg pipeline. It implements a
Mixture-of-Experts (MoE) layer: a router selects a small subset of experts per
token, and the chosen experts process the token. This keeps compute sparse
while the model capacity stays large.

- Public header: `include/neural_core/architecture/sparse_expert_routing.h`
- Implementation: `algorithms/ser/core/`
- CUDA extensions: `algorithms/ser/cuda/`

## Configuration

```c
SNEPPXSERConfig cfg = SNEPPX_ser_config_default();
cfg.num_experts    = 8;    // total experts in the pool
cfg.num_active     = 2;    // experts actually used per token
cfg.input_dim      = 512;
cfg.expert_dim     = 1024; // hidden width inside each expert MLP
cfg.output_dim     = 512;
cfg.top_k_method   = SNEPPX_TOPK_GREEDY; // or SNEPPX_TOPK_NOISY
cfg.load_balance_coef = 0.01f;           // aux load-balancing weight
cfg.dropout_rate   = 0.1f;
cfg.use_mlp_gater  = 0;    // MLP-based gate vs. linear router
cfg.gater_hidden_dim = 128;
```

## The gating / routing path

Routing happens in `algorithms/ser/core/gater.c` and `route.c`:

1. `SNEPPX_ser_gate_forward(layer, input, &gate_weights, &expert_indices,
   &gate_logits, temperature)` — computes per-token gate logits (via the
   `router`/`router_bias` linear layer, or the optional MLP gate), applies
   softmax, and selects the top-`k` experts.
   - `SNEPPX_TOPK_GREEDY` picks the highest logits.
   - `SNEPPX_TOPK_NOISY` adds training noise to the logits before selection
     (encourages exploration and better load balance).
2. `SNEPPX_ser_route(layer, input, &gate_weights, &expert_indices)` — the
   all-to-all dispatch: tokens are scattered to their selected experts.
   `SNEPPX_ser_route_mlp_gated` is the variant used with `use_mlp_gater`.
3. `SNEPPX_ser_expert_forward(expert, input, output)` — each selected expert
   runs its two-layer MLP (`w1/b1` → activation → `w2/b2`). Activations are
   `SNEPPX_ACT_RELU`, `SNEPPX_ACT_GELU`, or `SNEPPX_ACT_SWISH`.
4. The expert outputs are combined, weighted by `gate_weights`.

The public entry point:

```c
SNEPPXSERModel* model = SNEPPX_ser_model_create(&cfg, seed, num_layers);
SNEPPX_ser_forward(layer, input, &output);
```

## Load balancing

Sparse MoE collapses if all tokens pick the same few experts. SER provides
four auxiliary losses:

| Function | Purpose |
|----------|---------|
| `SNEPPX_ser_load_balance_loss` | coefficient-of-variation style balance across experts |
| `SNEPPX_ser_z_loss` | discourages huge gate logits (z-loss) |
| `SNEPPX_ser_importance_loss` | balances total importance per expert |
| `SNEPPX_ser_aux_load_balance` | standard MoE auxiliary balance loss |
| `SNEPPX_ser_aux_loss` | weighted combination: `load_balance_coef · load_balance + z_loss_coef · z_loss` |

`SNEPPX_ser_expert_capacity_balance` enforces a hard token-per-expert capacity
cap, dropping overflow tokens (used with expert parallelism).

## Training graph

```c
int rc = SNEPPX_ser_build_train_graph(model, tape,
                                      input_var, weight_vars, num_weights,
                                      &output_var);
```

`SNEPPX_ser_get_params()` flattens all expert MLP weights, router weights, and
gater weights in a fixed order for the optimizer.

## Minimal example

```c
#include "sparse_expert_routing.h"

SNEPPXSERConfig cfg = SNEPPX_ser_config_default();
cfg.num_experts = 4; cfg.num_active = 2;
cfg.input_dim = 8; cfg.expert_dim = 16; cfg.output_dim = 8;

SNEPPXSERModel* model = SNEPPX_ser_model_create(&cfg, 42, 1);
SNEPPXTensor* in = /* build (4, 8) token tensor */;
SNEPPXTensor* out = NULL;
SNEPPX_ser_forward(model->layers[0], in, &out);
SNEPPX_ser_model_destroy(model);
```

## Public API summary

- Lifecycle: `SNEPPX_ser_config_default`, `SNEPPX_expert_create/destroy`,
  `SNEPPX_ser_layer_create/destroy`, `SNEPPX_ser_model_create/destroy`.
- Routing: `SNEPPX_ser_route`, `SNEPPX_ser_route_mlp_gated`,
  `SNEPPX_ser_gate_forward`, `SNEPPX_ser_expert_forward`,
  `SNEPPX_ser_forward`.
- Losses: `SNEPPX_ser_load_balance_loss`, `SNEPPX_ser_z_loss`,
  `SNEPPX_ser_importance_loss`, `SNEPPX_ser_aux_load_balance`,
  `SNEPPX_ser_aux_loss`, `SNEPPX_ser_expert_capacity_balance`.
- Training: `SNEPPX_ser_get_params`, `SNEPPX_ser_build_train_graph`.
