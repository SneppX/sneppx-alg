# Code Walkthrough — HSS (Hierarchical State Space)

HSS is the first stage of the SNEPPX-Alg pipeline. It maps input sequences to a
latent state using a recurrent state-space model (SSM) inspired by structured
state-space sequence models such as Mamba/S6.

- Public header: `include/neural_core/architecture/hierarchical_state_space.h`
- Implementation: `algorithms/hss/core/`
- CUDA extensions: `algorithms/hss/cuda/`

## The model

An HSS model is a stack of layers, each with a recurrent state and a set of
SSM parameters:

| Field | Meaning |
|-------|---------|
| `A` | state transition matrix |
| `B` | input → state projection |
| `C` | state → output projection |
| `D` | feedthrough (skip) term |
| `dt` | per-step sampling interval |
| `x_proj`, `x_proj_bias` | input projection (input_dim → 2·state_dim) |
| `h` | the current recurrent state |
| `A_bar`, `B_bar` | discretized (ZOH or bilinear) versions of A and B |
| `norm_gamma`, `norm_beta` | per-layer normalization parameters |

## Configuration

```c
SNEPPXHSSConfig cfg = SNEPPX_hss_config_default();
cfg.state_dim         = 64;   // latent state width per layer
cfg.input_dim         = 512;  // input feature width
cfg.output_dim        = 512;  // output feature width
cfg.num_layers        = 2;
cfg.seq_len           = 1024;
cfg.dt_min            = 0.001f;
cfg.dt_max            = 0.1f;
cfg.use_hierarchical  = 1;    // multi-resolution state levels
cfg.use_parallel_scan = 1;    // associative scan instead of loop
```

`SNEPPX_hss_config_default()` returns state_dim=64, input/output 512,
2 layers, seq_len 1024, and hierarchical + parallel scan enabled.

## Forward path

The forward pass is the sequence of calls in `algorithms/hss/core/forward.c`:

1. `SNEPPX_hss_discretize(layer)` — converts continuous `(A, B, dt)` into the
   discrete `(A_bar, B_bar)` used for step-by-step recurrence. Two underlying
   methods exist:
   - `SNEPPX_hss_discretize_zoh()` — zero-order hold.
   - `SNEPPX_hss_discretize_bilinear()` — bilinear (Tustin) transform.
2. `SNEPPX_hss_scan(layer, x_seq, h_seq, y_seq)` — sequential scan over the
   input sequence, computing hidden states `h_t` and outputs `y_t` step by
   step.
3. `SNEPPX_hss_parallel_scan(layer, x_seq, h_seq, y_seq)` — associative-scan
   variant that computes the same recurrence in parallel (much faster on long
   sequences). Used when `use_parallel_scan` is set.
4. `SNEPPX_hss_hierarchical_scan(layer, x_seq, y_seq)` — multi-resolution scan
   used when `use_hierarchical` is set. The state space is decomposed into
   `SNEPPX_hss_hierarchical_levels(state_dim, min_dim)` coarse-to-fine levels.

The public entry point is:

```c
SNEPPXHSSModel* model = SNEPPX_hss_model_create(&cfg, seed);
int rc = SNEPPX_hss_forward(model, input_tensor, &output_tensor);
```

`SNEPPX_hss_step(layer, x, h_next)` performs a single-token recurrent update
and is used for autoregressive generation.

## Training graph

Training uses the autodiff framework:

```c
SNEPPXTape* tape = /* created by the trainer */;
SNEPPXVariable* input_var, **weight_vars;
SNEPPXVariable* output_var;
int rc = SNEPPX_hss_build_train_graph(model, tape,
                                      input_var, weight_vars, num_weights,
                                      &output_var);
```

Parameters are flattened in a fixed order by `SNEPPX_hss_get_params()` (see the
9-parameter table in [hss_training.md](hss_training.md)).

## Minimal example

```c
#include "hierarchical_state_space.h"

SNEPPXHSSConfig cfg = SNEPPX_hss_config_default();
cfg.state_dim = 4; cfg.input_dim = 4; cfg.output_dim = 4;
cfg.num_layers = 1; cfg.seq_len = 8;

SNEPPXHSSModel* model = SNEPPX_hss_model_create(&cfg, 42);
SNEPPXTensor* in = /* build (8, 4) input tensor */;
SNEPPXTensor* out = NULL;
SNEPPX_hss_forward(model, in, &out);
SNEPPX_hss_model_destroy(model);
```

## Public API summary

- Lifecycle: `SNEPPX_hss_config_default`, `SNEPPX_hss_layer_create/destroy`,
  `SNEPPX_hss_model_create/destroy`.
- Discretization: `SNEPPX_hss_discretize`, `SNEPPX_hss_discretize_zoh`,
  `SNEPPX_hss_discretize_bilinear`.
- Recurrence: `SNEPPX_hss_step`, `SNEPPX_hss_scan`,
  `SNEPPX_hss_parallel_scan`, `SNEPPX_hss_hierarchical_scan`,
  `SNEPPX_hss_hierarchical_levels`.
- Inference: `SNEPPX_hss_forward`.
- Training: `SNEPPX_hss_get_params`, `SNEPPX_hss_build_train_graph`.
