# Code Walkthrough — ARC (Adversarial Robustness Certification)

ARC is the third stage of the SNEPPX-Alg pipeline. It hardens the model
against adversarial inputs by combining three defense mechanisms: an input
guard, gradient obfuscation, and output verification. It also simulates
adversarial attacks (FGSM/PGD) so the model can be adversarially trained.

- Public header: `include/neural_core/architecture/adversarial_robustness_certification.h`
- Implementation: `algorithms/arc/core/`
- CUDA extensions: `algorithms/arc/cuda/`

## Configuration

```c
SNEPPXARCConfig cfg = SNEPPX_arc_config_default();
cfg.input_guard_strength        = 0.1f;
cfg.gradient_obfuscation_method = SNEPPX_OBF_NOISE; // NONE|NOISE|CLAMP|MIXED
cfg.gradient_noise_scale        = 1e-4f;
cfg.gradient_clip_max           = 1.0f;
cfg.output_verify_layers        = 2;
cfg.output_verify_threshold     = 0.95f;
cfg.adversarial_training        = 1;
cfg.attack_simulation_types     = SNEPPX_ATTACK_FGSM | SNEPPX_ATTACK_PGD;
cfg.attack_epsilon              = 0.01f;
```

## The three defense mechanisms

An ARC layer (`SNEPPXARCLayer`) is composed of three sub-objects:

### 1. Input guard (`SNEPPXInputGuard`)

`SNEPPX_arc_input_guard_forward(guard, input, &sanitized, &anomaly_score)`
projects the input onto a learned subspace (`projection_matrix`) and scores how
anomalous it is against training statistics (`norm_stats_mean`/`var`).
Anomalous samples are sanitized before they reach the rest of the model.

### 2. Gradient obfuscator (`SNEPPXGradientObfuscator`)

`SNEPPX_arc_obfuscate_gradients(obf, gradients, method)` hides gradient signal
from attackers during training. Four methods:

| Method | Behavior |
|--------|----------|
| `SNEPPX_OBF_NONE` | no modification |
| `SNEPPX_OBF_NOISE` | add Gaussian noise (scale `gradient_noise_scale`) |
| `SNEPPX_OBF_CLAMP` | clip to `gradient_clip_max` |
| `SNEPPX_OBF_MIXED` | noise + clamp |

### 3. Output verifier (`SNEPPXOutputVerifier`)

`SNEPPX_arc_verify_output(verifier, output, &verified_output, &confidence)`
runs the output through a small verification network and checks consistency
against a history of recent outputs. Low-confidence or inconsistent outputs are
corrected before being returned.

## Attack simulation

`SNEPPX_arc_simulate_attack(clean_input, attack_type, epsilon, &adversarial)`
generates adversarial examples:

- `SNEPPX_ATTACK_FGSM` — one-step sign-gradient perturbation.
- `SNEPPX_ATTACK_PGD` — iterative projected gradient descent.
- `SNEPPX_ATTACK_CW` — Carlini–Wagner style attack (declared; bit flag 4).

Attack types are bit flags, so `SNEPPX_ATTACK_FGSM | SNEPPX_ATTACK_PGD`
enables both.

## Forward and adversarial training

The plain forward pass:

```c
SNEPPXARCLayer* layer = SNEPPX_arc_layer_create(&cfg, input_dim, output_dim, seed);
float security_metrics = 0.0f;
SNEPPX_arc_forward(layer, input, &output, &security_metrics);
```

`security_metrics` aggregates the guard's anomaly score, obfuscation activity,
and verifier confidence so callers can monitor defensive posture.

Two training graphs exist:

```c
// Standard graph: input -> guard -> obfuscated training
int rc = SNEPPX_arc_build_train_graph(layer, tape, input_var,
                                      weight_vars, num_weights, &output_var);

// Adversarial graph: builds BOTH clean and adversarial branches so the
// loss can combine clean accuracy with robustness.
int rc2 = SNEPPX_arc_build_adversarial_train_graph(layer, tape, input_var,
                                        weight_vars, num_weights,
                                        &clean_output, &adv_output);
```

## Minimal example

```c
#include "adversarial_robustness_certification.h"

SNEPPXARCConfig cfg = SNEPPX_arc_config_default();
cfg.attack_epsilon = 0.05f;

SNEPPXARCLayer* layer = SNEPPX_arc_layer_create(&cfg, 8, 8, 42);
SNEPPXTensor* clean = /* build (4, 8) input */;
SNEPPXTensor* adv = NULL;
SNEPPX_arc_simulate_attack(clean, SNEPPX_ATTACK_PGD, 0.05f, &adv);
SNEPPXTensor* out = NULL;
float sec = 0.0f;
SNEPPX_arc_forward(layer, adv, &out, &sec);
SNEPPX_arc_layer_destroy(layer);
```

## Public API summary

- Lifecycle: `SNEPPX_arc_config_default`, `SNEPPX_input_guard_create/destroy`,
  `SNEPPX_gradient_obfuscator_create/destroy`,
  `SNEPPX_arc_output_verifier_create/destroy`, `SNEPPX_arc_layer_create/destroy`.
- Defense: `SNEPPX_arc_input_guard_forward`, `SNEPPX_arc_obfuscate_gradients`,
  `SNEPPX_arc_verify_output`, `SNEPPX_arc_forward`.
- Attacks: `SNEPPX_arc_simulate_attack`.
- Training: `SNEPPX_arc_get_params`, `SNEPPX_arc_build_train_graph`,
  `SNEPPX_arc_build_adversarial_train_graph`.
