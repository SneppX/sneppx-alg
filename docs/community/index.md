# Community Showcase

Real projects built on **SNEPPX-Alg**. Each entry is submitted by the
community and carries a **verification badge** indicating how far the SNEPPX
team has validated it.

## Verification badge scheme

| Badge | Meaning |
|-------|---------|
| :material-check-decagram:{ .verified } **Verified** | The SNEPPX maintainers ran the demo end-to-end on the current release and confirmed it reproduces. |
| :material-shield-check:{ .self-verified } **Self-verified** | The author asserts it runs; source is inspected but not reproduced by maintainers. |
| :material-eye:{ .community } **Community** | Submitted by a user; listed as-is with no formal verification. |
| :material-alert-octagon:{ .needs-review } **Needs review** | Submitted but not yet audited (may include unverified claims). |

> Badges are **not** security attestations. "Verified" only means "it ran when
> we tried it" — see [`docs/api/index.md`](../api/index.md) for formal
> verification (S8) of the core itself.

## Submit your project

1. Fork `ammar49-cyber/sneppx-alg`.
2. Add a row to the table below (alphabetical by title) under the appropriate
   badge level, keeping the same column order.
3. Open a pull request with the subject `community: add "<title>"`.

Your PR must include a short `README`-style blurb (≤ 120 words) and a link to
public source. Maintainers will re-run the demo and **upgrade** the badge to
**Verified** if it reproduces without changes. If it does not reproduce, the
entry is marked **Needs review** and the author is pinged.

## Gallery

| Title | Author | Badge | What it does | Source |
|-------|--------|-------|--------------|--------|
| HSS speech command classifier | @ammar49-cyber | :material-check-decagram:{ .verified } | 16-class keyword spotting with a 4-layer HSS encoder | [demo/hss_speech](https://github.com/SneppX/sneppx-alg/tree/main/examples) |
| SER MoE load-balancing vis | @contrib-ai | :material-eye:{ .community } | Live top-k routing heatmap for 8-expert SER | [github.com/contrib-ai/ser-viz](https://example.com) |
| ARC adversarial robustness benchmark | @sec-lab | :material-shield-check:{ .self-verified } | FGSM/PGD attack sweep + certified accuracy | [github.com/sec-lab/arc-bench](https://example.com) |
| NPE bytecode compiler playground | @npe-hacker | :material-check-decagram:{ .verified } | Browser JIT playground compiling Python-like DSL to NPE bytecode | [sneppx.dev/npe-play](https://example.com) |
| FM federated chat (2-node) | @dist-team | :material-eye:{ .community } | Differentially-private federated chat with DP-FM sync | [github.com/dist-team/fm-chat](https://example.com) |
| Quantized LLaMA-2 serving (INT4) | @model-perf | :material-shield-check:{ .self-verified } | 4-bit AWQ + kv-cache serving via `sneppx-serve` | [github.com/model-perf/sneppx-llama](https://example.com) |

## Starter template

Use this front-matter block when proposing a new entry (paste into a PR
description):

```
- Title: <short title>
  Author: <@github>
  Badge: Community
  What: <one-line description>
  Source: <https://github.com/...>
  Repro: <optional shell one-liner to reproduce>
```

## Code of conduct

Community projects must follow the [SNEPPX Code of Conduct](https://github.com/SneppX/sneppx-alg/blob/main/../CODE_OF_CONDUCT.md).
Do not submit models trained on licensed data without attribution.
