# Python API — Auto-generated Reference

This page is produced automatically from the `SneppX_ALG` source by
[`mkdocstrings`](https://mkdocstrings.github.io/) (Google docstring style). It
mirrors the curated [Python API](python.md) guide and is always in sync with the
installed package.

The C backend (`_SNEPPX_c` / `_arix_c`) is optional: wrappers use native
compute where an implementation is available and otherwise fall back to the
pure-NumPy engine. The post-quantum crypto implementations are experimental
and are not yet fully interoperable with finalized FIPS 203–205. See
[`docs/pqc_kat.md`](../pqc_kat.md) for measured ACVP coverage and known
signature-format gaps. Release/update signing currently uses Ed25519.

For the full method-level reference, see the curated
[Python API](python.md) guide. The blocks below surface each public class's
docstring and constructor signature.

## Core tensors & autodiff

::: SneppX_ALG.Tensor
    :members: false

::: SneppX_ALG.Linear
    :members: false

::: SneppX_ALG.Sequential
    :members: false

## Optimizers & training

::: SneppX_ALG.AdamW
    :members: false

::: SneppX_ALG.Optimizer
    :members: false

## Pipeline components

::: SneppX_ALG.HSSModel
    :members: false

::: SneppX_ALG.SERModel
    :members: false

::: SneppX_ALG.ARCModel
    :members: false

::: SneppX_ALG.QuantizedLinear
    :members: false

## Persistence & observability

::: SneppX_ALG.CheckpointWriter
    :members: false

::: SneppX_ALG.Profiler
    :members: false

## Model zoo

::: SneppX_ALG.from_pretrained
    :members: false
