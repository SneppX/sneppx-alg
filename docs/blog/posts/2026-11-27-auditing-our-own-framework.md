---
date: 2026-11-27
categories:
  - Security
  - Engineering
description: "We audited SneppX-ALG's own security and training code and found 68 issues, including 7 critical. Here is what we found, what we fixed, and what is still open."
slug: auditing-our-own-framework
---

# We audited our own ML framework and found 68 problems

On November 22, 2026 we ran a structured security and correctness review of
SneppX-ALG at commit `db256e27`. It found **68 issues: 7 Critical, 16 High,
26 Medium/Low, and 19 correctness bugs** in training code. This post explains
what went wrong, what we changed, and what we are still checking.

<!-- more -->

## Why publish this

SneppX-ALG includes experimental security components alongside its tensor and
autograd engine. Security code is only useful if people can see how it was
tested and where it failed. Publishing the audit lets you judge the project on
evidence instead of feature lists.

## What the audit found

The most serious problems were not immature features. They were placeholders
that *looked* like working security controls:

- **Simulation-grade crypto presented as real.** A KEM fallback returned
  unrelated random secrets, an AEAD fallback returned unencrypted output, and
  one transport mode labelled AES-GCM was a repeating-key XOR with no MAC.
- **Verification that did not verify.** A container signature check ignored
  the public key and accepted an empty signature, and the Python update path
  never enforced its own `verified` flag.
- **Fail-open defaults.** An inference server skipped its authentication setup,
  and the model hub shipped with default admin credentials.
- **Silent training errors.** Learning-rate schedules never reached the
  optimizer, multi-rank collectives were no-ops, and some code paths produced
  wrong results instead of raising an error.

## What we changed

The follow-up addendum on November 23 records all seven Critical findings as
resolved, with each mapped to its remediation commit. A second pass on
November 24 changed every remaining placeholder so that it either **fails
closed** (raises an error instead of pretending to succeed) or is clearly
documented as non-security code.

We also changed how the project describes itself:

- The README, documentation home page, and `llms.txt` now state the maturity
  of each area plainly: Alpha, not a drop-in PyTorch replacement, and not a
  certified security product.
- Distributed training now fails explicitly when `WORLD_SIZE` is greater than
  one, instead of appearing to train with unsynchronized gradients.
- Post-quantum code is documented with its known interoperability gaps, and no
  FIPS validation is claimed.

## What is still open

The Medium/Low findings and several numerical-correctness items still need
individual verification against current source and tests. The fixes listed
above are our own follow-up work, not an independent audit. We will publish
updates in the [audit archive](../../security/audits/index.md) as items are
confirmed.

## Read the details

- [Security audit archive](../../security/audits/index.md), including the full
  November 22 report
- [Security overview](../../security.md) and
  [security layer limitations](../../security_layers.md)
- [Project scope and limitations](../../why-sneppx-alg.md)

If you find a security issue, please report it privately by following the
[security policy](https://github.com/SneppX/sneppx-alg/blob/main/SECURITY.md).
