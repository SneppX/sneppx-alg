# SNEPPX-Algo Commenting Conventions

This document defines the commenting standard for SNEPPX-Algo. It exists because
this is an open-source codebase: a reader who opens `kyber.c` or `scan.c` should
be able to understand *what* the code does, *why* it works, and *how* it fits the
larger system without first reading a separate design doc.

The standard has four layers. They are listed in priority order.

## Layer 1 — File header block (every source file)

Every `.c`, `.h`, `.cpp`, `.cu`, `.cuh`, `.cl` file starts with a block comment
immediately after the include guard (headers) or after the includes (sources).
It must answer:

- **What**: what this file implements, in one or two sentences.
- **Concept**: the algorithm/idea in plain language (no math notation required).
- **Role**: how it fits in the pipeline (e.g. HSS → SER → ARC → NPE → FM) or in
  the module tree.
- **Callers**: where the public entry points are invoked from (optional if obvious).
- **Standard/Reference**: FIPS/NIST/paper references where relevant.

Template:

```c
/*
 * SNEPPX — <Module Name>
 *
 * WHAT
 *   <one-to-two-sentence summary of what this file implements.>
 *
 * CONCEPT
 *   <plain-language description of the algorithm or idea behind the code.>
 *
 * ROLE
 *   <where this sits in the pipeline or module tree, and what depends on it.>
 *
 * REFERENCES
 *   <FIPS / NIST / paper reference, or "None".>
 */
```

Example from the security domain:

```c
/*
 * SNEPPX — Kyber ML-KEM (Post-Quantum Key Encapsulation)
 *
 * WHAT
 *   Module-lattice (ML-KEM) key generation, encapsulation, and decapsulation
 *   for the FIPS 203 Kyber-512/768/1024 parameter sets.
 *
 * CONCEPT
 *   Kyber is a post-quantum KEM whose security rests on the Module-LWE
 *   problem. Polynomials live in R_q = Z_q[x]/(x^256 + 1) and are multiplied
 *   in the NTT domain for speed. Key generation samples a secret from a
 *   centered binomial distribution; encapsulation wraps an ephemeral key
 *   with a seed; decapsulation unwraps it with the secret key.
 *
 * ROLE
 *   Layer S0 (post-quantum crypto) of the S0–S9 security stack. Used by the
 *   key-vault and secure-transport components for establishing a shared
 *   secret that cannot be broken by a quantum adversary.
 *
 * REFERENCES
 *   FIPS 203 (Kyber), NIST PQC Round 3 finalist.
 */
```

## Layer 2 — Concept blocks before non-obvious algorithms

When a file contains a non-trivial algorithm, add a short block comment **right
before the implementation** explaining the idea in plain language, including the
math only as needed to follow the code. This is different from API docs: it
explains *why the algorithm works*, not *what the function takes*.

Example:

```c
/*
 * Associative scan (Blelloch):
 * A sequential state-space recurrence h_t = A h_{t-1} + B x_t can be
 * parallelized by pairing each step into (A, b) and combining pairs with the
 * associative operation (A2,b2)∘(A1,b1) = (A2·A1, A2·b1 + b2). This lets an
 * entire sequence be reduced in O(log n) parallel steps.
 */
```

Rules for concept blocks:

- Only write one where the surrounding code would otherwise be opaque.
- Keep it short — two to six lines. If it grows, move it to `docs/`.
- Always keep it adjacent to the code it explains; a drifting comment is a lie.

## Layer 3 — Inline "why" comments

Use inline comments only to record **decisions the reader cannot recover from
the code**: why a magic constant is what it is, why an operation is reordered,
why a memory fence is needed, why a fallback path exists.

Good:

```c
// Bounds in log-space to avoid underflow in the softmax denominator.
float m = -FLT_MAX;
```

```c
// delta_0 holds the running offset so each row's shifts accumulate once.
size_t delta_0 = 0;
```

Bad (restates the code — always avoid):

```c
// increment i by one
i++;
```

```c
// allocate memory
float* buf = malloc(n * sizeof(float));
```

## Layer 4 — API documentation (every public `SNEPPX_*` function)

Every function, struct, macro, and enum that is exported (public `SNEPPX_`
symbol, or declared in a public header) carries a brief documentation block.
Use Doxygen `@brief`, `@param`, `@return`, `@note` tags so the docs can be
generated automatically.

Function template:

```c
/**
 * @brief One-line summary of what the function does.
 * @param arg1 What arg1 is and the meaning of its valid range.
 * @param arg2 What arg2 is.
 * @return 0 on success, -1 on error. Optionally: pointer/void semantics.
 * @note Any caveat the caller must know (ownership, aliasing, thread safety).
 */
int SNEPPX_thing(const float* arg1, size_t arg2);
```

Struct template:

```c
/**
 * @brief What the struct represents.
 * @note Ownership rules for any pointers it holds (e.g. "caller frees").
 */
typedef struct {
    ...
} SNEPPXTensor;
```

Macro/enum template (only for non-obvious ones):

```c
/** @brief Variant 768 parameter set (K=3): ~3x Classic McEliece-sized keys. */
#define KYBER_VARIANT_768 3
```

Rules for API docs:

- **Mandatory** for all public symbols in headers and public `.c` entry points.
- `@brief` is required; `@param`/`@return` are required when the function has
  parameters/return values.
- `@note` for ownership transfer, thread-safety, or aliasing caveats.
- A `@param` that merely repeats the parameter name adds noise — write what the
  value means.

## Enforcement

`sneppx-format` enforces Layers 1 and 4 automatically:

- `DOC001` (MEDIUM) — public function without a `@brief` doc block directly
  above it.
- `DOC002` (MEDIUM) — source file missing a Layer-1 header block in the first
  15 lines.

These are advisory in the CLI (`--docs` flag) so legacy code can be brought up
to standard incrementally; new files must pass them before merge.

## Anti-patterns

- **Comment rot**: comment says one thing, code does another. Update both.
- **Chinese-menu comments**: `// calculate sum` above a sum. Never.
- **Comment walls**: 50 lines of prose for 10 lines of code. Split the content
  between the code (short concept) and `docs/` (long form).
- **Unsafe claims in security code**: never write "constant-time" or "secure"
  without the corresponding implementation evidence.
- **Comments with out-of-date references**: update the FIPS/paper reference when
  the implementation changes.
