# Testing Framework

## Overview

The sneppx-alg test suite uses the CTest framework (via CMake) with custom test harness macros defined in `tests/test_harness.h`. Tests cover all 5 supported architectures and 5 sneppx primitives.

## Test organization

```
tests/
├── test_harness.h              # Shared macros (ASSERT_EQ, TEST, etc.)
├── test_context_extension.c    # 128K context extension (all architectures)
├── test_mha_forward.c          # MHA forward pass (all architectures)
├── test_model_zoo.c            # Model zoo presets, JSON, weights
├── test_llama.c / test_llama3.c
├── test_mistral.c / test_qwen2.c / test_deepseek_v2.c
├── test_hss.c / test_ser.c / test_arc.c / test_npe.c / test_fm.c
└── ...
```

## Running tests

```bash
cmake --preset release
cmake --build build --config Release
cd build && ctest -C Release --output-on-failure
```

## Test macros

| Macro                     | Description                      |
|---------------------------|----------------------------------|
| `ASSERT_EQ(a, b)`         | Assert a == b                    |
| `ASSERT_NEAR(a, b, eps)`  | Assert |a - b| < eps              |
| `ASSERT_NULL(p)`          | Assert p == NULL                 |
| `ASSERT_NOT_NULL(p)`      | Assert p != NULL                 |
| `ASSERT_STREQ(s1, s2)`    | Assert string equality           |
| `ASSERT_SUCCESS(ret)`     | Assert ret == 0                  |
| `ASSERT_FAILURE(ret)`     | Assert ret != 0                  |
| `ASSERT_TRUE(cond)`       | Assert cond is truthy            |
| `ASSERT_FALSE(cond)`      | Assert cond is falsy             |

## Architecture test pattern

Tests for each architecture follow an `ARCH_test_*` naming convention with automatic fixture setup/teardown:

```c
static SNEPPXLlamaConfig LLLAMA_CFG;

// Called before each Llama test
void LLAMA_setup(void) {
    SNEPPX_llm_config_from_name("llama3", "8B", &LLAMA_CFG);
}

void LLAMA_test_forward(void) {
    // ... test body ...
}
```

## What's tested

| Feature                   | File                          | Architectures          |
|---------------------------|-------------------------------|------------------------|
| 128K context extension    | test_context_extension.c      | LLaMA2/3, Mistral, Qwen2, DeepSeek |
| MHA forward pass          | test_mha_forward.c            | LLaMA2/3, Mistral, Qwen2, DeepSeek |
| Model zoo presets         | test_model_zoo.c              | LLaMA2/3, Mistral, Qwen2, DeepSeek |
| JSON serialization        | test_model_zoo.c              | All                    |
| Weight name mapping       | test_model_zoo.c              | All                    |
