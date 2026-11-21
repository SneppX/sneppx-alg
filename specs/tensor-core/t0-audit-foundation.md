# T0 — Tensor Core Audit & Foundation

## Objective
Audit all existing tensor code, verify/extend the struct and enums, ensure all declarations exist in the header, add stub implementations for any missing functions, create the progress tracker, and verify the build.

## Critical Rules
- Read every existing tensor file before modifying. Edit, never duplicate.
- If a function or field already exists, do not recreate it. Extend or skip.
- All new code must compile. No broken headers.
- Tests must pass before declaring T0 complete.
- All memory allocation uses `SNEPPX_malloc` / `SNEPPX_free` from `SNEPPX_memory.h`.
- MSVC C11 — no `_Atomic`, no VLAs.

---

## STEP 0 — Audit Existing Tensor Files

### 0.1 Read `include/neural_core/kernel/multidimensional_tensor_engine.h`

Read the header completely. Report:

#### SNEPPXDtype enum
All 13 values must be present:

| Enum Value | Meaning | Byte Size |
|------------|---------|-----------|
| `SNEPPX_FLOAT8` | 8-bit floating point | 1 |
| `SNEPPX_FLOAT16` | 16-bit floating point | 2 |
| `SNEPPX_BFLOAT16` | Brain float 16 | 2 |
| `SNEPPX_FLOAT32` | 32-bit floating point | 4 |
| `SNEPPX_FLOAT64` | 64-bit floating point | 8 |
| `SNEPPX_INT8` | 8-bit signed integer | 1 |
| `SNEPPX_INT16` | 16-bit signed integer | 2 |
| `SNEPPX_INT32` | 32-bit signed integer | 4 |
| `SNEPPX_INT64` | 64-bit signed integer | 8 |
| `SNEPPX_UINT8` | 8-bit unsigned integer | 1 |
| `SNEPPX_BOOL` | Boolean | 1 |
| `SNEPPX_COMPLEX64` | 64-bit complex | 8 |
| `SNEPPX_COMPLEX128` | 128-bit complex | 16 |

If any are missing, add them. The ordering doesn't matter for correctness but should be kept logical (floats grouped, ints grouped, etc.).

#### SNEPPXLayout enum
| Enum Value | Meaning |
|------------|---------|
| `SNEPPX_LAYOUT_ROW_MAJOR` | C-style, last dimension contiguous |
| `SNEPPX_LAYOUT_COL_MAJOR` | Fortran-style, first dimension contiguous |
| `SNEPPX_LAYOUT_CHANNELS_LAST` | Channels-last for image data |
| `SNEPPX_LAYOUT_TILED` | Tiled for cache efficiency |

#### SNEPPXDevice enum
| Enum Value | Meaning |
|------------|---------|
| `SNEPPX_DEVICE_CPU` | Central processing unit |
| `SNEPPX_DEVICE_CUDA` | NVIDIA GPU |
| `SNEPPX_DEVICE_METAL` | Apple GPU via Metal |
| `SNEPPX_DEVICE_VULKAN` | Cross-platform GPU via Vulkan |
| `SNEPPX_DEVICE_TPU` | Google TPU (stub) |
| `SNEPPX_DEVICE_NPU` | Neural processing unit (stub) |

#### SNEPPXTensor struct
All 12 fields must be present:

| Field | Type | Purpose |
|-------|------|---------|
| `data` | `void*` | Pointer to raw data buffer |
| `shape` | `size_t*` | Array of dimension sizes |
| `strides` | `size_t*` | Array of stride values (bytes between elements) |
| `ndim` | `size_t` | Number of dimensions |
| `size` | `size_t` | Total number of elements |
| `item_size` | `size_t` | Bytes per element |
| `dtype` | `SNEPPXDtype` | Data type enum |
| `device` | `SNEPPXDevice` | Device enum |
| `device_id` | `int` | Device index (0 for CPU) |
| `layout` | `SNEPPXLayout` | Memory layout enum |
| `owns_data` | `int` | Whether tensor owns its memory (1 = destroy frees data) |
| `backend_handle` | `void*` | Opaque backend-specific handle (NULL for CPU) |

#### Every function declaration
List every function: its name, parameters, return type. Group by category (creation, shape, math, comparison, reduction, linalg, nn, io, utility). Each declaration must be on its own line ending with semicolon.

#### Every macro
List every `#define` (excluding the include guard). Known macros to add:
- `SNEPPX_DTYPE_SIZE(dtype)` — delegates to `SNEPPX_tensor_dtype_size(dtype)`
- `SNEPPX_DTYPE_IS_FLOAT(dtype)` — expands to 1 for FLOAT32/FLOAT64/FLOAT16/BFLOAT16/FLOAT8
- `SNEPPX_DTYPE_IS_INT(dtype)` — expands to 1 for INT32/INT64/INT16/INT8/UINT8/BOOL
- `SNEPPX_DTYPE_IS_COMPLEX(dtype)` — expands to 1 for COMPLEX64/COMPLEX128

#### Total line count
Report the exact line count of the header.

### 0.2 Read `kernel/tensor/tensor.c`

Read the source completely. Report:

#### Every implemented function
For each function declared in the header, check if it has a **real implementation** (not a stub). A real implementation:
- Validates inputs (null checks, bounds checks)
- Allocates memory via `SNEPPX_malloc` / `aligned_alloc_wrapper`
- Sets struct fields
- Returns result or NULL on error

#### Every stub function
A stub is a function that:
- Returns `NULL` unconditionally (for pointer-returning functions)
- Returns `0` or `0.0f` unconditionally (for value-returning functions)
- Uses `(void)param;` to suppress unused-parameter warnings
- Has a `/* TODO */` or similar comment

Known stubs (from original codebase):
- `SNEPPX_tensor_inverse` — returns NULL
- `SNEPPX_tensor_det` — returns 0.0f
- `SNEPPX_tensor_conv1d` — returns NULL
- `SNEPPX_tensor_conv2d` — returns NULL
- `SNEPPX_tensor_pool1d` — returns NULL
- `SNEPPX_tensor_pool2d` — returns NULL

#### Every helper/static function
List all `static` helper functions:
- `compute_offset` — computes byte offset from n-dimensional indices using strides
- `compute_offset_flat` — computes byte offset from flat index using strides
- `aligned_alloc_wrapper` — calls `SNEPPX_malloc(size, alignment)`
- `SNEPPX_tensor_fill_scalar` — fills entire tensor with a double value, casting per dtype
- `compare_op` — generic comparison operator returning SNEPPX_BOOL tensor
- `cmp_eq`, `cmp_ne`, `cmp_lt`, `cmp_le`, `cmp_gt`, `cmp_ge` — comparison callbacks
- `unary_op_f32` — generic unary f32 operator
- `negate_f32`, `sign_f32` — unary helper callbacks
- `uniform_01` — LCG-based uniform random in [0,1)
- `lcg_state` — global LCG state variable

#### Total line count
Report the exact line count of the source file.

### 0.3 List All Tensor-Related Files

Find every file that references `SNEPPXTensor` or `SNEPPX_tensor`:

- **Header**: `include/neural_core/kernel/multidimensional_tensor_engine.h`
- **Source**: `kernel/tensor/tensor.c`
- **Unit tests**: `tests/unit/test_tensor.c`, `tests/unit/test_tensor_edge.c`
- **Benchmarks**: `tests/benchmark/bench_tensor.c`
- **Other consumers**: autodiff (`ops.c`, `tape.c`, `variable.c`), optimizer (`optimizer.c`), all arch components (HSS, SER, ARC, NPE, FM), all Python bindings, all integration tests

### 0.4 Report Total Existing Tensor LOC

Sum the lines of:
- `multidimensional_tensor_engine.h` (header declarations)
- `tensor.c` (implementations)
- `test_tensor.c` (unit tests)
- `test_tensor_edge.c` (edge case tests)
- `bench_tensor.c` (benchmarks)

Do NOT proceed to Step 1 until this audit is complete and reported to the user.

---

## STEP 1 — Extend SNEPPXTensor Struct

### 1.1 Verify Existing Fields
Read the `SNEPPXTensor` struct definition. Check each of these 12 fields:

```
void* data;            — pointer to raw data
size_t* shape;         — array of dimension sizes
size_t* strides;       — array of stride values
size_t ndim;           — number of dimensions
size_t size;           — total number of elements
size_t item_size;      — bytes per element
SNEPPXDtype dtype;       — data type enum
SNEPPXDevice device;     — device enum
int device_id;         — device index
SNEPPXLayout layout;     — memory layout enum
int owns_data;         — whether tensor owns its memory
void* backend_handle;  — opaque backend-specific handle
```

### 1.2 Add Missing Fields
If any field is missing, add it to the struct. Maintain the existing field ordering (logical grouping: identity → shape → dtype → device → layout → metadata).

### 1.3 Report
After verification: all 12 fields present. Report which (if any) were missing and added.

---

## STEP 2 — Extend SNEPPXDtype Enum + Add Helper Macros

### 2.1 Verify Enum Values
Check that all 13 enum values exist. If any are missing, add them after the existing values. Do not reorder existing values (it changes their integer assignments and breaks binary compatibility).

### 2.2 Add Helper Macros
Insert these macros immediately after the `SNEPPXDtype` enum closing brace `} SNEPPXDtype;`:

```c
#define SNEPPX_DTYPE_SIZE(dtype) SNEPPX_tensor_dtype_size(dtype)

#define SNEPPX_DTYPE_IS_FLOAT(dtype) \
    ((dtype) == SNEPPX_FLOAT32 || (dtype) == SNEPPX_FLOAT64 || \
     (dtype) == SNEPPX_FLOAT16 || (dtype) == SNEPPX_BFLOAT16 || \
     (dtype) == SNEPPX_FLOAT8)

#define SNEPPX_DTYPE_IS_INT(dtype) \
    ((dtype) == SNEPPX_INT32 || (dtype) == SNEPPX_INT64 || \
     (dtype) == SNEPPX_INT16 || (dtype) == SNEPPX_INT8 || \
     (dtype) == SNEPPX_UINT8 || (dtype) == SNEPPX_BOOL)

#define SNEPPX_DTYPE_IS_COMPLEX(dtype) \
    ((dtype) == SNEPPX_COMPLEX64 || (dtype) == SNEPPX_COMPLEX128)
```

These macros enable compile-time dtype classification without switch statements.

---

## STEP 3 — Extend SNEPPXDevice Enum

### 3.1 Verify Enum Values
Check for existing values: `SNEPPX_DEVICE_CPU`, `SNEPPX_DEVICE_CUDA`, `SNEPPX_DEVICE_METAL`, `SNEPPX_DEVICE_VULKAN`.

### 3.2 Add Missing Values
Add these at the end of the enum:

```c
SNEPPX_DEVICE_TPU,   /* Google TPU — stub for future K0 backend */
SNEPPX_DEVICE_NPU,   /* Neural processing unit — stub for future K0 backend */
```

These are placeholders for the K0 backend system that will handle hardware abstraction. They don't need implementations in T0.

---

## STEP 4 — Extend SNEPPXLayout Enum

### 4.1 Verify Enum Values
Check for existing values: `SNEPPX_LAYOUT_ROW_MAJOR`, `SNEPPX_LAYOUT_COL_MAJOR`, `SNEPPX_LAYOUT_CHANNELS_LAST`.

### 4.2 Add Missing Values
Add at the end:

```c
SNEPPX_LAYOUT_TILED,  /* Tiled layout for cache-efficient access patterns */
```

Tiled layout divides the tensor into fixed-size blocks (tiles) that fit in L1 cache. The tile size is backend-defined. Stride calculation for tiled layouts is: `stride[i] = tile_size * (shape[i+1] / tile_size)` for the tiled dimension, with row-major or col-major for the intra-tile layout. For now, TILED is a declaration-only placeholder.

---

## STEP 5 — Add Missing Function Declarations

### 5.1 Verify All Declarations

Cross-reference every function from these categories against the header. For each function, verify:
- The function name exists
- The parameter types match
- The return type matches
- The `const` qualifiers match

### 5.2 Functions to Check

**Creation (8 functions):**
```c
SNEPPXTensor* SNEPPX_tensor_empty(const size_t* shape, size_t ndim, SNEPPXDtype dtype);
SNEPPXTensor* SNEPPX_tensor_zeros(const size_t* shape, size_t ndim, SNEPPXDtype dtype);
SNEPPXTensor* SNEPPX_tensor_ones(const size_t* shape, size_t ndim, SNEPPXDtype dtype);
SNEPPXTensor* SNEPPX_tensor_full(const size_t* shape, size_t ndim, SNEPPXDtype dtype, const void* value);
SNEPPXTensor* SNEPPX_tensor_arange(float start, float stop, float step, SNEPPXDtype dtype);
SNEPPXTensor* SNEPPX_tensor_linspace(float start, float stop, size_t steps, SNEPPXDtype dtype);
SNEPPXTensor* SNEPPX_tensor_eye(size_t n, SNEPPXDtype dtype);
SNEPPXTensor* SNEPPX_tensor_randn(const size_t* shape, size_t ndim, SNEPPXDtype dtype);
```

**Shape manipulation (16 functions):**
```c
SNEPPXTensor* SNEPPX_tensor_copy(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_clone(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_slice(const SNEPPXTensor* src, size_t dim, size_t start, size_t end);
SNEPPXTensor* SNEPPX_tensor_reshape(const SNEPPXTensor* src, const size_t* new_shape, size_t new_ndim);
SNEPPXTensor* SNEPPX_tensor_permute(const SNEPPXTensor* src, const size_t* axes);
SNEPPXTensor* SNEPPX_tensor_expand(const SNEPPXTensor* src, const size_t* new_shape, size_t new_ndim);
SNEPPXTensor* SNEPPX_tensor_squeeze(const SNEPPXTensor* src, size_t dim);
SNEPPXTensor* SNEPPX_tensor_unsqueeze(const SNEPPXTensor* src, size_t dim);
SNEPPXTensor* SNEPPX_tensor_concat(const SNEPPXTensor** tensors, size_t num_tensors, size_t dim);
SNEPPXTensor** SNEPPX_tensor_split(const SNEPPXTensor* src, size_t num_splits, size_t dim);
SNEPPXTensor* SNEPPX_tensor_tile(const SNEPPXTensor* src, const size_t* reps, size_t reps_ndim);
SNEPPXTensor* SNEPPX_tensor_repeat(const SNEPPXTensor* src, size_t repeats, size_t dim);
SNEPPXTensor* SNEPPX_tensor_gather(const SNEPPXTensor* src, size_t dim, const SNEPPXTensor* indices);
SNEPPXTensor* SNEPPX_tensor_scatter(SNEPPXTensor* dest, size_t dim, const SNEPPXTensor* indices, const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_masked_select(const SNEPPXTensor* src, const SNEPPXTensor* mask);
SNEPPXTensor* SNEPPX_tensor_masked_fill(SNEPPXTensor* src, const SNEPPXTensor* mask, const void* value);
SNEPPXTensor* SNEPPX_tensor_where(const SNEPPXTensor* condition, const SNEPPXTensor* x, const SNEPPXTensor* y);
```

**I/O and conversion (6 functions):**
```c
SNEPPXTensor* SNEPPX_tensor_cast(const SNEPPXTensor* src, SNEPPXDtype dtype);
SNEPPXTensor* SNEPPX_tensor_to_device(const SNEPPXTensor* src, SNEPPXDevice device);
SNEPPXTensor* SNEPPX_tensor_to_layout(const SNEPPXTensor* src, SNEPPXLayout layout);
int SNEPPX_tensor_save(const SNEPPXTensor* src, const char* path);     /* stub */
SNEPPXTensor* SNEPPX_tensor_load(const char* path);                    /* stub */
```

**Comparison (6 functions):**
```c
SNEPPXTensor* SNEPPX_tensor_eq(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_ne(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_lt(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_le(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_gt(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_ge(const SNEPPXTensor* a, const SNEPPXTensor* b);
```

**Element-wise math (16 functions):**
```c
SNEPPXTensor* SNEPPX_tensor_add(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_sub(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_mul(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_div(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_pow(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_neg(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_abs(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_sign(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_floor(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_ceil(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_round(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_trunc(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_exp(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_log(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_sqrt(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_sin(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_cos(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_tan(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_asin(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_acos(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_atan(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_sinh(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_cosh(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_tanh(const SNEPPXTensor* src);
```

**Reduction (10 functions):**
```c
SNEPPXTensor* SNEPPX_tensor_sum(const SNEPPXTensor* src, size_t dim);
SNEPPXTensor* SNEPPX_tensor_mean(const SNEPPXTensor* src, size_t dim);
SNEPPXTensor* SNEPPX_tensor_std(const SNEPPXTensor* src, size_t dim);
SNEPPXTensor* SNEPPX_tensor_var(const SNEPPXTensor* src, size_t dim);
float SNEPPX_tensor_min(const SNEPPXTensor* src);
float SNEPPX_tensor_max(const SNEPPXTensor* src);
size_t SNEPPX_tensor_argmin(const SNEPPXTensor* src);
size_t SNEPPX_tensor_argmax(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_cumsum(const SNEPPXTensor* src, size_t dim);
SNEPPXTensor* SNEPPX_tensor_cumprod(const SNEPPXTensor* src, size_t dim);
```

**Linear algebra (5 functions):**
```c
float SNEPPX_tensor_dot(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_matmul(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_transpose(const SNEPPXTensor* src, size_t dim1, size_t dim2);
SNEPPXTensor* SNEPPX_tensor_inverse(const SNEPPXTensor* src);   /* stub */
float SNEPPX_tensor_det(const SNEPPXTensor* src);             /* stub */
```

**Neural network (19 functions):**
```c
SNEPPXTensor* SNEPPX_tensor_softmax(const SNEPPXTensor* src, size_t dim);
SNEPPXTensor* SNEPPX_tensor_log_softmax(const SNEPPXTensor* src, size_t dim);
SNEPPXTensor* SNEPPX_tensor_relu(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_gelu(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_silu(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_sigmoid(const SNEPPXTensor* src);
SNEPPXTensor* SNEPPX_tensor_dropout(const SNEPPXTensor* src, float rate, unsigned int seed);
SNEPPXTensor* SNEPPX_tensor_layer_norm(const SNEPPXTensor* src, const SNEPPXTensor* gamma, const SNEPPXTensor* beta, float eps);
SNEPPXTensor* SNEPPX_tensor_batch_norm(const SNEPPXTensor* src, const SNEPPXTensor* gamma, const SNEPPXTensor* beta, const SNEPPXTensor* running_mean, const SNEPPXTensor* running_var, float eps);
SNEPPXTensor* SNEPPX_tensor_group_norm(const SNEPPXTensor* src, const SNEPPXTensor* gamma, const SNEPPXTensor* beta, size_t num_groups, float eps);
SNEPPXTensor* SNEPPX_tensor_instance_norm(const SNEPPXTensor* src, const SNEPPXTensor* gamma, const SNEPPXTensor* beta, float eps);
SNEPPXTensor* SNEPPX_tensor_embedding(const SNEPPXTensor* weight, const SNEPPXTensor* indices);
SNEPPXTensor* SNEPPX_tensor_cross_entropy(const SNEPPXTensor* pred, const SNEPPXTensor* target);
SNEPPXTensor* SNEPPX_tensor_mse_loss(const SNEPPXTensor* pred, const SNEPPXTensor* target);
SNEPPXTensor* SNEPPX_tensor_mae_loss(const SNEPPXTensor* pred, const SNEPPXTensor* target);
SNEPPXTensor* SNEPPX_tensor_nll_loss(const SNEPPXTensor* pred, const SNEPPXTensor* target);
SNEPPXTensor* SNEPPX_tensor_kl_div(const SNEPPXTensor* pred, const SNEPPXTensor* target);
SNEPPXTensor* SNEPPX_tensor_binary_cross_entropy(const SNEPPXTensor* pred, const SNEPPXTensor* target);
SNEPPXTensor* SNEPPX_tensor_conv1d(const SNEPPXTensor* input, const SNEPPXTensor* kernel, size_t stride, size_t padding);   /* stub */
SNEPPXTensor* SNEPPX_tensor_conv2d(const SNEPPXTensor* input, const SNEPPXTensor* kernel, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w);   /* stub */
SNEPPXTensor* SNEPPX_tensor_pool1d(const SNEPPXTensor* src, size_t kernel_size, size_t stride);   /* stub */
SNEPPXTensor* SNEPPX_tensor_pool2d(const SNEPPXTensor* src, size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w);   /* stub */
```

**Utility (5 functions):**
```c
void SNEPPX_tensor_print(const SNEPPXTensor* tensor);
size_t SNEPPX_tensor_dtype_size(SNEPPXDtype dtype);
size_t SNEPPX_tensor_numel(const SNEPPXTensor* tensor);
int SNEPPX_tensor_is_contiguous(const SNEPPXTensor* tensor);
const char* SNEPPX_tensor_dtype_name(SNEPPXDtype dtype);
```

**Accessors (8 functions):**
```c
float SNEPPX_tensor_get_f32(const SNEPPXTensor* tensor, const size_t* indices);
void SNEPPX_tensor_set_f32(SNEPPXTensor* tensor, const size_t* indices, float value);
double SNEPPX_tensor_get_f64(const SNEPPXTensor* tensor, const size_t* indices);
void SNEPPX_tensor_set_f64(SNEPPXTensor* tensor, const size_t* indices, double value);
int32_t SNEPPX_tensor_get_i32(const SNEPPXTensor* tensor, const size_t* indices);
void SNEPPX_tensor_set_i32(SNEPPXTensor* tensor, const size_t* indices, int32_t value);
int64_t SNEPPX_tensor_get_i64(const SNEPPXTensor* tensor, const size_t* indices);
void SNEPPX_tensor_set_i64(SNEPPXTensor* tensor, const size_t* indices, int64_t value);
uint8_t SNEPPX_tensor_get_bool(const SNEPPXTensor* tensor, const size_t* indices);
void SNEPPX_tensor_set_bool(SNEPPXTensor* tensor, const size_t* indices, uint8_t value);
```

**Lifecycle (2 functions):**
```c
SNEPPXTensor* SNEPPX_tensor_create(const size_t* shape, size_t ndim, SNEPPXDtype dtype);
void SNEPPX_tensor_destroy(SNEPPXTensor* tensor);
```

**Fill (2 functions):**
```c
void SNEPPX_tensor_fill_f32(SNEPPXTensor* tensor, float value);
void SNEPPX_tensor_fill_f64(SNEPPXTensor* tensor, double value);
```

### 5.2 Add Missing Declarations
If any declaration is missing, add it to the header in the appropriate section. Group by category with blank-line separators.

### 5.3 Signature Adaptation
If a function exists with a different signature from the spec:
- Do NOT change the existing signature (it will break callers)
- Report the difference
- Add any genuinely missing functions

---

## STEP 6 — Add Stub Implementations

### 6.1 Identify Missing Implementations
For every function declared in the header, check if it has a body in tensor.c. If a function is declared but NOT implemented, add a stub.

### 6.2 Stub Pattern
Use this exact pattern for stubs:

```c
int SNEPPX_tensor_save(const SNEPPXTensor* src, const char* path) {
    (void)src; (void)path;
    return -1;
}

SNEPPXTensor* SNEPPX_tensor_load(const char* path) {
    (void)path;
    return NULL;
}
```

The stub must:
- Suppress unused-parameter warnings via `(void)param;`
- Return a safe default (NULL for pointers, -1 for error-returning int, 0.0f for float)
- Compile without warnings

### 6.3 File Organization
Add stubs in a dedicated section at the end of tensor.c, with a comment:

```c
/* ============================================================
 * Stub implementations — replaced in later T phases
 * ============================================================ */
```

---

## STEP 7 — Create PROGRESS_TENSOR.md

Create `PROGRESS_TENSOR.md` in the project root:

```markdown
# Tensor Core Progress

Target: 6,000 LOC (existing 2,484 + 3,000 new)
Timeline: 6 phases, 6 days

| Phase | Day | Focus | LOC Target | LOC Actual | Status |
|-------|-----|-------|------------|------------|--------|
| T0 | 1 | Audit & Foundation | 500 | [actual] | Completed |
| T1 | 2 | Creation & Shape | 500 | | Pending |
| T2 | 3 | Element-wise Ops | 500 | | Pending |
| T3 | 4 | Reduction & LA | 500 | | Pending |
| T4 | 5 | NN Ops | 500 | | Pending |
| T5 | 6 | I/O & Tests | 500 | | Pending |

Total: 6,000 LOC

## T0 Deliverables
- [x] Audit existing tensor files ([LOC] LOC across header, source, tests, benchmarks)
- [x] Add 4 dtype helper macros: SNEPPX_DTYPE_SIZE, SNEPPX_DTYPE_IS_FLOAT, SNEPPX_DTYPE_IS_INT, SNEPPX_DTYPE_IS_COMPLEX
- [x] Extend SNEPPXDevice: +SNEPPX_DEVICE_TPU, +SNEPPX_DEVICE_NPU
- [x] Extend SNEPPXLayout: +SNEPPX_LAYOUT_TILED
- [x] Declare SNEPPX_tensor_save / SNEPPX_tensor_load
- [x] Add stub implementations for save/load in tensor.c
- [x] Build passes with zero errors
- [x] All existing tests pass
```

After T0 completes, update `LOC Actual` and mark all checkboxes.

---

## STEP 8 — Build and Verify

### 8.1 Build
```bash
mkdir -p build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DSNEPPX_BUILD_TESTS=ON
cmake --build . -j8
```

Verify:
- Zero compilation errors
- Zero linker errors
- All headers parseable
- No new warnings (pre-existing warnings are acceptable)

### 8.2 Test
```bash
ctest -C Release --output-on-failure
```

Verify:
- All existing tests still pass
- Any NEW tests (if added) pass
- Report total: passed/failed/not-run

### 8.3 Report
After completion, report to the user:
- Audit results: existing tensor LOC split by file, existing functions, existing dtypes, existing struct fields
- Changes made to `multidimensional_tensor_engine.h`: macros added, enums extended, functions declared
- Changes made to tensor.c: stubs added
- Build result: pass/fail, error count, warning count
- Test result: pass/fail counts
- Ready for T1

---

## Known Bugs Fixed During T0

### Bug 1: `SNEPPX_tensor_create` NULL shape + ndim > 0 crashes
The original code didn't check if `shape` is NULL before dereferencing it in the `for` loop that copies shape values. Fix: add early return `if (ndim > 0 && !shape) return NULL;` at the top of `SNEPPX_tensor_create`. Also handle ndim == 0 by allocating safe_ndim = 1 for shape/strides arrays (malloc(0) behavior is implementation-defined).

### Bug 2: `SNEPPX_tensor_eye` always uses float* regardless of dtype
The original code cast `tensor->data` to `float*` always, even for SNEPPX_FLOAT64. Fix: use `unsigned char*` and check dtype to write the correct width. For FLOAT64 use `double`, for integer types use `memset` to 1, for everything else use `float`.

### Bug 3: `SNEPPX_tensor_arange` negative step detection
The original code used `(stop - start) / step < 0` which fails when both numerator and denominator are negative (result is positive). Fix: explicit checks `if ((step > 0 && start >= stop) || (step < 0 && start <= stop)) return NULL;`.
