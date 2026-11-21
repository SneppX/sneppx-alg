# T1 — Tensor Core Creation & Shape Manipulation

## Objective
Implement creation functions (empty, zeros, ones, full, arange, linspace, eye, randn) and shape manipulation functions (copy, clone, slice, reshape, permute, expand, squeeze, unsqueeze, concat, split, tile, repeat, gather, scatter, masked_select, masked_fill, where), with comprehensive tests.

## Critical Rules
- Read existing code first. Edit, never duplicate.
- If a function already has a real implementation, do NOT rewrite it — just add a test for it if none exists.
- All new code must compile. Tests must pass.
- All memory: `SNEPPX_malloc` / `SNEPPX_free` from `SNEPPX_memory.h`.
- MSVC C11 — no `_Atomic`, no VLAs.
- Pre-existing tests must still pass after changes.
- `SNEPPX_tensor_destroy` must handle NULL safely (it already does: `free(tensor->shape); ...; free(tensor)`).
- The `test_tensor_edge.c` file (57 edge-case tests) must be preserved without modification.

---

## STEP 0 — Verify T0 Completion

### 0.1 Check Requirements
Before starting T1, verify:
1. `SNEPPX_tensor.h` has all 13 dtypes, 4 layouts, 6 devices, all 4 dtype macros (`SNEPPX_DTYPE_SIZE`, `SNEPPX_DTYPE_IS_FLOAT`, `SNEPPX_DTYPE_IS_INT`, `SNEPPX_DTYPE_IS_COMPLEX`)
2. `SNEPPX_tensor.h` has declarations for all 80+ functions (creation, shape, math, comparison, reduction, linalg, nn, io, utility)
3. `tensor.c` has `SNEPPX_tensor_save` and `SNEPPX_tensor_load` as stubs
4. `PROGRESS_TENSOR.md` exists
5. Build passes with zero errors
6. 48/50 tests pass (2 pre-existing crypto failures in test_argon2, test_ed25519)

If any of these fail, fix them before starting T1.

### 0.2 Read Existing Implementations
Read `tensor.c` to see which creation and shape functions already have real implementations (not stubs). This is critical — the original codebase had real implementations for ALL 8 creation functions and ALL 17 shape functions. Do NOT rewrite them.

For each function, determine:
- Does it exist and have a real body?
- Does it pass basic validation (null checks, bounds checks)?
- Are there any obvious bugs?

---

## STEP 1 — Creation Functions

### 1.1 Signature Reference (all exist in header)

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

### 1.2 Internal Helper: `SNEPPX_tensor_fill_scalar`

Add this static helper function to tensor.c. It fills an entire tensor with a given double value, handling all 13 dtypes by casting/writing the correct width.

**Implementation requirements:**
- Parameters: `SNEPPXTensor* t`, `double value`
- Uses `unsigned char*` base pointer
- For float types (FLOAT32, FLOAT64, FLOAT16, BFLOAT16, FLOAT8): write the cast value at the correct byte width
- For int types (INT32, INT64, INT16, INT8): write the cast value
- For UINT8: write `(uint8_t)value`
- For BOOL: write `value != 0.0`
- For COMPLEX64: write (float)value as real, 0.0f as imag (8 bytes total per element)
- For COMPLEX128: write (double)value as real, 0.0 as imag (16 bytes total per element)
- Iterates over all `t->size` elements using byte pointer arithmetic: `base + i * item_size`

```c
static void SNEPPX_tensor_fill_scalar(SNEPPXTensor* t, double value) {
    if (!t || !t->data) return;
    unsigned char* data = (unsigned char*)t->data;
    size_t count = t->size;
    size_t sz = t->item_size;
    SNEPPXDtype dt = t->dtype;

    for (size_t i = 0; i < count; i++) {
        unsigned char* dst = data + i * sz;
        switch (dt) {
            case SNEPPX_FLOAT32:
                *(float*)dst = (float)value;
                break;
            case SNEPPX_FLOAT64:
                *(double*)dst = value;
                break;
            case SNEPPX_FLOAT16:
            case SNEPPX_BFLOAT16:
                /* 16-bit floats: store lower 16 bits of float representation */
                {
                    float f = (float)value;
                    uint16_t h;
                    /* For BFLOAT16: truncate upper 16 bits */
                    memcpy(&h, (uint8_t*)&f + (dt == SNEPPX_BFLOAT16 ? 2 : 0), 2);
                    memcpy(dst, &h, 2);
                }
                break;
            case SNEPPX_FLOAT8:
                /* 8-bit float: clamp to [-2, 2) range and encode as 1.4.3 */
                {
                    float clamped = (float)fmax(-2.0, fmin(1.9375, value));
                    uint8_t e4m3;
                    /* Convert to E4M3 format: sign(1) + exp(4) + mant(3) */
                    int sign = clamped < 0 ? 1 : 0;
                    float abs_v = fabsf(clamped);
                    int exp = 0;
                    while (abs_v >= 2.0f) { abs_v /= 2.0f; exp++; }
                    while (abs_v < 1.0f && exp > -7) { abs_v *= 2.0f; exp--; }
                    uint8_t mant = (uint8_t)((abs_v - 1.0f) * 8.0f);
                    e4m3 = (sign << 7) | ((exp + 7) << 3) | (mant & 0x7);
                    memcpy(dst, &e4m3, 1);
                }
                break;
            case SNEPPX_INT32:
                *(int32_t*)dst = (int32_t)value;
                break;
            case SNEPPX_INT64:
                *(int64_t*)dst = (int64_t)value;
                break;
            case SNEPPX_INT16:
                *(int16_t*)dst = (int16_t)value;
                break;
            case SNEPPX_INT8:
                *(int8_t*)dst = (int8_t)value;
                break;
            case SNEPPX_UINT8:
                *dst = (uint8_t)value;
                break;
            case SNEPPX_BOOL:
                *dst = (value != 0.0) ? 1 : 0;
                break;
            case SNEPPX_COMPLEX64:
                *(float*)dst = (float)value;
                *((float*)dst + 1) = 0.0f;
                break;
            case SNEPPX_COMPLEX128:
                *(double*)dst = value;
                *((double*)dst + 1) = 0.0;
                break;
        }
    }
}
```

Note: The FLOAT8 and BFLOAT16 conversions above are simplified. The actual implementation in the codebase may differ. The key requirement is that all 13 dtypes produce correct output for the fill operation.

### 1.3 Verify and Fix Existing Implementations

Read the existing implementations of each creation function. Fix any bugs found:

#### `SNEPPX_tensor_create`
- **Bug**: When `shape` is NULL and `ndim > 0`, dereferences NULL pointer.
- **Fix**: Add null check: `if (ndim > 0 && !shape) return NULL;`
- **Bug**: When `ndim == 0`, `malloc(0)` for shape/strides is implementation-defined.
- **Fix**: Use `safe_ndim = (ndim == 0) ? 1 : ndim` for allocation. Store 0 as the actual ndim.
- **Memory**: Must allocate shape + strides arrays via `SNEPPX_malloc`. Must zero-initialize data via `memset(data, 0, total_bytes)`.

#### `SNEPPX_tensor_empty`
- Calls `SNEPPX_tensor_create`, then does NOT initialize data buffer (hence "empty" — data is uninitialized).
- **Validation**: Should check `shape != NULL` and `ndim > 0`.

#### `SNEPPX_tensor_zeros`
- Calls `SNEPPX_tensor_create`, which already zeroes the data via `memset`. This is correct.

#### `SNEPPX_tensor_ones`
- Calls `SNEPPX_tensor_create`, then fills all elements with 1.0f via `SNEPPX_tensor_fill_f64` or manual loop. Verify dtype handling.

#### `SNEPPX_tensor_full`
- Calls `SNEPPX_tensor_create`, then fills all elements with the provided value pointer. The value pointer is cast to the appropriate dtype. Verify it handles all dtypes.

#### `SNEPPX_tensor_arange`
- **Bug**: The negative-step detection uses `(stop - start) / step < 0` which is wrong when both numerator and denominator are negative (result is positive from C's integer division).
- **Fix**:
  ```c
  if ((step > 0 && start >= stop) || (step < 0 && start <= stop)) return NULL;
  ```
- **Validation**: If step is 0, return NULL. Support positive and negative steps.
- The number of elements = `(size_t)((stop - start) / step)`, adjusted for floating-point precision (add small epsilon like `1e-12`).

#### `SNEPPX_tensor_linspace`
- Creates tensor with `steps` elements. Steps must be >= 2 (or handle steps=1 gracefully).
- Fills linearly from start to stop inclusive.
- **Validation**: `steps < 2` return NULL (or handle steps == 1 as a single-element tensor).

#### `SNEPPX_tensor_eye`
- **Bug**: Always writes `float*` regardless of dtype. For FLOAT64, writes 4 bytes instead of 8, corrupting the buffer.
- **Fix**: Use `unsigned char*` base pointer. Write `(int32_t)1` for integer types, `(double)1.0` for FLOAT64, `(float)1.0f` for float types, etc.
- For BOOL, write 1 byte with value 1.
- For COMPLEX64, write 1.0f as real, 0.0f as imag.
- For COMPLEX128, write 1.0 as real, 0.0 as imag.

#### `SNEPPX_tensor_randn`
- Uses a LCG (`uniform_01`) to generate uniform random numbers in [0,1), then applies Box-Muller transform for normal distribution.
- **Validation**: shape == NULL or ndim == 0 returns NULL.
- **Box-Muller**: `z0 = sqrt(-2 * ln(u1)) * cos(2 * pi * u2)`.

### 1.4 Fix Bug 1: `SNEPPX_tensor_create` NULL shape + ndim > 0

In `SNEPPX_tensor_create`, before the shape-copy loop:

```c
if (ndim > 0 && !shape) return NULL;

size_t safe_ndim = (ndim == 0) ? 1 : ndim;
/* Use safe_ndim for allocation of shape/strides */
tensor->shape = (size_t*)SNEPPX_malloc(safe_ndim * sizeof(size_t));
if (!tensor->shape) { SNEPPX_free(tensor); return NULL; }
```

### 1.5 Fix Bug 2: `SNEPPX_tensor_eye` dtype handling

Use the `SNEPPX_tensor_fill_scalar` pattern: iterate with byte pointer arithmetic, switch on dtype to write the correct width.

### 1.6 Fix Bug 3: `SNEPPX_tensor_arange` negative step detection

Replace the ambiguous division with explicit sign checks.

---

## STEP 2 — Shape Manipulation Functions

### 2.1 Signature Reference

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

### 2.2 Implementation Details

#### `SNEPPX_tensor_copy`
- Creates a new tensor with same shape, dtype, device, layout as src.
- Deep-copies data via `memcpy(dst->data, src->data, src->size * src->item_size)`.
- Does NOT copy `owns_data` or `backend_handle`.
- **Validation**: src != NULL, src->data != NULL.

#### `SNEPPX_tensor_clone`
- Same as copy, but additionally copies `owns_data = 1` and `backend_handle = NULL`.
- Semantically: clone produces an independent tensor object.

#### `SNEPPX_tensor_slice`
- Extracts a contiguous sub-tensor along one dimension.
- Creates new tensor with ndim == src->ndim (same rank).
- New shape: same as src, but shape[dim] = (end - start).
- Data: copies the slice from src's data buffer using stride-based offset.
- **Validation**: dim < ndim, start < end, end <= shape[dim].

#### `SNEPPX_tensor_reshape`
- Returns a view (new tensor header) with different shape but shared data buffer.
- The new tensor has the same data pointer as src (no copy).
- `owns_data` is set to 0 on the new tensor so destroy doesn't free src's data.
- **Validation**: total elements must match (`numel(new_shape) == src->size`). new_ndim > 0.

#### `SNEPPX_tensor_permute`
- Transposes dimensions according to axes array.
- Creates new tensor with reordered shape AND data copied in the new order.
- **Validation**: ndim of axes matches src->ndim. Each axis in [0, ndim). No duplicate axes.

#### `SNEPPX_tensor_expand`
- Broadcasts size-1 dimensions to larger sizes (like NumPy broadcasting).
- Creates new tensor with expanded shape. Does NOT copy data (shares buffer).
- `owns_data` = 0 on result.
- **Validation**: new_ndim >= src->ndim. For each src dim, either src->shape[i] == new_shape[...] or src->shape[i] == 1.

#### `SNEPPX_tensor_squeeze`
- Removes a dimension of size 1.
- Result ndim = src->ndim - 1 (or returns copy if dim != 1).
- Shares data buffer (`owns_data` = 0).
- **Validation**: dim < ndim.

#### `SNEPPX_tensor_unsqueeze`
- Adds a dimension of size 1 at position dim.
- Result ndim = src->ndim + 1.
- Shares data buffer (`owns_data` = 0).
- **Validation**: dim <= ndim (can add at end: dim == ndim).

#### `SNEPPX_tensor_concat`
- Concatenates multiple tensors along one dimension.
- Creates new tensor with sum of sizes along dim.
- All tensors must have same ndim and same dtype AND same shape except along dim.
- Copies data from each tensor into the result buffer.
- **Validation**: num_tensors >= 2, all tensors non-NULL, dtype matches, ndim matches, shapes match except at dim.

#### `SNEPPX_tensor_split`
- Split is the inverse of concat.
- Returns an array of `num_splits` tensors.
- If `src->shape[dim]` is not evenly divisible by `num_splits`, the last split gets the remainder.
- SNEPPXTensor** array is `SNEPPX_malloc`'d. Each element is a new tensor (owns_data = 1).
- **Validation**: num_splits > 0, num_splits <= shape[dim], dim < ndim.

#### `SNEPPX_tensor_tile`
- Repeats the entire tensor along multiple dimensions (like NumPy `tile`).
- New shape = `src->shape[i] * reps[i]` for each dimension i.
- The reps array can have fewer dimensions than src (prepend 1s).
- Creates new tensor with copied data (owns_data = 1).
- **Validation**: reps != NULL, all reps > 0.

#### `SNEPPX_tensor_repeat`
- Repeats along a single dimension (like PyTorch `repeat_interleave`).
- Each element along dim is repeated `repeats` times consecutively.
- Creates new tensor with copied data.
- **Validation**: repeats > 0, dim < ndim.

#### `SNEPPX_tensor_gather`
- Gathers values from src along dim using indices (like PyTorch `torch.gather`).
- indices shape must match src shape in all dims except dim.
- Result shape = indices shape.
- For each index tuple: `result[i][j]...[k] = src[i][j]...[indices[i][j]...[k]]...[k]`.
- **Validation**: indices->ndim == src->ndim. indices->shape == src->shape except at dim.

#### `SNEPPX_tensor_scatter`
- In-place scatter: `dest[indices[...]] = src[...]`.
- dest is modified in place and returned.
- Same shape rules as gather (indices shape matches dest shape except at dim).
- **Validation**: same as gather.

#### `SNEPPX_tensor_masked_select`
- Returns a 1D tensor of values where mask is true (nonzero).
- Creates new tensor. Result size = count of nonzero mask elements.
- **Validation**: mask->shape == src->shape, same dtype for mask (must be BOOL or non-zero = true).

#### `SNEPPX_tensor_masked_fill`
- In-place: sets elements of src to value where mask is true.
- Returns the src pointer (not a new tensor). IMPORTANT: the caller should NOT destroy both.
- **Validation**: mask->shape == src->shape.

#### `SNEPPX_tensor_where`
- Element-wise select: where condition is true, take from x, else from y.
- All three tensors must have the same shape.
- Creates new tensor.
- **Validation**: condition->shape == x->shape == y->shape, all non-NULL.

---

## STEP 3 — Test: Creation

### 3.1 File: `tests/unit/test_tensor_creation.c`

Create a new test file. Must include:
```c
#include "SNEPPX_tensor.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
```

Test framework conventions:
- Each test is a `void test_xxx(void)` function.
- `assert(condition)` for pass/fail.
- Clean up: destroy all created tensors after each test.
- Print test name + status with `printf` and `++pass` / `++fail`.
- Return count of failures.
- `main` returns 0 if all pass, 1 if any fail.

### 3.2 Test Cases (19 tests)

| # | Test Name | What It Tests |
|---|-----------|---------------|
| 1 | `test_empty_basic` | Create 3x4 F32 empty, check ndim=2, shape matches, size=12, item_size=4 |
| 2 | `test_empty_1d` | Create 1D tensor, size matches |
| 3 | `test_empty_ndim0` | Create scalar (0-dim) tensor, size=1 |
| 4 | `test_empty_null_shape` | SNEPPX_tensor_create(NULL, 0, SNEPPX_FLOAT32) does not crash, returns valid tensor |
| 5 | `test_zeros_basic` | Create 2x3 zeros, verify all elements == 0.0f |
| 6 | `test_ones_basic` | Create 2x3 ones F32, verify all elements == 1.0f |
| 7 | `test_ones_f64` | Create 2x3 ones F64, verify all elements == 1.0 |
| 8 | `test_ones_i32` | Create 2x3 ones INT32, verify all elements == 1 |
| 9 | `test_full_basic` | Full with value 42.0f, verify all elements == 42.0f |
| 10 | `test_arange_basic` | arange(0, 10, 1), verify 10 elements, 0..9 |
| 11 | `test_arange_negative_step` | arange(10, 0, -2), verify 5 elements: 10,8,6,4,2 |
| 12 | `test_arange_float` | arange(0.0f, 1.0f, 0.2f), verify approximately correct |
| 13 | `test_arange_invalid` | arange(0, 10, -1) returns NULL |
| 14 | `test_linspace_basic` | linspace(0, 1, 5), verify 5 elements: 0, 0.25, 0.5, 0.75, 1.0 |
| 15 | `test_linspace_f64` | linspace(0.0, 1.0, 3) with F64 dtype |
| 16 | `test_eye_basic` | eye(3) F32, verify 3x3 with 1.0f on diagonal |
| 17 | `test_eye_f64` | eye(3) F64, verify 1.0 on diagonal |
| 18 | `test_randn_basic` | randn(1000,), verify mean ~0, std ~1 |
| 19 | `test_eye_i32` | eye(3) INT32, verify 1 on diagonal, 0 elsewhere |

### 3.3 Helper: Create tensor and verify all elements

For verifying all elements, use typed accessors:

```c
static int check_f32(const SNEPPXTensor* t, size_t i, float expected, float tol) {
    size_t indices[] = {i}; /* For 1D */
    float actual = SNEPPX_tensor_get_f32(t, indices);
    return fabsf(actual - expected) < tol;
}
```

For ND tensors, compute indices manually:
```c
static void flat_to_indices(size_t flat_idx, const size_t* shape, size_t ndim, size_t* indices) {
    for (size_t d = ndim; d > 0; d--) {
        indices[d - 1] = flat_idx % shape[d - 1];
        flat_idx /= shape[d - 1];
    }
}
```

### 3.4 Edge Cases in Tests
- `test_empty_null_shape`: ndim=0, shape=NULL should work.
- `test_arange_invalid`: step in wrong direction returns NULL.
- `test_arange_negative_step`: verify negative step works with F32.
- ndim=0 (scalar) creation: `SNEPPX_tensor_create(NULL, 0, SNEPPX_FLOAT32)`.

---

## STEP 4 — Test: Shape Manipulation

### 4.1 File: `tests/unit/test_tensor_shape.c`

Same framework as test_tensor_creation.c.

### 4.2 Test Cases (29 tests)

| # | Test Name | What It Tests |
|---|-----------|---------------|
| 1 | `test_copy_basic` | Copy F32 tensor, verify data matches, check shape, dtype |
| 2 | `test_copy_independent` | Modify original, verify copy unchanged |
| 3 | `test_clone_basic` | Clone tensor, verify fields match |
| 4 | `test_slice_1d` | Slice [2:5] from 1D tensor of size 7, verify 3 elements: 2,3,4 |
| 5 | `test_slice_2d` | Slice dim=0, start=1, end=3 from 4x3 tensor, verify 2x3 result |
| 6 | `test_slice_invalid` | start >= end returns NULL |
| 7 | `test_reshape_basic` | Reshape 1x12 to 3x4, verify shape and data sharing |
| 8 | `test_reshape_element_mismatch` | Reshape to wrong total size returns NULL |
| 9 | `test_permute_basic` | Permute 2x3x4 with axes {2,0,1}, verify new shape 4x2x3 |
| 10 | `test_permute_invalid_axes` | Duplicate axes returns NULL |
| 11 | `test_expand_basic` | Expand 1x3 to 2x3, verify data sharing |
| 12 | `test_expand_invalid` | Non-1 dim to different size returns NULL |
| 13 | `test_squeeze_basic` | Squeeze 1x3x1x4 to 3x4 |
| 14 | `test_squeeze_non_one` | Squeeze dim=1 from 3x4 (dim=1 is 4, not 1) shouldn't change |
| 15 | `test_unsqueeze_basic` | Unsqueeze 3x4 at dim=1 to 3x1x4 |
| 16 | `test_concat_2d` | Concatenate two 2x3 tensors along dim=0 to get 4x3 |
| 17 | `test_concat_invalid_shape` | Mismatched non-concat dim returns NULL |
| 18 | `test_split_equal` | Split 2x6 into 3 along dim=1, verify 3 tensors of 2x2 |
| 19 | `test_split_uneven` | Split 2x7 into 3 along dim=1, verify sizes 2,2,3 |
| 20 | `test_tile_basic` | Tile 1x2 with reps {2,3} to get 2x6 |
| 21 | `test_repeat_basic` | Repeat [1,2,3] along dim=0 with repeats=2 → [1,1,2,2,3,3] |
| 22 | `test_gather_basic` | Gather from 2x3 tensor using indices |
| 23 | `test_scatter_basic` | Scatter into zeros tensor using indices |
| 24 | `test_masked_select_basic` | Select from [1,2,3,4] with mask [1,0,1,0] → [1,3] |
| 25 | `test_masked_fill_basic` | Fill [1,2,3,4] with mask [0,1,0,1] value 99 → [1,99,3,99] |
| 26 | `test_masked_fill_returns_src` | Verify result pointer == src pointer |
| 27 | `test_where_basic` | where([1,0,1], [10,20,30], [40,50,60]) → [10,50,30] |
| 28 | `test_split_invalid` | num_splits > shape[dim] returns NULL |
| 29 | `test_concat_single` | Concatenate single tensor returns copy |

### 4.3 Test Data Patterns
For tests that need specific tensor data, create helpers:

```c
static SNEPPXTensor* make_test_tensor_1d(size_t n) {
    size_t shape = n;
    SNEPPXTensor* t = SNEPPX_tensor_empty(&shape, 1, SNEPPX_FLOAT32);
    if (!t) return NULL;
    for (size_t i = 0; i < n; i++) {
        size_t idx = i;
        SNEPPX_tensor_set_f32(t, &idx, (float)i);
    }
    return t;
}
```

### 4.4 Edge Cases
- `test_expand_invalid`: confirm NULL return
- `test_split_uneven`: check last tensor gets remainder
- `test_masked_fill_returns_src`: critical — prevents double-free
- `test_squeeze_non_one`: returns same-size tensor (or copy with same shape)
- `test_concat_single`: degenerate case, should return a copy

---

## STEP 5 — Build and Verify

### 5.1 Build
```bash
mkdir -p build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DSNEPPX_BUILD_TESTS=ON
cmake --build . -j8
```

**Requirements:**
- Zero compilation errors
- Zero linker errors
- All new files compiled and linked
- No new warnings (pre-existing warnings acceptable)

### 5.2 Test
```bash
ctest -C Release --output-on-failure
```

**Requirements:**
- All 48 new tests pass (19 creation + 29 shape)
- All existing tests still pass (48 original after T0 + 2 pre-existing failures = 50/52)
- Final result: 98/100 total tests passing
- The 2 pre-existing crypto failures (test_argon2, test_ed25519) remain as known failures

### 5.3 Verify Specifics
- Run the edge test file separately to confirm no regressions in existing 57 edge tests
- Run the basic test file to confirm no regressions
- Run benchmarks to confirm no crashes

### 5.4 Fix Any Failures
If a test fails:
1. Check the test logic first — is the expected value wrong?
2. Check the implementation — is there a bug in the function?
3. Fix bugs in the implementation (not in the test, unless test is wrong)
4. Rebuild and retest

---

## STEP 6 — Update PROGRESS

### 6.1 Update PROGRESS_TENSOR.md

```markdown
| T1 | 2 | Creation & Shape | 500 | [actual LOC] | Completed |
```

Checklist:
- [x] All creation functions working (empty, zeros, ones, full, arange, linspace, eye, randn)
- [x] All shape functions working (copy, clone, slice, reshape, permute, expand, squeeze, unsqueeze, concat, split, tile, repeat, gather, scatter, masked_select, masked_fill, where)
- [x] 19 creation tests pass
- [x] 29 shape tests pass
- [x] All pre-existing tests still pass
- [x] Build passes with zero errors
- [x] No new warnings

### 6.2 Update README

If the README lists test counts, update:
- "48/50 tests pass" → "98/100 tests pass" (48 existing + 48 new, minus 2 pre-existing failures)
- Update total ops count if any were added

---

## Known Bugs Fixed During T1

### Bug 1: `SNEPPX_tensor_create` NULL shape + ndim > 0
The original code in `SNEPPX_tensor_create` didn't check if shape is NULL when ndim > 0. This caused a null-pointer dereference in the for-loop that copies shape values.

**Fix**: Add `if (ndim > 0 && !shape) return NULL;` before the allocation.

### Bug 2: `SNEPPX_tensor_create` malloc(0) for ndim == 0
When ndim == 0 (scalar), the code called `malloc(0)` for shape and strides arrays, which has implementation-defined behavior (may return NULL or a non-NULL pointer, but with zero usable size).

**Fix**: Use `safe_ndim = (ndim == 0) ? 1 : ndim` for the allocation sizes, while storing the actual ndim as 0 in the struct. This ensures shape[0] and strides[0] are safe to access.

### Bug 3: `SNEPPX_tensor_eye` dtype width
The original code always cast `tensor->data` to `float*` and wrote 1.0f. For FLOAT64, this wrote only 4 bytes instead of 8, partially corrupting adjacent values.

**Fix**: Use `unsigned char*` base pointer and switch on dtype to write the correct width. Use `memset(dst, 0, item_size * n)` for off-diagonal zeros (handles all widths uniformly), then write 1.0 at the width of each dtype on the diagonal.

### Bug 4: `SNEPPX_tensor_arange` negative step detection
The expression `(stop - start) / step < 0` fails when both numerator and denominator are negative, because C's integer division produces a positive result from two negative operands.

**Fix**: Check sign explicitly: `if ((step > 0 && start >= stop) || (step < 0 && start <= stop)) return NULL;`

### Bug 5: `SNEPPX_tensor_ones` / `SNEPPX_tensor_full` dtype width
These functions called `SNEPPX_tensor_fill_f32` which always wrote float* regardless of dtype. This had the same width problem as eye for FLOAT64.

**Fix**: The `SNEPPX_tensor_fill_scalar` helper handles all 13 dtypes by switching on dtype and writing at the correct byte width.
