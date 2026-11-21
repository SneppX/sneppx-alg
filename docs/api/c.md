# C API Reference

Full C API for all SNEPPX-Algo components.

> **Commenting Standard**: All public `SNEPPX_*` functions in this API follow the four-layer commenting standard in [COMMENTING.md](../COMMENTING.md). Every function has a `@brief/@param/@return` Doxygen block, and every source file has a Layer-1 header block.

## Tensor API

**Header**: `#include "SNEPPX_tensor.h"`

### Types

```c
typedef struct SNEPPXTensor SNEPPXTensor;
typedef enum {
    SNEPPX_BOOL,
    SNEPPX_INT8,
    SNEPPX_INT16,
    SNEPPX_INT32,
    SNEPPX_INT64,
    SNEPPX_UINT8,
    SNEPPX_FLOAT16,
    SNEPPX_BFLOAT16,
    SNEPPX_FLOAT32,
    SNEPPX_FLOAT64,
    SNEPPX_FLOAT8,
    SNEPPX_COMPLEX64,
    SNEPPX_COMPLEX128,
} SNEPPXDType;
```

### Creation

```c
SNEPPXTensor* SNEPPX_tensor_create(const size_t* shape, size_t ndim, SNEPPXDtype dtype);
SNEPPXTensor* SNEPPX_tensor_zeros(const size_t* shape, size_t ndim, SNEPPXDtype dtype);
SNEPPXTensor* SNEPPX_tensor_ones(const size_t* shape, size_t ndim, SNEPPXDtype dtype);
SNEPPXTensor* SNEPPX_tensor_full(const size_t* shape, size_t ndim, SNEPPXDtype dtype, double value);
SNEPPXTensor* SNEPPX_tensor_randn(const size_t* shape, size_t ndim, SNEPPXDtype dtype);
SNEPPXTensor* SNEPPX_tensor_from_buffer(const void* data, const size_t* shape, size_t ndim, SNEPPXDtype dtype);
SNEPPXTensor* SNEPPX_tensor_arange(double start, double step, size_t n, SNEPPXDtype dtype);
SNEPPXTensor* SNEPPX_tensor_identity(size_t n, SNEPPXDtype dtype);
SNEPPXTensor* SNEPPX_tensor_copy(const SNEPPXTensor* src);
```

### Shape Operations

```c
SNEPPXTensor* SNEPPX_tensor_reshape(const SNEPPXTensor* a, const size_t* shape, size_t ndim);
SNEPPXTensor* SNEPPX_tensor_transpose(const SNEPPXTensor* a);
SNEPPXTensor* SNEPPX_tensor_permute(const SNEPPXTensor* a, const int32_t* axes, size_t ndim);
SNEPPXTensor* SNEPPX_tensor_expand(const SNEPPXTensor* a, const size_t* shape, size_t ndim);
SNEPPXTensor* SNEPPX_tensor_squeeze(const SNEPPXTensor* a);
SNEPPXTensor* SNEPPX_tensor_unsqueeze(const SNEPPXTensor* a, int32_t axis);
SNEPPXTensor* SNEPPX_tensor_slice(const SNEPPXTensor* a, size_t dim, size_t start, size_t end);
SNEPPXTensor* SNEPPX_tensor_pad(const SNEPPXTensor* a, const size_t* pad_before, const size_t* pad_after, size_t ndim);
```

### Arithmetic

```c
SNEPPXTensor* SNEPPX_tensor_add(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_sub(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_mul(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_div(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_pow(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_neg(const SNEPPXTensor* a);
SNEPPXTensor* SNEPPX_tensor_sqrt(const SNEPPXTensor* a);
```

### Matmul

```c
SNEPPXTensor* SNEPPX_tensor_matmul(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_batch_matmul(const SNEPPXTensor* a, const SNEPPXTensor* b);
```

### Reduction

```c
SNEPPXTensor* SNEPPX_tensor_sum(const SNEPPXTensor* a, size_t dim);
SNEPPXTensor* SNEPPX_tensor_mean(const SNEPPXTensor* a, size_t dim);
SNEPPXTensor* SNEPPX_tensor_var(const SNEPPXTensor* a, size_t dim);
float SNEPPX_tensor_max(const SNEPPXTensor* a);
float SNEPPX_tensor_min(const SNEPPXTensor* a);
size_t SNEPPX_tensor_argmax(const SNEPPXTensor* a);
size_t SNEPPX_tensor_argmin(const SNEPPXTensor* a);
```

### Unary

```c
SNEPPXTensor* SNEPPX_tensor_exp(const SNEPPXTensor* a);
SNEPPXTensor* SNEPPX_tensor_log(const SNEPPXTensor* a);
SNEPPXTensor* SNEPPX_tensor_sigmoid(const SNEPPXTensor* a);
SNEPPXTensor* SNEPPX_tensor_relu(const SNEPPXTensor* a);
SNEPPXTensor* SNEPPX_tensor_gelu(const SNEPPXTensor* a);
SNEPPXTensor* SNEPPX_tensor_silu(const SNEPPXTensor* a);
SNEPPXTensor* SNEPPX_tensor_softmax(const SNEPPXTensor* a, size_t dim);
SNEPPXTensor* SNEPPX_tensor_layer_norm(const SNEPPXTensor* a, const SNEPPXTensor* gamma, const SNEPPXTensor* beta, float eps);
```

---

## Autodiff API

**Header**: `#include "SNEPPX_autodiff.h"`

### Tape

```c
SNEPPXTape* SNEPPX_tape_create(void);
void SNEPPX_tape_destroy(SNEPPXTape* tape);
void SNEPPX_tape_record(SNEPPXTape* tape, SNEPPXVariable* var);
void SNEPPX_tape_backward(SNEPPXTape* tape, SNEPPXVariable* loss);
void SNEPPX_tape_zero_grad(SNEPPXTape* tape);
float SNEPPX_tape_global_norm(SNEPPXTape* tape);
void SNEPPX_tape_clip_grad_norm(SNEPPXTape* tape, float max_norm);
```

### Variable

```c
SNEPPXVariable* SNEPPX_variable_create(SNEPPXTensor* data, int requires_grad);
void SNEPPX_variable_destroy(SNEPPXVariable* var);
void SNEPPX_variable_set_requires_grad(SNEPPXVariable* var, int requires_grad);
SNEPPXVariable* SNEPPX_variable_detach(SNEPPXVariable* var);
SNEPPXVariable* SNEPPX_variable_copy(SNEPPXVariable* var);
void SNEPPX_variable_zero_grad(SNEPPXVariable* var);
float SNEPPX_variable_item(SNEPPXVariable* var);
size_t SNEPPX_variable_numel(SNEPPXVariable* var);
```

### Gradient Ops

```c
SNEPPXVariable* SNEPPX_add(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b);
SNEPPXVariable* SNEPPX_sub(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b);
SNEPPXVariable* SNEPPX_mul(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b);
SNEPPXVariable* SNEPPX_div(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b);
SNEPPXVariable* SNEPPX_matmul(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* b);
SNEPPXVariable* SNEPPX_mse_loss(SNEPPXTape* tape, SNEPPXVariable* pred, SNEPPXVariable* target);
SNEPPXVariable* SNEPPX_relu(SNEPPXTape* tape, SNEPPXVariable* a);
SNEPPXVariable* SNEPPX_sigmoid(SNEPPXTape* tape, SNEPPXVariable* a);
SNEPPXVariable* SNEPPX_tanh(SNEPPXTape* tape, SNEPPXVariable* a);
SNEPPXVariable* SNEPPX_softmax(SNEPPXTape* tape, SNEPPXVariable* a, size_t dim);
SNEPPXVariable* SNEPPX_layer_norm(SNEPPXTape* tape, SNEPPXVariable* a, SNEPPXVariable* gamma, SNEPPXVariable* beta, float eps);
SNEPPXVariable* SNEPPX_dropout(SNEPPXTape* tape, SNEPPXVariable* a, float rate, unsigned int seed);
SNEPPXVariable* SNEPPX_concat(SNEPPXTape* tape, SNEPPXVariable** vars, size_t num_vars, size_t dim);
SNEPPXVariable* SNEPPX_conv2d(SNEPPXTape* tape, SNEPPXVariable* input, SNEPPXVariable* kernel, size_t stride, size_t pad);
void SNEPPX_no_grad_enter(void);
void SNEPPX_no_grad_exit(void);
```

---

## Optimizer API

**Header**: `#include "SNEPPX_optimizer.h"`

```c
typedef enum {
    SNEPPX_OPTIMIZER_SGD,
    SNEPPX_OPTIMIZER_ADAM,
    SNEPPX_OPTIMIZER_ADAMW,
    SNEPPX_OPTIMIZER_RMSPROP,
    SNEPPX_OPTIMIZER_ADAGRAD,
} SNEPPXOptimizerType;

SNEPPXOptimizerConfig SNEPPX_optimizer_config_default(void);
SNEPPXOptimizer* SNEPPX_optimizer_create(const SNEPPXOptimizerConfig* config);
void SNEPPX_optimizer_destroy(SNEPPXOptimizer* opt);
void SNEPPX_optimizer_step(SNEPPXOptimizer* opt, SNEPPXTensor** params, SNEPPXTensor** grads, size_t num_params);

// Factory functions
SNEPPXOptimizer* SNEPPX_sgd_create(float lr, float momentum, float weight_decay);
SNEPPXOptimizer* SNEPPX_adam_create(float lr, float beta1, float beta2, float eps, float weight_decay);
SNEPPXOptimizer* SNEPPX_adamw_create(float lr, float beta1, float beta2, float eps, float weight_decay);
SNEPPXOptimizer* SNEPPX_rmsprop_create(float lr, float alpha, float eps, float momentum, float weight_decay);
SNEPPXOptimizer* SNEPPX_adagrad_create(float lr, float eps, float weight_decay);

// LR Schedulers
SNEPPXLRScheduler* SNEPPX_lr_scheduler_step_lr(float* lr_ptr, float gamma, size_t step_size);
SNEPPXLRScheduler* SNEPPX_lr_scheduler_cosine(float* lr_ptr, float min_lr, float max_lr, size_t total_steps);
void SNEPPX_lr_scheduler_destroy(SNEPPXLRScheduler* sched);
void SNEPPX_lr_scheduler_step(SNEPPXLRScheduler* sched, float current_loss);
```

---

## Algorithm APIs

### HSS

**Header**: `#include "SNEPPX_hss.h"`

```c
SNEPPXHSSConfig SNEPPX_hss_config_default(void);
SNEPPXHSSLayer* SNEPPX_hss_layer_create(const SNEPPXHSSConfig* config, unsigned int seed);
void SNEPPX_hss_layer_destroy(SNEPPXHSSLayer* layer);
SNEPPXHSSModel* SNEPPX_hss_model_create(const SNEPPXHSSConfig* config, unsigned int seed);
void SNEPPX_hss_model_destroy(SNEPPXHSSModel* model);
int SNEPPX_hss_forward(SNEPPXHSSModel* model, const SNEPPXTensor* input, SNEPPXTensor** output);
int SNEPPX_hss_build_train_graph(SNEPPXHSSModel* model, SNEPPXTape* tape, SNEPPXVariable* input, SNEPPXVariable** params, int n_params, SNEPPXVariable** output);
SNEPPXTensor** SNEPPX_hss_get_params(SNEPPXHSSModel* model, int* n);
```

### SER

**Header**: `#include "SNEPPX_ser.h"`

```c
SNEPPXSERConfig SNEPPX_ser_config_default(void);
SNEPPXExpert* SNEPPX_expert_create(const SNEPPXSERConfig* config, unsigned int seed, SNEPPXActivation activation);
void SNEPPX_expert_destroy(SNEPPXExpert* expert);
SNEPPXSERLayer* SNEPPX_ser_layer_create(const SNEPPXSERConfig* config, unsigned int seed);
void SNEPPX_ser_layer_destroy(SNEPPXSERLayer* layer);
int SNEPPX_ser_forward(SNEPPXSERLayer* layer, const SNEPPXTensor* input, SNEPPXTensor** output);
float SNEPPX_ser_load_balance_loss(const SNEPPXSERLayer* layer, const SNEPPXTensor* routing_weights);
int SNEPPX_ser_route_mlp_gated(SNEPPXSERLayer* layer, const SNEPPXTensor* input, SNEPPXTensor** routing_weights);
SNEPPXSERModel* SNEPPX_ser_model_create(const SNEPPXSERConfig* config, unsigned int seed, size_t num_layers);
void SNEPPX_ser_model_destroy(SNEPPXSERModel* model);
```

### ARC

**Header**: `#include "SNEPPX_arc.h"`

```c
SNEPPXARCConfig SNEPPX_arc_config_default(void);
SNEPPXARCLayer* SNEPPX_arc_layer_create(const SNEPPXARCConfig* config, size_t input_dim, size_t output_dim, unsigned int seed);
void SNEPPX_arc_layer_destroy(SNEPPXARCLayer* layer);
int SNEPPX_arc_forward(SNEPPXARCLayer* layer, const SNEPPXTensor* input, SNEPPXTensor** output, float metrics[4]);
float SNEPPX_arc_guard_score(const SNEPPXARCLayer* layer, const SNEPPXTensor* input);
float SNEPPX_arc_verify_score(const SNEPPXARCLayer* layer, const SNEPPXTensor* output);
int SNEPPX_arc_simulate_attack(SNEPPXARCLayer* layer, const SNEPPXTensor* input, int attack_type, float eps, SNEPPXTensor** adversarial);
int SNEPPX_arc_build_adversarial_train_graph(SNEPPXARCModel* model, SNEPPXTape* tape,
    SNEPPXVariable* input, SNEPPXVariable** weights, int n_weights,
    SNEPPXVariable** clean_out, SNEPPXVariable** adv_out);
```

### NPE

**Header**: `#include "SNEPPX_npe.h"`

```c
SNEPPXNPEConfig SNEPPX_npe_config_default(void);
SNEPPXNPEProgram* SNEPPX_npe_program_create(size_t max_instructions);
void SNEPPX_npe_program_destroy(SNEPPXNPEProgram* prog);
void SNEPPX_npe_program_append(SNEPPXNPEProgram* prog, SNEPPXNPEInstruction inst);
SNEPPXNPEVM* SNEPPX_npe_vm_create(const SNEPPXNPEConfig* config);
void SNEPPX_npe_vm_destroy(SNEPPXNPEVM* vm);
void SNEPPX_npe_vm_load(SNEPPXNPEVM* vm, SNEPPXNPEProgram* program);
int SNEPPX_npe_vm_run(SNEPPXNPEVM* vm, const SNEPPXTensor* input, SNEPPXTensor** output);
int SNEPPX_npe_vm_step(SNEPPXNPEVM* vm, const SNEPPXTensor* input, SNEPPXTensor** output);
int SNEPPX_npe_vm_optimize(SNEPPXNPEVM* vm);

// JIT pipeline
SNEPPXNPEProgram* SNEPPX_npe_jit_pipeline_compose(SNEPPXNPEProgram* input);
int SNEPPX_npe_jit_dce(SNEPPXNPEProgram* prog);
int SNEPPX_npe_jit_constant_fold(SNEPPXNPEProgram* prog);
int SNEPPX_npe_jit_fuse(SNEPPXNPEProgram* prog);

// Compilers
SNEPPXNPEProgram* SNEPPX_npe_compile_mlp(size_t input_dim, size_t hidden_dim);
SNEPPXNPEProgram* SNEPPX_npe_compile_attention(size_t dim);
int SNEPPX_npe_verify_program(const SNEPPXNPEProgram* prog);
```

### FM

**Header**: `#include "SNEPPX_fm.h"`

```c
SNEPPXFMConfig SNEPPX_fm_config_default(void);
SNEPPXFMMemoryBank* SNEPPX_fm_memory_bank_create(size_t capacity, size_t dim);
void SNEPPX_fm_memory_bank_destroy(SNEPPXFMMemoryBank* bank);
int SNEPPX_fm_memory_bank_query(SNEPPXFMMemoryBank* bank, const SNEPPXTensor* key, SNEPPXTensor** value, float* similarity);
int SNEPPX_fm_memory_bank_write(SNEPPXFMMemoryBank* bank, const SNEPPXTensor* key, const SNEPPXTensor* value);
SNEPPXFMNode* SNEPPX_fm_node_create(size_t node_id, size_t mem_capacity, size_t mem_dim);
void SNEPPX_fm_node_destroy(SNEPPXFMNode* node);
SNEPPXFMController* SNEPPX_fm_controller_create(const SNEPPXFMConfig* config);
void SNEPPX_fm_controller_destroy(SNEPPXFMController* ctrl);
int SNEPPX_fm_sync_all_reduce(SNEPPXFMController* ctrl, SNEPPXFMNode** nodes, size_t num_nodes, float privacy_epsilon);
int SNEPPX_fm_sync_nccl(SNEPPXFMController* ctrl, SNEPPXFMSyncCallback callback, void* context);
SNEPPXTensor* SNEPPX_fm_compress_gradients(const SNEPPXTensor* grad, float compression_ratio);
int SNEPPX_fm_forward(SNEPPXFMController* ctrl, size_t node_id, const SNEPPXTensor* input, SNEPPXTensor** output);
```

### Model Zoo

**Header**: `#include "model_zoo.h"`

```c
typedef enum { SNEPPX_MODEL_LLAMA_2 = 0, SNEPPX_MODEL_LLAMA_3 = 1,
               SNEPPX_MODEL_MISTRAL = 2, SNEPPX_MODEL_QWEN_2 = 3,
               SNEPPX_MODEL_DEEPSEEK_V2 = 4, SNEPPX_MODEL_UNKNOWN = 255
} SNEPPXModelFamily;

// Config structs: SNEPPXLlamaConfig, SNEPPXMistralConfig,
//                 SNEPPXQwen2Config, SNEPPXDeepSeekV2Config
// Wrapper:        SNEPPXLLMConfig (tagged union)

// Create preset config by family + size (e.g. "llama3", "8B")
int SNEPPX_llm_config_from_name(const char* family, const char* size,
                                 SNEPPXLLMConfig* out);

// Serialize to JSON (caller must free)
char* SNEPPX_llm_config_to_json(const SNEPPXLLMConfig* cfg);

// Parse from JSON
int SNEPPX_llm_config_from_json(const char* json, SNEPPXLLMConfig* out);

// Extend context to target_len using NTK-aware/YaRN scaling
int SNEPPX_llm_config_extend_context(SNEPPXLLMConfig* cfg, size_t target_len);

// Weight name mapping
const char* SNEPPX_llm_weight_prefix(SNEPPXModelFamily family);
int SNEPPX_llm_num_weight_tensors(SNEPPXModelFamily family, size_t num_layers);
```

### CUDA Trainer Bridge

**Header**: `#include "trainer_cuda.h"`

```c
int SNEPPX_trainer_cuda_available(void);
int SNEPPX_trainer_cuda_init(size_t mem_pool_size);
void SNEPPX_trainer_cuda_shutdown(void);
int SNEPPX_trainer_cuda_transfer(void* dst, const void* src, size_t bytes);
int SNEPPX_trainer_cuda_to_host(void* dst, const void* src, size_t bytes);
int SNEPPX_trainer_cuda_optimizer_step(float* params, const float* grads, size_t n, float lr, float weight_decay);
```

---

## Utility API

**Header**: `#include "SNEPPX_memory.h"`

```c
void* SNEPPX_malloc(size_t size, size_t alignment);
void SNEPPX_free(void* ptr, size_t size);
void* SNEPPX_calloc(size_t count, size_t size, size_t alignment);
void SNEPPX_secure_zero(void* ptr, size_t size);
```
