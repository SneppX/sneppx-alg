# SNEPPX-ALG API Quick Reference

## C API (Core Engine)

### Tensor

```c
SNEPPXTensor* SNEPPX_tensor_create(const size_t* shape, size_t ndim, SNEPPXDtype dtype);
SNEPPXTensor* SNEPPX_tensor_zeros(const size_t* shape, size_t ndim, SNEPPXDtype dtype);
SNEPPXTensor* SNEPPX_tensor_ones(const size_t* shape, size_t ndim, SNEPPXDtype dtype);
SNEPPXTensor* SNEPPX_tensor_randn(const size_t* shape, size_t ndim, SNEPPXDtype dtype);
void          SNEPPX_tensor_destroy(SNEPPXTensor* t);

// Ops
SNEPPXTensor* SNEPPX_tensor_add(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_matmul(const SNEPPXTensor* a, const SNEPPXTensor* b);
SNEPPXTensor* SNEPPX_tensor_relu(const SNEPPXTensor* t);
SNEPPXTensor* SNEPPX_tensor_softmax(const SNEPPXTensor* t, size_t dim);
SNEPPXTensor* SNEPPX_tensor_reshape(const SNEPPXTensor* t, const size_t* shape, size_t ndim);
SNEPPXTensor* SNEPPX_tensor_transpose(const SNEPPXTensor* t, size_t dim1, size_t dim2);

// I/O
int           SNEPPX_tensor_save(const SNEPPXTensor* t, const char* path);
SNEPPXTensor* SNEPPX_tensor_load(const char* path);
```

### Autograd

```c
SNEPPXVariable* SNEPPX_variable_create(SNEPPXTensor* data, int requires_grad);
void            SNEPPX_variable_destroy(SNEPPXVariable* v);
SNEPPXTape*     SNEPPX_tape_create();
void            SNEPPX_tape_backward(SNEPPXTape* tape, SNEPPXVariable* loss);
void            SNEPPX_tape_zero_grad(SNEPPXTape* tape);

// Gradient ops
SNEPPXVariable* SNEPPX_add(SNEPPXTape*, SNEPPXVariable*, SNEPPXVariable*);
SNEPPXVariable* SNEPPX_matmul(SNEPPXTape*, SNEPPXVariable*, SNEPPXVariable*);
SNEPPXVariable* SNEPPX_mse_loss(SNEPPXTape*, SNEPPXVariable*, SNEPPXVariable*);
SNEPPXVariable* SNEPPX_relu(SNEPPXTape*, SNEPPXVariable*);
SNEPPXVariable* SNEPPX_layer_norm(SNEPPXTape*, SNEPPXVariable*, SNEPPXVariable*, SNEPPXVariable*, float);
SNEPPXVariable* SNEPPX_dropout(SNEPPXTape*, SNEPPXVariable*, float, unsigned int);
```

### Optimizer

```c
SNEPPXOptimizer* SNEPPX_sgd_create(float lr, float momentum, float weight_decay);
SNEPPXOptimizer* SNEPPX_adam_create(float lr, float beta1, float beta2, float eps, float weight_decay);
SNEPPXOptimizer* SNEPPX_adamw_create(float lr, float beta1, float beta2, float eps, float weight_decay);
void             SNEPPX_optimizer_step(SNEPPXOptimizer*, SNEPPXTensor** params, SNEPPXTensor** grads, size_t n);
```

### HSS

```c
SNEPPXHSSConfig SNEPPX_hss_config_default();
SNEPPXHSSModel* SNEPPX_hss_model_create(const SNEPPXHSSConfig* cfg, unsigned int seed);
int             SNEPPX_hss_forward(SNEPPXHSSModel*, const SNEPPXTensor* input, SNEPPXTensor** output);
int             SNEPPX_hss_build_train_graph(SNEPPXHSSModel*, SNEPPXTape*, SNEPPXVariable*, SNEPPXVariable**, int, SNEPPXVariable**);
```

### SER

```c
SNEPPXSERConfig SNEPPX_ser_config_default();
SNEPPXSERModel* SNEPPX_ser_model_create(const SNEPPXSERConfig* cfg, unsigned int seed, size_t n_layers);
int  SNEPPX_ser_forward(SNEPPXSERModel*, const SNEPPXTensor*, SNEPPXTensor**);
int  SNEPPX_ser_route_mlp_gated(SNEPPXSERLayer*, const SNEPPXTensor*, SNEPPXTensor**);
```

### ARC

```c
SNEPPXARCLayer* SNEPPX_arc_layer_create(const SNEPPXARCConfig* cfg, size_t in_dim, size_t out_dim, unsigned int seed);
int  SNEPPX_arc_forward(SNEPPXARCLayer*, const SNEPPXTensor*, SNEPPXTensor**, float metrics[4]);
int  SNEPPX_arc_simulate_attack(SNEPPXARCLayer*, const SNEPPXTensor*, int attack_type, float eps, SNEPPXTensor**);
int  SNEPPX_arc_build_adversarial_train_graph(SNEPPXARCModel*, SNEPPXTape*, SNEPPXVariable*, SNEPPXVariable**, int, SNEPPXVariable**, SNEPPXVariable**);
```

### NPE

```c
SNEPPXNPEVM*     SNEPPX_npe_vm_create(const SNEPPXNPEConfig* cfg);
int              SNEPPX_npe_vm_run(SNEPPXNPEVM*, const SNEPPXTensor*, SNEPPXTensor**);
int              SNEPPX_npe_vm_optimize(SNEPPXNPEVM*);
SNEPPXNPEProgram* SNEPPX_npe_jit_pipeline_compose(SNEPPXNPEProgram*);
int              SNEPPX_npe_jit_fuse(SNEPPXNPEProgram*);
int              SNEPPX_npe_compile_mlp(size_t in_dim, size_t hidden_dim);
```

### FM

```c
SNEPPXFMController* SNEPPX_fm_controller_create(const SNEPPXFMConfig* cfg);
int  SNEPPX_fm_sync_all_reduce(SNEPPXFMController*, SNEPPXFMNode**, size_t, float);
int  SNEPPX_fm_sync_nccl(SNEPPXFMController*, SNEPPXFMSyncCallback, void*);
int  SNEPPX_fm_forward(SNEPPXFMController*, size_t node_id, const SNEPPXTensor*, SNEPPXTensor**);
```

### CUDA Trainer

```c
int  SNEPPX_trainer_cuda_available();
int  SNEPPX_trainer_cuda_init(size_t pool_size);
void SNEPPX_trainer_cuda_shutdown();
int  SNEPPX_trainer_cuda_transfer(void* dst, const void* src, size_t bytes);
int  SNEPPX_trainer_cuda_to_host(void* dst, const void* src, size_t bytes);
int  SNEPPX_trainer_cuda_optimizer_step(float* params, const float* grads, size_t n, float lr, float wd);
```

## Python API

### Tensor

```python
Tensor.create(shape, dtype)
Tensor.zeros(shape, dtype)
Tensor.ones(shape, dtype)
Tensor.randn(shape, dtype)
t + y, t - y, t * y, t / y
t @ y                         # matmul
t.relu(), t.softmax(dim)
t.sum(dim), t.mean(dim)
t.reshape(new_shape)
```

### Neural Network

```python
Linear(in_features, out_features, bias=True)
LayerNorm(normalized_shape, eps=1e-5)
Dropout(p=0.5)
Sequential(layers...)
```

### Optimizer

```python
AdamW(params, lr=3e-4, weight_decay=0.01)
SGD(params, lr=0.01, momentum=0.9)
```

### ARC

```python
ARCLayer(input_dim, output_dim)
layer.forward(input) -> output
layer.simulate_attack(input, attack_type, eps)
ARCAdversarialTrainGraph(attack_epsilon)
graph.build(weights, x_clean) -> (clean_out, adv_out)
```

### NPE + JIT

```python
NPECompiler()
compiler.compile(instrs) -> NPEProgram
compiler.jit_optimize(prog) -> NPEProgram
NPEVM()
vm.load(program)
vm.run(input) -> output
vm.jit_optimize()
```

### FM + NCCL

```python
FMController(num_nodes, memory_dim, mem_capacity)
ctrl.forward(node_id, input) -> output
ctrl.sync()
FMSyncNCCL()
nccl.sync(data, callback=fn) -> result
```

### Training

```python
TrainConfig()
config.use_cuda_optimizer = True/False
Trainer(model, config)
trainer.train_step(input, target) -> loss
```

### Model Zoo

```python
from_pretrained("meta-llama/Llama-2-7b-hf")
get_model_config("llama2-7b")
build_transformer_from_config(config)
```

### Quantization

```python
quantize_int8_sym(tensor)
awq_quantize(weight, act_scales)
gptq_quantize(weight, dataset, bits=4)
QuantizedLinear.from_float(linear, mode)
```

### Checkpoint / Profiler

```python
CheckpointWriter(path).save({"weights": w})
CheckpointReader(path).load()
Profiler().profile("name"): ...
@timeit(profiler)
```

### Security

```python
SecureAllocator()
DDoSMitigation(config)
TransportSecurity(cert, key)
KeyVault()
RLHFSafety()
SecurityMiddleware(config)
```

## Enums

| Enum | Values |
|------|--------|
| Dtype | FLOAT32, FLOAT64, FLOAT16, BFLOAT16, FLOAT8, INT32, INT64, INT16, INT8, UINT8, BOOL |
| OptimizerType | SGD, ADAM, ADAMW, RMSPROP, ADAGRAD |
| NPE OpCode | NOP, LOAD, MOV, MATMUL, ADD, RELU, SOFTMAX, LAYERNORM, SIGMOID, TANH, ATTENTION, HALT |
| NPE Fused Ops | MATMUL_RELU (0x80000000), MATMUL_ADD (0x40000000), MATMUL_ADD_RELU (0x20000000), ADD_RELU (0x80000001) |
| AttackType | FGSM, PGD, CW |
| FMSyncMethod | ALL_REDUCE, NCCL, GOSSIP, RING |

See [api/c.md](api/c.md) and [api/python.md](api/python.md) for full API references.
