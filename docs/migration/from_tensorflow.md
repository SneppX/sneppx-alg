# Migrating from TensorFlow / Keras to SNEPPX-Algo

SNEPPX-Algo provides a small **Keras-compatible** layer API
(`keras_api.py`) so TensorFlow users can reuse familiar class names. Below is
the canonical mapping; under the hood every layer delegates to the same
`SneppX_ALG` `Tensor` and NN core.

## Installation

```bash
pip install sneppx-alg
# from source (C backend required for training):
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
$env:PYTHONPATH = "bindings/python"
```

## Layers

| TensorFlow / Keras | SNEPPX-Algo |
|--------------------|-------------|
| `tf.keras.Input` | `Input` |
| `tf.keras.layers.Dense` | `Dense` |
| `tf.keras.layers.Conv2D` | `Conv2D` |
| `tf.keras.layers.MaxPool2D` | `MaxPool2D` |
| `tf.keras.layers.AveragePool2D` | `AveragePool2D` |
| `tf.keras.layers.Flatten` | `Flatten` |
| `tf.keras.layers.Dropout` | `Dropout` |
| `tf.keras.layers.BatchNormalization` | `BatchNormalization` |
| `tf.keras.layers.LayerNormalization` | `LayerNorm` |
| `tf.keras.layers.ReLU` | `ReLU` |
| `tf.keras.layers.Sigmoid` | `Sigmoid` |
| `tf.keras.layers.Softmax` | `Softmax` |
| `tf.keras.layers.Activation` | `Activation` |
| `tf.keras.Sequential` | `Sequential` (also Keras-style `Model`) |
| `tf.keras.Model` | `Model` |

## Optimizer

| TensorFlow | SNEPPX-Algo |
|------------|-------------|
| `tf.keras.optimizers.SGD` | `SGD` |
| `tf.keras.optimizers.Adam` | `AdamW` (weight decay decoupled) |
| `tf.keras.optimizers.schedules.CosineDecay` | `CosineAnnealingLR` |
| `tf.keras.losses.MeanSquaredError` | `MSELoss` |
| `tf.keras.losses.SparseCategoricalCrossentropy` | `CrossEntropyLoss` |

## Training

| TensorFlow | SNEPPX-Algo |
|------------|-------------|
| `model.compile(loss=..., optimizer=...)` | `Trainer(model, TrainConfig)` |
| `model.fit(dataset, epochs=...)` | `Trainer.fit(loader, epochs=N)` |
| `model.evaluate(...)` | `Trainer.evaluate(x, y)` |
| `tf.GradientTape` | `loss.backward()` (tape autodiff) |
| `tf.config.optimizer.set_jit(True)` | CUDA kernels auto-selected when `_HAS_CUDA` |

```python
# Keras
import tensorflow as tf
inputs = tf.keras.Input(shape=(784,))
x = tf.keras.layers.Dense(256, activation="relu")(inputs)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(10)(x)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
model.fit(ds, epochs=10)

# SNEPPX (Keras-style API)
from SneppX_ALG import Input, Dense, Dropout, Model, AdamW, CrossEntropyLoss, Trainer, TrainConfig
from SneppX_ALG import TensorDataset, DataLoader

inputs = Input(shape=(784,))
x      = Dense(256, activation="relu")(inputs)
x      = Dropout(0.2)(x)
outputs = Dense(10)(x)
model   = Model(inputs, outputs)

cfg = TrainConfig(); cfg.learning_rate = 1e-3
trainer = Trainer(model, cfg)
loader  = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
for epoch in range(10):
    trainer.fit(loader)
```

## Data pipeline

| TensorFlow | SNEPPX-Algo |
|------------|-------------|
| `tf.data.Dataset` | `Dataset` / `TensorDataset` |
| `tf.data.Dataset.from_tensor_slices` | `TensorDataset(*tensors)` |
| `.batch(32).shuffle(1000)` | `DataLoader(..., batch_size=32, shuffle=True)` |
| `tf.keras.layers.TextVectorization` | `Tokenizer` / `SimpleTokenizer` |

## Distribution strategy

| TensorFlow | SNEPPX-Algo |
|------------|-------------|
| `tf.distribute.MirroredStrategy` | `init_process_group(backend="nccl")` |
| `tf.distribute.MultiWorkerMirroredStrategy` | `launch(train_fn, num_nodes=2, num_gpus=8)` |
| `tf.keras.utils.get_custom_objects()` | import a layer class directly |

## tf.function ↔ JIT

SNEPPX exposes a JAX-style `jit`/`grad` transform rather than graph
tracing:

```python
from SneppX_ALG.interface_bindings.jit import jit, grad
from SneppX_ALG import Tensor

@jit
def loss_fn(params, x, y):        # XLA-style compile when C backend present
    ...
g = grad(loss_fn)(params, x, y)
```

## Checkpoint format

| TensorFlow | SNEPPX-Algo |
|------------|-------------|
| `model.save_weights("model.h5")` | `model.save_checkpoint("model.ckpt")` |
| `model.load_weights("model.h5")` | `model.load_checkpoint("model.ckpt")` |
| SavedModel | not supported (use `.ckpt` + `CheckpointReader`) |

## Gotchas

- Keras `Model`/layer outputs are `Tensor` objects — call `.numpy()` to
  interop with TF or NumPy.
- `Dense` weight layout is `(in, out)` (transpose of TF's `(out, in)`),
  matching the C core `SNEPPXLinear`.
- Without the C backend, `Trainer` raises
  `RuntimeError: C backend not available for training`; build the extension.
