# API Explorer

Quick reference for SNEPPX-Alg's Python API.

!!! tip "Interactive pages"
    - [Matmul visualizer](matmul-visualizer.md) — see the row · column dot product
    - [Conv2d visualizer](conv-visualizer.md) — sliding window, stride and padding
    - [Loss gallery](loss-gallery.md) — every loss with its gradient curve
    - [Optimizer race](optimizer-race.md) — SGD vs Adam vs AdamW vs NAdam


## Tensor

```python
from SneppX_ALG import Tensor
x = Tensor(data, requires_grad=True)
x.backward()
x.grad  # gradient
x.data  # numpy array
```

## nn modules

| Module | Description |
|--------|-------------|
| `nn.Linear` | Fully connected layer |
| `nn.Conv1d` | 1D convolution (groups, dilation) |
| `nn.Conv2d` | 2D convolution (groups, dilation) |
| `nn.Conv3d` | 3D convolution (groups, dilation) |
| `nn.ConvTranspose1d/2d/3d` | Transposed convolution |
| `nn.LayerNorm` | Layer normalization |
| `nn.RMSNorm` | RMS normalization |
| `nn.BatchNorm1d/2d/3d` | Batch normalization |
| `nn.InstanceNorm1d/2d/3d` | Instance normalization |
| `nn.GroupNorm` | Group normalization |
| `nn.Embedding` | Embedding lookup |
| `nn.EmbeddingBag` | Embedding bag (sum/mean/max) |
| `nn.LSTM` / `nn.GRU` / `nn.RNN` | Recurrent layers |
| `nn.MultiheadAttention` | Multi-head attention |
| `nn.MaxPool1d/2d/3d` | Max pooling |
| `nn.AvgPool1d/2d/3d` | Average pooling |
| `nn.AdaptiveAvgPool1d/2d/3d` | Adaptive average pooling |
| `nn.AdaptiveMaxPool1d/2d/3d` | Adaptive max pooling |
| `nn.Upsample` | Upsampling (nearest/bilinear/trilinear) |
| `nn.Unfold` / `nn.Fold` | Im2col / col2im |
| `nn.MaxUnpool1d/2d/3d` | Max unpooling |
| `nn.Dropout` | Dropout |
| `nn.Flatten` | Flatten |
| `nn.Bilinear` | Bilinear layer |
| `nn.ChannelShuffle` | Channel shuffle |
| `nn.PixelShuffle` / `nn.PixelUnshuffle` | Pixel shuffle |

## Optimizers

`SGD`, `Adam`, `AdamW`, `Adamax`, `NAdam`, `RMSprop`, `Adagrad`, `Adadelta`, `Rprop`, `ASGD`, `LBFGS`, `SparseAdam`

## Losses

`MSELoss`, `CrossEntropyLoss`, `BCELoss`, `BCEWithLogitsLoss`, `NLLLoss`, `KLDivLoss`, `SmoothL1Loss`, `HuberLoss`, `L1Loss`, `GaussianNLLLoss`, `HingeEmbeddingLoss`, `PoissonNLLLoss`, `SoftMarginLoss`, `MultiMarginLoss`, `MultiLabelSoftMarginLoss`, `CosineEmbeddingLoss`

## Autograd ops

`Add`, `Sub`, `Mul`, `Div`, `MatMul`, `Transpose`, `Sum`, `Mean`, `ReLU`, `Sigmoid`, `Tanh`, `Sqrt`, `Exp`, `Log`, `Pow`, `Reshape`, `GetItem`, `Square`, `Reciprocal`, `Sign`, `Clamp`, `Remainder`, `Prod`, `LogSumExp`, `Trace`, `Diagonal`, `MinDim`, `MaxDim`, `Neg`, `Abs`, `Softmax`, `LogSoftmax`, `LayerNorm`, `RMSNorm`, `Conv1d`, `Conv2d`, `Conv3d`, `Embedding`, `Permute`, `UpsampleNearest`, `AdaptiveAvgPool`, `AdaptiveMaxPool`, `Bilinear`, `Unfold`, `Fold`, `MaxUnpool`, `CosineEmbeddingLoss`