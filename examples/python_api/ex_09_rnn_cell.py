"""
Example 9: Recurrent Neural Network (RNN) cell step with SNEPPX-Alg API
"""
import numpy as np
from SneppX_ALG.interface_bindings.pytorch_api import Tensor, Module, Linear, Adam

class RNNCell(Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.w_ih = Linear(input_dim, hidden_dim)
        self.w_hh = Linear(hidden_dim, hidden_dim)

    def forward(self, x, h):
        # h_new = tanh(W_ih * x + W_hh * h + b)
        combined = self.w_ih(x) + self.w_hh(h)
        return Tensor(np.tanh(combined.data), requires_grad=True)

if __name__ == "__main__":
    rnn = RNNCell(input_dim=8, hidden_dim=16)
    x = Tensor(np.random.randn(1, 8).astype(np.float32))
    h = Tensor(np.zeros((1, 16)).astype(np.float32))
    
    h_new = rnn(x, h)
    print("RNN cell step output shape:", h_new.shape)
