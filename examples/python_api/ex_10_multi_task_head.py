"""
Example 10: Multi-Task Learning Head with SNEPPX-Alg API
"""
import numpy as np
from SneppX_ALG.interface_bindings.pytorch_api import Tensor, Module, Linear, Sequential

class MultiTaskModel(Module):
    def __init__(self):
        super().__init__()
        self.shared = Linear(16, 32)
        self.head1 = Linear(32, 2)  # Classification
        self.head2 = Linear(32, 1)  # Regression

    def forward(self, x):
        shared_feat = self.shared(x)
        return self.head1(shared_feat), self.head2(shared_feat)

if __name__ == "__main__":
    model = MultiTaskModel()
    x = Tensor(np.random.randn(5, 16).astype(np.float32))
    
    class_logits, reg_val = model(x)
    print("Class logits shape:", class_logits.shape)
    print("Regression value shape:", reg_val.shape)
    print("Multi-task model initialized!")
