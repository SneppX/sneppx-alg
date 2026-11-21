"""
Example 8: Simple Reinforcement Learning (DQN-like) step with SNEPPX-Alg API
"""
import numpy as np
from SneppX_ALG.interface_bindings.pytorch_api import Tensor, Module, Linear, Sequential, Adam

class QNetwork(Module):
    def __init__(self, state_dim=4, action_dim=2):
        super().__init__()
        self.net = Sequential(Linear(state_dim, 16), Linear(16, action_dim))

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    model = QNetwork()
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Simple DQN training step
    state = Tensor(np.random.randn(1, 4).astype(np.float32))
    action_idx = 1
    reward = 1.0
    
    # Q-value estimate
    q_values = model(state)
    # Bellman update: target = reward + gamma * max(next_q)
    target = reward
    
    pred = Tensor(q_values.data[0, action_idx], requires_grad=True)
    loss = Tensor((pred.data - target)**2, requires_grad=True)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("DQN Training Step completed!")
