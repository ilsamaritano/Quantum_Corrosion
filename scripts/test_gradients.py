import torch
import torch.nn as nn
from src.classical_models import build_mobilenetv2
import pennylane as qml

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(10, 8)
        self.dev = qml.device("default.qubit", wires=8)
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def qnode(inputs, weights):
            qml.AngleEmbedding(torch.pi * inputs, wires=range(8), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(8))
            return [qml.expval(qml.PauliZ(i)) for i in range(8)]
        
        self.q_weights = nn.Parameter(0.01 * torch.randn(3, 8, 3, dtype=torch.float32))
        self.qnode = qnode
        self.head = nn.Linear(8, 5)

    def forward(self, x):
        f = self.encoder(x)
        # return self.head(f) # if bypassed
        q = self.qnode(f, self.q_weights)
        q = torch.stack(q, dim=-1).to(f.dtype)
        return self.head(q)

m = Model()
x = torch.rand(4, 10)
out = m(x)
loss = out.sum()
loss.backward()
print("Encoder weight grad:", m.encoder.weight.grad is not None and m.encoder.weight.grad.abs().sum() > 0)
print("Q weights grad:", m.q_weights.grad is not None and m.q_weights.grad.abs().sum() > 0)
