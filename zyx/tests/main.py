import torch
from safetensors.torch import load_file
from torch import nn


class MnistNet(nn.Module):
    def __init__(self, state_dict):
        self.l1_weight = state_dict["l1.weight"]
        self.l1_bias = state_dict["l1.bias"]
        self.l2_weight = state_dict["l2.weight"]
        self.l2_bias = state_dict["l2.bias"]

    def forward(self, x):
        x = x.reshape([-1, 784])
        x = x.matmul(self.l1_weight.T) + self.l1_bias
        x = x.relu()
        x = x.matmul(self.l2_weight.T) + self.l2_bias
        return x


state_dict = load_file("../zyx-examples/models/mnist.safetensors")
net = MnistNet(state_dict)

x = torch.arange(0.0, 784, 1) / 784

x = net.forward(x)

print(f"{x}")
