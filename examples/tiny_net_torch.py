import torch
from torch.nn import Linear, Module
from torch.optim import SGD

from time import time

class TinyNet(Module):
  def __init__(self):
    super().__init__()
    self.l0 = Linear(1024, 1024)
    self.l1 = Linear(1024, 1024)

  def forward(self, x):
    x = self.l0(x).tanh()
    return self.l1(x)

tiny_net = TinyNet()
tiny_net = torch.compile(tiny_net)

opt = SGD(tiny_net.parameters(), lr=0.01)

x = torch.randn(1024, 1024)
label = torch.randn(1024)

# Compile
for _ in range(5):
  out = tiny_net(x)
  loss = (out - label) ** 2
  loss.sum().backward()
  opt.step()

now = time()
for _ in range(100):
  out = tiny_net(x)
  loss = (out - label) ** 2
  loss.sum().backward()
  opt.step()
elapsed = time() - now

print(f"Taken {elapsed:.2f}s")

