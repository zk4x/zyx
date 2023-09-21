from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes
from tinygrad.nn import Linear
from tinygrad.nn.optim import SGD
from time import time

class TinyNet:
  def __init__(self):
    self.l0 = Linear(1024, 1024)
    self.l1 = Linear(1024, 1024)

  def __call__(self, x):
    x = self.l0(x).tanh()
    return self.l1(x)

tiny_net = TinyNet()

opt = SGD([tiny_net.l0.weight, tiny_net.l0.bias, tiny_net.l1.weight, tiny_net.l1.bias], lr=0.01)

x = Tensor.randn(1024, 1024)
label = Tensor.randn(1024)

# Compile kernels
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
tiny_net.l0.weight.numpy()
tiny_net.l0.bias.numpy()
tiny_net.l1.weight.numpy()
tiny_net.l1.bias.numpy()
elapsed = time() - now

print(f"Taken {elapsed:.2f}s")

