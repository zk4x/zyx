# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

"""MNIST MLP training benchmark, mirroring `examples/mnist/src/main.rs` step for step.

Same arena as the zyx example so the numbers are comparable:

- 200 steps, batch 128, indices drawn on the host and uploaded per step
  (`train_x.index_select(0, &samples)` in zyx, advanced indexing here),
- one-hot cross entropy over the freshly initialized net,
- SGD(lr=0.01, momentum=0.9, nesterov=False),
- a warm-up step before the timing loop (optimizer state must exist, and it
  is what lets PT2 capture the step),
- `loss.item()` only on the steps that print, and outside the timed window,
- average over steps 100-200, printed with three decimals.

`mode="reduce-overhead"` is the PT2 equivalent of zyx's `Tape::freeze` +
replay: both capture the whole step once and replay it without tracing.
"""

import time

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn

BATCH = 128
STEPS = 200


class MnistNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(784, 128, bias=True)
        self.l2 = nn.Linear(128, 10, bias=True)

    def forward(self, x):
        x = x.view(x.shape[0], 784)
        x = torch.relu(self.l1(x))
        return self.l2(x)


def cross_entropy_one_hot(logits, targets):
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()


def main():
    device = torch.device("cuda")
    train_dataset = load_file("data/mnist_dataset.safetensors")
    train_x = train_dataset["train_x"].float().to(device)
    train_y = train_dataset["train_y"].long().to(device)
    n_train = train_x.shape[0]

    net = MnistNet().to(device)
    net.compile(mode="reduce-overhead")
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, nesterov=False)

    def train_step(samples):
        optimizer.zero_grad()
        x = train_x[samples]
        y = train_y[samples]
        logits = net(x)
        y_one_hot = torch.nn.functional.one_hot(y, num_classes=10).float()
        loss = cross_entropy_one_hot(logits, y_one_hot)
        loss.backward()
        optimizer.step()
        return loss

    def draw():
        # Host draw + upload, exactly like `Tensor::randint(..).to(dev)` in zyx.
        return torch.from_numpy(np.random.randint(0, n_train, size=BATCH, dtype=np.int64)).to(device)

    # Warm-up step: creates the optimizer state and captures the compiled step.
    train_step(draw())

    total_ms = 0.0
    count = 0
    for step in range(1, STEPS + 1):
        now = time.perf_counter()
        loss = train_step(draw())
        elapsed_ms = (time.perf_counter() - now) * 1000.0
        if step >= 100:
            total_ms += elapsed_ms
            count += 1
        if step % 5 == 0:
            print(f"step {step}, loss {loss.item():.6f}, step_time {elapsed_ms:.3f}ms")

    print(f"\nAverage step time (steps 100-{STEPS}): {total_ms / count:.3f}ms")


if __name__ == "__main__":
    main()
