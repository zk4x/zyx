# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

"""MNIST MLP training benchmark in tinygrad, mirroring `examples/mnist/src/main.rs`.

Same arena as the zyx and torch scripts:

- 200 steps, batch 128, indices drawn on the host and uploaded per step,
- one-hot cross entropy over the freshly initialized net,
- SGD(lr=0.01, momentum=0.9),
- warm-up step before timing (optimizer state + JIT capture),
- `loss.item()` only on printing steps, outside the timed window,
- average over steps 100-200, printed with three decimals.

`TinyJit` is tinygrad's `Tape::freeze` equivalent: the first call traces, the
second captures, every later call only patches the input buffers. `train_x`
and `train_y` are captured constants, like the leaves zyx's frozen tape holds.
"""

import os
import time

# tinygrad 0.11+ gates the optimizer behind a context flag; set it before the
# import so the same script works on released builds and on git main.
os.environ.setdefault("TRAINING", "1")

import numpy as np
from safetensors.numpy import load_file
from tinygrad import Device, Tensor, TinyJit, nn
from tinygrad.nn.optim import SGD

BATCH = 128
STEPS = 200


def cross_entropy_one_hot(logits: Tensor, targets: Tensor) -> Tensor:
    return -(targets * logits.log_softmax()).sum(axis=-1).mean()


def main():
    device = Device.DEFAULT
    dataset = load_file("data/mnist_dataset.safetensors")
    train_x = Tensor(dataset["train_x"].astype(np.float32)).to(device).realize()
    train_y = Tensor(dataset["train_y"].astype(np.int64)).to(device).realize()
    n_train = train_x.shape[0]

    l1 = nn.Linear(784, 128, bias=True)
    l2 = nn.Linear(128, 10, bias=True)
    params = nn.state.get_parameters((l1, l2))
    optimizer = SGD(params, lr=0.01, momentum=0.9)

    @TinyJit
    def train_step(samples: Tensor) -> Tensor:
        optimizer.zero_grad()
        x = train_x[samples].reshape(BATCH, 784)
        y = train_y[samples]
        x = l1(x).relu()
        logits = l2(x)
        y_one_hot = y.one_hot(10).float()
        loss = cross_entropy_one_hot(logits, y_one_hot)
        loss.backward()
        optimizer.step()
        return loss.realize()

    def draw() -> Tensor:
        # Host draw + upload, exactly like `Tensor::randint(..).to(dev)` in zyx.
        return Tensor(np.random.randint(0, n_train, size=BATCH, dtype=np.int64)).to(device)

    # Warm-up step: creates the optimizer state and triggers the JIT capture.
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
