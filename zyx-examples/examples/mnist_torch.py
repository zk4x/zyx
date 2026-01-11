import time
from turtle import window_height

import torch
from safetensors.torch import load_file
from torch import nn


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
    loss = -(targets * log_probs).sum(dim=-1)
    return loss.mean()


device = torch.device("cpu")

train_dataset = load_file("data/mnist_dataset.safetensors")

train_x = train_dataset["train_x"].float().to(device) / 255.0
train_y = train_dataset["train_y"].long().to(device)
test_x = train_dataset["test_x"].float().to(device) / 255.0
test_y = train_dataset["test_y"].long().to(device)

batch_size = 64
num_train = train_x.shape[0]

net = MnistNet().to(device)

# net.load_state_dict(torch.load("models/mnist.safetensors"))

optimizer = torch.optim.SGD(
    net.parameters(),
    lr=0.01,
    momentum=0.6,
    nesterov=False,
)

for epoch in range(1, 6):
    total_loss = 0.0
    iters = 0

    for i in range(0, num_train, batch_size):
        end = min(i + batch_size, num_train)

        x = train_x[i:end]
        y = train_y[i:end]

        optimizer.zero_grad()

        logits = net(x)
        y_one_hot = torch.nn.functional.one_hot(y, num_classes=10).float()
        loss = cross_entropy_one_hot(logits, y_one_hot)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(f"{loss.item()}")
        # time.sleep(3)

        iters += 1
