import time

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F


def cross_entropy_one_hot(logits, targets):
    log_probs = torch.log_softmax(logits, dim=-1)
    loss = -(targets * log_probs).sum(dim=-1)
    return loss.mean()


class MnistNet(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        self.l1_weight = nn.Parameter(state_dict["l1.weight"])
        self.l1_bias = nn.Parameter(state_dict["l1.bias"])
        self.l2_weight = nn.Parameter(state_dict["l2.weight"])
        self.l2_bias = nn.Parameter(state_dict["l2.bias"])

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

# tensor([[ 0.1464, -0.0082, -0.2147, -0.1245,  0.0447,  0.1138,  0.0383,  0.0569, -0.0029,  0.0660]])

state_dict = load_file("../zyx-examples/models/mnist.safetensors")

train_dataset = load_file("../zyx-examples/data/mnist_dataset.safetensors")

train_x = train_dataset["train_x"].float() / 255.0
train_y = train_dataset["train_y"].long()
test_x = train_dataset["test_x"].float() / 255.0
test_y = train_dataset["test_y"].long()

batch_size = 64
num_train = train_x.shape[0]


optimizer = torch.optim.SGD(
    net.parameters(),
    lr=0.1,
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
        y_one_hot = F.one_hot(y, num_classes=10).float()
        loss = cross_entropy_one_hot(logits, y_one_hot)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(f"{loss.item()}")
        time.sleep(1)

        iters += 1
