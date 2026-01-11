import time
from operator import ne

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
        # print(f"x={x.reshape(-1, 28, 28)[0, 15:20, 15:20]}")
        # print(f"{self.l1_weight[0, 0:10]}")
        x = x.matmul(self.l1_weight.T) + self.l1_bias
        x = x.relu()
        x = x.matmul(self.l2_weight.T) + self.l2_bias
        return x


state_dict = load_file("../zyx-examples/models/mnist.safetensors")
net = MnistNet(state_dict)

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
        print(f"{logits[0:4, :4]}")
        y_one_hot = F.one_hot(y, num_classes=10).float()
        loss = cross_entropy_one_hot(logits, y_one_hot)
        print(f"loss={loss.item()}")

        loss.backward()

        print(net.l2_bias.grad)
        # tensor([-0.0065, -0.0215,  0.0218, -0.0262, -0.0322,  0.0448,  0.0106,  0.0165, 0.0262, -0.0336])

        raise Exception

        optimizer.step()

        total_loss += loss.item()
        print(f"{loss.item()}")
        time.sleep(1)

        iters += 1
