import torch
from safetensors.torch import load_file
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function


# ----- MNIST model -----
class MnistNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(784, 128)
        self.l2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.shape[0], 784)  # reshape
        x = torch.relu(self.l1(x))  # first layer + activation
        return self.l2(x)  # second layer


# ----- Custom cross-entropy for one-hot -----
def cross_entropy_one_hot(logits, targets):
    log_probs = torch.log_softmax(logits, dim=-1)
    loss = -(targets * log_probs).sum(dim=-1)
    return loss.mean()


# ----- Load dataset -----
device = torch.device("cpu")
train_dataset = load_file("data/mnist_dataset.safetensors")
train_x = train_dataset["train_x"].float().to(device) / 255.0
train_y = train_dataset["train_y"].long().to(device)

batch_size = 128
num_train = train_x.shape[0]

# ----- Model and optimizer -----
net = MnistNet().to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.6)

# ----- Profile the full epoch -----
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    for i in range(0, num_train, batch_size):
        # ----- Batch picking (indexing) -----
        end = min(i + batch_size, num_train)
        x = train_x[i:end]  # <-- this indexing triggers CPU kernels
        y = train_y[i:end]  # <-- indexing triggers CPU kernels

        optimizer.zero_grad()

        # Forward
        logits = net(x)

        # One-hot encoding (elementwise kernels)
        y_one_hot = torch.nn.functional.one_hot(y, num_classes=10).float()

        # Loss
        loss = cross_entropy_one_hot(logits, y_one_hot)

        # Backward
        loss.backward()

        # Optimizer step
        optimizer.step()

# ----- Print all kernels -----
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=200))

# Approximate total CPU kernel count
print("Approx. total CPU kernels in this epoch:", len(prof.key_averages()))
