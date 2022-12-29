import torch
import math

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(1, 2000)
        self.a1 = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(2000, 5000)
        self.a2 = torch.nn.Tanh()
        self.l3 = torch.nn.Linear(5000, 1)
        self.a3 = torch.nn.Sigmoid()

    def forward(self, x):
        return self.a3(self.l3(self.a2(self.l2(self.a1(self.l1(x))))))

network = Network()

mse_loss = torch.nn.MSELoss()

optimizer = torch.optim.SGD(network.parameters(), lr = 0.03)

for _ in range(1):
    for i in range(100):
        optimizer.zero_grad()

        x = torch.tensor([[i/10.]])
        y = torch.tensor([[math.sin(i/10.)]])

        y_p = network(x)

        loss = mse_loss(y_p, y)

        loss.backward()

        optimizer.step()
