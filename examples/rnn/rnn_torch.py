import torch
import torch.nn as nn
import torch.optim as optim
import time

input_size = 16
hidden_size = 32
batch_size = 64
seq_len = 8

train_x = torch.randn(batch_size, seq_len, input_size)
target = torch.randn(batch_size, hidden_size)

rnn = nn.RNNCell(input_size, hidden_size, bias=True)

optim = optim.SGD(rnn.parameters(), lr=0.05, momentum=0.9, nesterov=True)

# Warmup
for _ in range(3):
    hidden = torch.zeros(batch_size, hidden_size)
    for t in range(seq_len):
        x_t = train_x[:, t, :]
        hidden = rnn(x_t, hidden)
    loss = nn.functional.mse_loss(hidden, target)
    loss.backward()
    optim.step()
    optim.zero_grad()

print("Training RNN...")
start = time.time()
for step in range(50):
    hidden = torch.zeros(batch_size, hidden_size)
    for t in range(seq_len):
        x_t = train_x[:, t, :]
        hidden = rnn(x_t, hidden)
    
    loss = nn.functional.mse_loss(hidden, target)
    loss.backward()
    optim.step()
    optim.zero_grad()
    
    print(f"step {step}, loss {loss.item():.8f}")

elapsed = time.time() - start
print(f"Total time: {elapsed:.3f}s for 50 steps")
print(f"Average: {elapsed/50*1000:.2f}ms per step")