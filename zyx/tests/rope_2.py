import torch

x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]).view([1, 2, 4])
base = 10000.

batch_size, seq_len, embed_dim = x.shape

position = torch.arange(0, seq_len, 1).unsqueeze(1)

freqs = torch.arange(0, embed_dim/2, 1)
freqs = freqs.pow(freqs * (2/embed_dim))

pos_enc = position * freqs

print(pos_enc)

sin_enc = pos_enc.sin()
cos_enc = pos_enc.cos()

print(sin_enc)
print(cos_enc)
