import torch
import math

# Create input tensor
x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.float32).view(1, 2, 4)
base = 10000.0

batch_size, seq_len, embed_dim = x.shape

# Generate position indices
position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)  # (seq_len, 1)

# Create frequency tensor
freqs = torch.arange(0, embed_dim // 2, dtype=torch.float32)  # (embed_dim//2)
freqs = base ** (freqs * (2 / embed_dim))  # Apply scaling

# Create positional encoding
pos_enc = position * freqs  # (seq_len, embed_dim//2)

# Apply sin and cos
sin_enc = torch.sin(pos_enc)
cos_enc = torch.cos(pos_enc)

print(sin_enc)
print(cos_enc)

# Expand for batch dimension
sin_enc = sin_enc.unsqueeze(0).expand(batch_size, -1, -1)
cos_enc = cos_enc.unsqueeze(0).expand(batch_size, -1, -1)

# Split x into even and odd dimensions
x_even = x[..., 0::2]  # (batch, seq, embed_dim//2)
x_odd = x[..., 1::2]   # (batch, seq, embed_dim//2)

# Apply RoPE rotation
x_rotated_even = x_even * cos_enc - x_odd * sin_enc
x_rotated_odd = x_even * sin_enc + x_odd * cos_enc

# Interleave results
x_rotated = torch.zeros_like(x)
x_rotated[..., 0::2] = x_rotated_even
x_rotated[..., 1::2] = x_rotated_odd

print("RoPE result:")
print(x_rotated)
print("\nExpected values (rounded to 6 decimals):")
print("[[[ 1.000000,  2.000000,  3.000000,  4.000000],")
print("  [-2.347314,  7.449169, 10.087157,  3.353991]]]")