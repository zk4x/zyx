# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only

"""Download and convert MNIST dataset to safetensors format."""

import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
from safetensors.numpy import save_file

# Create data directory
os.makedirs("data", exist_ok=True)

print("Downloading MNIST dataset...")

# Download MNIST
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transforms.ToTensor()
)

test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transforms.ToTensor()
)

# Convert to numpy arrays
train_x = np.array([img.numpy() * 255 for img, _ in train_dataset]).astype(np.uint8)
train_y = np.array([label for _, label in train_dataset]).astype(np.int64)
test_x = np.array([img.numpy() * 255 for img, _ in test_dataset]).astype(np.uint8)
test_y = np.array([label for _, label in test_dataset]).astype(np.int64)

print(f"train_x shape: {train_x.shape}")
print(f"train_y shape: {train_y.shape}")
print(f"test_x shape: {test_x.shape}")
print(f"test_y shape: {test_y.shape}")

# Save to safetensors
save_file(
    {
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "test_y": test_y,
    },
    "data/mnist_dataset.safetensors",
)

print("Saved to data/mnist_dataset.safetensors")
