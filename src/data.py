from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import v2

# data/ sits in the repo root, next to src/
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])


def get_datasets():
    training_data = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    test_data = torchvision.datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    return training_data, test_data


def get_dataloaders(batch_size=64):
    training_data, test_data = get_datasets()
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    return train_dataloader, test_dataloader
