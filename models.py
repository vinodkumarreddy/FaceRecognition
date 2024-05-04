import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F

class SimpleImageClassifier(nn.Module):
    def __init__(self, n_channels = 3, n_classes = 5):
        super().__init__()
        self.cnn_block = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding = 2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 16, 5, padding = 2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 8, 5, padding = 2),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 1, 5, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.linear_block = nn.Sequential(
            nn.Linear(225, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )
        
    def forward(self, x):
        x = self.cnn_block(x)
        return self.linear_block(x)

    def train(self, train_dataloader, valid_dataloader = None, ):
        