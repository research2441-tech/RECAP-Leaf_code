"""Compact three-block CNN baseline representative of lightweight Raspberry-Pi leaf classifiers."""
import torch.nn as nn


class SourceCNN(nn.Module):
    """Re-implementation of the three-block CNN described in the source manuscript
    (5x5 filters, 20/50/80 maps, max pooling, dense 500, dropout 0.5)."""
    def __init__(self, k):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(3, 20), blk(20, 50), blk(50, 80), nn.AdaptiveAvgPool2d(8))
        self.h = nn.Sequential(nn.Flatten(), nn.Linear(80 * 64, 500), nn.ReLU(), nn.Dropout(0.5), nn.Linear(500, k))
    def forward(self, x): return self.h(self.f(x))
