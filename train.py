"""Trains and validates various ResNet architectures against ImageNet."""

import torch
import models
import torchvision.transforms as T
from torchvision.datasets.imagenet import ImageNet
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[~] Using {device}')

writer = SummaryWriter()


model = models.ResNet18(option='A')
loss_function = torch.nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

imagenet = ImageNet(
    root='data',
)
test_partition = 50_000
train_set, test_set = random_split(imagenet, (len(imagenet) - test_partition, test_partition))

print(next(train_set).shape)
