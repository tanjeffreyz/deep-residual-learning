"""Trains and validates various ResNet architectures against CIFAR-10."""

import torch
import models
import ssl
import os
import torchvision.transforms as T
import numpy as np
from torchvision.datasets.cifar import CIFAR10
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()

model = models.CifarResNet(20, option='A').to(device)
loss_function = torch.nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Load dataset
ssl._create_default_https_context = ssl._create_unverified_context      # Patch expired certificate
transforms = T.Compose([
    T.ToTensor(),
    model.transform
])
train_set = CIFAR10(root='data', train=True, download=True, transform=transforms)
test_set = CIFAR10(root='data', train=False, transform=transforms)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=128)

# Create folders
root = os.path.join('models', str(model))
now = datetime.now()
branch = os.path.join(root, now.strftime('%m_%d_%Y'), now.strftime('%H_%M_%S'))
weight_dir = os.path.join(branch, 'weights')
if not os.path.isdir(weight_dir):
    os.makedirs(weight_dir)


#####################
#       Train       #
#####################
train_losses = np.empty((2, 0))
test_losses = np.empty((2, 0))
train_accuracies = np.empty((2, 0))
test_accuracies = np.empty((2, 0))
for epoch in tqdm(range(200), desc='Epoch'):
    train_loss = 0
    accuracy = 0
    for data, labels in tqdm(train_loader, desc='Train', leave=False):
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions = model.forward(data)
        loss = loss_function(predictions, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() / len(train_loader)
        accuracy += labels.eq(torch.argmax(predictions, 1)).sum().item() / len(train_set)
        del data, labels
    np.append(train_losses, [[epoch], [train_loss]], axis=1)
    np.append(train_accuracies, [[epoch], [accuracy]], axis=1)
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', accuracy, epoch)

    if epoch % 5 == 0:
        with torch.no_grad():
            test_loss = 0
            accuracy = 0
            for data, labels in tqdm(test_loader, desc='Test', leave=False):
                data = data.to(device)
                labels = labels.to(device)

                predictions = model.forward(data)
                loss = loss_function(predictions, labels)

                test_loss += loss.item() / len(test_loader)
                accuracy += labels.eq(torch.argmax(predictions, 1)).sum().item() / len(test_set)
                del data, labels
        np.append(test_losses, [[epoch], [test_loss]], axis=1)
        np.append(test_accuracies, [[epoch], [accuracy]], axis=1)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', accuracy, epoch)

        # Save metrics and checkpoint
        np.save(os.path.join(branch, 'train_losses'), train_losses)
        np.save(os.path.join(branch, 'test_losses'), test_losses)
        np.save(os.path.join(branch, 'train_accuracies'), train_accuracies)
        np.save(os.path.join(branch, 'test_accuracies'), test_accuracies)
        torch.save(model.state_dict(), os.path.join(weight_dir, f'cp_{epoch}'))

torch.save(model.state_dict(), os.path.join(weight_dir, 'final'))
