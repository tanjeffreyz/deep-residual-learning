"""Trains and validates various ResNet architectures against CIFAR-10."""

import torch
import models
import ssl
import os
import argparse
import torchvision.transforms as T
import numpy as np
from torchvision.datasets.cifar import CIFAR10
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('n', type=int, choices=(20, 32, 44, 56, 110))
parser.add_argument('-r', '--residual', action='store_true')
parser.add_argument('-o', '--option', type=str, choices=('A', 'B'), default=None)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()
now = datetime.now()

model = models.CifarResNet(args.n, residual=args.residual, option=args.option).to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,                      # Mutates the optimizer at each milestone
    milestones=(32_000, 48_000),
    gamma=0.1                       # Multiplies learning rate by 0.1 at every milestone
)

# Load dataset
ssl._create_default_https_context = ssl._create_unverified_context      # Patch expired certificate error
train_set = CIFAR10(
    root='data', train=True, download=True,
    transform=T.Compose([
        T.ToTensor(),
        model.transform,
        T.RandomCrop(32, padding=4)         # Pad each side by 4 pixels and randomly sample 32x32 image
    ])
)
test_set = CIFAR10(
    root='data', train=False,
    transform=T.Compose([
        T.ToTensor(),
        model.transform
    ])
)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=128)

# Create folders
root = os.path.join(
    'models',
    str(model),
    now.strftime('%m_%d_%Y'),
    now.strftime('%H_%M_%S')
)
weight_dir = os.path.join(root, 'weights')
if not os.path.isdir(weight_dir):
    os.makedirs(weight_dir)

train_losses = np.empty((2, 0))
test_losses = np.empty((2, 0))
train_errors = np.empty((2, 0))
test_errors = np.empty((2, 0))


def save_metrics():
    np.save(os.path.join(root, 'train_losses'), train_losses)
    np.save(os.path.join(root, 'test_losses'), test_losses)
    np.save(os.path.join(root, 'train_errors'), train_errors)
    np.save(os.path.join(root, 'test_errors'), test_errors)


#####################
#       Train       #
#####################
for epoch in tqdm(range(160), desc='Epoch'):
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
        scheduler.step()        # Step scheduler every iteration, not epoch

        train_loss += loss.item() / len(train_loader)
        accuracy += labels.eq(torch.argmax(predictions, 1)).sum().item() / len(train_set)
        del data, labels
    train_losses = np.append(train_losses, [[epoch], [train_loss]], axis=1)
    train_errors = np.append(train_errors, [[epoch], [1 - accuracy]], axis=1)
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Error/train', 1 - accuracy, epoch)

    if epoch % 4 == 0:
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
        test_losses = np.append(test_losses, [[epoch], [test_loss]], axis=1)
        test_errors = np.append(test_errors, [[epoch], [1 - accuracy]], axis=1)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Error/test', 1 - accuracy, epoch)

        save_metrics()
        if epoch % 20 == 0:
            torch.save(model.state_dict(), os.path.join(weight_dir, f'cp_{epoch}'))

save_metrics()
torch.save(model.state_dict(), os.path.join(weight_dir, 'final'))
