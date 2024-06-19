import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import v2

import pandas as pd

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

transform = v2.Compose([
    v2.PILToTensor(),
    v2.ToDtype(torch.float32),
    v2.Normalize((0.1307,), (0.3081,)),
])
train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
print(f"Mean: {train_dataset.data.float().mean() / 255}, Std: {train_dataset.data.float().std() / 255}")
# print(train_dataset.__getitem__(0))
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False)
# print(train_dataloader)
# x_train, _ = next(iter(train_dataloader))
# print(x_train)

simple_net = nn.Sequential(
    nn.Linear(784, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)
simple_net.to(device)

optimizer = torch.optim.SGD(simple_net.parameters(), lr=1e-3)
predicted = []

for x_train, y_train in train_dataloader:
    optimizer.zero_grad()
    x_train, y_train = x_train.to(device), y_train.to(device)
    # print(x_train.view(784))
    # print(F.softmax(simple_net(x_train.view(-1, 784)), dim=-1), y_train)
    print(simple_net(x_train.view(-1, 784)).shape, y_train.shape)
    loss = F.cross_entropy(simple_net(x_train.view(-1, 784)), y_train)
    print(loss.item())
    # if loss.item() < 1:
    #     break
    loss.backward()
    optimizer.step()

import sys; sys.exit(0)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(len(y_train))
simple_net = nn.Sequential(
    nn.Linear(784, 30),
    nn.ReLU(),
    nn.Linear(30, 10)
)
simple_net.to(device)

optimizer = torch.optim.SGD(simple_net.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for input, label in zip(x_train, y_train):
    optimizer.zero_grad()
    target = torch.tensor(cats[label], dtype=torch.float32, device=device)
    # print(cats[label])
    input = torch.tensor(input, dtype=torch.float32, device=device).view(784)
    output = simple_net(input)
    # print(output)
    loss = criterion(output, target)
    print(loss)
    loss.backward()
    optimizer.step()
    # break


