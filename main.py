import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) #Normalise the data
])

trainDataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testDataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainLoader = DataLoader(trainDataset, batch_size=64, shuffle=True)
testLoader = DataLoader(testDataset, batch_size=64, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32*7*7, 128) #2 max pools reduces size from 28*28 down to 7*7
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN().to(device)
optimiser = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

fileName = 'weights.pth'
if os.path.exists(fileName):
    model.load_state_dict(torch.load('weights.pth'))
else:
    epochs = 10
    for epoch in range(1, epochs+1):
            model.train()
            runningLoss = 0

            for inputs, labels in trainLoader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimiser.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimiser.step()

                runningLoss += loss.item()

            avgLoss = runningLoss / len(trainLoader)
            print(f'Epoch {epoch} | Average loss {avgLoss}')
            runningLoss = 0

    torch.save(model.state_dict(), 'weights.pth')

model.eval()
with torch.no_grad():
    pass
