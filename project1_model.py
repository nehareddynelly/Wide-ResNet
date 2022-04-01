import numpy as np
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
batch_size = 128

cifar_train = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
cifar_test = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
train_loader_cifar = DataLoader(
    cifar_train, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader_cifar = DataLoader(
    cifar_test, batch_size=batch_size, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1,
                          stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' % (depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(
            wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(
            wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(
            wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def project1_model():
    return Wide_ResNet(16, 5, 0, 10)


model = project1_model().half()
for layer in model.modules():
    if isinstance(layer, nn.BatchNorm2d):
        layer.float()
model.to(device)
trainable_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
print(trainable_params)

loss_train = []
error_train = []
loss_test = []
error_test = []


def epoch(loader, model, opt=None, sch=None):
    global loss_train
    global error_train
    global loss_test
    global error_test

    total_err = 0
    running_loss = 0
    for X, y in loader:
        X = X.to(device).half()
        y = y.to(device)
        output = model(X)
        criterion = nn.CrossEntropyLoss().cuda().half()
        loss = criterion(output, y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total_err += (output.max(dim=1)[1] != y).sum().item()
        running_loss += loss.item()*X.shape[0]
        if opt:
            loss_train.append(running_loss)
            error_train.append(total_err/X.size(0))
        else:
            loss_test.append(running_loss)
            error_test.append(total_err/X.size(0))
        if sch:
            sch.step()
    return total_err/len(loader.dataset), running_loss/len(loader.dataset)


epochs = 200
optimizer = optim.SGD(model.parameters(), lr=0.2,
                      momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, total_steps=(epochs * math.ceil(50000/batch_size)),
                                                pct_start=0.3,
                                                anneal_strategy='cos',
                                                cycle_momentum=True,
                                                base_momentum=0.85,
                                                max_momentum=0.95,
                                                )
for e in range(epochs):
    train_error, train_loss = epoch(
        train_loader_cifar, model, opt=optimizer, sch=scheduler)
    test_error, test_loss = epoch(test_loader_cifar, model)
    print(f"{e+1} {train_loss:.6f}, {train_error:.6f}, {test_loss:.6f}, {test_error:.6f}")

torch.save(model.state_dict(), 'project1_model.pt')
model.load_state_dict(torch.load('project1_model.pt'))

total_examples = np.zeros((10, 1))
error_cifar = np.zeros((10, 1))
for X_cifar, Y_cifar in test_loader_cifar:
    X_cifar = X_cifar.to(device).half()
    Y_cifar = Y_cifar.to(device)
    for i in range(10):
        r = (Y_cifar == i).nonzero()
        r = r.squeeze(1)
        if r.nelement() != 0:
            output = model(X_cifar)[r]
            error_cifar[i] += (output.max(dim=1)[1] != Y_cifar[r]).sum().item()
        total_examples[i] += len(r)
for i in range(10):
    error_cifar[i] = error_cifar[i]/total_examples[i]
    print('Accuracy of class: %s is %f' % (classes[i], 100-100*error_cifar[i]))
print(f"total accuracy: {100-100*(error_cifar.sum()/10):.6f}%")
