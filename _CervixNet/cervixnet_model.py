import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out1, out3, out5):
        super(InceptionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out1, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out3, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out5, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm2d(out1 + out3 + out5)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x3 = F.relu(self.conv3(x))
        x5 = F.relu(self.conv5(x))
        return self.bn(torch.cat([x1, x3, x5], dim=1))


class CervixNET(nn.Module):
    def __init__(self, num_classes=3):
        super(CervixNET, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2)
        self.pool1 = nn.MaxPool2d(3, stride=2)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(3, stride=2)

        self.parallel1 = InceptionBlock(128, 32, 64, 128)
        self.pool3 = nn.MaxPool2d(3, stride=2)

        self.parallel2 = InceptionBlock(224, 32, 64, 128)
        self.parallel3 = InceptionBlock(224, 32, 64, 128)
        self.pool4 = nn.MaxPool2d(3, stride=2)

        self.parallel4 = InceptionBlock(224, 32, 64, 128)
        self.pool5 = nn.MaxPool2d(5, stride=1)

        self.fc1 = nn.Linear(224 * 2 * 2, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout2 = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = self.parallel1(x)
        x = self.pool3(x)
        x = self.parallel2(x)
        x = self.parallel3(x)
        x = self.pool4(x)
        x = self.parallel4(x)
        x = self.pool5(x)
        x = torch.flatten(x, 1)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x
