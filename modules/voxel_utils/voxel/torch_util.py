import torch.nn as nn

class Conv2d(nn.Module):
    def __init__(self, inplanes, planes, stride, padding):
        super(Conv2d, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out

class Conv3d(nn.Module):
    def __init__(self, inplanes, planes, stride, padding):
        super(Conv3d, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, 3, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out