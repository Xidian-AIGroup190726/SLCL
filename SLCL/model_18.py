import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18


class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        self.f1 = []
        for name, module in resnet18().named_children():
            if name == 'conv1':
                module = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f1.append(module)
        # encoder
        self.f1 = nn.Sequential(*self.f1)
        # projection head
        self.g1 = nn.Sequential(nn.Linear(512, 256, bias=False), nn.BatchNorm1d(256),
                               nn.ReLU(inplace=True), nn.Linear(256, feature_dim, bias=True))

        self.f2 = []
        for name, module in resnet18().named_children():
            if name == 'conv1':
                module = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f2.append(module)
        # encoder
        self.f2 = nn.Sequential(*self.f2)
        # projection head
        self.g2 = nn.Sequential(nn.Linear(512, 256, bias=False), nn.BatchNorm1d(256),
                               nn.ReLU(inplace=True), nn.Linear(256, feature_dim, bias=True))

    def forward(self, ms, pan):
        # encoder
        ms = self.f1(ms)
        feature_ms = torch.flatten(ms, start_dim=1)
        pan = self.f2(pan)
        feature_pan = torch.flatten(pan, start_dim=1)

        out_ms = self.g1(feature_ms)
        out_pan = self.g2(feature_pan)

        return F.normalize(feature_ms, dim=-1), F.normalize(out_ms, dim=-1), F.normalize(feature_pan, dim=-1), F.normalize(out_pan, dim=-1)


model=Model()
print(model)