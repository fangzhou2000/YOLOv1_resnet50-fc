import numpy as np
import torch
import torch.nn as nn
import torchvision

from model.darknet import DarkNet
from model.resnet import YOLOResNet


class yolo(nn.Module):
    """
    s: 7*7
    """
    def __init__(self, s, cell_out_ch, backbone_name, pretrain=None):
        super(yolo, self).__init__()
        self.s = s
        self.backbone_name = backbone_name
        self.backbone = None
        if backbone_name == 'darknet':
            self.backbone = DarkNet()
        elif backbone_name == 'resnet':
            self.backbone == YOLOResNet()
        self.conv = None

        assert self.backbone is not None, 'backbone type not support'

        self.fc = nn.Sequential(
            nn.Linear(1024*s*s, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, s*s*cell_out_ch)
        )

    def forward(self, x):
        # if x is a picture, batch_size == 1
        batch_size = x.size(0)
        x = self.backbone(x)
        x = torch.flatten(x)
        x = self.fc(x)
        # shift the dim, -1 means auto shift
        x = x.view(batch_size, self.s**2, -1)
        return x