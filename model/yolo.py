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


class yolo_loss:
    def __init__(self, device, s, b, image_size, num_classes):
        self.device = device
        self.s = s
        self.b = b
        self.image_size = image_size
        self.num_classes = num_classes

    def __call__(self, input, target):
        """
        :param input: tensor[s, s, b*5 + n_class]  b: b * (c_x, c_y, w, h, obj_conf), class1_p, class2_p..
        :param target: (dataset) tensor[n_bbox] bbox: x_min, ymin, xmax, ymax, class
        :return: loss tensor

        grid type: [[bbox, ..], [], ..] -> bbox: c_x, c_y, w, h, class
        
        """
        self.batch_size = input.size(0)

        # label preprocess
        target = [self.label_direct2grid(target[i]) for i in range(self.batch_size)]

        # IOU match predictor and label
        # x, y, w, h, c
        match = []
        conf = []
        for i in range(self.batch_size):
            m, c = self.match_pred_target(input[i], target[i])
            match.append(m)
            conf.append(c)
        
        loss = torch.zeros([self.batch_size], dtype=torch.float, device=self.device)
        xy_loss = torch.zeros_like(loss)
        wh_loss = torch.zeros_like(loss)
        conf_loss = torch.zeros_like(loss)
        class_loss = torch.zeros_like(loss)
        for i in range(self.batch_size):
            loss[i], xy_loss[i], wh_loss[i], conf_loss[i], class_loss[i] = \
                self.compute_loss(input[i], target[i], match[i], conf[i])
        return torch.mean(loss), torch.mean(xy_loss), torch.mean(wh_loss), torch.mean(conf_loss), torch.mean(class_loss)

    def label_direct2grid(self, label):
        """
        :param:
        """