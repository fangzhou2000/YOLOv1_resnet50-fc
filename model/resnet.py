import torch
from torchvision.models.resnet import ResNet, Bottleneck

class YOLOResNet(ResNet):
    def __init__(self, block, layers):
        super(YOLOResNet, self).__init__(block=block, layers=layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        return x

def yolo_resnet(pretrained=None) -> YOLOResNet:
    # setup([3 4 6 3]) for resnet50
    model = YOLOResNet(Bottleneck, [3, 4, 6, 3], pretrained)
    if pretrained is not None:
        state_dict = torch.load(pretrained)
        model.load_state_dict(state_dict)
    return model
