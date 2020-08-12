import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FPN(nn.Module):
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        self.in_planes = 64

        self.resnet = nn.Sequential()

        self.resnet.add_module("conv1", nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False))
        self.resnet.add_module("bn1", nn.BatchNorm2d(64))
        self.resnet.add_module("relu", nn.ReLU(inplace=True))
        self.resnet.add_module("maxpool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # Bottom-up layers
        self.resnet.add_module("layer1", self._make_layer(block, 64, num_blocks[0], stride=1))
        self.resnet.add_module("layer2", self._make_layer(block, 128, num_blocks[1], stride=2))
        self.resnet.add_module("layer3", self._make_layer(block, 256, num_blocks[2], stride=2))
        self.resnet.add_module("layer4", self._make_layer(block, 512, num_blocks[3], stride=2))

        # Lateral layers
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # c5 -> p5
        self.flatlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)  # c4 -> p4
        self.flatlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)  # c3 -> p3
        self.flatlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)  # c2 -> p2
        # smooth
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # smooth p4
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # smooth p3
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # smooth p2

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: top feature map to be upsampled.
          y: lateral feature map.

        Returns:
          added feature map.
        '''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='nearest', align_corners=None) + y  # bilinear, False

    def freeze_resnet(self):
        self.resnet.eval()

    def forward(self, x):
        # Bottom-up
        out = self.resnet.conv1(x)
        out = self.resnet.bn1(out)
        out = self.resnet.relu(out)
        out = self.resnet.maxpool(out)

        c2 = self.resnet.layer1(out)
        c3 = self.resnet.layer2(c2)
        c4 = self.resnet.layer3(c3)
        c5 = self.resnet.layer4(c4)

        fp5 = self.toplayer(c5)
        fp4 = self._upsample_add(fp5, self.flatlayer1(c4))
        fp3 = self._upsample_add(fp4, self.flatlayer2(c3))
        fp2 = self._upsample_add(fp3, self.flatlayer3(c2))

        fp4 = self.smooth1(fp4)
        fp3 = self.smooth2(fp3)
        fp2 = self.smooth3(fp2)

        return [fp2, fp3, fp4, fp5]


def FPN50():
    return FPN(Bottleneck, [3, 4, 6, 3])


def FPN101():
    return FPN(Bottleneck, [3, 4, 23, 3])
