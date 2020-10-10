import torch
import torch.nn as nn
import torch.nn.functional as F


class s11_david(nn.Module):

    def __init__(self, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(s11_david, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        ############## basic block1 ###############
        in_planes = 128
        self.bconv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1,
                                padding=1)
        self.bbn1 = nn.BatchNorm2d(in_planes)
        self.bconv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1,
                                padding=1)
        self.bbn2 = nn.BatchNorm2d(in_planes)
        ############## basic block2 ###############
        in_planes2 = 512
        self.bconv3 = nn.Conv2d(in_planes2, in_planes2, kernel_size=3, stride=1,
                                padding=1)
        self.bbn3 = nn.BatchNorm2d(in_planes2)
        self.bconv4 = nn.Conv2d(in_planes2, in_planes2, kernel_size=3, stride=1,
                                padding=1)
        self.bbn4 = nn.BatchNorm2d(in_planes2)
        ################################################

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.inplanes2 = 128
        self.conv2 = nn.Conv2d(64, self.inplanes2, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(self.inplanes2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.inplanes3 = 256
        self.conv3 = nn.Conv2d(128, self.inplanes3, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(self.inplanes3)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.inplanes4 = 512
        self.conv4 = nn.Conv2d(256, self.inplanes4, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn4 = nn.BatchNorm2d(self.inplanes4)
        self.maxpool4 = nn.MaxPool2d(kernel_size=4, stride=1)

        self.fc = nn.Linear(512, num_classes)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # layer1
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.bn2(x)
        x = F.relu(x)

        identity = x

        out = self.bconv1(x)
        out = self.bbn1(out)
        out = F.relu(out)

        out = self.bconv2(out)
        out = self.bbn2(out)
        out = F.relu(out)

        out += identity
        x = out
        # layer2
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.bn3(x)
        x = F.relu(x)

        # layer 3
        x = self.conv4(x)
        x = self.maxpool3(x)
        x = self.bn4(x)
        x = F.relu(x)

        identity = x

        out = self.bconv3(x)
        out = self.bbn3(out)
        out = F.relu(out)

        out = self.bconv4(out)
        out = self.bbn4(out)
        out = F.relu(out)

        out += identity
        x = out

        x = self.maxpool4(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=-1)
        return x

    def forward(self, x):
        return self._forward_impl(x)