import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from batchnorm import GhostBatchNorm


class Netquiz(nn.Module):
    # BN_flag 0: normal batchnorm; 1: Ghost batchnorm
    def batch_norm(self, channels, BN_flag):
        if BN_flag == 1:
            return GhostBatchNorm(channels, num_splits=2, weight=False)
        else:
            return nn.BatchNorm2d(channels)

    def __init__(self, flag):
        super(Netquiz, self).__init__()
        # 32,32,3

        self.conv1 = nn.Conv2d(
            3, 8, kernel_size=3, stride=1, padding=1, bias=False)  # 32,32,8
        self.Batchnorm1 = self.batch_norm(8, flag)


        self.conv2 = nn.Conv2d(
            11, 32, kernel_size=3, stride=1, padding=1, bias=False)  # 32,32,8
        self.Batchnorm2 = self.batch_norm(32, flag)

        self.pool1 = nn.MaxPool2d(2, 2)  # 16,16,8

        self.conv3 = nn.Conv2d(
            43, 32, kernel_size=3, stride=1, padding=1, bias=False)  # 16,16,8
        self.Batchnorm3 = self.batch_norm(32, flag)

        self.conv4 = nn.Conv2d(
            75, 32, kernel_size=3, stride=1, padding=1, bias=False)  # 16,16,8
        self.Batchnorm4 = self.batch_norm(32, flag)

        self.conv5 = nn.Conv2d(
            107, 32, kernel_size=3, stride=1, padding=1, bias=False)  # 16,16,8
        self.Batchnorm5 = self.batch_norm(32, flag)

        self.pool2 = nn.MaxPool2d(2, 2)  # 8,8,8

        self.conv6 = nn.Conv2d(
            96, 32, kernel_size=3, stride=1, padding=1, bias=False)  # 8,8,8
        self.Batchnorm6 = self.batch_norm(32, flag)

        self.conv7 = nn.Conv2d(
            128, 32, kernel_size=3, stride=1, padding=1, bias=False)  # 8,8,8
        self.Batchnorm7 = self.batch_norm(32, flag)

        self.conv8 = nn.Conv2d(
            160, 32, kernel_size=3, stride=1, padding=1, bias=False)  # 8,8,8

        self.gap = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Conv2d(32, 10, kernel_size=1)

    def forward(self, x):
        x1 = x  # 3
        x2 = F.relu(self.conv1(x1))  # 8
        x2 = self.Batchnorm1(x2)
        x3 = torch.cat((x1, x2), dim=1)  # 11
        x3 = F.relu(self.conv2(x3))  # 8
        x3 = self.Batchnorm2(x3)

        x4 = torch.cat((x1, x2, x3), dim=1)  # 19
        x4 = self.pool1(x4)

        x5 = F.relu(self.conv3(x4))
        x5 = self.Batchnorm3(x5)
        x6 = torch.cat((x4, x5), dim=1)
        x6 = F.relu(self.conv4(x6))
        x6 = self.Batchnorm4(x6)

        x7 = torch.cat((x4, x5, x6), dim=1)
        x7 = F.relu(self.conv5(x7))
        x7 = self.Batchnorm5(x7)

        x8 = torch.cat((x5, x6, x7), dim=1)
        x8 = self.pool2(x8)

        x9 = F.relu(self.conv6(x8))
        x9 = self.Batchnorm6(x9)
        x10 = torch.cat((x8, x9), dim=1)
        x10 = F.relu(self.conv7(x10))
        x10 = self.Batchnorm7(x10)

        x11 = torch.cat((x8, x9, x10), dim=1)
        x11 = F.relu(self.conv8(x11))

        x12 = self.gap(x11)
        x13 = self.fc(x12)

        x13 = x13.view(-1, 10)
        return F.log_softmax(x13, dim=-1)
