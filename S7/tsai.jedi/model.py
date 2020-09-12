# Create and view model architecture
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# sys.path.append(".")
from com.tsai.jedi.batchnorm import GhostBatchNorm, depthwise_separable_conv
from com.tsai.jedi import config



def model_summary(model_, input_):

    use_cuda = config.use_cuda
    device = config.device
    arch = model_.to(device)
    return summary(arch, input_)


class Net(nn.Module):
    # BN_flag 0: normal batchnorm; 1: Ghost batchnorm
    def batch_norm(self, channels, BN_flag):
        if BN_flag == 1:
            return GhostBatchNorm(channels, num_splits=2, weight=False)
        else:
            return nn.BatchNorm2d(channels)

    def __init__(self, BN_flag):
        super(Net, self).__init__()

        # Convolution Block-1 ###################################
        # Input:32x32  Outout:32x32 RF:3x3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)  # input -? OUtput? #32,32
        self.batchnorm1 = self.batch_norm(32, BN_flag=1)

        # Input:32x32  Outout:32x32 RF:5x5
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)  # 32,32
        self.batchnorm2 = self.batch_norm(32, BN_flag=1)

        self.dp2 = nn.Dropout(0.10)
        # Transition Block-1 ###################################
        # Input:32x32  Outout:16x16 RF:10x10
        self.pool1 = nn.MaxPool2d(2, 2)  # 32,16

        # Input:16x16  Outout:16x16 RF:10x10
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, padding=1)  # 16,16
        self.batchnorm3 = self.batch_norm(16, BN_flag=1)
        self.dp3 = nn.Dropout(0.10)

        # Convolution Block-2 ###################################
        # Input:16x16  Outout:16x16 RF:12x12

        self.conv4 = depthwise_separable_conv(nin=16, nout=64, kernel_size_=3, padding_=1)  # 16,16
        self.batchnorm4 = self.batch_norm(64, BN_flag=1)
        self.dp4 = nn.Dropout(0.10)

        # Input:16x16  Outout:8x8 RF:24x24
        self.pool2 = nn.MaxPool2d(2, 2)

        # Transition Block-2 ###################################
        # Input:8x8  Outout:8x8 RF:26x26
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=1)  # 8,8
        self.batchnorm6 = self.batch_norm(32, BN_flag=1)
        self.dp6 = nn.Dropout(0.10)

        # Convolution Block-3 ###################################
        # Input:8x8  Outout:8x8 RF:31x31
        self.conv7 = nn.Conv2d(in_channels=32, out_channels=128, dilation=2, kernel_size=3, padding=2)  # 8,8
        self.batchnorm7 = self.batch_norm(128, BN_flag=1)
        self.dp7 = nn.Dropout(0.10)

        # Input:8x8  Outout:8x8 RF:36x36
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=128, dilation=2, kernel_size=3, padding=1)  # 8,8
        self.batchnorm8 = self.batch_norm(128, BN_flag=1)
        self.dp8 = nn.Dropout(0.10)

        # MP
        # Input:8x8  Outout:4x4 RF:72x72
        self.pool3 = nn.MaxPool2d(2, 2)

        # Input:4x4  Outout:4x4 RF:72x72
        self.conv9 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1)

        self.avgp = nn.AvgPool2d(kernel_size=4)
        self.conv11 = nn.Conv2d(64, 10, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batchnorm1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchnorm2(x)
        x = self.dp2(x)

        x = self.pool1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchnorm3(x)
        x = self.dp3(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchnorm4(x)
        x = self.dp4(x)

        # x = self.conv5(x)
        # x = F.relu(x)
        # x = self.batchnorm5(x)
        # x = self.dp5(x)

        x = self.pool2(x)

        x = self.conv6(x)
        x = F.relu(x)
        x = self.batchnorm6(x)
        x = self.dp6(x)

        x = self.conv7(x)
        x = F.relu(x)
        x = self.batchnorm7(x)
        x = self.dp7(x)

        x = self.conv8(x)
        x = F.relu(x)
        x = self.batchnorm8(x)
        x = self.dp8(x)

        x = self.pool3(x)

        x = self.conv9(x)
        x = F.relu(x)
        # x = self.batchnorm9(x)

        x = self.conv10(x)
        x = F.relu(x)
        # x = self.batchnorm10(x)
        # x = self.dp8(x)

        x = self.avgp(x)
        x = self.conv11(x)

        x = x.view(-1, 10)
        return F.log_softmax(x)