#Create and view model architecture
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('tsai.jedi')
from batchnorm import GhostBatchNorm
import config


def model_summary(model_, input_):
    from torchsummary import summary
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

        self.conv1 = nn.Conv2d(1, 8, 3)  # input -? OUtput? RF #28,26,3
        self.batchnorm1 = self.batch_norm(8, BN_flag)
        # self.dp1 = nn.Dropout(0.20)

        self.conv2 = nn.Conv2d(8, 20, 3)  # 26,24,5
        self.batchnorm2 = self.batch_norm(20, BN_flag)

        self.dp2 = nn.Dropout(0.10)

        self.pool1 = nn.MaxPool2d(2, 2)  # 24,12,10

        self.conv3 = nn.Conv2d(20, 10, 1)  # 12,12,10
        self.batchnorm3 = self.batch_norm(10, BN_flag)
        self.dp3 = nn.Dropout(0.10)

        self.conv4 = nn.Conv2d(10, 14, 3)  # 12,10,12
        self.batchnorm4 = self.batch_norm(14, BN_flag)
        self.dp4 = nn.Dropout(0.10)

        # self.pool2 = nn.MaxPool2d(2, 2)#10,5,24

        self.conv5 = nn.Conv2d(14, 18, 3)  # 10,8,12
        self.batchnorm5 = self.batch_norm(18, BN_flag)
        self.dp5 = nn.Dropout(0.10)

        self.conv6 = nn.Conv2d(18, 24, 3)  # 8,6,14
        self.batchnorm6 = self.batch_norm(24, BN_flag)
        self.dp6 = nn.Dropout(0.10)

        self.conv7_avgp = nn.AvgPool2d(kernel_size=6)
        self.conv8 = nn.Conv2d(24, 10, 1)  # 6,4,18

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

        x = self.conv5(x)
        x = F.relu(x)
        x = self.batchnorm5(x)
        x = self.dp5(x)

        x = self.conv6(x)
        x = F.relu(x)
        x = self.batchnorm6(x)
        x = self.dp6(x)

        x = self.conv7_avgp(x)
        x = self.conv8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)



