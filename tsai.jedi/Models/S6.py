#Create and view model architecture
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append(".")
from batchnorm import GhostBatchNorm
import config


def model_summary(model_, input_):
    from torchsummary import summary
    use_cuda = config.use_cuda
    device = config.device
    arch = model_.to(device)
    return summary(arch, input_)


class Net(nn.Module):

    # Function to fetch GhostBatchNorm
    # flag = 1 for GhostBatchNorm and flag = 0 for batchnorm2d
    def batch_norm(self, channels, flag):
        if flag == 1:
            return GhostBatchNorm(channels, num_splits=2, weight=False)
        else:
            return nn.BatchNorm2d(channels)

    def __init__(self, flag):
        super(Net, self).__init__()

        # block1
        self.conv1 = nn.Conv2d(1, 8, 3, padding=True)  # o/p size:28; rf: 3
        self.Batchnorm1 = self.batch_norm(8, flag)
        self.conv2 = nn.Conv2d(8, 15, 3)  # o/p size: 26; rf: 5
        self.Batchnorm2 = self.batch_norm(15, flag)

        # transition block
        self.pool1 = nn.MaxPool2d(2, 2)  # o/p size: 13; rf: 6
        self.pool1trns = nn.Conv2d(15, 10, 1)  # o/p size: 13; rf: 6
        self.Batchnormtrns1 = self.batch_norm(10, flag)

        # block2
        self.conv3 = nn.Conv2d(10, 14, 3)  # o/p size: 11; rf: 10
        self.Batchnorm3 = self.batch_norm(14, flag)
        self.dp3 = nn.Dropout(p=0.10)
        self.conv4 = nn.Conv2d(14, 16, 3)  # o/p size: 9; rf: 14
        self.Batchnorm4 = self.batch_norm(16, flag)
        self.dp4 = nn.Dropout(p=0.10)
        self.conv5 = nn.Conv2d(16, 20, 3)  # o/p size: 7; rf: 18
        self.Batchnorm5 = self.batch_norm(20, flag)
        self.dp5 = nn.Dropout(p=0.10)

        # gap and 1X1
        self.conv6_avgp = nn.AvgPool2d(kernel_size=7)  # o/p size: 1; rf: 30
        self.pool2trns = nn.Conv2d(20, 10, 1)  # 6

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.Batchnorm1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.Batchnorm2(x)

        x = self.pool1(x)
        x = self.pool1trns(x)
        x = F.relu(x)
        x = self.Batchnormtrns1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.Batchnorm3(x)
        x = self.dp3(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.Batchnorm4(x)
        x = self.dp4(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.Batchnorm5(x)
        x = self.dp5(x)

        x = self.conv6_avgp(x)

        x = self.pool2trns(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
