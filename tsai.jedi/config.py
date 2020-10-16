import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
#batch_size is 128 for S6-S10, 512-s11
batch_size = 500
num_workers = 0
pin_memory = True
input_size_MNIST = (1, 28, 28)
input_size_CIFAR10 = (3, 32, 32)
