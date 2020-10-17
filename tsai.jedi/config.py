import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# batch_size is 128 for S6-S10, 512-s11
batch_size = 64
num_workers = 0
pin_memory = True
input_size_MNIST = (1, 28, 28)
input_size_CIFAR10 = (3, 32, 32)

imagenet_directory = '/content/JEDI/data'


def update_var():
    global imagenet_path
    imagenet_path = imagenet_directory + '/tiny-imagenet-200/'
