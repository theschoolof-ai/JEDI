import torch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
batch_size = 256
num_workers = 1
pin_memory = True
input_size = (3,32,32)