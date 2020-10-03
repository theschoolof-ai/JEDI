import sys
sys.path.append("tsai.jedi")
#from __future__ import print_function
import torch
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from datatransforms import train_transform_alb, test_transform_alb
import config

torch.manual_seed(1)
kwargs = {'num_workers': config.num_workers, 'pin_memory': config.pin_memory} if config.use_cuda else {}

#MNIST
train_loader_MNIST = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.RandomRotation((-8.0, 8.0), fill=(1,)),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=config.batch_size, shuffle=True, **kwargs)
test_loader_MNIST = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=config.batch_size, shuffle=True, **kwargs)
#CIFAR10
train_loader_CIFAR10 = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=True, download=True,
                     transform=transforms.Compose([
                         transforms.RandomCrop(32, padding=4),
                         transforms.RandomHorizontalFlip(),
                         transforms.RandomRotation((-8.0, 8.0)),
                         transforms.ToTensor(),
                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                     ])),
    batch_size=config.batch_size, shuffle=True, **kwargs)
test_loader_CIFAR10 = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])),
    batch_size=config.batch_size, shuffle=True, **kwargs)

#CIFAR-10 Albumentation
datasets.CIFAR10('../data', train=True, download=True)


def load_cifar10_data(filename):
    with open('../data/cifar-10-batches-py/' + filename, 'rb') as file:
        batch = pickle.load(file, encoding='latin1')

    features = batch['data']
    labels = batch['labels']
    return features, labels


batch_1, labels_1 = load_cifar10_data('data_batch_1')
batch_2, labels_2 = load_cifar10_data('data_batch_2')
batch_3, labels_3 = load_cifar10_data('data_batch_3')
batch_4, labels_4 = load_cifar10_data('data_batch_4')
batch_5, labels_5 = load_cifar10_data('data_batch_5')

test, label_test = load_cifar10_data('test_batch')

X_train = np.concatenate([batch_1, batch_2, batch_3, batch_4, batch_5], 0)
Y_train = np.concatenate([labels_1, labels_2, labels_3, labels_4, labels_5], 0)


def return_photo(batch_file):
    assert batch_file.shape[1] == 3072
    dim = np.sqrt(1024).astype(int)
    r = batch_file[:, 0:1024].reshape(batch_file.shape[0], dim, dim, 1)
    g = batch_file[:, 1024:2048].reshape(batch_file.shape[0], dim, dim, 1)
    b = batch_file[:, 2048:3072].reshape(batch_file.shape[0], dim, dim, 1)
    photo = np.concatenate([r, g, b], -1)
    return photo


X_train = return_photo(X_train)
X_test = return_photo(test)
Y_test = np.array(label_test)

classes_CIFAR10 = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

trainset = train_transform_alb(image_list=X_train, label=Y_train)
testset = test_transform_alb(image_list=X_test, label=Y_test)

train_loader_CIFAR10_alb = torch.utils.data.DataLoader(trainset,
                                                   batch_size=config.batch_size, shuffle=True, **kwargs)

test_loader_CIFAR10_alb = torch.utils.data.DataLoader(testset,
                                                  batch_size=config.batch_size, shuffle=True, **kwargs)
