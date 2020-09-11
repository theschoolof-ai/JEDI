from __future__ import print_function
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import sys

sys.path.append(".")

import config
from model import Net, model_summary
from Engine_train_test import train, test
from dataloader import train_loader,test_loader

print(model_summary(Net(1), config.input_size))

accu = []
loss_test = []

for idx in range(0, 5):
    # L1 regularization with Batch Normalization
    if idx == 0:
        use_cuda = config.use_cuda
        device = config.device
        model_ = Net(BN_flag=0).to(device)
        l1_regularization = [1, 0.001]
        optimizer = optim.SGD(model_.parameters(), lr=0.016, momentum=0.9)
        scheduler = StepLR(optimizer, step_size=2, gamma=0.92)

    # L2 regularization with Batch Normalization
    if idx == 1:
        use_cuda = config.use_cuda
        device = config.device
        model_ = Net(BN_flag=0).to(device)
        l1_regularization = [0, 0]
        optimizer = optim.SGD(model_.parameters(), lr=0.016, momentum=0.9, weight_decay=0.001)

    # L1 and L2 with Batch Normalization
    if idx == 2:
        use_cuda = config.use_cuda
        device = config.device
        model_ = Net(BN_flag=0).to(device)
        l1_regularization = [1, 0.001]
        optimizer = optim.SGD(model_.parameters(), lr=0.016, momentum=0.9, weight_decay=0.001)

    # with GBN
    if idx == 3:
        use_cuda = config.use_cuda
        device = config.device
        model_ = Net(BN_flag=1).to(device)
        optimizer = optim.SGD(model_.parameters(), lr=0.016, momentum=0.9)
        scheduler = StepLR(optimizer, step_size=2, gamma=0.92)

    # with L1 and L2 with GBN
    if idx == 4:
        use_cuda = config.use_cuda
        device = config.device
        model_ = Net(BN_flag=1).to(device)
        l1_regularization = [1, 0.001]
        optimizer = optim.SGD(model_.parameters(), lr=0.016, momentum=0.9, weight_decay=0.001)
        scheduler = StepLR(optimizer, step_size=2, gamma=0.92)

    valid_acc = []
    loss_test_ = []

    for epoch in range(1, 26):
        train(model_, device, train_loader, optimizer, epoch, l1_regularization)
        scheduler.step()
        valid_a, valid_l = test(model_, device, test_loader)
        # Appending to loss and accuracy lists
        valid_acc.append(valid_a)
        loss_test_.append(valid_l)

    accu.append(valid_acc)
    loss_test.append(loss_test_)

import pickle
with open("D:/ML/EVA/JEDI/model_objects/model_op_params.pickle","wb") as f:
    pickle.dump(accu, f)
    pickle.dump(loss_test, f)
#with open("C:/temp/test.pickle", "rb") as f:
#   testout1 = pickle.load(f)
#   testout2 = pickle.load(f)



