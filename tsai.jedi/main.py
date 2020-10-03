import sys

sys.path.append('D:/ML/EVA/JEDI/tsai.jedi/Models')
# from S8_resnet import ResNet18
from S7 import model_summary

from S9_resnet import resnet18

# from QuizDNN import Netquiz

sys.path.append("tsai.jedi")
import torch
import config
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from Engine_train_test import train, test
from dataloader import train_loader_CIFAR10_alb, test_loader_CIFAR10_alb

print(config.input_size_CIFAR10)
# model_ = Netquiz(0).to(config.device)
model_ = resnet18().to(config.device)
print(model_summary(model_, config.input_size_CIFAR10))

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

accu = []
loss_test = []

# optimizer = optim.SGD(model_.parameters(), lr=0.016, momentum=0.9, weight_decay=5e-4)
optimizer = optim.SGD(model_.parameters(), lr=3.94E-03, momentum=0.9, weight_decay=5e-4)

# scheduler = StepLR(optimizer, step_size=2, gamma=0.92)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', verbose=False)

valid_acc = []
loss_test_ = []
l1_regularization = [0, 0]
for epoch in range(1, 50):
    train(model_, config.device, train_loader_CIFAR10_alb, optimizer, epoch, l1_regularization=[0, 1])
    #scheduler.step()
    valid_a, valid_l = test(model_, config.device, test_loader_CIFAR10_alb)
    scheduler.step(valid_l)
    # Appending to loss and accuracy lists
    valid_acc.append(valid_a)
    loss_test_.append(valid_l)

accu.append(valid_acc)
loss_test.append(loss_test_)

# torch.save(model_, 'D:/ML/EVA/JEDI/tsai.jedi/model_objects/s9_resnet_albu_v2.pt')
