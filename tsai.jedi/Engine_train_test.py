from __future__ import print_function
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

test_loss_ = []
train_loss = 0


def train(model, device, train_loader, optimizer, epoch, l1_regularization=[1, 0.001]):
    model.train()
    train_correct = 0
    train_loss = 0
    type = l1_regularization[0]
    l = l1_regularization[1]
    pbar = tqdm(train_loader, leave=False, position=0)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        output = model(data)
        optimizer.zero_grad()

        if type == 1:
            l1_regularization = 0
            for param in model.parameters():
                l1_regularization += torch.sum(abs(param))
            train_loss = F.nll_loss(output, target) + l * l1_regularization
        else:
            train_loss = F.nll_loss(output, target)

        train_loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        train_correct += pred.eq(target.view_as(pred)).sum().item()
        pbar.set_description(desc=f'loss={train_loss.item()} batch_id={batch_idx}')

    print('Epoch: {:.0f},LR: {}.\nTrain set: train Average loss: {:.4f}, train_Accuracy: {}/{} ({:.4f}%)\n'.format(
        epoch, optimizer.param_groups[0]['lr'], train_loss, train_correct, len(train_loader.dataset),
        100. * train_correct / len(train_loader.dataset)))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return 100. * correct / len(test_loader.dataset), test_loss