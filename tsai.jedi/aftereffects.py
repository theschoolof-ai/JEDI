import torch
import config
from model import Net, model_summary
from Engine_train_test import train, test
from dataloader import train_loader,test_loader
from model import Net, model_summary
from batchnorm import GhostBatchNorm

model = torch.load('model_objects/GBN_model.pt')
model.eval()


tot = 0
data_ = []
target_ = []
pred_ = []
correct = 0
tot_correct = 0
indx = []
torch.manual_seed(1)
for data, target in test_loader:
    data, target = data.to(config.device), target.to(config.device)
    output = model(data)
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    equals = pred.eq(target.view_as(pred)).tolist()

