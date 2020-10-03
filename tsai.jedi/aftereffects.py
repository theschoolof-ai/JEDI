import torch
import config
import matplotlib.pyplot as plt
from dataloader import classes_CIFAR10
import numpy as np


def get_image_with_target(model_path, number_of_img, dataloader, return_torch=False, is_classified=True):
    import numpy as np
    model = torch.load(model_path)
    model.eval()
    tot = 0
    data_ = []
    target_ = []
    pred_ = []
    correct = 0
    tot_correct = 0
    indx = []
    test_loader = dataloader
    for data, target in test_loader:
        data, target = data.to(config.device), target.to(config.device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        equals = pred.eq(target.view_as(pred)).tolist()

        for idx in range(0, len(equals)):
            if equals[idx] == [is_classified]:
                indx.append(idx)
                img = data[idx].cpu()
                img = img.numpy()
                data_.append(img.transpose(1, 2, 0))
                target_.append(target[idx].cpu().data.numpy())
                pred_.append(pred[idx][0].cpu().data.numpy())

        if len(indx) > number_of_img:
            if return_torch:
                return torch.tensor(data_[:number_of_img]), torch.tensor(target_[:number_of_img]), torch.tensor(
                    pred_[:number_of_img])
            else:
                return data_[:number_of_img], target_[:number_of_img], pred_[:number_of_img]


def plots(ims, figsize=(15, 15), rows=5, interp=False, titles=None, class_names=classes_CIFAR10):
    f = plt.figure(figsize=figsize)
    cols = len(ims) // rows if len(ims) % 2 == 0 else len(ims) // rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i + 1)
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        sp.axis('Off')
        plt.imshow(ims[i])
