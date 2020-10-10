import albumentations as A
import torchvision.transforms as trns
from albumentations.pytorch import ToTensorV2
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import Dataset


class train_transform_alb(Dataset):
    def __init__(self, image_list, label):
        self.image_list = image_list
        self.label = label
        self.aug = A.Compose([
            A.RandomCrop(32, 32, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Rotate((-8.0, 8.0), p=0.5),
            A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=[0.4914, 0.4822, 0.4465], p=0.5),
            # fill_value=[0.4914, 0.4822, 0.4465]
            A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        image = Image.fromarray(self.image_list[i]).convert('RGB')
        image = self.aug(image=np.array(image))['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.tensor(image, dtype=torch.float)
        label = self.label[i]
        label = torch.tensor(label, dtype=torch.long)
        return image, label


class test_transform_alb(Dataset):
    def __init__(self, image_list, label):
        self.image_list = image_list
        self.label = label
        self.aug = A.Compose([
            A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        image = Image.fromarray(self.image_list[i]).convert('RGB')
        image = self.aug(image=np.array(image))['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.tensor(image, dtype=torch.float)
        label = self.label[i]
        label = torch.tensor(label, dtype=torch.long)
        return image, label


class train_transform_s11(Dataset):
    def __init__(self, image_list, label):
        self.image_list = image_list
        self.label = label
        self.aug = A.Compose([
            A.PadIfNeeded(min_height=40, min_width=40,
                             always_apply=True, border_mode=0, value=[0, 0, 0]),
            A.RandomCrop(32, 32, p=1),
            A.HorizontalFlip(p=0.5),
            A.Rotate((-8.0, 8.0), p=0.5),
            A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=[0.4914, 0.4822, 0.4465], p=0.5),
            A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        image = Image.fromarray(self.image_list[i]).convert('RGB')
        image = self.aug(image=np.array(image))['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.tensor(image, dtype=torch.float)
        label = self.label[i]
        label = torch.tensor(label, dtype=torch.long)
        return image, label
