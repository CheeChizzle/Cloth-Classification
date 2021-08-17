from typing import ForwardRef
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random



def seed_all(seed):
    print(f"SEEDING WITH {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SingleViewNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(6, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 20, 3),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(3380, 120),
            nn.BatchNorm1d(120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Linear(84, 6)
        )

    def forward(self, x):
        return self.net(x)

class MultiViewNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = SingleViewNet()
    
    def forward(self,images):
        # images.shape == B V C H W
        first_view = images[:,0,...] # B C H W
        second_view = images[:,1,...]
        third_view = images[:,2,...]
        fourth_view = images[:,3,...]

        first_view_score = self.net(first_view)
        second_view_score = self.net(second_view)
        third_view_score = self.net(third_view)
        fourth_view_score = self.net(fourth_view)

        score = first_view_score + second_view_score + third_view_score + fourth_view_score

        # scores.shape  == B K
        return score

loss_func = nn.CrossEntropyLoss()


class SingleViewResNet(nn.Module):
    def __init__(self):
        super().__init__()
        temporary_model = torchvision.models.resnet50(pretrained=True)
        layers = list(temporary_model.children())
        layers = layers[:-1]
        shape = layers[-1].shape[1]
        self.net = nn.Sequential(
            *layers,
            nn.Linear(shape, 6)
        )

    def forward(self, x):
        return self.net(x)

# class MultiViewResNet(nn.Module):