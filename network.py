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
    def __init__(self,  freeze_layers = False): # add another parameter that acts as a flag
        super().__init__()
        self.freeze_layers = freeze_layers
        temporary_model = torchvision.models.resnet50(pretrained=True)
        layers = list(temporary_model.children())
        layers = layers[:-1]
        if self.freeze_layers:
            for layer in layers:
                for param in layer.parameters():
                    param.requires_grad = False
        layers.append(nn.Flatten())
        layers.append(nn.Linear(2048, 128))
        layers.append(nn.BatchNorm1d(128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 128))
        layers.append(nn.BatchNorm1d(128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 128))
        layers.append(nn.BatchNorm1d(128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 128))
        layers.append(nn.BatchNorm1d(128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 84))
        layers.append(nn.BatchNorm1d(84))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(84, 6))
        self.net = nn.Sequential(
            *layers
        )
        

    def forward(self, x):
        result = self.net(x)
        # result = self.fc3(temp)
        return result 

class MultiViewResNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.net1 = SingleViewResNet(**kwargs)
        self.net2 = SingleViewResNet(**kwargs)
        self.net3 = SingleViewResNet(**kwargs)
        self.net4 = SingleViewResNet(**kwargs)
    
    def forward(self,images):
        # images.shape == B V C H W
        first_view = images[:,0,...] # B C H W
        second_view = images[:,1,...]
        third_view = images[:,2,...]
        fourth_view = images[:,3,...]

        first_view_score = self.net1(first_view)
        second_view_score = self.net2(second_view)
        third_view_score = self.net3(third_view)
        fourth_view_score = self.net4(fourth_view)

        score = first_view_score + second_view_score + third_view_score + fourth_view_score

        # scores.shape  == B K
        return score

# net = load_net()
# scores = [net(view) for view in views]
# maj voting or score adding
# 1 sv model for multiview classification

# my_emsemble = [load_net(net_path) for net_path in net_paths]
# scores = [net(img) for net in my_ensemble]
# multiple sv models for singleview classification


# class SingleViewRGBD(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = 

# class MultiViewRGBD(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = SingleViewRGBD()

# Dropout
# input: p (probability of randomly dropping out a unit)
# 1-p = probability of not being set to zero, p = probability of being set to zero
# if set to zero,
# set weight unit to zero and multiply weight vector by input
# else, pass?

# input: layer, p, is_training = F/T
# P.S. random.random() (gives # between 0.0-1.0 w/ equal probability)
# if is_training:
    # for weights and biases in layer:
    #      for unit in weights and biases:       
            # if random.random() <= p:
            #   unit =0
# else:
# (return original layer)

