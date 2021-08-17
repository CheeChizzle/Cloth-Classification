# imports
from loader import ClothDataset
import matplotlib
import torch
import numpy as np
import torchvision
matplotlib.use('MacOSX')

trainset = ClothDataset(32, keys="train.pkl", use_single_view=False)