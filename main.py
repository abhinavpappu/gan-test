from __future__ import print_function

import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets

from utils import Logger

def mnist_data():
    compose = transforms.Compose([
        transforms.ToTensor()
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])
    
    out_dir = './dataset'
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

# Load Data
