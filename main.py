from __future__ import print_function

import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets

from utils import Logger

def mnist_data():
    compose = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    out_dir = './dataset'
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

# Load Data
data = mnist_data()

# Create loader with data, so that we can iterate over it
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)

# Num batches
num_batches = len(data_loader)

# print(num_batches)
# print(data[0])

class DiscriminatorNet(torch.nn.Module):
    # A three-layer discriminative neural network
    def __init__(self):
        super(DiscriminatorNet, self).__init__()

        n_features = 784  # 28x28 images
        n_out = 1

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.out = nn.Sequential(
            nn.Linear(256, n_out),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.hidden0(x),
        x = self.hidden1(x),
        x = self.hidden2(x),
        x = self.out(x)
        return x

discriminator = DiscriminatorNet()

# Flatten 28x28 images
def images_to_vectors(images):
    return images.view(images.size(0), 784)

# Unflatten images
def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)

class GeneratorNet(torch.nn.Module):
    # A three hidden-layer generative neural netword

    def __init__(self):
        super(GeneratorNet, self).__init__()

        n_features = 100
        n_out = 784

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2)
        )

        self.hidden1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )

        self.out = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

generator = GeneratorNet()

# Generate random noise
def noise(size):
    # Generates a 1D vector of gaussian sampled values
    n = Variable(torch.randn(size, 100))
    return n


d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

loss = nn.BCELoss()

# Generate ones (the output for real images)
def ones_target(size):
    # Tensor containing ones, with shape = size
    data = Variable(torch.ones(size, 1))
    return data

# Generate zeros (the output for fake images)
def zeros_target(size):
    # Tensor containing zeros with shape = size
    data = Variable(torch.zeros(size, 1))
    return data


def train_discriminator(optimizer, real_data, fake_data):
    N = real_data.size(0)

    # Reset gradients
    optimizer.zero_grad()


    # 1.1 Train on real data
    prediction_real = discriminator(real_data)

    # Calculate error and backpropagate
    error_real = loss(prediction_real, ones_target(N))
    error_real.backward()


    # 1.2 Train on fake data
    prediction_fake = discriminator(fake_data)

    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, zeros_target(N))
    error_fake.backward()


    # 1.3 Update weights with gradients
    optimizer.step()

    
    # Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake


