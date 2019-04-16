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
        print('d')
        print(type(x))
        x = self.hidden0(x),
        print(type(x))
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

        n_features = 100 # i think this is arbitrary
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
        print('g')
        print(type(x))
        x = self.hidden0(x)
        print(type(x))
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
    N = real_data.size(0) # batch size

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


def train_generator(optimizer, fake_data):
    N = fake_data.size(0) # batch size

    # Reset gradients
    optimizer.zero_grad()

    # Sample noise and generate fake data
    # See what the discriminator thinks of the fake data
    prediction = discriminator(fake_data)

    # Calculate error and backpropagate
    # We want the discriminator to output ones (it thinks the data is real)
    error = loss(prediction, ones_target(N))
    error.backward()

    # Update weights with gradients
    optimizer.step()

    return error

# We want to visualize how the training process develops as the GAN learns
num_test_samples = 16
test_noise = noise(num_test_samples)


# Create Logger instance
logger = Logger(model_name='VGAN', data_name='MNIST')

# Total number of epochs to train
num_epochs = 200

for epoch in range(num_epochs):
    for n_batch, (real_batch, _) in enumerate(data_loader):
        N = real_batch.size(0)

        real_data = Variable(images_to_vectors(real_batch))

        # 1. Train Discriminator

        # Generate fake data and detach (so gradients are not calculated for the generator)
        fake_data = generator(noise(N))

        # Train discriminator
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data)


        # 2. Train Generator

        # Generate fake data (not detaching this time since we do want the gradients calculated to train)
        fake_data = generator(noise(N))

        # Train generator
        g_error = train_generator(g_optimizer, fake_data)


        # Log batch error
        logger.log(d_error, g_error, epoch, n_batch, num_batches)

        # Display progress every few batches
        if (n_batch) % 100 == 0:
            test_images = vectors_to_images(generator(test_noise)).data

            logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches)

            # Display status logs
            logger.display_status(epoch, num_epochs, n_batch, num_batches, d_error, g_error, d_pred_real, d_pred_fake)

