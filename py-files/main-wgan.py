# main.py
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from model import Generator, DiscriminatorWithAutoencoder
from dataloader import get_dataloader
from trainergan import train_loop
# Hyperparameters
img_size = 64  # Image size
latent_dim = 128  # Latent dimension for the generator
img_channels = 1
batch_size = 64
epochs = 175
learning_rate = 0.0002
beta1 = 0.9
beta2 = 0.999
n_critic = 5  # Number of discriminator updates per generator update
lambda_gp = 10  # Gradient penalty lambda hyperparameter

# DataLoader
dataset = get_dataloader()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
generator = Generator(latent_dim, img_channels, img_size).to(device)
discriminator = DiscriminatorWithAutoencoder(img_channels, img_size).to(device)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, beta2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, beta2))



if __name__ == '__main__':  # Ensure this block is properly used
    # Call train loop function with all necessary arguments
    train_loop(generator, discriminator, optimizer_G, optimizer_D, dataset, device, epochs, n_critic, lambda_gp, latent_dim)

    # Save trained models
    torch.save(generator.state_dict(), 'generatorwithCNN-res.pth')
    torch.save(discriminator.state_dict(), 'discriminatorwithCNN-res.pth')
'''
# Call train loop function with all necessary arguments
train_loop(generator, discriminator, optimizer_G, optimizer_D, dataset, device, epochs, n_critic, lambda_gp, latent_dim)

# Save trained models
torch.save(generator.state_dict(), 'generatorwithCNN-res.pth')
torch.save(discriminator.state_dict(), 'discriminatorwithCNN-res.pth')
'''