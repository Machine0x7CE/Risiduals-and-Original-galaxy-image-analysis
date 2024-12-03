import torch
import torch.optim as optim
from model import Generator, DiscriminatorWithAutoencoder, Encoder
from trainer import display_images_and_mse, train
from dataloader import get_dataloader

# Hyperparameters
img_size = 64  # Image size
latent_dim = 128  # Latent dimension for the generator
img_channels = 1
batch_size = 64
learning_rate = 0.0002
beta1 = 0.9
beta2 = 0.999
n_critic = 5  # Number of discriminator updates per generator update
lambda_gp = 10  # Gradient penalty lambda hyperparameter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from dataloader import get_dataloader  # Ensure both train and val loaders are accessible

def main():
    # Load data
    train_dataloader, val_dataloader = get_dataloader()  # Fetch both loaders

    # Initialize models
    generator = Generator(latent_dim, img_channels, img_size).to(device)
    discriminator = DiscriminatorWithAutoencoder(img_channels, img_size).to(device)
    encoder = Encoder(img_channels, img_size, latent_dim).to(device)

    # Load pre-trained models
    generator.load_state_dict(torch.load('generatorwithCNN-res.pth'))
    discriminator.load_state_dict(torch.load('discriminatorwithCNN-res.pth'))

    # Set models to evaluation mode
    generator.eval()
    discriminator.eval()

    # Train and display
    train(train_dataloader, generator, discriminator, encoder)
    torch.save(encoder.state_dict(), 'trained_encoder_cnn-res.pth')


if __name__ == "__main__":
    main()
