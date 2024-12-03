import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from model import Generator, DiscriminatorWithAutoencoder, Encoder
from dataloader import get_dataloader

# Constants
img_size = 64  # Image size (64x64)
latent_dim = 128  # Latent dimension for Generator and Encoder
criterion = nn.MSELoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to save losses as JSON
def save_losses_to_json(losses, filename="losses_enc.json"):
    with open(filename, "w") as f:
        json.dump({"total_loss_per_epoch": losses}, f)

def display_images_and_mse(generator, discriminator, encoder, dataloader):
    with torch.no_grad():
        data_sample, = next(iter(dataloader))
        data_sample = data_sample.cuda()  # Move to GPU
        latent_encoder_sample = encoder(data_sample)
        images_from_encoder = generator(latent_encoder_sample).detach()

        latent_random_sample = torch.randn(data_sample.size(0), 128).cuda()  # Move to GPU
        images_from_generator = generator(latent_random_sample).detach()

        output_discriminator_encoder, _, _ = discriminator(images_from_encoder)
        output_discriminator_generator, _, _ = discriminator(images_from_generator)

        output_discriminatorcriminator_encoder = output_discriminatorcriminator_encoder.detach()
        output_discriminator_generator = output_discriminator_generator.detach()

        mse_images = criterion(images_from_encoder, images_from_generator).item()

        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for i in range(5):
            axes[0, i].imshow(images_from_encoder[i].cpu().squeeze(0), cmap='gray')
            axes[0, i].set_title(f"Enc: {output_discriminator_encoder[i].item():.2f}")
            axes[0, i].axis('off')

            axes[1, i].imshow(images_from_generator[i].cpu().squeeze(0), cmap='gray')
            axes[1, i].set_title(f"generator: {output_discriminator_generator[i].item():.2f}")
            axes[1, i].axis('off')
        plt.suptitle(f'Epoch {epoch + 1}, MSE Between Images: {mse_images:.4f}')
        plt.show()

# Training Loop with Checkpoints
def train(dataloader, generator, discriminator, encoder, num_epochs=250):
    losses_per_epoch = []
    criterion = nn.MSELoss()
    optimizer = optim.Adam(encoder.parameters(), lr=0.001, betas=(0.9, 0.99))

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0  # Track total loss for the epoch
        num_batches = 0

        with tqdm(dataloader, unit="batch", desc=f"Epoch {epoch + 1}/{num_epochs}") as tepoch:
            for data_batch in tepoch:
                # Fetch the data tensor (assuming no labels in TensorDataset)
                data_batch = data_batch[0].to(device)  # Move the data to the GPU

                # Generate latent vectors using the Encoder
                latent_vectors = encoder(data_batch)

                # Generate images using the Generator
                generated_images = generator(latent_vectors)

                # Compute image loss
                img_loss = torch.mean((data_batch - generated_images) ** 2)

                # Pass the original image through Encoder and Generator
                latent_representation = encoder(data_batch)
                generated_img = generator(latent_representation)

                # Get feature vectors from Discriminator
                _, original_features, _ = discriminator(data_batch)
                _, generated_features, _ = discriminator(generated_img)

                # Compute feature loss
                feat_loss = torch.mean((original_features - generated_features) ** 2)

                # Combine losses
                loss = img_loss + feat_loss

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update progress bar
                tepoch.set_postfix(loss=loss.item(), img_loss=img_loss.item(), feat_loss=feat_loss.item())

                # Accumulate total loss
                total_loss += loss.item()
                num_batches += 1

        # Store average loss for the epoch
        avg_loss = total_loss / num_batches
        losses_per_epoch.append(avg_loss)

    # Plot the total loss per epoch
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), losses_per_epoch, marker='o')
    plt.title('Total Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.grid(True)
    plt.show()

    # Save losses as JSON
    save_losses_to_json(losses_per_epoch)

if __name__ == "__main__":
    dataloader = get_dataloader('norm-res-95.npy')
    train(dataloader)