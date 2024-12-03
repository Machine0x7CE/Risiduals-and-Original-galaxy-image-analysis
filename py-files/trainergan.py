import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from dataloader import get_dataloader



# Loss functions
adversarial_loss = nn.BCELoss()
pixelwise_loss = nn.MSELoss()


lambda_gp = 10  # Gradient penalty lambda hyperparamete

def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    device = real_samples.device
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = discriminator(interpolates)[0]  # Extract first element
    fake = torch.ones(d_interpolates.shape, device=device, requires_grad=False)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Define the loss functions
def discriminator_loss(real_images, fake_images, discriminator):
    real_validity = discriminator(real_images)[0]  # Extract first element
    fake_validity = discriminator(fake_images)[0]  # Extract first element
    gradient_penalty = compute_gradient_penalty(discriminator, real_images, fake_images)
    d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
    return d_loss, gradient_penalty.item()  # Return both loss and gradient penalty

def generator_loss(fake_images, discriminator):
    fake_validity = discriminator(fake_images)[0]  # Extract first element
    g_loss = -torch.mean(fake_validity)
    return g_loss

def plot_losses(d_losses, g_losses, val_losses, gp_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label="Discriminator Loss")
    plt.plot(g_losses, label="Generator Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.plot(gp_losses, label="Gradient Penalty")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Generator, Discriminator, Validation Loss, and Gradient Penalty per Epoch")
    plt.show()

def save_losses_to_json(d_losses, g_losses, val_losses, gp_losses, filename='losses.json'):
    losses = {
        'discriminator_loss': d_losses,
        'generator_loss': g_losses,
        'validation_loss': val_losses,
        'gradient_penalty': gp_losses
    }
    with open(filename, 'w') as f:
        json.dump(losses, f)

def train_loop(generator, discriminator, optimizer_G, optimizer_D, dataset, device, epochs, n_critic, lambda_gp, latent_dim):
    # Initialize lists to store losses and generated samples
    train_dataloader, val_dataloader = get_dataloader()  # Call the function to get these
    d_losses = []
    g_losses = []
    val_losses = []
    gp_losses = []
    generated_samples = []  # List to hold generated samples per epoch

    for epoch in range(epochs):
        d_loss_total = 0.0
        g_loss_total = 0.0
        gp_loss_total = 0.0
        epoch_samples = []  # Temporarily store samples for the current epoch

        for i, (real_imgs,) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")):
            real_imgs = real_imgs.to(device)  # Move tensor to the correct device
            batch_size_actual = real_imgs.size(0)

            # Train Discriminator
            for _ in range(n_critic):
                optimizer_D.zero_grad()
                z = torch.randn(batch_size_actual, latent_dim).to(device)  # Move z to device
                gen_imgs = generator(z)
                d_loss, gp_loss = discriminator_loss(real_imgs, gen_imgs.detach(), discriminator)
                d_loss.backward()
                optimizer_D.step()
                d_loss_total += d_loss.item()
                gp_loss_total += gp_loss

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(batch_size_actual, latent_dim).to(device)  # Move z to device
            gen_imgs = generator(z)
            g_loss = generator_loss(gen_imgs, discriminator)
            g_loss.backward()
            optimizer_G.step()
            g_loss_total += g_loss.item()
            epoch_samples.append(gen_imgs.detach().cpu().numpy())  # Append generated images for this batch

        # Calculate average losses for the epoch
        d_losses.append(d_loss_total / len(train_dataloader))
        g_losses.append(g_loss_total / len(train_dataloader))
        gp_losses.append(gp_loss_total / (len(train_dataloader) * n_critic))
        generated_samples.append(epoch_samples)  # Append all samples from this epoch

        # Validation phase
        val_loss_total = 0.0
        with torch.no_grad():
            for real_imgs in val_dataloader:
                real_imgs = real_imgs[0].to(device)  # Move tensor to device
                batch_size_actual = real_imgs.size(0)
                z = torch.randn(batch_size_actual, latent_dim).to(device)  # Move z to device
                gen_imgs = generator(z)
                val_loss = generator_loss(gen_imgs, discriminator)
                val_loss_total += val_loss.item()

        val_losses.append(val_loss_total / len(val_dataloader))

                # Save generated images and compare real vs. fake at specific epochs
        if epoch == epochs - 1:
            samples = generator(torch.randn(16, latent_dim).to(device))
            samples = (samples + 1) / 2.0  # Rescale the images to [0, 1]
            samples = samples.detach().cpu().numpy()

            fig, axs = plt.subplots(4, 4, figsize=(8, 8))
            count = 0
            for i in range(4):
                for j in range(4):
                    axs[i, j].imshow(samples[count, 0, :, :], cmap='gray')
                    axs[i, j].axis('off')
                    count += 1
            axs[0, 0].set_title('Generated Images')

            # Plot real images
            real_samples = real_imgs[:16].detach().cpu().numpy()
            fig, axs = plt.subplots(4, 4, figsize=(8, 8))
            count = 0
            for i in range(4):
                for j in range(4):
                    axs[i, j].imshow(real_samples[count, 0, :, :], cmap='gray')
                    axs[i, j].axis('off')
                    count += 1
            axs[0, 0].set_title('Real Images')
            plt.show()

    # Save losses to a JSON file
    save_losses_to_json(d_losses, g_losses, val_losses, gp_losses)
    plot_losses(d_losses, g_losses, val_losses, gp_losses)

'''
        # Save generated images and compare real vs. fake at specific epochs
        if epoch == int(epochs * 0.5) - 1 or epoch == int(epochs * 0.75) - 1 or epoch == epochs - 1:
            samples = generator(torch.randn(16, latent_dim).to(device))
            samples = (samples + 1) / 2.0  # Rescale the images to [0, 1]
            samples = samples.detach().cpu().numpy()

            fig, axs = plt.subplots(4, 4, figsize=(8, 8))
            count = 0
            for i in range(4):
                for j in range(4):
                    axs[i, j].imshow(samples[count, 0, :, :], cmap='gray')
                    axs[i, j].axis('off')
                    count += 1
            axs[0, 0].set_title('Generated Images')

            # Plot real images
            real_samples = real_imgs[:16].detach().cpu().numpy()
            fig, axs = plt.subplots(4, 4, figsize=(8, 8))
            count = 0
            for i in range(4):
                for j in range(4):
                    axs[i, j].imshow(real_samples[count, 0, :, :], cmap='gray')
                    axs[i, j].axis('off')
                    count += 1
            axs[0, 0].set_title('Real Images')
            plt.show()
'''




'''
    # Save generated samples as images at the end of training
    for epoch_idx, epoch_samples in enumerate(generated_samples):
        for i, gen_sample in enumerate(epoch_samples):
            fig, axs = plt.subplots(1, 1, figsize=(4, 4))
            axs.imshow(gen_sample[0, 0, :, :], cmap='gray')  # Display the first image from the generated batch
            axs.axis('off')
            axs.set_title(f'Epoch {epoch_idx + 1} - Sample {i + 1}')
            plt.savefig(f'generated_sample_epoch{epoch_idx + 1}_sample{i + 1}.png')
            plt.close()

    # Save model checkpoints after training
    torch.save(generator.state_dict(), 'generator_final.pth')
    torch.save(discriminator.state_dict(), 'discriminator_final.pth')
'''

