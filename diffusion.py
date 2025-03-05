import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
import os
from sentence_transformers import SentenceTransformer
from torchvision.datasets import ImageFolder
from torch.utils.checkpoint import checkpoint


# local imports
from models.unet import ConditionalUNet
from preprocess import load_data, load_labels
from plot_func import visualize_forward_process, plot_training_progress, create_diffusion_gif
from models.text_encoder import get_word_embeddings

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
batch_size = 256
image_size = 128
channels = 3
epochs = 200
lr = 1e-4
num_timesteps = 1000
min_beta = 1e-4
max_beta = 0.01
num_classes = 10


# Diffusion hyperparameters
def linear_beta_schedule(timesteps):
    """
    Linear beta schedule from min_beta to max_beta
    """
    return torch.linspace(min_beta, max_beta, timesteps)

# Calculate diffusion parameters
betas = linear_beta_schedule(num_timesteps)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

# Forward process (noising)
def q_sample(x_start, t, noise=None):
    """
    Add noise to the input image at timestep t
    """
    if noise is None:
        noise = torch.randn_like(x_start)
    
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def extract(a, t, shape):
    """
    Extract appropriate indices from 'a' based on 't' and reshape to match 'shape'
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(shape) - 1))).to(t.device)



# Loss function
def p_losses(denoise_model, x_start, t, y, noise=None):
    """
    Calculate the loss for training the diffusion model
    """
    if noise is None:
        noise = torch.randn_like(x_start)
        
    # Get noisy version of image at timestep t
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    # Predict the noise
    predicted_noise = denoise_model(x_noisy, t, y)
    
    # Calculate loss
    loss = F.mse_loss(predicted_noise, noise)
    return loss

# Sampling (denoising)
@torch.no_grad()
def p_sample(model, x, t, t_index, y):
    """
    Sample from the model at timestep t
    """
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    # Equation 11 in the paper
    # Use predicted noise to predict x_0
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t, y) / sqrt_one_minus_alphas_cumprod_t
    )
    
    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Add noise scaled by the variance for timestep t
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(model, shape, y, device, save_all=False):
    """
    Generate samples using the complete denoising process
    
    Args:
        model: The UNet model
        shape: Shape of the image to generate
        y: Class labels
        device: Device to run on
        save_all: Whether to save all intermediate steps (for visualization)
    
    Returns:
        A list of images from the sampling process
    """
    batch_size = shape[0]
    
    # Start from pure noise
    img = torch.randn(shape, device=device)
    imgs = []
    
    # Save steps for visualization
    save_interval = max(num_timesteps // 60, 1)  # Save ~60 frames
    
    for i in tqdm(reversed(range(0, num_timesteps)), desc="Sampling"):
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)
        img = p_sample(model, img, t, i, y)
        
        # Save intermediate steps
        if i % save_interval == 0 or i < 10 or i == num_timesteps - 1:
            imgs.append(img.cpu().clone())
            
    return imgs

@torch.no_grad()
def sample(model, image_size, batch_size, y, channels=1, device=device, save_all=False):
    """
    Generate samples with specified labels
    """
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size), 
                         y=y, device=device, save_all=save_all)



def main():

    # Create output directory
    output_dir = "output_ImageNet"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load data
    trainloader, dataset = load_data()

    # Obatin label enbeddings
    label2text =  load_labels(dataset)
    label2emb = get_word_embeddings(label2text)
    
    # Create model
    model = ConditionalUNet(in_channels=channels, out_channels=channels, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, foreach=True)
    scaler = torch.GradScaler('cuda')
    
    
    # For tracking loss history
    loss_history = []
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (images, labels_idx) in enumerate(progress_bar):
            label_emb = label2emb[labels_idx]
            images = images.to(device)
            label_emb = label_emb.to(device)
            
            # Reset gradients
            optimizer.zero_grad()
            
            with torch.autocast('cuda'):
                # Sample random timesteps
                t = torch.randint(0, num_timesteps, (images.shape[0],), device=device).long()
                
                # Calculate loss
                loss = p_losses(model, images, t, label_emb)
            
            # Update weights
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update progress bar
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=epoch_loss/(batch_idx+1))
        
        # Track average loss for this epoch
        avg_loss = epoch_loss/len(trainloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss}")
        
        # Model checkpoint
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(output_dir, f"model_epoch_{epoch+1}.pt"))
    
    plot_training_progress(loss_history, 
                        save_path=os.path.join(output_dir, f"loss.png"))

    print("Training completed!")
    
    # Final evaluation
    model.eval()

    save_path = os.path.join(output_dir, "generated_samples")
    visualize_forward_process(images[0], num_timesteps, q_sample, num_steps=10, output_dir=output_dir)

    # Generate a grid of samples for each digit
    print("Generating final samples for all digits...")
    for digit in range(10):
        # labels_idx = torch.ones(16, dtype=torch.long, device=device) * digit
        labels = label2emb[digit]
        labels = labels.repeat(16, 1).to(device)

        samples = sample(model, image_size, 16, labels, channels)
        grid = torchvision.utils.make_grid(samples[-1], nrow=4, normalize=True)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0).cpu())
        plt.title(f"Generated {label2text[digit]}")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"final_class_{label2text[digit]}.png"))
        plt.close()
         
    # Create animations for the generation process
    print("Creating animations for the diffusion process...")
    for digit in range(10):
        print(f"Creating animation for  {label2text[digit]}...")
        label = label2emb[digit].unsqueeze(0).to(device)
        samples = sample(model, image_size, 1, label, channels, save_all=True)
        create_diffusion_gif(samples, num_timesteps, filename=os.path.join(output_dir, f"class_{label2text[digit]}_generation.gif"))

if __name__ == "__main__":
    main()