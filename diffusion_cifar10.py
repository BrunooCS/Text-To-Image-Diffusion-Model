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

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
batch_size = 128
image_size = 28
channels = 3
epochs = 10
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


# Building blocks for U-Net
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ConditionalEmbedding(nn.Module):
    def __init__(self, num_classes, d_model):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, d_model)
        
    def forward(self, y):
        embeddings = self.embedding(y)
        return embeddings

# Modified Block class to better handle spatial dimensions
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            # Use kernel_size=2 instead of 4 for upsampling to avoid dimension issues
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 2, 2, 0)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            # Use kernel_size=2 instead of 4 for downsampling to avoid dimension issues
            self.transform = nn.Conv2d(out_ch, out_ch, 2, 2, 0)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        
    def forward(self, x, t):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend time embedding dimensions
        time_emb = time_emb[..., None, None]
        # Add time embedding
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)

# Modified UNet architecture to maintain appropriate spatial dimensions
class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=1, model_channels=64, out_channels=1, time_dim=256, embedding_size=384,num_classes=10):
        super().__init__()
        # Time embedding
        self.time_dim = time_dim
        self.embedding_size = embedding_size
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )
                
        # Initial projection
        self.conv0 = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Downsampling - reduced to 2 blocks instead of 3 for 28x28 images
        self.downs = nn.ModuleList([
            Block(model_channels, model_channels*2, time_dim+embedding_size),
            Block(model_channels*2, model_channels*4, time_dim+embedding_size)
        ])
        
        # Middle - uses standard convolution instead of another downsampling block
        self.middle_block = nn.Sequential(
            nn.Conv2d(model_channels*4, model_channels*4, 3, padding=1),
            nn.BatchNorm2d(model_channels*4),
            nn.ReLU(),
            nn.Conv2d(model_channels*4, model_channels*4, 3, padding=1),
            nn.BatchNorm2d(model_channels*4),
            nn.ReLU()
        )
        
        # Upsampling - reduced to 2 blocks to match the downsampling
        self.ups = nn.ModuleList([
            Block(model_channels*4, model_channels*2, time_dim+embedding_size, up=True),
            Block(model_channels*2, model_channels, time_dim+embedding_size, up=True)
        ])
        
        # Final layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(model_channels*2, model_channels, 3, padding=1),
            nn.BatchNorm2d(model_channels),
            nn.ReLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1)
        )
        
    def forward(self, x, t, y_emb):
        # Initial projection
        x = self.conv0(x)
        residual1 = x
        
        # Time embedding
        t = self.time_mlp(t)
        # Add conditional embedding
        t = torch.cat((t, y_emb), dim=1)
        
        # Down samples
        skips = []
        for layer in self.downs:
            x = layer(x, t)
            skips.append(x)
        
        # Middle - no time conditioning needed for simple convolutions
        x = self.middle_block(x)
        
        # Up samples
        for i, layer in enumerate(self.ups):
            # Concatenate with skip connection
            x = torch.cat([x, skips[-i-1]], dim=1)
            x = layer(x, t)
        
        # Concatenate with initial projection as residual
        x = torch.cat([x, residual1], dim=1)
        
        # Final convolution
        return self.final_conv(x)

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


# Load MNIST dataset
def load_data():
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    
    return trainloader, trainset

# Visualization Functions
def show_images(images, title="", save_path=None):
    """
    Show a batch of images
    
    Args:
        images: Tensor or list of tensors
        title: Title for the plot
        save_path: Path to save the image (optional)
    """
    if isinstance(images, list):
        images = torch.cat(images, dim=0)
    
    plt.figure(figsize=(10, 10))
    
    if len(images.shape) == 4:
        # Make a grid of multiple images
        grid_size = min(4, images.shape[0])
        grid = torchvision.utils.make_grid(images[:grid_size**2], nrow=grid_size, normalize=True)
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    else:
        # Single image
        plt.imshow(images.cpu().numpy())
    
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    # plt.show()

def visualize_forward_process(img, num_steps=10, output_dir='output'):
    """
    Visualize the forward (noising) process
    
    Args:
        img: Original image
        num_steps: Number of steps to visualize
    """
    plt.figure(figsize=(15, 3))
    
    # Show original image
    plt.subplot(1, num_steps+1, 1)
    plt.imshow(img.squeeze().cpu().permute(1, 2, 0))
    plt.title("Original")
    plt.axis('off')
    
    # Show noised images at different timesteps
    for i, t in enumerate(np.linspace(0, num_timesteps-1, num_steps).astype(int)):
        t_tensor = torch.tensor([t], device=device)
        noised_img = q_sample(img.unsqueeze(0), t_tensor)
        
        plt.subplot(1, num_steps+1, i+2)
        plt.imshow(noised_img.squeeze().cpu().permute(1, 2, 0))
        plt.title(f"t={t}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"forward_diffusion.png"), bbox_inches='tight')


def create_diffusion_gif(images, filename='diffusion_process.gif', fps=5):
    """
    Create a gif of the diffusion process
    
    Args:
        images: List of images from the sampling process
        filename: Output filename
        fps: Frames per second
    """
    # Make sure images are numpy arrays
    if isinstance(images, list) and isinstance(images[0], torch.Tensor):
        images = [img.squeeze().cpu().permute(1, 2, 0).numpy() for img in images]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.axis('off')
    
    def update(i):
        ax.clear()
        ax.imshow(images[i])
        ax.set_title(f'Diffusion Step {num_timesteps - i*10}', fontsize=15)
        ax.axis('off')
        return [ax]
    
    anim = FuncAnimation(fig, update, frames=len(images), interval=1000/fps)
    anim.save(filename, writer='pillow', fps=fps)
    plt.close()
    print(f"GIF saved as {filename}")

def visualize_batch_generation(model, class_labels, batch_size=4, save_dir='generated_samples'):
    """
    Generate and visualize a batch of images for each class
    
    Args:
        model: The trained diffusion model
        class_labels: List of class labels to generate
        batch_size: Number of images to generate per class
        save_dir: Directory to save the visualizations
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model.eval()
    
    plt.figure(figsize=(12, len(class_labels)*3))
    
    for i, label in enumerate(class_labels):
        # Create tensor of same class label
        labels = torch.ones(batch_size, dtype=torch.long, device=device) * label
        
        # Generate samples
        samples = sample(model, image_size, batch_size, labels, channels)
        
        # Get final samples
        final_samples = samples[-1]
        
        # Create grid
        grid = torchvision.utils.make_grid(final_samples, nrow=batch_size, normalize=True)
        
        # Display
        plt.subplot(len(class_labels), 1, i+1)
        plt.imshow(grid.permute(1, 2, 0).cpu())
        plt.title(f"Generated Samples for Class {label}")
        plt.axis('off')
        
        # Save class-specific grid
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0).cpu())
        plt.title(f"Generated Samples for Class {label}")
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f"class_{label}_samples.png"))
        plt.close()
        
        # Generate animation for this class (first sample only)
        anim_samples = sample(model, image_size, 1, torch.tensor([label], device=device), channels, save_all=True)
        create_diffusion_gif(anim_samples, filename=os.path.join(save_dir, f"class_{label}_animation.gif"))
    
    # Save the complete figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "all_classes.png"))
    plt.show()


def plot_training_progress(loss_history, save_path=None):
    """
    Plot training loss over epochs
    
    Args:
        loss_history: List of average losses per epoch
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, marker='o')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    
    # plt.show()


def get_word_embeddings(words):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(words, show_progress_bar=True)
    return torch.tensor(embeddings)


def main():

    label2name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    label2emb = get_word_embeddings(label2name)



    # Create output directory
    output_dir = "output_cifar10"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load data
    trainloader, dataset = load_data()
    
    # Create model
    model = ConditionalUNet(in_channels=channels, out_channels=channels, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Move diffusion parameters to device
    betas = linear_beta_schedule(num_timesteps).to(device)
    
    # For tracking loss history
    loss_history = []
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (images, labels_idx) in enumerate(progress_bar):
            labels = label2emb[labels_idx]
            images = images.to(device)
            labels = labels.to(device)
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Sample random timesteps
            t = torch.randint(0, num_timesteps, (images.shape[0],), device=device).long()
            
            # Calculate loss
            loss = p_losses(model, images, t, labels)
            
            # Update weights
            loss.backward()
            optimizer.step()
            
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
    visualize_forward_process(images[0], num_steps=10, output_dir=output_dir)

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
        plt.title(f"Generated {label2name[digit]}")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"final_class_{label2name[digit]}.png"))
        plt.close()
         
    # Create animations for the generation process
    print("Creating animations for the diffusion process...")
    for digit in range(10):
        print(f"Creating animation for  {label2name[digit]}...")
        label = label2emb[digit].unsqueeze(0).to(device)
        samples = sample(model, image_size, 1, label, channels, save_all=True)
        create_diffusion_gif(samples, filename=os.path.join(output_dir, f"class_{label2name[digit]}_generation.gif"))

if __name__ == "__main__":
    main()