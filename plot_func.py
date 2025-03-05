import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from matplotlib.animation import FuncAnimation
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def visualize_forward_process(img, num_timesteps, q_sample, num_steps=10, output_dir='output'):
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


def create_diffusion_gif(images, num_timesteps,  filename='diffusion_process.gif', fps=5):
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
    
