import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import os
import gradio as gr
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import time

# Local import for your model
from models.unet import ConditionalUNet

# Set random seed for reproducibility and define device
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
image_size = 128
channels = 3
num_timesteps = 1000
min_beta = 1e-4
max_beta = 0.01

# Diffusion hyperparameters and pre-calculations
def linear_beta_schedule(timesteps):
    return torch.linspace(min_beta, max_beta, timesteps)

betas = linear_beta_schedule(num_timesteps)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

def extract(a, t, shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(shape) - 1))).to(t.device)

@torch.no_grad()
def p_sample(model, x, t, t_index, y):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t, y) / sqrt_one_minus_alphas_cumprod_t)
    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

# Modified sampling loop to capture intermediate steps
@torch.no_grad()
def p_sample_loop(model, shape, y, device, return_intermediate=False, num_intermediate=10):
    batch_size = shape[0]
    img = torch.randn(shape, device=device)
    
    if return_intermediate:
        # Calculate which timesteps to save
        steps_to_save = np.linspace(0, num_timesteps-1, num_intermediate, dtype=int)
        steps_to_save = list(reversed(steps_to_save))
        intermediate_images = []
    
    for i in tqdm(reversed(range(num_timesteps)), desc="Sampling"):
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)
        img = p_sample(model, img, t, i, y)
        
        if return_intermediate and i in steps_to_save:
            intermediate_images.append(img.clone())
    
    if return_intermediate:
        return img, intermediate_images
    return img

# Sentence Transformer for text embeddings
sentence_model = None

def load_sentence_model():
    global sentence_model
    if sentence_model is None:
        print("Loading sentence transformer model...")
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    return sentence_model

def get_text_embedding(text):
    model = load_sentence_model()
    embedding = model.encode([text], show_progress_bar=False)
    return torch.tensor(embedding, dtype=torch.float32)

def load_model_and_label_map(model_path):
    print("Loading model from", model_path)
    model = ConditionalUNet(in_channels=channels, out_channels=channels, num_classes=1).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def tensor_to_image(tensor):
    # Normalize and convert tensor to numpy array
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    img_np = tensor[0].permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    return img_np

def generate_from_text(model, text_input, n_samples=1, show_process=False):
    embedding = get_text_embedding(text_input).to(device)
    embedding_batch = embedding.repeat(n_samples, 1)
    
    if show_process:
        sample, intermediates = p_sample_loop(
            model, 
            shape=(n_samples, channels, image_size, image_size), 
            y=embedding_batch, 
            device=device,
            return_intermediate=True,
            num_intermediate=10
        )
        return sample, intermediates
    else:
        sample = p_sample_loop(
            model, 
            shape=(n_samples, channels, image_size, image_size), 
            y=embedding_batch, 
            device=device
        )
        return sample, None

# Gradio Interface Functions
def generate_images(text_input, num_images, show_process, progress=gr.Progress()):
    progress(0, desc="Initializing...")
    try:
        n_samples = int(num_images)
        n_samples = max(1, min(4, n_samples))  # Limit to 4 samples max
    except:
        n_samples = 1
    
    progress(0.1, desc="Generating images...")
    samples, intermediates = generate_from_text(model, text_input, n_samples=n_samples, show_process=show_process)
    
    progress(0.8, desc="Processing results...")
    results = []
    
    for i in range(n_samples):
        if i < samples.shape[0]:
            img = tensor_to_image(samples[i:i+1])
            results.append(img)
    
    progress(0.9, desc="Finalizing...")
    
    if show_process and intermediates:
        process_images = [tensor_to_image(img[0:1]) for img in intermediates]
        return results, process_images
    else:
        return results, None

# Custom CSS for modern, minimalistic UI
custom_css = """
:root {
    --primary-color: #000000;
    --secondary-color: #f5f5f7;
    --accent-color: #0071e3;
    --text-color: #1d1d1f;
    --background-color: #ffffff;
    --border-radius: 8px;
    --shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

body {
    font-family: "SF Pro Text", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif;
    background-color: var(--background-color);
    color: var(--text-color);
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 40px 20px;
}

.header {
    text-align: center;
    margin-bottom: 40px;
}

.header h1 {
    font-size: 48px;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin-bottom: 16px;
    color: var(--primary-color);
}

.header p {
    font-size: 20px;
    color: var(--text-color);
    opacity: 0.8;
    max-width: 600px;
    margin: 0 auto;
}

.gr-button-primary {
    background-color: var(--accent-color) !important;
    border-color: var(--accent-color) !important;
}

.gr-form {
    border-radius: var(--border-radius) !important;
    box-shadow: var(--shadow) !important;
}

.gr-panel {
    border-radius: var(--border-radius) !important;
}

.gr-input {
    border-radius: var(--border-radius) !important;
}

.gr-prose h1, .gr-prose h2, .gr-prose h3 {
    font-weight: 600;
    letter-spacing: -0.5px;
}

.gr-gallery {
    gap: 16px !important;
}

.gr-gallery > div {
    border-radius: var(--border-radius) !important;
    overflow: hidden;
    box-shadow: var(--shadow) !important;
}

.gr-checkbox {
    background-color: var(--secondary-color) !important;
}

.gr-slider {
    --slider-color: var(--accent-color) !important;
}

.diffusion-process {
    display: flex;
    flex-direction: row;
    gap: 8px;
    overflow-x: auto;
    padding: 10px 0;
}

.diffusion-process img {
    height: 120px;
    border-radius: var(--border-radius);
    box-shadow: var(--shadow);
}

.gr-prose {
    max-width: 650px;
    margin: 0 auto;
}
"""

# Build the Gradio Interface
with gr.Blocks(css=custom_css, title="AI Image Generator") as app:
    with gr.Row(elem_classes="header"):
        gr.Markdown("# AI Image Generator")
        gr.Markdown("Generate beautiful images from text descriptions using diffusion models")
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                text_input = gr.Textbox(
                    label="", 
                    placeholder="Enter a description, like 'a red apple on a white table'",
                    lines=3
                )
                
                with gr.Row():
                    num_images = gr.Slider(
                        minimum=1, 
                        maximum=4, 
                        value=1, 
                        step=1, 
                        label="Number of Images"
                    )
                    show_process = gr.Checkbox(
                        label="Show Generation Process", 
                        value=False
                    )
                
                generate_btn = gr.Button("Generate", variant="primary")
                
        with gr.Column(scale=2):
            output_gallery = gr.Gallery(
                label="Generated Images",
                show_label=False,
                columns=2,
                height=500
            )
            
            with gr.Group(visible=False) as process_group:
                gr.Markdown("## Generation Process")
                gr.Markdown("See how the image evolves from random noise to the final result:")
                process_gallery = gr.Gallery(
                    label="Process Steps",
                    show_label=False,
                    columns=10,
                    height=150
                )
    
    # Handle visibility of process group
    show_process.change(
        fn=lambda x: gr.Group(visible=x), 
        inputs=show_process, 
        outputs=process_group
    )
    
    # Handle image generation
    generate_btn.click(
        generate_images,
        inputs=[text_input, num_images, show_process],
        outputs=[output_gallery, process_gallery]
    )
    
    gr.Markdown("""
    ### About
    
    This application uses a diffusion model to generate images from text descriptions. The model was trained on a dataset of images and their descriptions.
    
    - Enter a detailed text description
    - Select the number of images to generate
    - Enable "Show Generation Process" to see how the image evolves
    """)

if __name__ == "__main__":
    # Set your model path accordingly
    model_path = "output_ImageNet/model_epoch_250.pt"
    os.makedirs("output_visualization", exist_ok=True)
    model = load_model_and_label_map(model_path)
    app.launch()