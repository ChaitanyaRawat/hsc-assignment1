
import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import List, Tuple, Optional, Union
import os
import imageio
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def plot_reconstruction_samples(
    original_images: torch.Tensor,
    reconstructed_images: torch.Tensor,
    save_path: Optional[str] = None,
    title: str = "Original vs Reconstructed",
    max_samples: int = 8
) -> None:
    
    n_samples = min(len(original_images), max_samples)
    
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 2, 4))
    
    for i in range(n_samples):
        
        orig_img = original_images[i].numpy().reshape(28, 28)
        axes[0, i].imshow(orig_img, cmap='gray')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        
        recon_img = reconstructed_images[i].numpy().reshape(28, 28)
        axes[1, i].imshow(recon_img, cmap='gray')
        axes[1, i].set_title(f'Reconstructed {i+1}')
        axes[1, i].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()  


def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_recon_losses: List[float],
    val_recon_losses: List[float],
    train_kl_losses: List[float],
    val_kl_losses: List[float],
    save_path: Optional[str] = None
) -> None:
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(train_losses) + 1)
    
    
    axes[0].plot(epochs, train_losses, 'b-', label='Training', alpha=0.7)
    axes[0].plot(epochs, val_losses, 'r-', label='Validation', alpha=0.7)
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    
    axes[1].plot(epochs, train_recon_losses, 'b-', label='Training', alpha=0.7)
    axes[1].plot(epochs, val_recon_losses, 'r-', label='Validation', alpha=0.7)
    axes[1].set_title('Reconstruction Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    
    axes[2].plot(epochs, train_kl_losses, 'b-', label='Training', alpha=0.7)
    axes[2].plot(epochs, val_kl_losses, 'r-', label='Validation', alpha=0.7)
    axes[2].set_title('KL Divergence Loss')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()  


def plot_interpolation_grid(
    interpolated_images: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Latent Space Interpolation",
    max_cols: int = 10
) -> None:
    
    n_images = len(interpolated_images)
    n_cols = min(n_images, max_cols)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(n_images):
        row = i // n_cols
        col = i % n_cols
        
        img = interpolated_images[i].reshape(28, 28)
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(f'Step {i+1}')
        axes[row, col].axis('off')
    
    
    for i in range(n_images, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    
    plt.close()  


def plot_latent_space_2d(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    save_path: Optional[str] = None,
    max_samples: int = 1000
) -> None:
    
    if model.latent_dim != 2:
        print(f"Cannot plot {model.latent_dim}D latent space. Only 2D is supported.")
        return
    
    model.eval()
    latent_vectors = []
    labels = []
    
    with torch.no_grad():
        sample_count = 0
        for data, label in data_loader:
            if sample_count >= max_samples:
                break
            
            data = data.to(device)
            z = model.encode(data)
            
            latent_vectors.append(z.cpu().numpy())
            labels.append(label.numpy())
            
            sample_count += len(data)
    
    
    latent_vectors = np.concatenate(latent_vectors, axis=0)[:max_samples]
    labels = np.concatenate(labels, axis=0)[:max_samples]
    
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        latent_vectors[:, 0], 
        latent_vectors[:, 1], 
        c=labels, 
        cmap='tab10', 
        alpha=0.7,
        s=1
    )
    plt.colorbar(scatter)
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('2D Latent Space Visualization')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    
    plt.close()  


def create_morphing_video(
    frames_dir: str,
    output_path: str,
    fps: int = 10,
    loop_back: bool = True
) -> str:
    
    
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    
    if not frame_files:
        print(f"No frame files found in {frames_dir}")
        return None
    
    
    images = []
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        img = imageio.imread(frame_path)
        images.append(img)
    
    
    if loop_back:
        images.extend(images[-2:0:-1])  
    
    
    duration = 1.0 / fps
    
    if output_path.endswith('.gif'):
        imageio.mimsave(output_path, images, duration=duration, loop=0)
    else:
        imageio.mimsave(output_path, images, fps=fps)
    
    print(f"Video created: {output_path}")
    return output_path


def calculate_reconstruction_metrics(
    original_images: torch.Tensor,
    reconstructed_images: torch.Tensor
) -> dict:
   
    
    orig_imgs = original_images.cpu().numpy().reshape(-1, 28, 28)
    recon_imgs = reconstructed_images.cpu().numpy().reshape(-1, 28, 28)
    
    
    mse_values = []
    ssim_values = []
    
    for i in range(len(orig_imgs)):
        
        mse = np.mean((orig_imgs[i] - recon_imgs[i]) ** 2)
        mse_values.append(mse)
        
        
        ssim_val = ssim(
            orig_imgs[i], 
            recon_imgs[i], 
            data_range=orig_imgs[i].max() - orig_imgs[i].min()
        )
        ssim_values.append(ssim_val)
    
    return {
        'mse_mean': np.mean(mse_values),
        'mse_std': np.std(mse_values),
        'ssim_mean': np.mean(ssim_values),
        'ssim_std': np.std(ssim_values),
        'mse_values': mse_values,
        'ssim_values': ssim_values
    }


def plot_generated_samples(
    model: torch.nn.Module,
    device: torch.device,
    num_samples: int = 16,
    save_path: Optional[str] = None
) -> None:
    
    model.eval()
    
    with torch.no_grad():
        
        generated_images = model.sample(num_samples, device)
        generated_images = generated_images.cpu().numpy()
    
    
    n_cols = 4
    n_rows = (num_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        row = i // n_cols
        col = i % n_cols
        
        img = generated_images[i].reshape(28, 28)
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(f'Sample {i+1}')
        axes[row, col].axis('off')
    
    
    for i in range(num_samples, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle('Generated Samples from Random Latent Vectors', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    
    plt.close()  


def save_tensor_as_image(
    tensor: torch.Tensor,
    save_path: str,
    normalize: bool = True
) -> None:
    
    
    if tensor.dim() == 1:
        img_array = tensor.cpu().numpy().reshape(28, 28)
    else:
        img_array = tensor.cpu().numpy()
    
    
    if normalize:
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
    
    
    img_array = (img_array * 255).astype(np.uint8)
    img = Image.fromarray(img_array, mode='L')
    img.save(save_path)


def create_comparison_grid(
    image_sets: List[Tuple[torch.Tensor, str]],
    save_path: Optional[str] = None,
    title: str = "Image Comparison"
) -> None:
    
    n_sets = len(image_sets)
    n_samples = len(image_sets[0][0])
    
    fig, axes = plt.subplots(n_sets, n_samples, figsize=(n_samples * 1.5, n_sets * 1.5))
    
    if n_sets == 1:
        axes = axes.reshape(1, -1)
    elif n_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for set_idx, (images, label) in enumerate(image_sets):
        for img_idx in range(n_samples):
            img = images[img_idx].cpu().numpy().reshape(28, 28)
            axes[set_idx, img_idx].imshow(img, cmap='gray')
            
            if img_idx == 0:
                axes[set_idx, img_idx].set_ylabel(label, rotation=90)
            
            axes[set_idx, img_idx].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison grid saved to: {save_path}")
    
    plt.close()  