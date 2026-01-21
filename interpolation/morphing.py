
import torch
import numpy as np
from typing import Tuple, List, Optional
import os


import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt


import sys
import os.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vae import VAE
from utils.visualization import plot_interpolation_grid, create_morphing_video


class LatentInterpolator:
    
    
    def __init__(self, model: VAE, device: torch.device):
        
        self.model = model
        self.device = device
        self.model.eval()
    
    def interpolate_images(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        num_steps: int = 30,
        interpolation_method: str = 'linear'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        with torch.no_grad():
            
            if image_a.dim() == 1:
                image_a = image_a.unsqueeze(0)
            if image_b.dim() == 1:
                image_b = image_b.unsqueeze(0)
            
            image_a = image_a.to(self.device)
            image_b = image_b.to(self.device)
            
            
            z_a = self.model.encode(image_a)
            z_b = self.model.encode(image_b)
            
            
            alpha_values = torch.linspace(0, 1, num_steps).to(self.device)
            
            
            if interpolation_method == 'linear':
                interpolated_z = self._linear_interpolation(z_a, z_b, alpha_values)
            elif interpolation_method == 'spherical':
                interpolated_z = self._spherical_interpolation(z_a, z_b, alpha_values)
            else:
                raise ValueError(f"Unknown interpolation method: {interpolation_method}")
            
            
            interpolated_images = self.model.decode(interpolated_z)
            
            return interpolated_images, interpolated_z, alpha_values.cpu()
    
    def _linear_interpolation(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        alpha_values: torch.Tensor
    ) -> torch.Tensor:
        
        
        interpolated_z = []
        
        for alpha in alpha_values:
            z_interp = (1 - alpha) * z_a + alpha * z_b
            interpolated_z.append(z_interp)
        
        return torch.cat(interpolated_z, dim=0)
    
    def _spherical_interpolation(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        alpha_values: torch.Tensor
    ) -> torch.Tensor:
        
        
        z_a_norm = z_a / torch.norm(z_a, dim=1, keepdim=True)
        z_b_norm = z_b / torch.norm(z_b, dim=1, keepdim=True)
        
        
        dot_product = torch.sum(z_a_norm * z_b_norm, dim=1, keepdim=True)
        
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        omega = torch.acos(dot_product)
        
        interpolated_z = []
        
        for alpha in alpha_values:
            if torch.abs(omega) < 1e-6:
                
                z_interp = (1 - alpha) * z_a + alpha * z_b
            else:
                
                sin_omega = torch.sin(omega)
                weight_a = torch.sin((1 - alpha) * omega) / sin_omega
                weight_b = torch.sin(alpha * omega) / sin_omega
                z_interp = weight_a * z_a + weight_b * z_b
            
            interpolated_z.append(z_interp)
        
        return torch.cat(interpolated_z, dim=0)
    
    def batch_interpolate(
        self,
        images_a: torch.Tensor,
        images_b: torch.Tensor,
        num_steps: int = 30
    ) -> List[torch.Tensor]:
        
        interpolation_sequences = []
        
        for i in range(images_a.shape[0]):
            interpolated_imgs, _, _ = self.interpolate_images(
                images_a[i], images_b[i], num_steps
            )
            interpolation_sequences.append(interpolated_imgs)
        
        return interpolation_sequences
    
    def explore_latent_space(
        self,
        center_image: torch.Tensor,
        num_samples: int = 16,
        radius: float = 2.0
    ) -> torch.Tensor:
        
        with torch.no_grad():
            if center_image.dim() == 1:
                center_image = center_image.unsqueeze(0)
            
            center_image = center_image.to(self.device)
            
            
            center_z = self.model.encode(center_image)
            
            
            perturbations = torch.randn(num_samples, self.model.latent_dim).to(self.device)
            perturbations = perturbations * radius
            
            
            perturbed_z = center_z.repeat(num_samples, 1) + perturbations
            
            
            generated_images = self.model.decode(perturbed_z)
            
            return generated_images


class MorphingPipeline:
    
    
    def __init__(
        self,
        model: VAE,
        device: torch.device,
        output_dir: str = './outputs'
    ):
        
        self.interpolator = LatentInterpolator(model, device)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_morph_sequence(
        self,
        source_image: torch.Tensor,
        target_image: torch.Tensor,
        num_steps: int = 30,
        save_frames: bool = False,
        create_video: bool = True,
        sequence_name: str = "morph_sequence"
    ) -> Tuple[torch.Tensor, Optional[str], Optional[str]]:
        
        
        interpolated_images, latent_vectors, alpha_values = self.interpolator.interpolate_images(
            source_image, target_image, num_steps
        )
        
        
        if create_video:
            video_path = self._create_video_from_tensors(
                interpolated_images,
                os.path.join(self.output_dir, f"{sequence_name}.gif")
            )
        
        frames_dir = None
        
        
        if save_frames:
            frames_dir = os.path.join(self.output_dir, f"{sequence_name}_frames")
            os.makedirs(frames_dir, exist_ok=True)
            
            for i, img in enumerate(interpolated_images):
                frame_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
                self._save_image(img, frame_path)
        
        
        grid_path = os.path.join(self.output_dir, f"{sequence_name}_grid.png")
        plot_interpolation_grid(
            interpolated_images.cpu().numpy(),
            save_path=grid_path,
            title=f"Morphing Sequence: {sequence_name}"
        )
        
        return interpolated_images, frames_dir, video_path
    
    def _save_image(self, image_tensor: torch.Tensor, save_path: str):
        
        import matplotlib.pyplot as plt
        
        
        image = image_tensor.cpu().numpy().reshape(28, 28)
        
        
        plt.figure(figsize=(2, 2))
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()
    
    def _create_video_from_tensors(self, image_tensors: torch.Tensor, output_path: str, fps: int = 10) -> str:
        
        import imageio
        import matplotlib.pyplot as plt
        from io import BytesIO
        import numpy as np
        
        images = []
        
        
        for img_tensor in image_tensors:
            
            img_array = img_tensor.cpu().numpy().reshape(28, 28)
            
            
            img_array = (img_array * 255).astype(np.uint8)
            
            
            img_rgb = np.stack([img_array] * 3, axis=-1)
            
            images.append(img_rgb)
        
        
        images.extend(images[-2:0:-1])  
        
        
        if output_path.endswith('.gif'):
            
            imageio.mimsave(output_path, images, duration=0.2, loop=0, fps=5)
        else:
            imageio.mimsave(output_path, images, fps=fps)
        
        print(f"Video created: {output_path}")
        return output_path
    
    def create_multi_morph(
        self,
        images: List[torch.Tensor],
        num_steps: int = 20,
        sequence_name: str = "multi_morph"
    ) -> torch.Tensor:
        
        all_interpolations = []
        
        for i in range(len(images) - 1):
            interpolated, _, _ = self.interpolator.interpolate_images(
                images[i], images[i + 1], num_steps
            )
            
            
            all_interpolations.append(interpolated[:-1])
        
        
        all_interpolations.append(interpolated[-1:])
        
        
        complete_sequence = torch.cat(all_interpolations, dim=0)
        
        
        grid_path = os.path.join(self.output_dir, f"{sequence_name}_grid.png")
        plot_interpolation_grid(
            complete_sequence.cpu().numpy(),
            save_path=grid_path,
            title=f"Multi-Morph Sequence: {sequence_name}",
            max_cols=num_steps
        )
        
        return complete_sequence


def calculate_morphing_smoothness(interpolated_images: torch.Tensor) -> float:
    
    if len(interpolated_images) < 2:
        return 0.0
    
    differences = []
    for i in range(len(interpolated_images) - 1):
        diff = torch.mean(torch.abs(interpolated_images[i] - interpolated_images[i + 1]))
        differences.append(diff.item())
    
    return np.mean(differences)