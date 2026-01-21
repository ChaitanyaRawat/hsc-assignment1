import torch
import random
import numpy as np
import os
import json
from typing import Dict, Any, Optional
import argparse
from PIL import Image
import torchvision.transforms as transforms


def set_random_seeds(seed: int = 42) -> None:
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
   
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Apple Silicon) device")
    else:
        device = torch.device('cpu')
        print("Using CPU device")
    
    return device


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
   
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def save_config(config: Dict[str, Any], save_path: str) -> None:
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to: {save_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Configuration loaded from: {config_path}")
    return config


def create_experiment_dir(base_dir: str, experiment_name: Optional[str] = None) -> str:
    
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if experiment_name:
        exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    else:
        exp_dir = os.path.join(base_dir, f"experiment_{timestamp}")
    
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Experiment directory created: {exp_dir}")
    
    return exp_dir


def format_time(seconds: float) -> str:
    
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def print_model_summary(model: torch.nn.Module, input_size: tuple = (1, 784)) -> None:
    
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*60)
    
    
    param_info = count_parameters(model)
    print(f"Total Parameters: {param_info['total_parameters']:,}")
    print(f"Trainable Parameters: {param_info['trainable_parameters']:,}")
    
    
    print("\nModel Architecture:")
    print(model)
    
    
    try:
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(input_size)
            if hasattr(model, 'forward'):
                output = model(dummy_input)
                if isinstance(output, tuple):
                    print(f"\nOutput shapes:")
                    for i, out in enumerate(output):
                        print(f"  Output {i+1}: {list(out.shape)}")
                else:
                    print(f"\nOutput shape: {list(output.shape)}")
    except Exception as e:
        print(f"\nCould not determine output shapes: {e}")
    
    print("="*60 + "\n")


class ConfigParser:
    
    
    def __init__(self):
        
        self.parser = argparse.ArgumentParser(description='VAE Training and Morphing')
        self._add_arguments()
    
    def _add_arguments(self):
        
        
        mode_group = self.parser.add_mutually_exclusive_group(required=True)
        mode_group.add_argument('--train', action='store_true', help='Train VAE model')
        mode_group.add_argument('--morph', action='store_true', help='Generate morphing sequence')
        mode_group.add_argument('--evaluate', action='store_true', help='Evaluate trained model')
        
        
        self.parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
        self.parser.add_argument('--batch-size', type=int, default=64, help='Training batch size (optimized for stability)')
        self.parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
        self.parser.add_argument('--latent-dim', type=int, default=32, help='Latent space dimension (increased for better accuracy)')
        self.parser.add_argument('--beta', type=float, default=0.5, help='Beta parameter for β-VAE (reduced for better reconstruction)')
        
        
        self.parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
        self.parser.add_argument('--val-split', type=float, default=0.1, help='Validation split ratio')
        
        
        self.parser.add_argument('--model-path', type=str, help='Path to saved model')
        self.parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
        
        
        self.parser.add_argument('--source-idx', type=int, help='Source image index from MNIST test set')
        self.parser.add_argument('--target-idx', type=int, help='Target image index from MNIST test set')
        self.parser.add_argument('--source-image', type=str, help='Path to source image file (28x28 grayscale)')
        self.parser.add_argument('--target-image', type=str, help='Path to target image file (28x28 grayscale)')
        self.parser.add_argument('--steps', type=int, default=30, help='Number of intermediate transition frames (N)')
        self.parser.add_argument('--interpolation-method', type=str, default='linear', 
                               choices=['linear', 'spherical'], help='Interpolation method')
        
        
        self.parser.add_argument('--output-dir', type=str, default='./outputs', help='Output directory')
        self.parser.add_argument('--save-video', action='store_true', help='Save morphing sequence as video/GIF file')
        self.parser.add_argument('--save-frames', action='store_true', help='Save individual frame images (uses more disk space)')
        self.parser.add_argument('--output-name', type=str, help='Custom name for output sequence')
        
        
        self.parser.add_argument('--seed', type=int, default=42, help='Random seed')
        self.parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cuda, cpu)')
    
    def parse_args(self):
        
        return self.parser.parse_args()
    
    def get_config_dict(self, args) -> Dict[str, Any]:
        
        return {
            'mode': {
                'train': args.train,
                'morph': args.morph,
                'evaluate': args.evaluate
            },
            'training': {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'latent_dim': args.latent_dim,
                'beta': args.beta
            },
            'data': {
                'data_dir': args.data_dir,
                'val_split': args.val_split
            },
            'model': {
                'model_path': args.model_path,
                'checkpoint_dir': args.checkpoint_dir
            },
            'morphing': {
                'source_idx': args.source_idx,
                'target_idx': args.target_idx,
                'source_image': args.source_image,
                'target_image': args.target_image,
                'steps': args.steps,
                'interpolation_method': args.interpolation_method
            },
            'output': {
                'output_dir': args.output_dir,
                'save_video': args.save_video,
                'save_frames': args.save_frames,
                'output_name': args.output_name
            },
            'general': {
                'seed': args.seed,
                'device': args.device
            }
        }


def check_requirements() -> bool:
    
    required_packages = [
        'torch',
        'torchvision',
        'numpy',
        'matplotlib',
        'imageio',
        'tqdm',
        'cv2',
        'PIL',
        'skimage'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            elif package == 'skimage':
                from skimage import metrics
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("All required packages are installed.")
    return True


def load_custom_image(image_path: str) -> torch.Tensor:
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    
    image = Image.open(image_path).convert('L')
    
    
    if image.size != (28, 28):
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  
    ])
    
    tensor = transform(image)
    
    print(f"Loaded custom image from {image_path}: shape {tensor.shape}")
    return tensor


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    
    if not checkpoint_files:
        return None
    
    
    final_checkpoint = [f for f in checkpoint_files if 'final' in f]
    if final_checkpoint:
        return os.path.join(checkpoint_dir, final_checkpoint[0])
    
    
    epoch_checkpoints = [f for f in checkpoint_files if 'epoch' in f]
    if epoch_checkpoints:
        
        epoch_checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
        return os.path.join(checkpoint_dir, epoch_checkpoints[-1])
    
    return None