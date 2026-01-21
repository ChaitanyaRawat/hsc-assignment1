
import os
import sys
import time
from typing import Optional

import torch
import numpy as np


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataset import MNISTDataLoader, get_data_info
from models.vae import VAE, get_model_info
from training.trainer import VAETrainer, load_model
from interpolation.morphing import MorphingPipeline, calculate_morphing_smoothness
from utils.helpers import (
    set_random_seeds, get_device, ConfigParser, save_config,
    print_model_summary, check_requirements, get_latest_checkpoint,
    load_custom_image
)
from utils.visualization import (
    plot_generated_samples, plot_latent_space_2d, calculate_reconstruction_metrics,
    plot_reconstruction_samples
)


def train_vae(config: dict) -> str:
    
    print("Training VAE...")
    
    
    set_random_seeds(config['general']['seed'])
    
    
    device = get_device()
    
    
    data_loader = MNISTDataLoader(
        data_dir=config['data']['data_dir'],
        batch_size=config['training']['batch_size'],
        val_split=config['data']['val_split']
    )
    
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()
    
    
    model = VAE(
        input_size=784,
        latent_dim=config['training']['latent_dim']
    )
    
    
    print_model_summary(model)
    
    
    trainer = VAETrainer(
        model=model,
        device=device,
        learning_rate=config['training']['learning_rate'],
        beta=config['training']['beta'],
        checkpoint_dir=config['model']['checkpoint_dir']
    )
    
    
    config_path = os.path.join(config['model']['checkpoint_dir'], 'training_config.json')
    save_config(config, config_path)
    
    
    start_time = time.time()
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['training']['epochs'],
        early_stopping_patience=10,
        save_interval=5,
        visualize_interval=10
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    
    final_model_path = os.path.join(config['model']['checkpoint_dir'], 'vae_model_final.pth')
    
    
    print("\nEvaluating on test set...")
    test_loss, test_recon, test_kl = trainer.validate(test_loader)
    print(f"Test Loss: {test_loss:.4f} (Recon: {test_recon:.4f}, KL: {test_kl:.4f})")
    
    
    output_dir = config['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    
    trainer.model.eval()
    with torch.no_grad():
        test_data, _ = next(iter(test_loader))
        test_data = test_data.to(device)
        reconstructed, _, _ = trainer.model(test_data)
        
        
        metrics = calculate_reconstruction_metrics(test_data.cpu(), reconstructed.cpu())
        print(f"\nReconstruction Metrics:")
        print(f"  MSE: {metrics['mse_mean']:.6f} ± {metrics['mse_std']:.6f}")
        print(f"  SSIM: {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
        
        
        plot_reconstruction_samples(
            test_data.cpu()[:8],
            reconstructed.cpu()[:8],
            save_path=os.path.join(output_dir, 'final_reconstructions.png'),
            title='Final Model Reconstructions'
        )
    
    
    plot_generated_samples(
        trainer.model,
        device,
        num_samples=16,
        save_path=os.path.join(output_dir, 'generated_samples.png')
    )
    
    
    if config['training']['latent_dim'] == 2:
        plot_latent_space_2d(
            trainer.model,
            test_loader,
            device,
            save_path=os.path.join(output_dir, 'latent_space_2d.png')
        )
    
    print(f"Model saved to: {final_model_path}")
    return final_model_path


def generate_morphing(config: dict, model_path: str) -> None:
   
    print("Generating morphing sequence...")
    print("=" * 50)
    
    
    set_random_seeds(config['general']['seed'])
    
    
    device = get_device()
    
    
    if model_path and os.path.exists(model_path):
        model = load_model(model_path, device)
    else:
        print(f"Model not found at {model_path}")
        
        latest_checkpoint = get_latest_checkpoint(config['model']['checkpoint_dir'])
        if latest_checkpoint:
            print(f"Using latest checkpoint: {latest_checkpoint}")
            model = load_model(latest_checkpoint, device)
        else:
            print("No trained model found. Please train a model first.")
            return
    
    
    if config['morphing']['source_image'] and config['morphing']['target_image']:
        
        print("Loading custom image files...")
        try:
            source_image = load_custom_image(config['morphing']['source_image'])
            target_image = load_custom_image(config['morphing']['target_image'])
            source_label = "custom_A"
            target_label = "custom_B"
            print(f" Loaded custom images:")
            print(f"  Source: {config['morphing']['source_image']}")
            print(f"  Target: {config['morphing']['target_image']}")
        except Exception as e:
            print(f" Error loading custom images: {e}")
            return
    
    else:
        
        if config['morphing']['source_idx'] is None or config['morphing']['target_idx'] is None:
            print(" Either provide custom image paths (--source-image, --target-image)")
            print("   or MNIST indices (--source-idx, --target-idx)")
            return
            
        print("Loading images from MNIST test set...")
        data_loader = MNISTDataLoader(
            data_dir=config['data']['data_dir'],
            batch_size=128
        )
        
        source_idx = config['morphing']['source_idx']
        target_idx = config['morphing']['target_idx']
        
        source_image, source_label = data_loader.test_dataset[source_idx]
        target_image, target_label = data_loader.test_dataset[target_idx]
        
        print(f"Source: digit {source_label} (index {source_idx})")
        print(f"Target: digit {target_label} (index {target_idx})")
    
    
    N = config['morphing']['steps']
    print(f"\n🎬 Generating {N} intermediate transition frames...")
    
    
    pipeline = MorphingPipeline(
        model=model,
        device=device,
        output_dir=config['output']['output_dir']
    )
    
    
    if config['output']['output_name']:
        sequence_name = config['output']['output_name']
    else:
        sequence_name = f"morph_{source_label}to{target_label}"
    
    
    interpolated_images, frames_dir, video_path = pipeline.create_morph_sequence(
        source_image=source_image,
        target_image=target_image,
        num_steps=N,
        save_frames=config['output']['save_frames'],  
        create_video=config['output']['save_video'],
        sequence_name=sequence_name
    )
    
    
    smoothness = calculate_morphing_smoothness(interpolated_images)
    print(f"\n Results:")
    print(f"  Generated {len(interpolated_images)} transition frames")
    print(f"  Smoothness metric: {smoothness:.6f} (lower = smoother)")
    
    
    analysis_path = os.path.join(config['output']['output_dir'], f"{sequence_name}_analysis.txt")
    with open(analysis_path, 'w', encoding='utf-8') as f:
        f.write(f"Morphing Analysis\n")
        f.write(f"================\n\n")
        f.write(f"Input Images:\n")
        f.write(f"  Source (Image A): {source_label}\n")
        f.write(f"  Target (Image B): {target_label}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Transition frames (N): {N}\n")
        f.write(f"  Interpolation method: {config['morphing']['interpolation_method']}\n")
        f.write(f"  Model latent dimension: {model.latent_dim}\n\n")
        f.write(f"Output:\n")
        f.write(f"  Total frames generated: {len(interpolated_images)}\n")
        f.write(f"  Smoothness metric: {smoothness:.6f}\n")
        
        if video_path:
            f.write(f"  Video file saved to: {video_path}\n")
        if frames_dir:
            f.write(f"  Frame images saved to: {frames_dir}\n")
        if not video_path and not frames_dir:
            f.write(f"  Morphing generated in memory (no files saved to disk)\n")
    
    print(f"  Analysis saved to: {analysis_path}")
    if video_path:
        print(f"  Video file saved to: {video_path}")
    if frames_dir:
        print(f"  {N} frame images saved to: {frames_dir}")
    if not video_path and not frames_dir:
        print(f"  Morphing sequence generated in memory only")
    
    print("\n🎉 Morphing sequence generation completed!")
    print(" Structural features successfully morphed from Image A to Image B")


def evaluate_model(config: dict, model_path: str) -> None:
    
    print("Evaluating trained model...")
    
    
    device = get_device()
    
    
    if model_path and os.path.exists(model_path):
        model = load_model(model_path, device)
    else:
        latest_checkpoint = get_latest_checkpoint(config['model']['checkpoint_dir'])
        if latest_checkpoint:
            model = load_model(latest_checkpoint, device)
        else:
            print("No trained model found. Please train a model first.")
            return
    
    
    model_info = get_model_info(model)
    print(f"\nModel Information:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    
    data_loader = MNISTDataLoader(
        data_dir=config['data']['data_dir'],
        batch_size=128
    )
    
    _, _, test_loader = data_loader.get_data_loaders()
    
    
    model.eval()
    total_mse = 0
    total_ssim = 0
    num_samples = 0
    
    print("\nEvaluating reconstruction quality...")
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):
            if batch_idx >= 10:  
                break
            
            data = data.to(device)
            reconstructed, _, _ = model(data)
            
            
            metrics = calculate_reconstruction_metrics(data.cpu(), reconstructed.cpu())
            
            batch_size = data.shape[0]
            total_mse += metrics['mse_mean'] * batch_size
            total_ssim += metrics['ssim_mean'] * batch_size
            num_samples += batch_size
    
    avg_mse = total_mse / num_samples
    avg_ssim = total_ssim / num_samples
    
    print(f"Average Reconstruction MSE: {avg_mse:.6f}")
    print(f"Average Reconstruction SSIM: {avg_ssim:.4f}")
    
    
    output_dir = config['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    
    test_data, test_labels = next(iter(test_loader))
    test_data = test_data.to(device)
    
    with torch.no_grad():
        reconstructed, _, _ = model(test_data)
    
    plot_reconstruction_samples(
        test_data.cpu()[:10],
        reconstructed.cpu()[:10],
        save_path=os.path.join(output_dir, 'evaluation_reconstructions.png'),
        title='Model Evaluation - Reconstructions'
    )
    
    
    plot_generated_samples(
        model,
        device,
        num_samples=16,
        save_path=os.path.join(output_dir, 'evaluation_generated.png')
    )
    
    
    if model.latent_dim == 2:
        plot_latent_space_2d(
            model,
            test_loader,
            device,
            save_path=os.path.join(output_dir, 'evaluation_latent_space.png'),
            max_samples=1000
        )
    
    print("Model evaluation completed!")


def main():
    
    
    if not check_requirements():
        return
    
    
    parser = ConfigParser()
    args = parser.parse_args()
    config = parser.get_config_dict(args)
    
    
    os.makedirs(config['model']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['output']['output_dir'], exist_ok=True)
    
    
    if config['mode']['train']:
        model_path = train_vae(config)
        print(f"\nTraining completed. Model saved to: {model_path}")
        
    elif config['mode']['morph']:
        generate_morphing(config, config['model']['model_path'])
        
    elif config['mode']['evaluate']:
        evaluate_model(config, config['model']['model_path'])
    
    print("Program completed.")


if __name__ == "__main__":
    main()