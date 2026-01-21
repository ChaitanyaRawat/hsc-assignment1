
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from typing import Dict, List, Tuple, Optional
import json


import sys
import os.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vae import VAE, vae_loss_function
from utils.visualization import plot_reconstruction_samples, plot_loss_curves


class EarlyStopping:
    
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, restore_best_weights: bool = True):
        
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: VAE) -> bool:
        
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        
        return False
    
    def save_checkpoint(self, model: VAE):
         
        self.best_weights = model.state_dict().copy()


class VAETrainer:
    
    
    def __init__(
        self,
        model: VAE,
        device: torch.device,
        learning_rate: float = 0.001,
        beta: float = 1.0,
        checkpoint_dir: str = './checkpoints'
    ):
        
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.beta = beta
        self.checkpoint_dir = checkpoint_dir
        
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=1e-5,  
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.7, 
            patience=5,
            min_lr=1e-6
        )
        
        
        self.train_losses = []
        self.val_losses = []
        self.train_recon_losses = []
        self.train_kl_losses = []
        self.val_recon_losses = []
        self.val_kl_losses = []
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float, float]:
        
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        num_batches = len(train_loader)
        
        with tqdm(train_loader, desc="Training") as pbar:
            for batch_idx, (data, _) in enumerate(pbar):
                data = data.to(self.device)
                
                
                self.optimizer.zero_grad()
                
                
                reconstructed, mean, logvar = self.model(data)
                
                
                loss, recon_loss, kl_loss = vae_loss_function(
                    reconstructed, data, mean, logvar, self.beta
                )
                
                
                loss.backward()
                self.optimizer.step()
                
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Recon': f'{recon_loss.item():.4f}',
                    'KL': f'{kl_loss.item():.4f}'
                })
        
        
        avg_total_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        
        return avg_total_loss, avg_recon_loss, avg_kl_loss
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(self.device)
                
                
                reconstructed, mean, logvar = self.model(data)
                
                
                loss, recon_loss, kl_loss = vae_loss_function(
                    reconstructed, data, mean, logvar, self.beta
                )
                
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
        
        
        avg_total_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        
        return avg_total_loss, avg_recon_loss, avg_kl_loss
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        save_interval: int = 5,
        visualize_interval: int = 10
    ) -> Dict[str, List[float]]:
        
        
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            
            
            train_loss, train_recon, train_kl = self.train_epoch(train_loader)
            
            
            val_loss, val_recon, val_kl = self.validate(val_loader)
            
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_recon_losses.append(train_recon)
            self.train_kl_losses.append(train_kl)
            self.val_recon_losses.append(val_recon)
            self.val_kl_losses.append(val_kl)
            
            
            print(f"Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f})")
            print(f"Val Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.4f})")
            
            
            self.scheduler.step(val_loss)
            
            
            if epoch % save_interval == 0:
                self.save_checkpoint(epoch, val_loss)
            
            
            if epoch % visualize_interval == 0:
                self.visualize_reconstructions(val_loader, epoch)
            
            
            if early_stopping(val_loss, self.model):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break
        
        
        self.save_checkpoint(epoch, val_loss, is_final=True)
        
        
        self.plot_training_history()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_recon_losses': self.train_recon_losses,
            'train_kl_losses': self.train_kl_losses,
            'val_recon_losses': self.val_recon_losses,
            'val_kl_losses': self.val_kl_losses
        }
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_final: bool = False):
         
        suffix = 'final' if is_final else f'epoch_{epoch}'
        checkpoint_path = os.path.join(self.checkpoint_dir, f'vae_model_{suffix}.pth')
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'latent_dim': self.model.latent_dim,
            'input_size': self.model.input_size,
            'beta': self.beta,
            'learning_rate': self.learning_rate
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        
        history_path = os.path.join(self.checkpoint_dir, f'training_history_{suffix}.json')
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_recon_losses': self.train_recon_losses,
            'train_kl_losses': self.train_kl_losses,
            'val_recon_losses': self.val_recon_losses,
            'val_kl_losses': self.val_kl_losses
        }
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def visualize_reconstructions(self, data_loader: DataLoader, epoch: int):
         
        self.model.eval()
        with torch.no_grad():
            
            data, _ = next(iter(data_loader))
            data = data.to(self.device)
            
            
            reconstructed, _, _ = self.model(data)
            
            
            plot_reconstruction_samples(
                data.cpu()[:8], 
                reconstructed.cpu()[:8], 
                save_path=os.path.join(self.checkpoint_dir, f'reconstructions_epoch_{epoch}.png'),
                title=f'Reconstructions - Epoch {epoch}'
            )
    
    def plot_training_history(self):
         
        plot_loss_curves(
            self.train_losses,
            self.val_losses,
            self.train_recon_losses,
            self.val_recon_losses,
            self.train_kl_losses,
            self.val_kl_losses,
            save_path=os.path.join(self.checkpoint_dir, 'training_curves.png')
        )


def load_model(checkpoint_path: str, device: torch.device) -> VAE:
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    
    model = VAE(
        input_size=checkpoint.get('input_size', 784),
        latent_dim=checkpoint.get('latent_dim', 20)
    )
    
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f}")
    
    return model