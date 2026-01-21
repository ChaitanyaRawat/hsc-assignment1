
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any
import numpy as np


class Encoder(nn.Module):
    
    
    def __init__(self, input_size: int = 784, latent_dim: int = 20):

        super(Encoder, self).__init__()
        
        self.input_size = input_size
        self.latent_dim = latent_dim
        
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten()  
        )
        
        
        self.conv_output_size = 128 * 4 * 4
        
        
        self.fc_hidden1 = nn.Linear(self.conv_output_size, 512)
        self.fc_hidden2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.2)
        self.fc_mean = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        
        x = x.reshape(-1, 1, 28, 28)
        
        
        x = self.conv_layers(x)
        
        
        x = F.relu(self.fc_hidden1(x))
        x = self.dropout(x)
        x = F.relu(self.fc_hidden2(x))
        x = self.dropout(x)
        
        
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        
        return mean, logvar


class Decoder(nn.Module):
    
    
    def __init__(self, latent_dim: int = 20, output_size: int = 784):
        
        super(Decoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.output_size = output_size
        
        
        self.fc_hidden1 = nn.Linear(latent_dim, 256)
        self.fc_hidden2 = nn.Linear(256, 512)
        self.fc_conv_input = nn.Linear(512, 128 * 4 * 4)
        self.dropout = nn.Dropout(0.2)
        
        
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),   
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),   
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),   
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),   
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.ConvTranspose2d(16, 1, kernel_size=5, stride=1, padding=2),  
            nn.Sigmoid()  
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        
        
        x = F.relu(self.fc_hidden1(z))
        x = self.dropout(x)
        x = F.relu(self.fc_hidden2(x))
        x = self.dropout(x)
        x = F.relu(self.fc_conv_input(x))
        
        
        x = x.reshape(-1, 128, 4, 4)
        
        
        x = self.deconv_layers(x)
        
        
        x = x[:, :, :28, :28]
        x = x.reshape(-1, self.output_size)
        
        return x


class VAE(nn.Module):
    
    
    def __init__(self, input_size: int = 784, latent_dim: int = 20):
        
        super(VAE, self).__init__()
        
        self.input_size = input_size
        self.latent_dim = latent_dim
        
        self.encoder = Encoder(input_size, latent_dim)
        self.decoder = Decoder(latent_dim, input_size)
    
    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        reconstructed_x = self.decoder(z)
        
        return reconstructed_x, mean, logvar
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        
        mean, _ = self.encoder(x)
        return mean
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        
        return self.decoder(z)
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z)


def vae_loss_function(
    reconstructed_x: torch.Tensor, 
    x: torch.Tensor, 
    mean: torch.Tensor, 
    logvar: torch.Tensor,
    beta: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    
    reconstruction_loss = F.binary_cross_entropy(
        reconstructed_x, x, reduction='sum'
    )
    
    
    
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    
    
    total_loss = reconstruction_loss + beta * kl_loss
    
    return total_loss, reconstruction_loss, kl_loss


def get_model_info(model: VAE) -> Dict[str, Any]:
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'latent_dimension': model.latent_dim,
        'input_size': model.input_size,
        'encoder_layers': len(list(model.encoder.children())),
        'decoder_layers': len(list(model.decoder.children()))
    }