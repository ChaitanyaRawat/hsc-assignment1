
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from typing import Tuple, Dict, Any
import os


class MNISTDataLoader:
    
    
    def __init__(
        self, 
        data_dir: str = './data', 
        batch_size: int = 128, 
        val_split: float = 0.1,
        download: bool = True
    ):
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.download = download
        
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))  
        ])
        
        self._setup_data()
    
    def _setup_data(self) -> None:
         
        
        os.makedirs(self.data_dir, exist_ok=True)
        
        
        full_train_dataset = torchvision.datasets.MNIST(
            root=self.data_dir,
            train=True,
            transform=self.transform,
            download=self.download
        )
        
        
        self.test_dataset = torchvision.datasets.MNIST(
            root=self.data_dir,
            train=False,
            transform=self.transform,
            download=self.download
        )
        
        
        train_size = int((1 - self.val_split) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        
        self.train_dataset, self.val_dataset = random_split(
            full_train_dataset, [train_size, val_size]
        )
        
        print(f"Dataset loaded successfully!")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")
    
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2 if torch.cuda.is_available() else 0
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2 if torch.cuda.is_available() else 0
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2 if torch.cuda.is_available() else 0
        )
        
        return train_loader, val_loader, test_loader
    
    def get_sample_data(self, num_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        
        test_loader = DataLoader(self.test_dataset, batch_size=num_samples, shuffle=True)
        images, labels = next(iter(test_loader))
        return images, labels
    
    def get_specific_samples(self, indices: list) -> Tuple[torch.Tensor, torch.Tensor]:
        
        images = []
        labels = []
        
        for idx in indices:
            img, label = self.test_dataset[idx]
            images.append(img)
            labels.append(label)
        
        return torch.stack(images), torch.tensor(labels)


def get_data_info() -> Dict[str, Any]:
    
    return {
        'input_size': 784,  
        'image_shape': (28, 28),
        'num_classes': 10,
        'num_channels': 1
    }