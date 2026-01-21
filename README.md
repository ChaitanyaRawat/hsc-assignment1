# Digit Morphing with VAE

Morphing between handwritten digits using variational autoencoders.

## Getting Started

First train a model, then use either the GUI or command line.

### GUI (easier)
```bash
pip install -r requirements.txt
python main.py --train --epochs 5
python gui.py
```

### Command line
```bash
pip install -r requirements.txt
python main.py --train --epochs 5  
python main.py --morph --source-idx 2 --target-idx 7 --steps 15 --save-video
```

## How it works

Basically a VAE (Variational Autoencoder) that learns to compress 28x28 digit images into a smaller representation, then reconstructs them. For morphing, we interpolate between two images in this compressed space.

- Images: 28×28 pixels
- Latent space: 32 dimensions  
- Uses CNN layers with batch normalization

## Files

- `main.py` - run training and morphing from command line
- `gui.py` - graphical interface (easier to use)
- `models/` - the VAE neural network code
- `training/` - training loop and stuff
- `interpolation/` - morphing between images
- `data/` - loads MNIST dataset
- `utils/` - misc helper functions
- `outputs/` - where generated images/videos get saved

## Usage

### Training
```bash
python main.py --train --epochs 10 --batch-size 64 --latent-dim 32
```

### Morphing
GUI version (recommended):
```bash
python gui.py
```

Command line version:
```bash
python main.py --morph --model-path ./checkpoints/vae_model_final.pth
```

### Evaluation
```bash
python main.py --evaluate --model-path ./checkpoints/vae_model_final.pth
```

## Requirements

- Python 3.8-3.13 (PyTorch not compatible with 3.14 yet)
- PyTorch >= 1.12.0
- CUDA (optional, for GPU acceleration)

## Virtual Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
source code/
├── main.py              # Command line interface
├── gui.py               # GUI application
├── requirements.txt     # Python dependencies
├── models/
│   └── vae.py          # VAE architecture (2.97M parameters)
├── training/
│   └── trainer.py      # Training pipeline with early stopping
├── data/
│   └── dataset.py      # MNIST data loading
├── interpolation/
│   └── morphing.py     # Latent space interpolation
├── utils/
│   ├── helpers.py      # Config and utilities
│   └── visualization.py # Plotting functions
├── checkpoints/        # Saved models (created after training)
└── outputs/           # Generated images and GIFs
```

## Features

- **Enhanced VAE Architecture**: 6 conv layers + batch normalization + dropout
- **Real-time GUI**: Tkinter interface with morphing preview
- **Smooth Interpolation**: Linear interpolation in 32D latent space
- **GIF Generation**: Animated morphing sequences
- **Early Stopping**: Prevents overfitting during training
- **Learning Rate Scheduling**: Adaptive learning rate
- **Model Checkpointing**: Automatic saving of best models

## Technical Details

### Model Architecture
- **Input**: 28×28 grayscale MNIST images
- **Encoder**: CNN (1→32→64→128 channels) + FC layers
- **Latent Space**: 32 dimensions
- **Decoder**: FC layers + Deconv (128→64→32→1 channels)
- **Parameters**: 2.97 million trainable parameters

### Loss Function
```
Total Loss = Reconstruction Loss + β × KL Divergence
```
- Reconstruction: MSE between input and output
- KL Divergence: Regularizes latent space to N(0,1)
- β: Weighting parameter (default 1.0)

### Training Features
- Adam optimizer with learning rate scheduling
- Early stopping (patience=7 epochs)
- Batch normalization for stable training
- Dropout (0.2) for regularization

## Examples

### Basic Training
```bash
python main.py --train
```

### Custom Training
```bash
python main.py --train --epochs 20 --batch-size 128 --learning-rate 0.001
```

### Generate Morphing GIF
```bash
python main.py --morph --source-digit 3 --target-digit 8 --frames 20
```

## Troubleshooting

### Python Version Issues
If you get PyTorch compatibility errors:
```bash
python --version  # Check your Python version
# Install Python 3.11 if using 3.14
```

### GPU Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Memory Issues
Reduce batch size if running out of memory:
```bash
python main.py --train --batch-size 32
```

## Expected Results

- **Training Time**: 10-30 minutes (depending on epochs and hardware)
- **Model Size**: ~12MB
- **Morphing Quality**: Smooth transitions between digit classes
- **Reconstruction Quality**: High fidelity for MNIST digits

## Credits

Built for HSC Assignment 1 - exploring variational autoencoders and latent space interpolation for digit morphing.
