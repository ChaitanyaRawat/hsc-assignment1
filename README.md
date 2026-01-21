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
