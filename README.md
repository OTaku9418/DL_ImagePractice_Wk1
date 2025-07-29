# CIFAR-10 ResNet-18 Classification

A PyTorch implementation of ResNet-18 for CIFAR-10 classification with 224x224 input resolution.

## Project Structure

```text
├── models/          # Model definitions
├── modules/         # Core training/evaluation modules
├── utils/           # Utility functions
├── data/            # Dataset storage
├── logs/            # Training logs and plots
├── saved_models/    # Trained model checkpoints
└── main.py          # Main training script
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python main.py --mode train
```

Test the model:

```bash
python main.py --mode test
```

## Model Architecture

- ResNet-18 with BasicBlock residual connections
- Input size: 224x224x3 (grayscale converted to RGB)
- Output: 10 Fashion-MNIST classes
- Regularization: BatchNorm + Dropout

