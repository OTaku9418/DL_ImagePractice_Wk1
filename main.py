
import torch
import os
import sys
import time
import argparse

# Add project path to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from models.resnet import create_resnet18
from utils.data_loader import create_fashion_mnist_dataloaders
from modules.trainer import create_trainer
from modules.evaluator import create_evaluator
from utils.visualizer import create_visualizer


def parse_arguments():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fashion-MNIST ResNet-18 Training Program')
    
    # Data related parameters
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Data storage directory (default: ./data)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers (default: 0)')
    parser.add_argument('--validation_split', type=float, default=0.1,
                       help='Validation set ratio (default: 0.1)')
    
    # Model related parameters
    parser.add_argument('--num_classes', type=int, default=10,
                       help='Number of classes (default: 10)')
    
    # Training related parameters
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    
    # Save related parameters
    parser.add_argument('--save_dir', type=str, default='./saved_models',
                       help='Model save directory (default: ./saved_models)')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Log save directory (default: ./logs)')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save model every N epochs (default: 10)')
    
    # Device related parameters
    parser.add_argument('--device', type=str, default='auto',
                       help='Computing device (auto/cpu/cuda, default: auto)')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'test', 'both'],
                       help='Run mode (train/test/both, default: train)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume training from checkpoint path')
    
    return parser.parse_args()


def setup_device(device_arg):
    # Setup computing device
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_arg == 'cuda':
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, using CPU")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Display GPU information if using GPU
    if device.type == 'cuda':
        print(f"GPU model: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"CUDA version: {torch.version.cuda}")
    
    return device


def create_directories(args):
    # Create necessary directories
    directories = [args.data_dir, args.save_dir, args.log_dir]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory created/confirmed: {directory}")


def load_model_checkpoint(model, checkpoint_path, device):
    # Load model checkpoint
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded (Epoch {checkpoint['epoch']+1}, "
          f"Best validation accuracy: {checkpoint['best_val_acc']:.2f}%)")
    
    return checkpoint


def train_model(args, device):
    # Train model
    print("=" * 80)
    print("Starting Fashion-MNIST Classification Model Training")
    print("=" * 80)
    
    # 1. Create data loaders
    print("\n1. Creating data loaders...")
    train_loader, val_loader, test_loader, dataset_info = create_fashion_mnist_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation_split=args.validation_split
    )
    
    print("Dataset information:")
    for key, value in dataset_info.items():
        print(f"  {key}: {value}")
    
    # 2. Create model
    print("\n2. Creating ResNet-18 model...")
    model = create_resnet18(num_classes=args.num_classes)
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 3. Create trainer
    print("\n3. Creating trainer...")
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        save_dir=args.save_dir
    )
    
    # 4. Setup optimizer and loss function
    print("\n4. Setting up optimizer and loss function...")
    trainer.setup_optimizer_and_criterion(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 5. Resume training if checkpoint specified
    start_epoch = 0
    if args.resume:
        checkpoint = load_model_checkpoint(trainer.model, args.resume, device)
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.best_val_acc = checkpoint['best_val_acc']
        trainer.history = checkpoint['history']
        start_epoch = checkpoint['epoch'] + 1
        print(f"Continuing training from epoch {start_epoch+1}...")
    
    # 6. Start training
    print(f"\n5. Starting training (Target: 95%+ accuracy)...")
    trainer.train(num_epochs=args.epochs, save_every=args.save_every)
    
    # 7. Post-training evaluation
    print("\n6. Training completed, running final test...")
    test_metrics = trainer.test_model()
    
    # Check if target is achieved
    target_accuracy = 95.0
    if test_metrics['accuracy'] >= target_accuracy:
        print(f"\nTarget accuracy {target_accuracy}% achieved!")
        print(f"Final test accuracy: {test_metrics['accuracy']:.2f}%")
    else:
        print(f"\nTarget accuracy {target_accuracy}% not achieved")
        print(f"Current test accuracy: {test_metrics['accuracy']:.2f}%")
        print(f"Need improvement: {target_accuracy - test_metrics['accuracy']:.2f}%")
    
    return trainer, test_metrics


def test_model(args, device):
    # Test model
    print("=" * 80)
    print("Testing Fashion-MNIST Classification Model")
    print("=" * 80)
    
    # 1. Create data loaders
    print("\n1. Creating data loaders...")
    _, _, test_loader, dataset_info = create_fashion_mnist_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # 2. Create model
    print("\n2. Creating model...")
    model = create_resnet18(num_classes=args.num_classes)
    
    # 3. Load best model
    best_model_path = os.path.join(args.save_dir, 'best_model.pth')
    if not os.path.exists(best_model_path):
        print(f"Error: Best model file not found at {best_model_path}")
        print("Please run training mode first")
        return None
    
    checkpoint = load_model_checkpoint(model, best_model_path, device)
    
    # 4. Create evaluator and visualizer
    print("\n3. Creating evaluator and visualizer...")
    evaluator = create_evaluator(
        model=model,
        test_loader=test_loader,
        class_names=dataset_info['class_names'],
        device=device,
        save_dir=args.log_dir
    )
    
    visualizer = create_visualizer(save_dir=args.log_dir)
    
    # 5. Comprehensive evaluation
    print("\n4. Starting comprehensive evaluation...")
    history_path = os.path.join(args.save_dir, 'training_history.json')
    evaluator.comprehensive_evaluation(history_path=history_path, visualizer=visualizer)
    
    return evaluator


def main():
    """
    Main function
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup computing device
    device = setup_device(args.device)
    
    # Create directories
    create_directories(args)
    
    # Record program start time
    program_start_time = time.time()
    
    try:
        if args.mode == 'train':
            # Training only
            trainer, test_metrics = train_model(args, device)
            
        elif args.mode == 'test':
            # Testing only
            evaluator = test_model(args, device)
            
        elif args.mode == 'both':
            # Training then testing
            trainer, test_metrics = train_model(args, device)
            print("\n" + "="*80)
            evaluator = test_model(args, device)
        
        # Calculate total time
        total_time = time.time() - program_start_time
        print(f"\nProgram total time: {total_time/3600:.2f} hours ({total_time:.1f} seconds)")
        
        print("\nProgram execution completed!")
        
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
        
    except Exception as e:
        print(f"\nProgram execution error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
