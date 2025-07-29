
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import os


class CIFAR10DataLoader:
    """
    CIFAR-10 data loader class
    Handles data downloading, preprocessing, data augmentation and splitting
    Resizes images to 224x224 for ResNet compatibility
    """
    
    def __init__(self, data_dir="./data", batch_size=128, num_workers=4, validation_split=0.1):
        """
        Initialize data loader
        
        Args:
            data_dir (str): Data storage directory
            batch_size (int): Batch size
            num_workers (int): Number of worker processes for data loading
            validation_split (float): Validation set ratio
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_split = validation_split
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)

        # CIFAR-10 class names
        self.class_names = [
            'plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
        # Initialize data transforms
        self._setup_transforms()
        
        # Load datasets
        self._load_datasets()
        
        # Create data loaders
        self._create_dataloaders()
    
    def _setup_transforms(self):
        """
        Setup data preprocessing and augmentation transforms
        Resize images to 224x224 for ResNet compatibility
        """
        # Training set data augmentation transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),                    # Resize to 224x224
            transforms.RandomHorizontalFlip(p=0.5),          # Random horizontal flip
            transforms.RandomRotation(degrees=10),           # Random rotation
            transforms.RandomAffine(                         # Random affine transformation
                degrees=0, 
                translate=(0.1, 0.1), 
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),                           # Convert to tensor
            transforms.Normalize((0.485, 0.456, 0.406),      # ImageNet normalization
                               (0.229, 0.224, 0.225))        # for better transfer learning
        ])
        
        # Validation and test set transforms (no augmentation)
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),                   # Resize to 224x224
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),      # ImageNet normalization
                               (0.229, 0.224, 0.225))
        ])
    
    def _load_datasets(self):
        """
        Load CIFAR-10 dataset
        """
        print("Loading CIFAR-10 dataset...")
        
        # Load complete training set
        full_train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.train_transform
        )
        
        # Load test set
        self.test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.test_transform
        )
        
        # Split training set into train and validation
        train_size = int((1 - self.validation_split) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        
        self.train_dataset, val_dataset_with_augmentation = random_split(
            full_train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # Set random seed for reproducibility
        )
        
        # Create validation set without data augmentation
        val_dataset_base = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=False,
            transform=self.test_transform
        )
        
        # Get validation set indices
        val_indices = val_dataset_with_augmentation.indices
        self.val_dataset = torch.utils.data.Subset(val_dataset_base, val_indices)
        
        print(f"Dataset loading completed!")
        print(f"Training set size: {len(self.train_dataset)}")
        print(f"Validation set size: {len(self.val_dataset)}")
        print(f"Test set size: {len(self.test_dataset)}")
    
    def _create_dataloaders(self):
        """
        Create data loaders
        """
        # Training data loader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,  # Enable memory pinning for GPU acceleration
            drop_last=True    # Drop last incomplete batch
        )
        
        # Validation data loader
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        # Test data loader
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_dataloaders(self):
        """
        Get all data loaders
        
        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_class_names(self):
        """
        Get class names
        
        Returns:
            list: List of class names
        """
        return self.class_names
    
    def get_dataset_info(self):
        """
        Get dataset information
        
        Returns:
            dict: Dictionary containing dataset sizes and other info
        """
        return {
            "num_classes": len(self.class_names),
            "train_size": len(self.train_dataset),
            "val_size": len(self.val_dataset),
            "test_size": len(self.test_dataset),
            "input_shape": (3, 224, 224),  # Updated for 224x224 3-channel input
            "class_names": self.class_names
        }


def create_CIFAR10_dataloaders(data_dir="./data", batch_size=128, 
                                   num_workers=4, validation_split=0.1):
    # Factory function to create CIFAR-10 data loaders
    dataloader = CIFAR10DataLoader(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        validation_split=validation_split
    )
    
    train_loader, val_loader, test_loader = dataloader.get_dataloaders()
    dataset_info = dataloader.get_dataset_info()
    
    return train_loader, val_loader, test_loader, dataset_info


if __name__ == "__main__":
    # Test data loader
    print("Testing data loader...")
    
    # Create data loaders
    train_loader, val_loader, test_loader, info = create_CIFAR10_dataloaders(
        batch_size=32, num_workers=2
    )
    
    print("Dataset information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test one batch of data
    print("\nTesting data batch...")
    for images, labels in train_loader:
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")
        print(f"Image data type: {images.dtype}")
        print(f"Label data type: {labels.dtype}")
        print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
        break
    
    print("Data loader test successful!")
