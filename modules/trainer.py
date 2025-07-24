
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import os
import json
from collections import defaultdict
import psutil
import gc


class ModelTrainer:

    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device=None, save_dir="./saved_models"):
        # Initialize trainer
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Setup computing device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Move model to specified device
        self.model = self.model.to(self.device)
        
        # Save directory
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Training history record
        self.history = defaultdict(list)
        
        # Best model metrics
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        # Memory monitoring
        self.process = psutil.Process()
    
    def setup_optimizer_and_criterion(self, learning_rate=0.001, weight_decay=1e-4):
        # Setup optimizer and loss function
        
        # Use Adam optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Use cross-entropy loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler: StepLR, decay 0.2 every 2 epochs
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=2, 
            gamma=0.2
        )
        
        print(f"Optimizer: Adam (lr={learning_rate}, weight_decay={weight_decay})")
        print(f"Loss function: CrossEntropyLoss")
        print(f"Learning rate scheduler: StepLR (step_size=2, gamma=0.2)")

    def _get_memory_usage(self):
        
        # Get memory and GPU memory usage
       
        memory_info = {}
        
        # CPU memory usage
        memory_info['cpu_memory_mb'] = self.process.memory_info().rss / 1024 / 1024
        
        # GPU memory usage
        if torch.cuda.is_available():
            memory_info['gpu_memory_mb'] = torch.cuda.memory_allocated(self.device) / 1024 / 1024
            memory_info['gpu_memory_cached_mb'] = torch.cuda.memory_reserved(self.device) / 1024 / 1024
        else:
            memory_info['gpu_memory_mb'] = 0
            memory_info['gpu_memory_cached_mb'] = 0
        
        return memory_info
    
    def train_epoch(self, epoch):
        # Train one epoch
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Create progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]', 
                   leave=False, ncols=100)
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # Move data to device
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward propagation
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward propagation
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            current_acc = 100.0 * correct / total
            current_loss = running_loss / (batch_idx + 1)
            pbar.set_postfix({
                'Loss': f'{current_loss:.3f}',
                'Acc': f'{current_acc:.2f}%'
            })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc
        }
    
    def validate_epoch(self, epoch):
        # Validate one epoch
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Create progress bar
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]', 
                   leave=False, ncols=100)
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(pbar):
                # Move data to device
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward propagation
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                current_acc = 100.0 * correct / total
                current_loss = running_loss / (batch_idx + 1)
                pbar.set_postfix({
                    'Loss': f'{current_loss:.3f}',
                    'Acc': f'{current_acc:.2f}%'
                })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100.0 * correct / total
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc
        }
    
    def test_model(self):
        # Test model performance
        print("\nStarting model testing...")
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Create progress bar
        pbar = tqdm(self.test_loader, desc='Testing', ncols=100)
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(pbar):
                # Move data to device
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward propagation
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                current_acc = 100.0 * correct / total
                pbar.set_postfix({'Acc': f'{current_acc:.2f}%'})
        
        # Calculate test metrics
        test_loss = running_loss / len(self.test_loader)
        test_acc = 100.0 * correct / total
        
        print(f"\nTest results:")
        print(f"  Test loss: {test_loss:.4f}")
        print(f"  Test accuracy: {test_acc:.2f}%")
        
        return {
            'loss': test_loss,
            'accuracy': test_acc
        }
    
    def save_model(self, epoch, is_best=False):
        # Save model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': dict(self.history)
        }
        
        # Save current checkpoint
        checkpoint_path = os.path.join(self.save_dir, 'checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model separately if needed
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Best model saved to: {best_path}")
    
    def train(self, num_epochs=100, save_every=10):
        # Train model
        print(f"\nStarting training for {num_epochs} epochs...")
        print("=" * 80)
        
        # Record training start time
        train_start_time = time.time()
        
        # Initial memory status
        initial_memory = self._get_memory_usage()
        print(f"Initial memory usage: CPU {initial_memory['cpu_memory_mb']:.1f}MB, "
              f"GPU {initial_memory['gpu_memory_mb']:.1f}MB")
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Train one epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate one epoch
            val_metrics = self.validate_epoch(epoch)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['lr'].append(current_lr)
            
            # Check if this is the best model
            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
                self.best_epoch = epoch
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Get memory usage
            memory_info = self._get_memory_usage()
            
            # Print epoch results
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Train Acc: {train_metrics['accuracy']:6.2f}% | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:6.2f}% | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {epoch_time:.1f}s | "
                  f"GPU: {memory_info['gpu_memory_mb']:.0f}MB"
                  + (" *" if is_best else ""))
            
            # Save model periodically
            if (epoch + 1) % save_every == 0:
                self.save_model(epoch, is_best=False)
            
            # Save best model
            if is_best:
                self.save_model(epoch, is_best=True)
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # Training completed
        total_time = time.time() - train_start_time
        final_memory = self._get_memory_usage()
        
        print("=" * 80)
        print(f"Training completed!")
        print(f"Total training time: {total_time/3600:.2f} hours ({total_time:.1f} seconds)")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}% (Epoch {self.best_epoch+1})")
        print(f"Final memory usage: CPU {final_memory['cpu_memory_mb']:.1f}MB, "
              f"GPU {final_memory['gpu_memory_mb']:.1f}MB")
        
        # Save training history
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(dict(self.history), f, indent=2)
        print(f"Training history saved to: {history_path}")
        
        # Final model save
        self.save_model(num_epochs-1, is_best=False)


def create_trainer(model, train_loader, val_loader, test_loader, 
                  device=None, save_dir="./saved_models"):
    # Factory function to create trainer
    return ModelTrainer(model, train_loader, val_loader, test_loader, device, save_dir)


if __name__ == "__main__":
    print("Trainer module test completed!")
