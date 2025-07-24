
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
from sklearn.metrics import confusion_matrix


class TrainingVisualizer:
    
    # Training process visualization utilities
    
    
    def __init__(self, save_dir="./logs"):
        # Initialize visualizer
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_training_history(self, history_path):
        # Plot training curves
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
        except FileNotFoundError:
            print(f"Training history file not found: {history_path}")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate curve
        axes[1, 0].plot(epochs, history['lr'], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Accuracy difference (overfitting detection)
        acc_diff = np.array(history['train_acc']) - np.array(history['val_acc'])
        axes[1, 1].plot(epochs, acc_diff, 'm-', linewidth=2)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Training-Validation Accuracy Difference', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy Difference (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        history_plot_path = os.path.join(self.save_dir, "training_history.png")
        plt.savefig(history_plot_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {history_plot_path}")
        plt.close()
        
        # Print best results
        best_val_acc_idx = np.argmax(history['val_acc'])
        print(f"Best validation accuracy: {max(history['val_acc']):.2f}% (Epoch {best_val_acc_idx + 1})")
        print(f"Final validation accuracy: {history['val_acc'][-1]:.2f}%")
    
    def plot_confusion_matrix(self, targets, predictions, class_names):
        
        # Plot confusion matrix
        
        # Calculate confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot confusion matrix heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        cm_path = os.path.join(self.save_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {cm_path}")
        plt.close()
        
        # Calculate per-class accuracy
        print("Per-class accuracy:")
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        for i, (class_name, acc) in enumerate(zip(class_names, class_accuracies)):
            print(f"  {class_name}: {acc:.4f} ({acc*100:.2f}%)")


def create_visualizer(save_dir="./logs"):
    # Factory function to create visualizer
    
    return TrainingVisualizer(save_dir=save_dir)
