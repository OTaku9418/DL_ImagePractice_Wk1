
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import os


class ModelEvaluator:
    """
    Model evaluator for comprehensive performance analysis
    """
    
    def __init__(self, model, test_loader, class_names, device=None, save_dir="./logs"):
        """
        Initialize evaluator
        
        Args:
            model (nn.Module): Model to evaluate
            test_loader (DataLoader): Test data loader
            class_names (list): List of class names
            device (torch.device): Computing device
            save_dir (str): Results save directory
        """
        self.model = model
        self.test_loader = test_loader
        self.class_names = class_names
        self.save_dir = save_dir
        
        # Setup computing device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Move model to device and set to evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Evaluator initialized, using device: {self.device}")
    
    def predict_all(self):
        """
        Make predictions on all test data
        
        Returns:
            tuple: (true_labels, predicted_labels, prediction_probabilities)
        """
        print("Making model predictions...")
        
        all_targets = []
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc="Predicting"):
                # Move data to device
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward propagation
                outputs = self.model(inputs)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)
                
                # Collect results
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_targets), np.array(all_predictions), np.array(all_probabilities)
    
    def calculate_accuracy(self, targets, predictions):
        """
        Calculate accuracy
        
        Args:
            targets (np.array): True labels
            predictions (np.array): Predicted labels
            
        Returns:
            float: Accuracy percentage
        """
        return (targets == predictions).mean() * 100
    
    def generate_classification_report(self, targets, predictions, save=True):
        """
        Generate classification report
        
        Args:
            targets (np.array): True labels
            predictions (np.array): Predicted labels
            save (bool): Whether to save report
            
        Returns:
            str: Classification report string
        """
        report = classification_report(
            targets, predictions,
            target_names=self.class_names,
            digits=4
        )
        
        print("Classification Report:")
        print(report)
        
        if save:
            report_path = os.path.join(self.save_dir, "classification_report.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("Fashion-MNIST Classification Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(report)
            print(f"Classification report saved to: {report_path}")
        
        return report
    
    def analyze_misclassified_samples(self, targets, predictions, probabilities, 
                                    num_samples=10, save=True):
        """
        Analyze misclassified samples
        
        Args:
            targets (np.array): True labels
            predictions (np.array): Predicted labels
            probabilities (np.array): Prediction probabilities
            num_samples (int): Number of error samples to display
            save (bool): Whether to save analysis results
        """
        # Find misclassified samples
        misclassified_indices = np.where(targets != predictions)[0]
        
        if len(misclassified_indices) == 0:
            print("No misclassified samples found!")
            return
        
        print(f"\nMisclassified samples analysis (total {len(misclassified_indices)} errors):")
        
        # Sort by prediction confidence to show most "confident" wrong predictions
        confidences = probabilities[misclassified_indices, predictions[misclassified_indices]]
        sorted_indices = misclassified_indices[np.argsort(confidences)[::-1]]
        
        # Display top num_samples error samples
        print(f"\nTop {min(num_samples, len(sorted_indices))} most confident wrong predictions:")
        for i, idx in enumerate(sorted_indices[:num_samples]):
            true_label = targets[idx]
            pred_label = predictions[idx]
            confidence = probabilities[idx, pred_label]
            
            print(f"  {i+1}. True: {self.class_names[true_label]}, "
                  f"Predicted: {self.class_names[pred_label]}, "
                  f"Confidence: {confidence:.4f}")
        
        if save:
            # Save detailed error analysis
            analysis_path = os.path.join(self.save_dir, "misclassification_analysis.txt")
            with open(analysis_path, 'w', encoding='utf-8') as f:
                f.write("Misclassified Samples Analysis\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total error samples: {len(misclassified_indices)}\n")
                f.write(f"Error rate: {len(misclassified_indices)/len(targets)*100:.2f}%\n\n")
                
                f.write("Most confident wrong predictions:\n")
                for i, idx in enumerate(sorted_indices[:50]):  # Save top 50
                    true_label = targets[idx]
                    pred_label = predictions[idx]
                    confidence = probabilities[idx, pred_label]
                    f.write(f"{i+1:3d}. True: {self.class_names[true_label]:12s}, "
                           f"Predicted: {self.class_names[pred_label]:12s}, "
                           f"Confidence: {confidence:.4f}\n")
            
            print(f"Error analysis saved to: {analysis_path}")
    
    def comprehensive_evaluation(self, history_path=None, visualizer=None):
        """
        Comprehensive model performance evaluation
        
        Args:
            history_path (str): Path to training history file
            visualizer: Visualizer instance for plotting
        """
        print("Starting comprehensive evaluation...")
        print("=" * 60)
        
        # Get prediction results
        targets, predictions, probabilities = self.predict_all()
        
        # Calculate basic metrics
        accuracy = self.calculate_accuracy(targets, predictions)
        print(f"\nTest set accuracy: {accuracy:.2f}%")
        
        # Check if target accuracy is achieved
        target_accuracy = 95.0
        if accuracy >= target_accuracy:
            print(f"Target accuracy {target_accuracy}% achieved successfully!")
        else:
            print(f"Target accuracy {target_accuracy}% not achieved, need to improve by {target_accuracy-accuracy:.2f}%")
        
        # Generate classification report
        self.generate_classification_report(targets, predictions)
        
        # Plot confusion matrix if visualizer is provided
        if visualizer is not None:
            visualizer.plot_confusion_matrix(targets, predictions, self.class_names)
        
        # Analyze misclassified samples
        self.analyze_misclassified_samples(targets, predictions, probabilities)
        
        # Plot training history if provided and visualizer available
        if history_path and os.path.exists(history_path) and visualizer is not None:
            visualizer.plot_training_history(history_path)
        
        print("\nEvaluation completed!")


def create_evaluator(model, test_loader, class_names, device=None, save_dir="./logs"):
    """
    Factory function to create evaluator
    
    Args:
        model (nn.Module): Model to evaluate
        test_loader (DataLoader): Test data loader
        class_names (list): List of class names
        device (torch.device): Computing device
        save_dir (str): Results save directory
        
    Returns:
        ModelEvaluator: Initialized evaluator
    """
    return ModelEvaluator(model, test_loader, class_names, device, save_dir)


if __name__ == "__main__":
    print("Evaluation module test completed!")
