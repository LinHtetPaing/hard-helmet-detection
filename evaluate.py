"""
Model evaluation script
"""
import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from model import get_model
from utils import get_transforms


def evaluate_model(model, dataloader, device, class_names):
    """
    Evaluate model on a dataset
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to evaluate on
        class_names: List of class names
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    print("\nEvaluating model...")
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Classification report
    print("\n" + "=" * 70)
    print("Classification Report:")
    print("=" * 70)
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    return {
        'accuracy': accuracy,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities,
        'confusion_matrix': cm,
        'class_names': class_names
    }


def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png'):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage'})
    
    plt.title('Confusion Matrix (Normalized)', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add counts to annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count = cm[i, j]
            percentage = cm_normalized[i, j]
            plt.text(j + 0.5, i + 0.7, f'({count})',
                    ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved to {save_path}")
    plt.close()


def plot_roc_curve(labels, probabilities, class_names, save_path='roc_curve.png'):
    """
    Plot ROC curve for binary classification
    
    Args:
        labels: True labels
        probabilities: Predicted probabilities
        class_names: List of class names
        save_path: Path to save the plot
    """
    from sklearn.metrics import roc_curve, auc
    
    plt.figure(figsize=(10, 8))
    
    # Calculate ROC curve for each class
    for i, class_name in enumerate(class_names):
        # Binary labels for this class
        binary_labels = (labels == i).astype(int)
        class_probs = probabilities[:, i]
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(binary_labels, class_probs)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.plot(fpr, tpr, lw=2, 
                label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=16, pad=20)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ ROC curve saved to {save_path}")
    plt.close()


def plot_prediction_distribution(labels, probabilities, class_names, 
                                 save_path='prediction_distribution.png'):
    """
    Plot prediction confidence distribution
    
    Args:
        labels: True labels
        probabilities: Predicted probabilities
        class_names: List of class names
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, len(class_names), figsize=(15, 5))
    
    if len(class_names) == 1:
        axes = [axes]
    
    for i, class_name in enumerate(class_names):
        class_mask = labels == i
        class_probs = probabilities[class_mask, i]
        
        axes[i].hist(class_probs, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[i].set_xlabel('Confidence', fontsize=12)
        axes[i].set_ylabel('Count', fontsize=12)
        axes[i].set_title(f'{class_name}\n(n={class_mask.sum()})', fontsize=14)
        axes[i].grid(True, alpha=0.3)
        axes[i].axvline(x=0.5, color='r', linestyle='--', linewidth=2, label='Threshold')
        axes[i].legend()
    
    plt.suptitle('Prediction Confidence Distribution', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Prediction distribution saved to {save_path}")
    plt.close()


def main(args):
    """
    Main evaluation function
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    transform = get_transforms(train=False, img_size=args.img_size)
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"\nDataset: {args.data_dir}")
    print(f"Samples: {len(dataset)}")
    print(f"Classes: {dataset.classes}")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = get_model(num_classes=len(dataset.classes), pretrained=False)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"✓ Model loaded successfully!")
    print(f"Checkpoint info - Epoch: {checkpoint['epoch']}, Accuracy: {checkpoint['accuracy']:.2f}%")
    
    # Evaluate
    results = evaluate_model(model, dataloader, device, dataset.classes)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Evaluation Summary:")
    print("=" * 70)
    print(f"Overall Accuracy: {results['accuracy']:.2%}")
    print(f"Total Samples: {len(results['labels'])}")
    print("=" * 70)
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot confusion matrix
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(results['confusion_matrix'], results['class_names'], cm_path)
    
    # Plot ROC curve
    roc_path = os.path.join(output_dir, 'roc_curve.png')
    plot_roc_curve(results['labels'], results['probabilities'], results['class_names'], roc_path)
    
    # Plot prediction distribution
    dist_path = os.path.join(output_dir, 'prediction_distribution.png')
    plot_prediction_distribution(results['labels'], results['probabilities'], 
                                 results['class_names'], dist_path)
    
    print(f"\n✓ All evaluation plots saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate helmet detection model")
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to evaluation data directory')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Path to save evaluation results')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU for evaluation')
    
    args = parser.parse_args()
    main(args)

