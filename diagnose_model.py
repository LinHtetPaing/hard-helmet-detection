"""
Model Diagnostic Tool
Helps identify why model predictions are wrong
"""
import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from collections import Counter
import numpy as np

from model import get_model
from utils import get_transforms


def diagnose_model(checkpoint_path, data_dir, device='cpu'):
    """
    Diagnose model predictions
    """
    print("\n" + "="*70)
    print("MODEL DIAGNOSTIC TOOL")
    print("="*70)
    
    # Load dataset
    transform = get_transforms(train=False, img_size=224)
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    print(f"\nDataset: {data_dir}")
    print(f"Total images: {len(dataset)}")
    print(f"Classes: {dataset.classes}")
    print(f"Class to index: {dataset.class_to_idx}")
    
    # Count actual distribution
    actual_counts = Counter([label for _, label in dataset.samples])
    print(f"\nActual distribution:")
    for idx, class_name in enumerate(dataset.classes):
        count = actual_counts.get(idx, 0)
        print(f"  {class_name}: {count} images")
    
    # Load model
    print(f"\nLoading model from: {checkpoint_path}")
    model = get_model(num_classes=len(dataset.classes), pretrained=False)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded successfully!")
        print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Checkpoint accuracy: {checkpoint.get('accuracy', 'unknown')}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    model = model.to(device)
    model.eval()
    
    # Make predictions
    print(f"\nAnalyzing predictions...")
    print("="*70)
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_files = []
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # Get filenames
            start_idx = i * dataloader.batch_size
            end_idx = start_idx + inputs.size(0)
            batch_files = [dataset.samples[j][0] for j in range(start_idx, end_idx)]
            all_files.extend(batch_files)
    
    # Analysis
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Prediction distribution
    pred_counts = Counter(all_predictions)
    print(f"\nPrediction distribution:")
    for idx, class_name in enumerate(dataset.classes):
        count = pred_counts.get(idx, 0)
        percentage = 100 * count / len(all_predictions)
        print(f"  {class_name}: {count}/{len(all_predictions)} ({percentage:.1f}%)")
    
    # Check if model is stuck
    if len(pred_counts) == 1:
        print(f"\n⚠️  PROBLEM: Model ALWAYS predicts class {list(pred_counts.keys())[0]} ({dataset.classes[list(pred_counts.keys())[0]]})")
        print("   This is a severe issue!")
    
    # Accuracy per class
    print(f"\nAccuracy per class:")
    for idx, class_name in enumerate(dataset.classes):
        class_mask = all_labels == idx
        if class_mask.sum() > 0:
            class_correct = ((all_predictions == all_labels) & class_mask).sum()
            class_total = class_mask.sum()
            class_acc = 100 * class_correct / class_total
            print(f"  {class_name}: {class_correct}/{class_total} ({class_acc:.1f}%)")
        else:
            print(f"  {class_name}: No samples")
    
    # Overall accuracy
    accuracy = 100 * (all_predictions == all_labels).sum() / len(all_labels)
    print(f"\nOverall accuracy: {accuracy:.2f}%")
    
    # Confidence analysis
    print(f"\nConfidence analysis:")
    avg_confidence = np.max(all_probabilities, axis=1).mean()
    print(f"  Average confidence: {avg_confidence:.2%}")
    
    # Show some wrong predictions
    wrong_mask = all_predictions != all_labels
    wrong_indices = np.where(wrong_mask)[0]
    
    if len(wrong_indices) > 0:
        print(f"\nWrong predictions (showing first 10):")
        print("-"*70)
        for i, idx in enumerate(wrong_indices[:10]):
            true_class = dataset.classes[all_labels[idx]]
            pred_class = dataset.classes[all_predictions[idx]]
            confidence = all_probabilities[idx][all_predictions[idx]]
            filename = os.path.basename(all_files[idx])
            print(f"{i+1}. {filename}")
            print(f"   True: {true_class} | Predicted: {pred_class} (confidence: {confidence:.2%})")
    
    # Show some correct predictions
    correct_mask = all_predictions == all_labels
    correct_indices = np.where(correct_mask)[0]
    
    if len(correct_indices) > 0:
        print(f"\nCorrect predictions (showing first 5):")
        print("-"*70)
        for i, idx in enumerate(correct_indices[:5]):
            true_class = dataset.classes[all_labels[idx]]
            confidence = all_probabilities[idx][all_predictions[idx]]
            filename = os.path.basename(all_files[idx])
            print(f"{i+1}. {filename}")
            print(f"   Class: {true_class} (confidence: {confidence:.2%})")
    
    # Diagnosis
    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)
    
    issues = []
    
    if len(pred_counts) == 1:
        issues.append("❌ Model ALWAYS predicts the same class!")
        issues.append("   Possible causes:")
        issues.append("   - Insufficient training data")
        issues.append("   - Model didn't learn properly")
        issues.append("   - Class imbalance during training")
        issues.append("   - Wrong labels in dataset")
    
    if len(dataset) < 100:
        issues.append(f"⚠️  Very small dataset ({len(dataset)} images)")
        issues.append("   Recommendation: Collect at least 500+ images per class")
    
    if accuracy < 60:
        issues.append(f"⚠️  Low accuracy ({accuracy:.1f}%)")
        issues.append("   Model needs more training or better data")
    
    if avg_confidence < 0.7:
        issues.append(f"⚠️  Low confidence ({avg_confidence:.2%})")
        issues.append("   Model is uncertain about predictions")
    elif avg_confidence > 0.95:
        issues.append(f"⚠️  Very high confidence ({avg_confidence:.2%})")
        issues.append("   Might indicate overfitting")
    
    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(issue)
    else:
        print("\n✓ No major issues detected!")
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print("\n1. INCREASE DATASET SIZE:")
    print("   - You need at least 500-1000 images per class")
    print("   - Current: only 26-52 images per class")
    print("   - Download more data from Kaggle/Roboflow")
    print("\n2. CHECK YOUR DATA:")
    print("   - Manually verify images are in correct folders")
    print("   - Use: python image_sorter.py to review/sort images")
    print("\n3. RETRAIN WITH MORE DATA:")
    print("   - python train.py --epochs 30 --batch_size 16")
    print("   - Use --freeze_backbone for faster initial training")
    print("\n4. IF STUCK, TRY:")
    print("   - Lower learning rate: --lr 0.0001")
    print("   - Different batch size: --batch_size 8")
    print("   - More augmentation (already in training)")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Diagnose model predictions")
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to data directory to test on')
    parser.add_argument('--cpu', action='store_true',
                       help='Use CPU instead of GPU')
    
    args = parser.parse_args()
    
    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    diagnose_model(args.checkpoint, args.data_dir, device)


if __name__ == "__main__":
    main()

