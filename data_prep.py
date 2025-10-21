"""
Data preparation and validation script
"""
import os
import argparse
from pathlib import Path
import shutil
from collections import Counter


def validate_dataset(data_dir):
    """
    Validate dataset structure and show statistics
    
    Args:
        data_dir (str): Path to dataset directory
    """
    print(f"\nValidating dataset: {data_dir}")
    print("=" * 60)
    
    if not os.path.exists(data_dir):
        print(f"❌ Error: Directory {data_dir} does not exist")
        return False
    
    # Check for class directories
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    if len(subdirs) == 0:
        print(f"❌ Error: No subdirectories found in {data_dir}")
        return False
    
    print(f"Found {len(subdirs)} classes: {subdirs}")
    
    # Count images per class
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    class_counts = {}
    
    for subdir in subdirs:
        class_path = os.path.join(data_dir, subdir)
        images = [
            f for f in os.listdir(class_path)
            if os.path.splitext(f)[1].lower() in image_extensions
        ]
        class_counts[subdir] = len(images)
    
    # Print statistics
    print("\nClass Distribution:")
    print("-" * 60)
    total_images = sum(class_counts.values())
    
    for class_name, count in class_counts.items():
        percentage = 100 * count / total_images if total_images > 0 else 0
        print(f"  {class_name:20s}: {count:5d} images ({percentage:5.1f}%)")
    
    print("-" * 60)
    print(f"  {'Total':20s}: {total_images:5d} images")
    
    # Check for imbalance
    if len(class_counts) > 1:
        counts = list(class_counts.values())
        max_count = max(counts)
        min_count = min(counts)
        if max_count > 0:
            imbalance_ratio = max_count / min_count
            if imbalance_ratio > 2:
                print(f"\n⚠️  Warning: Class imbalance detected (ratio: {imbalance_ratio:.2f}:1)")
                print("   Consider balancing your dataset for better results")
    
    # Validation checks
    print("\nValidation Checks:")
    print("-" * 60)
    
    checks_passed = True
    
    if total_images == 0:
        print("❌ No images found in dataset")
        checks_passed = False
    else:
        print(f"✓ Found {total_images} total images")
    
    if len(subdirs) < 2:
        print("⚠️  Warning: Only one class found (need at least 2 for classification)")
    else:
        print(f"✓ Found {len(subdirs)} classes")
    
    if any(count < 10 for count in class_counts.values()):
        print("⚠️  Warning: Some classes have very few images (< 10)")
        print("   Recommend at least 100 images per class")
    else:
        print("✓ All classes have sufficient images")
    
    print("=" * 60)
    
    return checks_passed


def split_dataset(source_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split dataset into train/val/test sets
    
    Args:
        source_dir (str): Source directory with class subdirectories
        train_ratio (float): Ratio for training set
        val_ratio (float): Ratio for validation set
        test_ratio (float): Ratio for test set
        seed (int): Random seed for reproducibility
    """
    import random
    
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
        raise ValueError("Ratios must sum to 1.0")
    
    random.seed(seed)
    
    print(f"\nSplitting dataset from: {source_dir}")
    print(f"Train: {train_ratio:.1%}, Val: {val_ratio:.1%}, Test: {test_ratio:.1%}")
    print("=" * 60)
    
    # Get base directory
    base_dir = os.path.dirname(source_dir)
    
    # Create split directories
    split_dirs = {
        'train': os.path.join(base_dir, 'train'),
        'val': os.path.join(base_dir, 'val'),
        'test': os.path.join(base_dir, 'test')
    }
    
    # Get class directories
    classes = [d for d in os.listdir(source_dir) 
               if os.path.isdir(os.path.join(source_dir, d))]
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    
    for class_name in classes:
        print(f"\nProcessing class: {class_name}")
        
        # Get all images for this class
        class_path = os.path.join(source_dir, class_name)
        images = [
            f for f in os.listdir(class_path)
            if os.path.splitext(f)[1].lower() in image_extensions
        ]
        
        # Shuffle images
        random.shuffle(images)
        
        # Calculate split indices
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train + n_val],
            'test': images[n_train + n_val:]
        }
        
        # Copy files to split directories
        for split_name, split_images in splits.items():
            # Create class directory in split
            split_class_dir = os.path.join(split_dirs[split_name], class_name)
            os.makedirs(split_class_dir, exist_ok=True)
            
            # Copy images
            for img_name in split_images:
                src = os.path.join(class_path, img_name)
                dst = os.path.join(split_class_dir, img_name)
                shutil.copy2(src, dst)
            
            print(f"  {split_name:5s}: {len(split_images):4d} images → {split_class_dir}")
    
    print("\n" + "=" * 60)
    print("✓ Dataset split complete!")
    print("\nValidating splits...")
    
    # Validate each split
    for split_name, split_dir in split_dirs.items():
        print(f"\n{split_name.upper()} SET:")
        validate_dataset(split_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset preparation and validation")
    
    parser.add_argument('--mode', type=str, choices=['validate', 'split'], required=True,
                        help='Mode: validate or split dataset')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Training set ratio (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation set ratio (default: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Test set ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for splitting (default: 42)')
    
    args = parser.parse_args()
    
    if args.mode == 'validate':
        validate_dataset(args.data_dir)
    elif args.mode == 'split':
        split_dataset(
            args.data_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )

