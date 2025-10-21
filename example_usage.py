"""
Example usage script demonstrating the helmet detection system
"""
import torch
from model import get_model, HelmetDetector
from utils import count_parameters, get_transforms
from PIL import Image
import numpy as np


def example_1_model_creation():
    """Example 1: Creating and inspecting the model"""
    print("\n" + "=" * 70)
    print("Example 1: Model Creation and Architecture")
    print("=" * 70)
    
    # Create model with pretrained weights
    model = get_model(num_classes=2, pretrained=True, freeze_backbone=False)
    
    print(f"\nModel type: {type(model).__name__}")
    print(f"Number of trainable parameters: {count_parameters(model):,}")
    
    # Get model summary
    print("\nModel structure:")
    print("-" * 70)
    for name, module in model.named_children():
        if hasattr(module, 'named_children'):
            print(f"{name}:")
            for child_name, child_module in module.named_children():
                if child_name == 'fc':  # Print details of final layer
                    print(f"  {child_name}: {child_module}")
        else:
            print(f"{name}: {module}")
    
    print("\n✓ Model created successfully!")


def example_2_freeze_unfreeze():
    """Example 2: Freezing and unfreezing backbone"""
    print("\n" + "=" * 70)
    print("Example 2: Freezing/Unfreezing Backbone")
    print("=" * 70)
    
    model = get_model(num_classes=2, pretrained=True)
    
    # Count trainable parameters before freezing
    params_before = count_parameters(model)
    print(f"\nTrainable parameters (before freezing): {params_before:,}")
    
    # Freeze backbone
    model.freeze_backbone()
    params_frozen = count_parameters(model)
    print(f"Trainable parameters (after freezing): {params_frozen:,}")
    print(f"Reduction: {100*(params_before-params_frozen)/params_before:.1f}%")
    
    # Unfreeze backbone
    model.unfreeze_backbone()
    params_after = count_parameters(model)
    print(f"Trainable parameters (after unfreezing): {params_after:,}")
    
    print("\n✓ Freeze/unfreeze demonstrated!")


def example_3_forward_pass():
    """Example 3: Forward pass with dummy data"""
    print("\n" + "=" * 70)
    print("Example 3: Forward Pass with Dummy Data")
    print("=" * 70)
    
    model = get_model(num_classes=2, pretrained=True)
    model.eval()
    
    # Create dummy input (batch_size=4, channels=3, height=224, width=224)
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Input dtype: {dummy_input.dtype}")
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    
    # Apply softmax to get probabilities
    probabilities = torch.softmax(output, dim=1)
    predictions = torch.argmax(probabilities, dim=1)
    
    print(f"\nProbabilities shape: {probabilities.shape}")
    print(f"Predictions shape: {predictions.shape}")
    
    print("\nSample predictions:")
    print("-" * 70)
    class_names = ['without_helmet', 'with_helmet']
    for i in range(batch_size):
        pred_class = predictions[i].item()
        confidence = probabilities[i, pred_class].item()
        print(f"Sample {i+1}: {class_names[pred_class]:15s} (confidence: {confidence:.2%})")
    
    print("\n✓ Forward pass successful!")


def example_4_transforms():
    """Example 4: Image preprocessing transforms"""
    print("\n" + "=" * 70)
    print("Example 4: Image Preprocessing Transforms")
    print("=" * 70)
    
    # Get transforms
    train_transform = get_transforms(train=True, img_size=224)
    val_transform = get_transforms(train=False, img_size=224)
    
    print("\nTraining transforms:")
    print("-" * 70)
    print(train_transform)
    
    print("\nValidation transforms:")
    print("-" * 70)
    print(val_transform)
    
    # Create a dummy image
    dummy_image = Image.fromarray(
        np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    )
    
    print(f"\nOriginal image size: {dummy_image.size}")
    
    # Apply transforms
    train_tensor = train_transform(dummy_image)
    val_tensor = val_transform(dummy_image)
    
    print(f"Transformed tensor shape: {train_tensor.shape}")
    print(f"Tensor dtype: {train_tensor.dtype}")
    print(f"Tensor range: [{train_tensor.min():.3f}, {train_tensor.max():.3f}]")
    
    print("\n✓ Transforms applied successfully!")


def example_5_model_inference_simulation():
    """Example 5: Complete inference simulation"""
    print("\n" + "=" * 70)
    print("Example 5: Complete Inference Simulation")
    print("=" * 70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model = get_model(num_classes=2, pretrained=True)
    model = model.to(device)
    model.eval()
    
    # Prepare transforms
    transform = get_transforms(train=False, img_size=224)
    
    # Simulate processing an image
    print("\nSimulating image processing...")
    
    # Create dummy image
    dummy_image = Image.fromarray(
        np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
    )
    
    # Transform
    image_tensor = transform(dummy_image).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
    
    # Get results
    class_names = ['without_helmet', 'with_helmet']
    pred_class = prediction.item()
    pred_confidence = confidence.item()
    
    print("\nInference Results:")
    print("-" * 70)
    print(f"Prediction: {class_names[pred_class]}")
    print(f"Confidence: {pred_confidence:.2%}")
    print("\nClass probabilities:")
    for i, class_name in enumerate(class_names):
        prob = probabilities[0, i].item()
        print(f"  {class_name:15s}: {prob:.2%}")
    
    print("\n✓ Inference simulation complete!")


def example_6_batch_processing():
    """Example 6: Batch processing multiple images"""
    print("\n" + "=" * 70)
    print("Example 6: Batch Processing")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes=2, pretrained=True)
    model = model.to(device)
    model.eval()
    
    transform = get_transforms(train=False, img_size=224)
    
    # Simulate batch of images
    batch_size = 8
    print(f"\nProcessing batch of {batch_size} images...")
    
    # Create dummy images
    images = []
    for i in range(batch_size):
        img = Image.fromarray(
            np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        )
        images.append(transform(img))
    
    # Stack into batch
    batch = torch.stack(images).to(device)
    
    print(f"Batch shape: {batch.shape}")
    
    # Batch inference
    with torch.no_grad():
        outputs = model(batch)
        probabilities = torch.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probabilities, 1)
    
    # Display results
    class_names = ['without_helmet', 'with_helmet']
    print("\nBatch Results:")
    print("-" * 70)
    
    for i in range(batch_size):
        pred_class = predictions[i].item()
        confidence = confidences[i].item()
        print(f"Image {i+1:2d}: {class_names[pred_class]:15s} ({confidence:.2%})")
    
    # Statistics
    with_helmet_count = (predictions == 1).sum().item()
    without_helmet_count = batch_size - with_helmet_count
    
    print("\nBatch Statistics:")
    print("-" * 70)
    print(f"With helmet: {with_helmet_count}/{batch_size} ({100*with_helmet_count/batch_size:.1f}%)")
    print(f"Without helmet: {without_helmet_count}/{batch_size} ({100*without_helmet_count/batch_size:.1f}%)")
    
    print("\n✓ Batch processing complete!")


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("HELMET DETECTION SYSTEM - EXAMPLE USAGE")
    print("=" * 70)
    print("\nThis script demonstrates various features of the helmet detection system.")
    print("All examples use dummy data for demonstration purposes.")
    
    try:
        # Run examples
        example_1_model_creation()
        example_2_freeze_unfreeze()
        example_3_forward_pass()
        example_4_transforms()
        example_5_model_inference_simulation()
        example_6_batch_processing()
        
        # Summary
        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Prepare your dataset in data/train and data/val")
        print("  2. Run: python train.py --epochs 30")
        print("  3. Run: python detect.py --checkpoint checkpoints/best_model.pth \\")
        print("                           --mode image --image test.jpg")
        print("\nFor more information, see README.md")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

