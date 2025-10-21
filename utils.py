"""
Utility functions for helmet detection
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torchvision import transforms


def get_transforms(train=True, img_size=224):
    """
    Get data transforms for training and validation
    
    Args:
        train (bool): Training mode (applies augmentation)
        img_size (int): Image size for resizing
    
    Returns:
        transforms.Compose: Transform pipeline
    """
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


def save_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        accuracy: Current accuracy
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath, device):
    """
    Load model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        filepath: Path to checkpoint
        device: Device to load model on
    
    Returns:
        model, optimizer, epoch, loss, accuracy
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    accuracy = checkpoint['accuracy']
    
    print(f"Checkpoint loaded from {filepath}")
    print(f"Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    
    return model, optimizer, epoch, loss, accuracy


def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training history
    
    Args:
        history (dict): Dictionary containing training metrics
        save_path (str): Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy
    axes[1].plot(history['train_acc'], label='Train Accuracy')
    axes[1].plot(history['val_acc'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")
    plt.close()


def visualize_predictions(images, true_labels, pred_labels, class_names, num_images=8):
    """
    Visualize model predictions
    
    Args:
        images: Batch of images
        true_labels: True labels
        pred_labels: Predicted labels
        class_names: List of class names
        num_images: Number of images to display
    """
    num_images = min(num_images, len(images))
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.ravel()
    
    for i in range(num_images):
        # Denormalize image
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        true_class = class_names[true_labels[i]]
        pred_class = class_names[pred_labels[i]]
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        axes[i].set_title(f'True: {true_class}\nPred: {pred_class}', color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def preprocess_image(image_path, img_size=224):
    """
    Preprocess a single image for inference
    
    Args:
        image_path (str): Path to image
        img_size (int): Image size
    
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    transform = get_transforms(train=False, img_size=img_size)
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, image


def draw_detection_result(image, prediction, class_names, confidence):
    """
    Draw detection result on image
    
    Args:
        image: PIL Image or numpy array
        prediction: Predicted class index
        class_names: List of class names
        confidence: Prediction confidence
    
    Returns:
        numpy array: Image with detection result drawn
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Draw result
    label = class_names[prediction]
    text = f"{label}: {confidence:.2%}"
    
    # Choose color based on prediction
    color = (0, 255, 0) if prediction == 0 else (0, 0, 255)  # Green for with_helmet (index 0), Red for without_helmet (index 1)
    
    # Draw background rectangle
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
    )
    cv2.rectangle(image, (10, 10), (10 + text_width + 10, 10 + text_height + baseline + 10), color, -1)
    
    # Draw text
    cv2.putText(image, text, (15, 10 + text_height), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def count_parameters(model):
    """
    Count the number of trainable parameters in a model
    
    Args:
        model: PyTorch model
    
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

