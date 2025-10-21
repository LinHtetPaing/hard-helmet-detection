"""
Show misclassified images to help debug
"""
import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
import cv2
import numpy as np
from PIL import Image

from model import get_model
from utils import get_transforms


def show_misclassified(checkpoint_path, data_dir, device='cpu', max_show=20):
    """
    Show misclassified images for manual review
    """
    # Load dataset
    transform = get_transforms(train=False, img_size=224)
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Load model
    model = get_model(num_classes=len(dataset.classes), pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("\n" + "="*70)
    print("MISCLASSIFIED IMAGES VIEWER")
    print("="*70)
    print(f"\nClasses: {dataset.classes}")
    print("\nShowing misclassified images...")
    print("Press any key to see next image, 'q' to quit\n")
    print("="*70)
    
    misclassified_count = 0
    
    with torch.no_grad():
        for idx, (inputs, label) in enumerate(dataloader):
            inputs_gpu = inputs.to(device)
            outputs = model(inputs_gpu)
            probabilities = F.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            label = label.item()
            
            # Check if misclassified
            if prediction != label:
                misclassified_count += 1
                
                if misclassified_count > max_show:
                    print(f"\nShowed {max_show} misclassified images. Stopping.")
                    break
                
                # Get image path
                image_path = dataset.samples[idx][0]
                filename = os.path.basename(image_path)
                
                # Load original image
                img = cv2.imread(image_path)
                if img is None:
                    continue
                
                # Resize for display
                height, width = img.shape[:2]
                max_height = 800
                if height > max_height:
                    scale = max_height / height
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    img = cv2.resize(img, (new_width, new_height))
                
                # Add text overlay
                true_class = dataset.classes[label]
                pred_class = dataset.classes[prediction]
                confidence = probabilities[0][prediction].item()
                
                # Create info panel
                panel_height = 120
                panel = np.zeros((panel_height, img.shape[1], 3), dtype=np.uint8)
                
                # Add text
                cv2.putText(panel, f"File: {filename}", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(panel, f"TRUE CLASS: {true_class}", (10, 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(panel, f"PREDICTED: {pred_class} ({confidence:.1%})", (10, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(panel, f"Misclassified #{misclassified_count}", (10, 115),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Combine
                display_img = np.vstack([panel, img])
                
                # Show
                cv2.imshow('Misclassified Images - Press any key for next, Q to quit', display_img)
                print(f"\n[{misclassified_count}] {filename}")
                print(f"    TRUE: {true_class} | PREDICTED: {pred_class} ({confidence:.1%})")
                
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\nQuitting...")
                    break
    
    cv2.destroyAllWindows()
    
    print("\n" + "="*70)
    print(f"Total misclassified images shown: {misclassified_count}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Show misclassified images")
    
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/val',
                       help='Path to data directory')
    parser.add_argument('--max', type=int, default=20,
                       help='Maximum number of images to show')
    parser.add_argument('--cpu', action='store_true',
                       help='Use CPU')
    
    args = parser.parse_args()
    
    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    show_misclassified(args.checkpoint, args.data_dir, device, args.max)


if __name__ == "__main__":
    main()

