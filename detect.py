"""
Inference script for helmet detection
"""
import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import numpy as np

from model import get_model
from utils import preprocess_image, draw_detection_result


class HelmetDetectorInference:
    """
    Helmet detector inference class
    """
    
    def __init__(self, checkpoint_path, device='cpu', img_size=224):
        """
        Initialize the detector
        
        Args:
            checkpoint_path (str): Path to model checkpoint
            device (str): Device to run inference on
            img_size (int): Input image size
        """
        self.device = torch.device(device)
        self.img_size = img_size
        self.class_names = ['with_helmet', 'without_helmet']  # Must match training data order!
        
        # Load model
        print(f"Loading model from {checkpoint_path}...")
        self.model = get_model(num_classes=2, pretrained=False)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully!")
        print(f"Checkpoint info - Epoch: {checkpoint['epoch']}, Accuracy: {checkpoint['accuracy']:.2f}%")
    
    def predict(self, image_path):
        """
        Predict helmet presence in an image
        
        Args:
            image_path (str): Path to image
        
        Returns:
            prediction (int): Predicted class (0: no helmet, 1: helmet)
            confidence (float): Prediction confidence
            probabilities (dict): Probabilities for each class
        """
        # Preprocess image
        image_tensor, original_image = preprocess_image(image_path, self.img_size)
        image_tensor = image_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
        
        prediction = prediction.item()
        confidence = confidence.item()
        probs_dict = {
            self.class_names[i]: probabilities[0][i].item()
            for i in range(len(self.class_names))
        }
        
        return prediction, confidence, probs_dict, original_image
    
    def predict_batch(self, image_paths):
        """
        Predict helmet presence in multiple images
        
        Args:
            image_paths (list): List of image paths
        
        Returns:
            results (list): List of (prediction, confidence, probabilities) tuples
        """
        results = []
        for image_path in image_paths:
            pred, conf, probs, _ = self.predict(image_path)
            results.append((pred, conf, probs))
        return results


def detect_image(args):
    """
    Detect helmet in a single image
    
    Args:
        args: Command line arguments
    """
    # Initialize detector
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    detector = HelmetDetectorInference(args.checkpoint, device=device, img_size=args.img_size)
    
    # Predict
    print(f"\nProcessing image: {args.image}")
    prediction, confidence, probabilities, original_image = detector.predict(args.image)
    
    # Print results
    print("\n" + "=" * 50)
    print("Detection Results:")
    print("=" * 50)
    print(f"Prediction: {detector.class_names[prediction]}")
    print(f"Confidence: {confidence:.2%}")
    print("\nProbabilities:")
    for class_name, prob in probabilities.items():
        print(f"  {class_name}: {prob:.2%}")
    print("=" * 50)
    
    # Draw and save result
    if args.output:
        result_image = draw_detection_result(
            original_image, prediction, detector.class_names, confidence
        )
        
        # Save result
        result_image = Image.fromarray(result_image)
        result_image.save(args.output)
        print(f"\n✓ Result saved to {args.output}")
    
    # Display result if requested
    if args.display:
        result_image = draw_detection_result(
            original_image, prediction, detector.class_names, confidence
        )
        cv2.imshow('Helmet Detection Result', cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        print("\nPress any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def detect_folder(args):
    """
    Detect helmet in all images in a folder
    
    Args:
        args: Command line arguments
    """
    # Initialize detector
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    detector = HelmetDetectorInference(args.checkpoint, device=device, img_size=args.img_size)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = [
        os.path.join(args.folder, f)
        for f in os.listdir(args.folder)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]
    
    if not image_files:
        print(f"No image files found in {args.folder}")
        return
    
    print(f"\nFound {len(image_files)} images in {args.folder}")
    print("Processing images...\n")
    
    # Create output directory
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    
    # Process each image
    results = []
    for i, image_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processing {os.path.basename(image_path)}...")
        
        prediction, confidence, probabilities, original_image = detector.predict(image_path)
        results.append({
            'image': os.path.basename(image_path),
            'prediction': detector.class_names[prediction],
            'confidence': confidence,
            'probabilities': probabilities
        })
        
        print(f"  → {detector.class_names[prediction]} ({confidence:.2%})")
        
        # Save result
        if args.output:
            result_image = draw_detection_result(
                original_image, prediction, detector.class_names, confidence
            )
            output_path = os.path.join(args.output, f"result_{os.path.basename(image_path)}")
            result_image = Image.fromarray(result_image)
            result_image.save(output_path)
    
    # Print summary
    print("\n" + "=" * 50)
    print("Detection Summary:")
    print("=" * 50)
    with_helmet = sum(1 for r in results if r['prediction'] == 'with_helmet')
    without_helmet = len(results) - with_helmet
    print(f"Total images: {len(results)}")
    print(f"With helmet: {with_helmet} ({100*with_helmet/len(results):.1f}%)")
    print(f"Without helmet: {without_helmet} ({100*without_helmet/len(results):.1f}%)")
    print("=" * 50)
    
    if args.output:
        print(f"\n✓ Results saved to {args.output}")


def detect_webcam(args):
    """
    Real-time helmet detection from webcam
    
    Args:
        args: Command line arguments
    """
    # Initialize detector
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    detector = HelmetDetectorInference(args.checkpoint, device=device, img_size=args.img_size)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("\nStarting webcam detection...")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Save temporary image
        temp_path = '/tmp/temp_webcam_frame.jpg'
        Image.fromarray(frame_rgb).save(temp_path)
        
        # Predict
        prediction, confidence, _, _ = detector.predict(temp_path)
        
        # Draw result
        label = detector.class_names[prediction]
        text = f"{label}: {confidence:.2%}"
        color = (0, 255, 0) if prediction == 0 else (0, 0, 255)  # Green for with_helmet (index 0), Red for without (index 1)
        
        # Draw background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
        )
        cv2.rectangle(frame, (10, 10), (10 + text_width + 10, 10 + text_height + baseline + 10), color, -1)
        
        # Draw text
        cv2.putText(frame, text, (15, 10 + text_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Display
        cv2.imshow('Helmet Detection - Webcam', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nWebcam detection stopped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Helmet detection inference")
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU for inference')
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['image', 'folder', 'webcam'], default='image',
                        help='Detection mode')
    
    # Input parameters
    parser.add_argument('--image', type=str,
                        help='Path to input image (for image mode)')
    parser.add_argument('--folder', type=str,
                        help='Path to input folder (for folder mode)')
    
    # Output parameters
    parser.add_argument('--output', type=str,
                        help='Path to save output (image path or folder path)')
    parser.add_argument('--display', action='store_true',
                        help='Display the result (for image mode)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'image' and not args.image:
        parser.error("--image is required for image mode")
    if args.mode == 'folder' and not args.folder:
        parser.error("--folder is required for folder mode")
    
    # Run detection
    if args.mode == 'image':
        detect_image(args)
    elif args.mode == 'folder':
        detect_folder(args)
    elif args.mode == 'webcam':
        detect_webcam(args)

