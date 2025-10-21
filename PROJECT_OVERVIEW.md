# Helmet Detection Project Overview

## 📁 Project Structure

```
helmet_detection/
│
├── Core Files
│   ├── model.py                    # ResNet50-based helmet detection model
│   ├── train.py                    # Training script with full pipeline
│   ├── detect.py                   # Inference script (image/folder/webcam)
│   ├── evaluate.py                 # Model evaluation with metrics & plots
│   ├── utils.py                    # Utility functions and helpers
│   └── data_prep.py                # Dataset validation and splitting
│
├── Configuration & Setup
│   ├── requirements.txt            # Python dependencies
│   ├── .gitignore                  # Git ignore patterns
│   ├── quick_start.sh              # Quick setup script
│   └── example_usage.py            # Usage examples and demonstrations
│
├── Documentation
│   ├── README.md                   # Complete documentation
│   └── PROJECT_OVERVIEW.md         # This file
│
└── Directories (created automatically)
    ├── data/                       # Dataset storage
    │   ├── train/
    │   │   ├── with_helmet/
    │   │   └── without_helmet/
    │   ├── val/
    │   │   ├── with_helmet/
    │   │   └── without_helmet/
    │   └── test/
    ├── checkpoints/                # Saved model checkpoints
    └── models/                     # Additional model files
```

## 🎯 Key Features

### 1. **Model (model.py)**
- Uses pretrained ResNet50 from ImageNet
- Custom classifier for binary classification
- Support for freezing/unfreezing backbone
- ~23M trainable parameters

### 2. **Training (train.py)**
- Complete training pipeline
- Data augmentation (flip, rotate, color jitter)
- Learning rate scheduling
- Automatic checkpoint saving
- Training visualization
- Progress tracking with tqdm

### 3. **Inference (detect.py)**
- Three detection modes:
  - **Image mode**: Process single images
  - **Folder mode**: Batch process directories
  - **Webcam mode**: Real-time detection
- Confidence scores and visualization
- GPU/CPU support

### 4. **Evaluation (evaluate.py)**
- Comprehensive metrics:
  - Accuracy, precision, recall, F1-score
  - Confusion matrix visualization
  - ROC curves
  - Prediction confidence distribution
- Detailed classification report

### 5. **Data Preparation (data_prep.py)**
- Dataset validation
- Automatic train/val/test splitting
- Class balance checking
- Dataset statistics

### 6. **Utilities (utils.py)**
- Image preprocessing and augmentation
- Checkpoint management
- Visualization functions
- Helper utilities

## 🚀 Quick Start Workflow

### Step 1: Setup
```bash
cd helmet_detection
chmod +x quick_start.sh
./quick_start.sh
```

### Step 2: Prepare Dataset
```bash
# Validate your dataset
python data_prep.py --mode validate --data_dir data/train

# Or split an existing dataset
python data_prep.py --mode split --data_dir data/raw
```

### Step 3: Train Model
```bash
# Basic training
python train.py --epochs 30 --batch_size 32

# Or with frozen backbone (faster)
python train.py --epochs 20 --freeze_backbone
```

### Step 4: Evaluate Model
```bash
python evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --data_dir data/val
```

### Step 5: Inference
```bash
# Single image
python detect.py \
    --checkpoint checkpoints/best_model.pth \
    --mode image \
    --image test.jpg \
    --output result.jpg

# Batch processing
python detect.py \
    --checkpoint checkpoints/best_model.pth \
    --mode folder \
    --folder test_images/ \
    --output results/

# Real-time webcam
python detect.py \
    --checkpoint checkpoints/best_model.pth \
    --mode webcam
```

## 📊 Model Performance

### Expected Performance (with proper dataset)
- **Accuracy**: 90-95% (with 1000+ images per class)
- **Inference Speed**: 
  - GPU (CUDA): ~50-100 FPS
  - CPU: ~10-20 FPS
- **Model Size**: ~98 MB (full model)

### Training Time (approximate)
- **GPU (RTX 3080)**: ~2-5 minutes per epoch
- **CPU**: ~30-60 minutes per epoch

## 🔧 Customization Options

### Model Customization
```python
# In model.py
# Change backbone
from torchvision.models import resnet34, resnet101, resnet152

# Modify classifier
self.resnet50.fc = nn.Sequential(
    nn.Linear(num_features, 1024),  # Larger hidden layer
    nn.ReLU(),
    nn.Dropout(0.3),  # Less dropout
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)
```

### Training Customization
```bash
# Different optimizer
# Edit train.py: optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

# Custom learning rate schedule
# Edit train.py: scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

# Mixed precision training
# Add: from torch.cuda.amp import autocast, GradScaler
```

### Data Augmentation
```python
# In utils.py, modify get_transforms()
transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),  # Add vertical flip
    transforms.RandomRotation(30),  # Increase rotation
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Add perspective
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    # ... rest of transforms
])
```

## 📈 Monitoring Training

### Training Outputs
- **Console**: Real-time loss and accuracy
- **Checkpoints**: Saved in `checkpoints/`
- **Plots**: `training_history.png`
- **Metrics**: `training_history.json`

### Checkpoint Files
- `best_model.pth`: Best validation accuracy
- `final_model.pth`: Last epoch
- `checkpoint_epoch_N.pth`: Periodic checkpoints

## 🐛 Common Issues & Solutions

### Issue: CUDA Out of Memory
**Solution**: Reduce batch size
```bash
python train.py --batch_size 16  # or 8
```

### Issue: Low Accuracy
**Solutions**:
1. Check dataset quality and balance
2. Increase training epochs
3. Try lower learning rate
4. Use data augmentation
5. Collect more training data

### Issue: Overfitting
**Solutions**:
1. Add more dropout
2. Use data augmentation
3. Reduce model complexity
4. Collect more training data
5. Use early stopping

### Issue: Slow Training
**Solutions**:
1. Use GPU (CUDA)
2. Increase `num_workers`
3. Use `--freeze_backbone`
4. Reduce image size

## 📚 Dataset Recommendations

### Minimum Requirements
- **Images per class**: 100+ (preferably 500+)
- **Image quality**: Clear, varied lighting
- **Balance**: Similar number of images per class
- **Variety**: Different angles, distances, workers

### Public Datasets
1. **Hard Hat Workers Dataset** (Kaggle)
2. **Safety Helmet Detection** (Roboflow)
3. **Construction Site Safety** datasets
4. Custom data from construction sites

### Data Collection Tips
- Capture various lighting conditions
- Include different helmet colors/styles
- Vary camera angles and distances
- Include partial visibility scenarios
- Balance positive and negative examples

## 🔬 Advanced Usage

### Export Model for Production
```python
# Convert to TorchScript
model = get_model(num_classes=2, pretrained=False)
model.load_state_dict(torch.load('checkpoints/best_model.pth')['model_state_dict'])
model.eval()

traced_model = torch.jit.trace(model, torch.randn(1, 3, 224, 224))
traced_model.save('model_traced.pt')
```

### Model Quantization
```python
# Quantize model for faster inference
import torch.quantization

model = get_model(num_classes=2, pretrained=False)
model.load_state_dict(torch.load('checkpoints/best_model.pth')['model_state_dict'])
model.eval()

quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

### Integration with Video Processing
```python
import cv2

cap = cv2.VideoCapture('construction_video.mp4')
detector = HelmetDetectorInference('checkpoints/best_model.pth')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Save frame temporarily
    cv2.imwrite('temp_frame.jpg', frame)
    
    # Detect
    pred, conf, _, _ = detector.predict('temp_frame.jpg')
    
    # Annotate and save/display
    # ... your code here
```

## 📝 Best Practices

1. **Always validate your dataset** before training
2. **Start with frozen backbone** for quick experimentation
3. **Monitor validation metrics** to detect overfitting
4. **Save checkpoints regularly**
5. **Use GPU** when available
6. **Test on diverse images** before deployment
7. **Version control** your trained models
8. **Document** your training configurations

## 🎓 Learning Resources

- **PyTorch Documentation**: https://pytorch.org/docs/
- **ResNet Paper**: "Deep Residual Learning for Image Recognition"
- **Transfer Learning**: PyTorch tutorials
- **Computer Vision**: Fast.ai course

## 📞 Support

For issues or questions:
1. Check this documentation
2. Review error messages carefully
3. Test with example_usage.py
4. Validate your dataset with data_prep.py

---

**Built with PyTorch 🔥 | Powered by ResNet50 🚀**

