# Helmet Detection System

A deep learning-based system for detecting safety helmets on construction workers using PyTorch and ResNet50.

## 🎯 Overview

This project implements a binary classification model to detect whether construction workers are wearing safety helmets. It uses a pretrained ResNet50 model fine-tuned on helmet detection data.

## ✨ Features

- **Transfer Learning**: Uses pretrained ResNet50 from ImageNet
- **Binary Classification**: Detects "with_helmet" vs "without_helmet"
- **Multiple Inference Modes**:
  - Single image detection
  - Batch folder detection
  - Real-time webcam detection
- **Training Tools**: Complete training pipeline with:
  - Data augmentation
  - Learning rate scheduling
  - Checkpoint saving
  - Training visualization
- **Easy to Use**: Simple command-line interface

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

## 🚀 Installation

1. Clone or download this project:
```bash
cd helmet_detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install torch torchvision pillow numpy matplotlib opencv-python tqdm scikit-learn
```

## 📁 Project Structure

```
helmet_detection/
├── model.py              # ResNet50-based model definition
├── train.py              # Training script
├── detect.py             # Inference script
├── utils.py              # Utility functions
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── data/                # Dataset directory
│   ├── train/
│   │   ├── with_helmet/
│   │   └── without_helmet/
│   ├── val/
│   │   ├── with_helmet/
│   │   └── without_helmet/
│   └── test/
├── checkpoints/         # Saved model checkpoints
└── models/             # Additional model files
```

## 📊 Dataset Preparation

Organize your dataset in the following structure:

```
data/
├── train/
│   ├── with_helmet/       # Images of workers wearing helmets
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── without_helmet/    # Images of workers without helmets
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── val/
    ├── with_helmet/
    └── without_helmet/
```

### Dataset Sources

You can use publicly available datasets such as:
- **Hard Hat Detection Dataset** from Roboflow
- **Safety Helmet Detection** from Kaggle
- **Custom dataset** from construction sites (with proper permissions)

## 🏋️ Training

### Basic Training

```bash
python train.py --train_dir data/train --val_dir data/val --epochs 30
```

### Advanced Training Options

```bash
python train.py \
  --train_dir data/train \
  --val_dir data/val \
  --checkpoint_dir checkpoints \
  --epochs 50 \
  --batch_size 32 \
  --lr 0.001 \
  --img_size 224 \
  --num_workers 4
```

### Training with Frozen Backbone (Faster)

For faster training, freeze the ResNet50 backbone and only train the classifier:

```bash
python train.py --train_dir data/train --val_dir data/val --freeze_backbone --epochs 20
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--train_dir` | `data/train` | Path to training data |
| `--val_dir` | `data/val` | Path to validation data |
| `--checkpoint_dir` | `checkpoints` | Path to save checkpoints |
| `--epochs` | `30` | Number of training epochs |
| `--batch_size` | `32` | Batch size for training |
| `--lr` | `0.001` | Learning rate |
| `--img_size` | `224` | Input image size |
| `--freeze_backbone` | `False` | Freeze ResNet50 backbone |
| `--num_workers` | `4` | Number of data loading workers |
| `--save_freq` | `10` | Save checkpoint every N epochs |

## 🔍 Inference

### Single Image Detection

```bash
python detect.py \
  --checkpoint checkpoints/best_model.pth \
  --mode image \
  --image path/to/image.jpg \
  --output result.jpg \
  --display
```

### Batch Folder Detection

Process all images in a folder:

```bash
python detect.py \
  --checkpoint checkpoints/best_model.pth \
  --mode folder \
  --folder path/to/images/ \
  --output results/
```

### Real-time Webcam Detection

```bash
python detect.py \
  --checkpoint checkpoints/best_model.pth \
  --mode webcam
```

Press 'q' to quit webcam mode.

### Inference Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--checkpoint` | Yes | Path to model checkpoint |
| `--mode` | Yes | Detection mode: `image`, `folder`, or `webcam` |
| `--image` | For image mode | Path to input image |
| `--folder` | For folder mode | Path to input folder |
| `--output` | No | Path to save output |
| `--display` | No | Display result (image mode) |
| `--img_size` | No | Input image size (default: 224) |
| `--cpu` | No | Use CPU instead of GPU |

## 🧪 Testing the Model

Test the model architecture:

```bash
python model.py
```

This will:
- Print the model architecture
- Show the number of parameters
- Test a forward pass with dummy data

## 📈 Model Architecture

The model uses **ResNet50** as the backbone with a custom classifier:

```
ResNet50 (pretrained on ImageNet)
    ↓
[2048 features]
    ↓
Linear(2048 → 512)
    ↓
ReLU + Dropout(0.5)
    ↓
Linear(512 → 2)
    ↓
[Output: with_helmet, without_helmet]
```

## 🎨 Data Augmentation

Training uses the following augmentations:
- Random horizontal flip
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation)
- Resize to 224×224
- Normalization (ImageNet mean/std)

## 💾 Model Checkpoints

During training, the following checkpoints are saved:
- `best_model.pth` - Best model based on validation accuracy
- `final_model.pth` - Model at the end of training
- `checkpoint_epoch_N.pth` - Checkpoint every N epochs
- `training_history.json` - Training metrics
- `training_history.png` - Training plots

## 📊 Results Visualization

After training, you'll find:
- **Training history plot**: Shows loss and accuracy curves
- **Model checkpoints**: Best and final models
- **Training metrics**: JSON file with detailed metrics

## 🔧 Customization

### Modify the Model

Edit `model.py` to:
- Change the number of classes
- Modify the classifier architecture
- Use a different backbone (ResNet34, ResNet101, etc.)

### Adjust Data Augmentation

Edit `utils.py` in the `get_transforms()` function to:
- Add more augmentations
- Change augmentation parameters
- Implement custom transforms

### Change Training Strategy

Edit `train.py` to:
- Use different optimizers (SGD, AdamW, etc.)
- Implement custom learning rate schedules
- Add early stopping
- Implement mixed precision training

## 🐛 Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
python train.py --batch_size 16
```

### Low Training Accuracy

Try:
1. Increase number of epochs
2. Lower learning rate: `--lr 0.0001`
3. Use frozen backbone first: `--freeze_backbone`
4. Check dataset quality and balance

### Slow Training

1. Use GPU if available
2. Increase num_workers: `--num_workers 8`
3. Use frozen backbone: `--freeze_backbone`

## 📝 Example Workflow

1. **Prepare your dataset**:
   ```bash
   # Organize images into data/train and data/val folders
   ```

2. **Train the model**:
   ```bash
   python train.py --epochs 30 --batch_size 32
   ```

3. **Test on a single image**:
   ```bash
   python detect.py --checkpoint checkpoints/best_model.pth \
                    --mode image --image test.jpg --display
   ```

4. **Process a folder of images**:
   ```bash
   python detect.py --checkpoint checkpoints/best_model.pth \
                    --mode folder --folder test_images/ --output results/
   ```

5. **Real-time detection**:
   ```bash
   python detect.py --checkpoint checkpoints/best_model.pth --mode webcam
   ```

## 🎯 Performance Tips

1. **For best accuracy**:
   - Use large, balanced dataset (1000+ images per class)
   - Train for more epochs (50-100)
   - Fine-tune the entire model (don't freeze backbone)

2. **For faster training**:
   - Use `--freeze_backbone` flag
   - Reduce image size: `--img_size 128`
   - Use smaller batch size on CPU

3. **For production deployment**:
   - Export model to ONNX or TorchScript
   - Use model quantization
   - Implement batch processing

## 🤝 Contributing

Feel free to:
- Report issues
- Suggest improvements
- Submit pull requests
- Share your trained models

## 📄 License

This project is provided as-is for educational and commercial use.

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- torchvision for pretrained models
- ResNet paper: "Deep Residual Learning for Image Recognition" (He et al., 2015)

## 📧 Contact

For questions or issues, please open an issue in the repository.

---

**Happy Detecting! Stay Safe! 👷‍♂️🪖**

