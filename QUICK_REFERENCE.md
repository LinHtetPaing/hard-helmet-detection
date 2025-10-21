# Helmet Detection - Quick Reference Guide

## 🚀 Installation (One-time Setup)

```bash
cd helmet_detection
pip install -r requirements.txt
```

## 📂 Dataset Structure

Your dataset should be organized like this:

```
data/
├── train/
│   ├── with_helmet/       # Training images WITH helmets
│   └── without_helmet/    # Training images WITHOUT helmets
└── val/
    ├── with_helmet/       # Validation images WITH helmets
    └── without_helmet/    # Validation images WITHOUT helmets
```

## 🎯 Common Commands

### 1. Validate Dataset
```bash
python data_prep.py --mode validate --data_dir data/train
```

### 2. Train Model (Quick - 20 epochs with frozen backbone)
```bash
python train.py --epochs 20 --freeze_backbone --batch_size 32
```

### 3. Train Model (Full - 50 epochs, fine-tune all layers)
```bash
python train.py --epochs 50 --batch_size 32 --lr 0.0001
```

### 4. Evaluate Model
```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --data_dir data/val
```

### 5. Detect on Single Image
```bash
python detect.py --checkpoint checkpoints/best_model.pth \
                 --mode image --image photo.jpg \
                 --output result.jpg --display
```

### 6. Detect on Folder of Images
```bash
python detect.py --checkpoint checkpoints/best_model.pth \
                 --mode folder --folder my_images/ \
                 --output results/
```

### 7. Real-time Webcam Detection
```bash
python detect.py --checkpoint checkpoints/best_model.pth --mode webcam
```

## 📊 Training Parameters Quick Reference

| Parameter | Default | Description | Recommendation |
|-----------|---------|-------------|----------------|
| `--epochs` | 30 | Training epochs | 20-50 |
| `--batch_size` | 32 | Batch size | 16-64 (GPU), 8-16 (CPU) |
| `--lr` | 0.001 | Learning rate | 0.001 (frozen), 0.0001 (full) |
| `--freeze_backbone` | False | Freeze ResNet50 | True for quick, False for best |
| `--img_size` | 224 | Image size | 224 (standard) |

## 🎨 Example Workflows

### Workflow 1: Quick Test (Small Dataset)
```bash
# 1. Validate
python data_prep.py --mode validate --data_dir data/train

# 2. Quick train (10 epochs, frozen)
python train.py --epochs 10 --freeze_backbone

# 3. Test on image
python detect.py --checkpoint checkpoints/best_model.pth \
                 --mode image --image test.jpg --display
```

### Workflow 2: Production Model (Large Dataset)
```bash
# 1. Split dataset
python data_prep.py --mode split --data_dir data/raw

# 2. Train with frozen backbone first (20 epochs)
python train.py --epochs 20 --freeze_backbone --batch_size 32

# 3. Fine-tune all layers (30 more epochs)
python train.py --epochs 30 --batch_size 16 --lr 0.0001

# 4. Evaluate
python evaluate.py --checkpoint checkpoints/best_model.pth \
                   --data_dir data/val

# 5. Test on folder
python detect.py --checkpoint checkpoints/best_model.pth \
                 --mode folder --folder test_images/ --output results/
```

### Workflow 3: Real-time Deployment
```bash
# 1. Train best model
python train.py --epochs 50 --batch_size 32

# 2. Evaluate thoroughly
python evaluate.py --checkpoint checkpoints/best_model.pth \
                   --data_dir data/test

# 3. Deploy with webcam
python detect.py --checkpoint checkpoints/best_model.pth --mode webcam
```

## 🔧 Troubleshooting Quick Fixes

### Problem: "CUDA out of memory"
```bash
# Solution: Reduce batch size
python train.py --batch_size 8  # or even 4
```

### Problem: "No module named 'torch'"
```bash
# Solution: Install requirements
pip install -r requirements.txt
```

### Problem: "FileNotFoundError: data/train"
```bash
# Solution: Create directory structure
mkdir -p data/train/{with_helmet,without_helmet}
mkdir -p data/val/{with_helmet,without_helmet}
# Then add your images
```

### Problem: Training is too slow
```bash
# Solution 1: Use frozen backbone
python train.py --freeze_backbone

# Solution 2: Reduce image size
python train.py --img_size 128

# Solution 3: Increase workers (if you have multiple CPUs)
python train.py --num_workers 8
```

### Problem: Low accuracy
```bash
# Solution 1: Train longer
python train.py --epochs 50

# Solution 2: Lower learning rate
python train.py --lr 0.0001

# Solution 3: Validate your dataset
python data_prep.py --mode validate --data_dir data/train
# Make sure you have balanced, quality data
```

## 📈 Monitoring Training

During training, you'll see:
```
Epoch [1/30]
Training: 100%|████████| Loss: 0.234 | Acc: 89.5%
Validation: 100%|████████| Loss: 0.198 | Acc: 92.3%

Epoch Summary:
Train Loss: 0.2340 | Train Acc: 89.50%
Val Loss: 0.1980 | Val Acc: 92.30%
✓ New best model saved! (Val Acc: 92.30%)
```

Outputs:
- `checkpoints/best_model.pth` - Best model
- `checkpoints/training_history.png` - Training curves
- `checkpoints/training_history.json` - Metrics data

## 🎯 Expected Results

With a good dataset (500+ images per class):
- **Training time**: 2-5 min/epoch (GPU) or 30-60 min/epoch (CPU)
- **Expected accuracy**: 90-95%
- **Inference speed**: 50-100 FPS (GPU) or 10-20 FPS (CPU)

## 📝 File Purposes

| File | Purpose |
|------|---------|
| `model.py` | ResNet50 model definition |
| `train.py` | Training script |
| `detect.py` | Inference (image/folder/webcam) |
| `evaluate.py` | Evaluation with metrics |
| `utils.py` | Helper functions |
| `data_prep.py` | Dataset validation/splitting |
| `example_usage.py` | Usage demonstrations |
| `requirements.txt` | Python dependencies |

## 💡 Tips

1. **Always start with `--freeze_backbone`** for fast experimentation
2. **Use `data_prep.py`** to check dataset before training
3. **Monitor validation accuracy** to detect overfitting
4. **Use GPU** when available for 10-20x speedup
5. **Save your best models** - they're in `checkpoints/`
6. **Test incrementally**: single image → folder → webcam

## 🆘 Getting Help

1. **Read error messages** carefully
2. **Check dataset** with `data_prep.py --mode validate`
3. **Test with examples**: `python example_usage.py`
4. **Review logs** in console output
5. **Check README.md** for detailed documentation

## 🎓 Learning Path

1. **Day 1**: Setup, dataset preparation, quick training
2. **Day 2**: Full training, evaluation, parameter tuning
3. **Day 3**: Real-time detection, optimization, deployment

---

**Keep this guide handy! Bookmark it for quick reference.** 🔖

