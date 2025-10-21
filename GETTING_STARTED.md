# Getting Started with Helmet Detection

Welcome! This guide will help you get started with the helmet detection system in just a few minutes.

## ✅ Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- (Optional) CUDA-capable GPU for faster training

## 🚀 Step-by-Step Setup

### Step 1: Install Dependencies (2 minutes)

```bash
cd helmet_detection
pip install -r requirements.txt
```

This will install:
- PyTorch (deep learning framework)
- torchvision (computer vision tools)
- OpenCV (image processing)
- Other utilities

### Step 2: Test the Installation (1 minute)

```bash
python example_usage.py
```

This will run several examples demonstrating the system's capabilities.

### Step 3: Prepare Your Dataset (10-30 minutes)

Create the directory structure:

```bash
mkdir -p data/train/with_helmet
mkdir -p data/train/without_helmet
mkdir -p data/val/with_helmet
mkdir -p data/val/without_helmet
```

Then add your images:
- Put training images of workers WITH helmets in `data/train/with_helmet/`
- Put training images of workers WITHOUT helmets in `data/train/without_helmet/`
- Put validation images similarly in `data/val/`

**Recommendations:**
- Minimum: 100 images per class
- Good: 500+ images per class
- Better: 1000+ images per class

**Where to get data:**
1. Public datasets: Kaggle, Roboflow (search "hard hat detection" or "helmet detection")
2. Custom data from construction sites (with proper permissions)
3. Web scraping (respect copyright)

Validate your dataset:

```bash
python data_prep.py --mode validate --data_dir data/train
python data_prep.py --mode validate --data_dir data/val
```

### Step 4: Train Your First Model (10-60 minutes)

**Quick training** (frozen backbone, 20 epochs):

```bash
python train.py --epochs 20 --freeze_backbone --batch_size 32
```

**Full training** (fine-tune all layers, 30 epochs):

```bash
python train.py --epochs 30 --batch_size 32
```

Training outputs:
- `checkpoints/best_model.pth` - Your trained model
- `checkpoints/training_history.png` - Training curves
- Console shows real-time progress

### Step 5: Test Your Model (1 minute)

**On a single image:**

```bash
python detect.py \
    --checkpoint checkpoints/best_model.pth \
    --mode image \
    --image path/to/test_image.jpg \
    --display
```

**On a folder of images:**

```bash
python detect.py \
    --checkpoint checkpoints/best_model.pth \
    --mode folder \
    --folder path/to/test_images/ \
    --output results/
```

**Real-time webcam detection:**

```bash
python detect.py \
    --checkpoint checkpoints/best_model.pth \
    --mode webcam
```

Press 'q' to quit webcam mode.

### Step 6: Evaluate Your Model (2 minutes)

```bash
python evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --data_dir data/val \
    --output_dir evaluation_results
```

This generates:
- Confusion matrix
- ROC curves
- Detailed classification report
- Prediction distribution plots

## 🎯 What You've Built

You now have a complete helmet detection system that can:

✅ Detect helmets in images with 90-95% accuracy (with good data)  
✅ Process single images or entire folders  
✅ Run real-time detection from webcam  
✅ Provide confidence scores for each prediction  
✅ Generate comprehensive evaluation metrics  

## 📚 Next Steps

### Improve Your Model

1. **Collect more data** - More diverse, high-quality images
2. **Balance your dataset** - Equal images per class
3. **Train longer** - Increase epochs: `--epochs 50`
4. **Fine-tune hyperparameters** - Learning rate, batch size
5. **Use data augmentation** - Already included in training!

### Deploy Your Model

1. **Export to production format** (TorchScript, ONNX)
2. **Integrate with video streams** 
3. **Build a web API** (Flask, FastAPI)
4. **Create a mobile app** (PyTorch Mobile)
5. **Deploy to edge devices** (Raspberry Pi, Jetson Nano)

### Explore the Code

- `model.py` - Understand the ResNet50 architecture
- `train.py` - Customize training process
- `detect.py` - Modify inference pipeline
- `utils.py` - Add your own utility functions

## 💡 Tips for Success

1. **Start small**: Test with a small dataset first (100 images)
2. **Use frozen backbone**: Faster training, good baseline
3. **Monitor validation metrics**: Avoid overfitting
4. **Test incrementally**: Image → Folder → Webcam
5. **Save your work**: Keep track of best models
6. **Document experiments**: Note what works and what doesn't

## 🐛 Common First-Time Issues

### "CUDA out of memory"
```bash
# Reduce batch size
python train.py --batch_size 8
```

### "ImportError: No module named torch"
```bash
# Install requirements again
pip install -r requirements.txt
```

### "FileNotFoundError: data/train"
```bash
# Create directories
mkdir -p data/train/{with_helmet,without_helmet}
mkdir -p data/val/{with_helmet,without_helmet}
```

### "RuntimeError: Model file doesn't exist"
```bash
# Train a model first
python train.py --epochs 10 --freeze_backbone
```

### Training is very slow
```bash
# Option 1: Use frozen backbone
python train.py --freeze_backbone

# Option 2: Reduce image size
python train.py --img_size 128

# Option 3: Smaller model for testing
python train.py --epochs 5 --batch_size 16
```

## 🎓 Learning Resources

### Understanding the Project
1. Read `PROJECT_OVERVIEW.md` for architecture details
2. Read `QUICK_REFERENCE.md` for command cheatsheet
3. Run `python example_usage.py` to see code examples

### Deep Learning Concepts
- **Transfer Learning**: Using pre-trained models
- **ResNet50**: Residual neural network architecture
- **Data Augmentation**: Improving generalization
- **Fine-tuning**: Adapting models to new tasks

### PyTorch Resources
- Official PyTorch tutorials: pytorch.org/tutorials
- Transfer learning guide: pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- torchvision models: pytorch.org/vision/stable/models.html

## 🆘 Getting Help

1. **Check console output** - Error messages are usually clear
2. **Validate your data** - `python data_prep.py --mode validate --data_dir data/train`
3. **Review documentation** - README.md, PROJECT_OVERVIEW.md
4. **Start fresh** - Delete checkpoints, retrain with known-good data
5. **Test with examples** - `python example_usage.py`

## ✨ Success Checklist

Before moving to production, ensure:

- [ ] Dataset validated with `data_prep.py`
- [ ] Model trained for sufficient epochs (30+)
- [ ] Validation accuracy > 90%
- [ ] Tested on diverse images
- [ ] Evaluated with `evaluate.py`
- [ ] Tested real-time performance
- [ ] Documented your training configuration
- [ ] Saved best model checkpoint

## 🎉 Congratulations!

You've successfully set up a state-of-the-art helmet detection system using deep learning!

### What's Impressive About This:
- Uses cutting-edge ResNet50 architecture
- Transfer learning from ImageNet (14M images)
- Real-time inference capability
- Production-ready code structure
- Comprehensive evaluation metrics

### Share Your Results:
- Test accuracy percentages
- Real-world deployment scenarios
- Interesting edge cases
- Performance optimizations

## 📬 Feedback

This project is designed to be:
- ✅ Easy to understand
- ✅ Simple to use
- ✅ Ready for production
- ✅ Fully customizable

Enjoy building with AI! 🚀

---

**Need a quick command? Check QUICK_REFERENCE.md**  
**Want to understand the architecture? Read PROJECT_OVERVIEW.md**  
**Looking for detailed docs? See README.md**

