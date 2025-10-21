#!/bin/bash

# Helmet Detection - Quick Start Script
# This script helps you get started with the helmet detection project

echo "=================================="
echo "Helmet Detection - Quick Start"
echo "=================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Install dependencies
echo "Installing dependencies..."
echo "------------------------"
pip3 install -r requirements.txt
echo ""

# Check if data directory exists
if [ ! -d "data/train" ]; then
    echo "⚠️  Warning: Training data directory not found"
    echo ""
    echo "Please prepare your dataset in the following structure:"
    echo "  data/"
    echo "  ├── train/"
    echo "  │   ├── with_helmet/"
    echo "  │   └── without_helmet/"
    echo "  └── val/"
    echo "      ├── with_helmet/"
    echo "      └── without_helmet/"
    echo ""
    echo "You can use the data_prep.py script to validate and split your data:"
    echo "  python3 data_prep.py --mode validate --data_dir data/train"
    echo ""
else
    echo "✓ Training data directory found"
    echo ""
    
    # Validate dataset
    echo "Validating training dataset..."
    python3 data_prep.py --mode validate --data_dir data/train
    echo ""
    
    if [ -d "data/val" ]; then
        echo "Validating validation dataset..."
        python3 data_prep.py --mode validate --data_dir data/val
        echo ""
    fi
fi

# Test model
echo "Testing model architecture..."
echo "------------------------"
python3 model.py
echo ""

echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Prepare your dataset (if not done):"
echo "   - Organize images into data/train and data/val folders"
echo "   - Use: python3 data_prep.py --mode validate --data_dir data/train"
echo ""
echo "2. Train the model:"
echo "   python3 train.py --epochs 30 --batch_size 32"
echo ""
echo "3. Test on an image:"
echo "   python3 detect.py --checkpoint checkpoints/best_model.pth \\"
echo "                     --mode image --image test.jpg --display"
echo ""
echo "4. Real-time detection:"
echo "   python3 detect.py --checkpoint checkpoints/best_model.pth --mode webcam"
echo ""
echo "For more information, see README.md"
echo ""

