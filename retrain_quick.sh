#!/bin/bash
# Quick Retrain Script - Use this to retrain with your new data

echo "========================================================================"
echo "RETRAINING HELMET DETECTION MODEL"
echo "========================================================================"
echo ""

# Check dataset
echo "Validating current dataset..."
python data_prep.py --mode validate --data_dir data/train
echo ""

# Backup old checkpoints
if [ -d "checkpoints" ]; then
    echo "Backing up old checkpoints..."
    mkdir -p checkpoints_old
    cp checkpoints/*.pth checkpoints_old/ 2>/dev/null
    echo "✓ Old checkpoints backed up to checkpoints_old/"
    echo ""
fi

# Start training
echo "Starting training with new dataset..."
echo "========================================================================"
echo ""

# Train with good settings for your dataset size (200 images)
python train.py \
    --epochs 40 \
    --batch_size 16 \
    --lr 0.001 \
    --freeze_backbone \
    --num_workers 2

echo ""
echo "========================================================================"
echo "TRAINING COMPLETE!"
echo "========================================================================"
echo ""
echo "Your new model is saved in: checkpoints/best_model.pth"
echo ""
echo "Test it now:"
echo "  python detect.py --checkpoint checkpoints/best_model.pth \\"
echo "                   --mode image --image YOUR_IMAGE.png --display"
echo ""
echo "Or evaluate it:"
echo "  python evaluate.py --checkpoint checkpoints/best_model.pth \\"
echo "                     --data_dir data/val --cpu"
echo ""

