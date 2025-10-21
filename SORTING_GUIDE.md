# Image Sorting Guide

You have 4,999 images in `data/train/with_helmet/` that need to be properly separated into "with helmet" and "without helmet" categories.

## 🎯 Available Tools

I've created two tools to help you:

### 1. **image_sorter.py** - Interactive Visual Sorter (RECOMMENDED)
Shows each image one-by-one, you classify them with keyboard shortcuts.

### 2. **batch_move.py** - Batch Move Tool
For moving multiple images at once based on patterns or lists.

---

## 🖼️ Method 1: Interactive Visual Sorting (Best for Accuracy)

This tool shows you each image and lets you classify it with keyboard shortcuts.

### Start the Interactive Sorter:

```bash
cd helmet_detection
python image_sorter.py --source data/train/with_helmet
```

### Keyboard Controls:

- **H** or **W** → Image has helmet (keep in WITH helmet folder)
- **N** → Image has NO helmet (move to WITHOUT helmet folder)
- **S** → Skip this image (decide later)
- **U** → Undo last action
- **R** → Show remaining count
- **Q** → Quit and save progress

### Tips:
- Images are shown full-screen with instructions
- Your progress is saved - you can quit and resume anytime
- Use 'U' to undo if you make a mistake
- Take breaks - press 'Q' to quit, run again to continue

---

## 📦 Method 2: Batch Move (Fast but Manual)

### Option A: Move by File List

If you have a list of files to move, create a text file:

**Create `without_helmet_list.txt`:**
```
hard_hat_workers100.png
hard_hat_workers250.png
hard_hat_workers399.png
```

**Then move them:**
```bash
python batch_move.py list without_helmet_list.txt \
    --source data/train/with_helmet \
    --dest data/train/without_helmet
```

### Option B: Move by Pattern

If files follow a naming pattern:

```bash
# Dry run first (preview what would be moved)
python batch_move.py pattern "*without*" \
    --source data/train/with_helmet \
    --dest data/train/without_helmet \
    --dry-run

# Actually move them
python batch_move.py pattern "*without*" \
    --source data/train/with_helmet \
    --dest data/train/without_helmet
```

### Option C: Move by Index Range

Move files by their position (after sorting):

```bash
# Move files at index 0-99
python batch_move.py range 0 99 \
    --source data/train/with_helmet \
    --dest data/train/without_helmet
```

---

## 💡 Recommended Workflow

### For Manual Review (Most Accurate):

1. **Start the interactive sorter:**
   ```bash
   python image_sorter.py --source data/train/with_helmet
   ```

2. **Review images:**
   - Press **N** for workers WITHOUT helmets
   - Press **H** for workers WITH helmets
   - Press **S** to skip if unsure

3. **Take breaks:**
   - Press **Q** to quit anytime
   - Your progress is saved
   - Run the command again to continue

4. **Monitor progress:**
   - Press **R** to see remaining count
   - The window shows current image number

### For Quick Batch Processing:

1. **Manually identify a few images to move**

2. **Create a list file:**
   ```bash
   # Create without_helmet_files.txt with filenames
   nano without_helmet_files.txt
   ```

3. **Move them in batch:**
   ```bash
   python batch_move.py list without_helmet_files.txt
   ```

---

## 📊 Check Your Progress

### Count images in each folder:

```bash
# Count with_helmet images
ls data/train/with_helmet/*.png | wc -l

# Count without_helmet images
ls data/train/without_helmet/*.png | wc -l
```

### Validate your dataset:

```bash
python data_prep.py --mode validate --data_dir data/train
```

This will show:
- Number of images per class
- Class balance
- Warnings about imbalances

---

## 🎯 After Sorting

Once you've properly separated your images:

### 1. Validate the Dataset
```bash
python data_prep.py --mode validate --data_dir data/train
```

### 2. Create Validation Set
You also need to create validation data. You can either:

**Option A: Split your current data**
```bash
# First, move all images to a raw folder
mkdir data/raw
mv data/train/with_helmet data/raw/
mv data/train/without_helmet data/raw/

# Split into train/val/test (80/10/10)
python data_prep.py --mode split --data_dir data/raw
```

**Option B: Sort validation data separately**
```bash
# Create validation directories
mkdir -p data/val/with_helmet
mkdir -p data/val/without_helmet

# Add validation images, then sort them
python image_sorter.py --source data/val/with_helmet
```

### 3. Train Your Model
```bash
python train.py --epochs 20 --freeze_backbone --batch_size 32
```

---

## ⚡ Quick Start Example

**The fastest way to get started:**

```bash
# Navigate to project
cd helmet_detection

# Start interactive sorting
python image_sorter.py --source data/train/with_helmet

# When you see an image:
# - Press 'N' if worker has NO helmet
# - Press 'H' if worker HAS helmet
# - Press 'S' if you're unsure
# - Press 'Q' when you want to stop

# Check your progress
python data_prep.py --mode validate --data_dir data/train

# Continue sorting or start training!
```

---

## 🛠️ Troubleshooting

### "Image window doesn't show"
Make sure OpenCV is installed:
```bash
pip install opencv-python
```

### "Can't find images"
Check your path:
```bash
ls data/train/with_helmet/*.png | head
```

### "Window is too small/large"
The sorter automatically resizes large images to fit your screen.

### "Made a mistake"
Press **U** to undo the last action in the interactive sorter.

---

## 📈 Sorting Tips

1. **Look for visible helmets** - Usually bright colors (yellow, orange, white)
2. **Check heads of workers** - Is there a hard hat on their head?
3. **When in doubt, skip** - Press 'S' and review later
4. **Take breaks** - Sorting 5000 images takes time
5. **Work in sessions** - Do 100-200 at a time, quit with 'Q', resume later

---

## ✅ Quality Check

After sorting, ensure:
- [ ] Both folders have images
- [ ] Images in `with_helmet` actually show helmets
- [ ] Images in `without_helmet` show workers without helmets
- [ ] Class balance is reasonable (not 99% vs 1%)
- [ ] Validation shows no warnings

**Run validation:**
```bash
python data_prep.py --mode validate --data_dir data/train
```

---

## 🎉 Ready to Train!

Once your data is properly sorted:

```bash
# Validate everything looks good
python data_prep.py --mode validate --data_dir data/train

# Train your model
python train.py --epochs 30 --batch_size 32

# Test it
python detect.py --checkpoint checkpoints/best_model.pth \
                 --mode image --image test_image.jpg --display
```

Good luck sorting! 🚀

