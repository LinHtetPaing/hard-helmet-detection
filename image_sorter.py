"""
Interactive Image Sorter
Helps manually separate images into with_helmet and without_helmet folders
"""
import os
import shutil
import cv2
import argparse
from pathlib import Path


class ImageSorter:
    """Interactive image sorting tool"""
    
    def __init__(self, source_dir, with_helmet_dir, without_helmet_dir):
        self.source_dir = Path(source_dir)
        self.with_helmet_dir = Path(with_helmet_dir)
        self.without_helmet_dir = Path(without_helmet_dir)
        
        # Create directories if they don't exist
        self.with_helmet_dir.mkdir(parents=True, exist_ok=True)
        self.without_helmet_dir.mkdir(parents=True, exist_ok=True)
        
        # Get list of images
        self.images = sorted(list(self.source_dir.glob('*.png')) + 
                           list(self.source_dir.glob('*.jpg')) +
                           list(self.source_dir.glob('*.jpeg')))
        
        self.current_index = 0
        self.stats = {
            'with_helmet': 0,
            'without_helmet': 0,
            'skipped': 0,
            'total_processed': 0
        }
    
    def display_instructions(self):
        """Display instructions to user"""
        print("\n" + "="*70)
        print("IMAGE SORTER - Interactive Classification Tool")
        print("="*70)
        print("\nInstructions:")
        print("  Press 'h' or 'w' - Image has helmet (WITH helmet)")
        print("  Press 'n' - Image has NO helmet (WITHOUT helmet)")
        print("  Press 's' - Skip this image")
        print("  Press 'q' - Quit and save progress")
        print("  Press 'u' - Undo last action")
        print("  Press 'r' - Show remaining count")
        print("\n" + "="*70)
        print(f"Total images to sort: {len(self.images)}")
        print("="*70 + "\n")
    
    def move_image(self, image_path, destination):
        """Move image to destination folder"""
        dest_path = destination / image_path.name
        shutil.move(str(image_path), str(dest_path))
        return dest_path
    
    def show_image(self, image_path):
        """Display image with OpenCV"""
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Error: Could not load image {image_path}")
            return False
        
        # Resize if too large
        height, width = img.shape[:2]
        max_height = 800
        if height > max_height:
            scale = max_height / height
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
        
        # Add text overlay with instructions
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], 60), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
        
        text = f"Image {self.current_index + 1}/{len(self.images)} | H/W=Helmet | N=No Helmet | S=Skip | Q=Quit"
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        filename_text = f"File: {image_path.name}"
        cv2.putText(img, filename_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow('Image Sorter - Classify Image', img)
        return True
    
    def run(self):
        """Run the interactive sorting session"""
        self.display_instructions()
        
        if len(self.images) == 0:
            print("No images found in source directory!")
            return
        
        history = []  # For undo functionality
        
        while self.current_index < len(self.images):
            image_path = self.images[self.current_index]
            
            # Show image
            if not self.show_image(image_path):
                self.current_index += 1
                continue
            
            # Wait for key press
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                # Quit
                print("\nQuitting...")
                break
            
            elif key == ord('h') or key == ord('H') or key == ord('w') or key == ord('W'):
                # With helmet
                if image_path.parent == self.with_helmet_dir:
                    print(f"[{self.current_index + 1}] Already in with_helmet folder, skipping...")
                else:
                    new_path = self.move_image(image_path, self.with_helmet_dir)
                    history.append(('with_helmet', image_path, new_path))
                    self.stats['with_helmet'] += 1
                    print(f"[{self.current_index + 1}] ✓ Moved to WITH helmet folder")
                
                self.stats['total_processed'] += 1
                self.current_index += 1
            
            elif key == ord('n') or key == ord('N'):
                # Without helmet
                if image_path.parent == self.without_helmet_dir:
                    print(f"[{self.current_index + 1}] Already in without_helmet folder, skipping...")
                else:
                    new_path = self.move_image(image_path, self.without_helmet_dir)
                    history.append(('without_helmet', image_path, new_path))
                    self.stats['without_helmet'] += 1
                    print(f"[{self.current_index + 1}] ✓ Moved to WITHOUT helmet folder")
                
                self.stats['total_processed'] += 1
                self.current_index += 1
            
            elif key == ord('s') or key == ord('S'):
                # Skip
                print(f"[{self.current_index + 1}] Skipped")
                self.stats['skipped'] += 1
                self.current_index += 1
            
            elif key == ord('u') or key == ord('U'):
                # Undo last action
                if history:
                    last_action, original_path, current_path = history.pop()
                    shutil.move(str(current_path), str(original_path))
                    
                    if last_action == 'with_helmet':
                        self.stats['with_helmet'] -= 1
                    else:
                        self.stats['without_helmet'] -= 1
                    
                    self.stats['total_processed'] -= 1
                    self.current_index = max(0, self.current_index - 1)
                    print(f"[{self.current_index + 1}] ↶ Undone - moved back to {original_path.parent.name}")
                else:
                    print("Nothing to undo!")
            
            elif key == ord('r') or key == ord('R'):
                # Show remaining count
                remaining = len(self.images) - self.current_index
                print(f"\nRemaining images: {remaining}")
                print(f"Progress: {self.current_index}/{len(self.images)} ({100*self.current_index/len(self.images):.1f}%)")
                continue  # Don't advance
            
            else:
                # Invalid key
                print(f"Invalid key. Press H=helmet, N=no helmet, S=skip, Q=quit")
                continue  # Don't advance
        
        cv2.destroyAllWindows()
        self.show_summary()
    
    def show_summary(self):
        """Show final statistics"""
        print("\n" + "="*70)
        print("SORTING SUMMARY")
        print("="*70)
        print(f"Total processed: {self.stats['total_processed']}")
        print(f"  With helmet: {self.stats['with_helmet']}")
        print(f"  Without helmet: {self.stats['without_helmet']}")
        print(f"  Skipped: {self.stats['skipped']}")
        print(f"Remaining: {len(self.images) - self.current_index}")
        print("="*70)
        
        # Count files in directories
        with_count = len(list(self.with_helmet_dir.glob('*.png'))) + \
                    len(list(self.with_helmet_dir.glob('*.jpg'))) + \
                    len(list(self.with_helmet_dir.glob('*.jpeg')))
        without_count = len(list(self.without_helmet_dir.glob('*.png'))) + \
                       len(list(self.without_helmet_dir.glob('*.jpg'))) + \
                       len(list(self.without_helmet_dir.glob('*.jpeg')))
        
        print(f"\nCurrent folder counts:")
        print(f"  {self.with_helmet_dir}: {with_count} images")
        print(f"  {self.without_helmet_dir}: {without_count} images")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Interactive image sorter for helmet detection")
    
    parser.add_argument('--source', type=str, default='data/train/with_helmet',
                       help='Source directory containing images to sort')
    parser.add_argument('--with_helmet', type=str, default='data/train/with_helmet',
                       help='Destination for images WITH helmets')
    parser.add_argument('--without_helmet', type=str, default='data/train/without_helmet',
                       help='Destination for images WITHOUT helmets')
    
    args = parser.parse_args()
    
    sorter = ImageSorter(args.source, args.with_helmet, args.without_helmet)
    
    try:
        sorter.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")
        sorter.show_summary()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

