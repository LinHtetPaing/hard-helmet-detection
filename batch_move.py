"""
Batch Move Tool
For moving multiple images at once based on a list
"""
import os
import shutil
import argparse
from pathlib import Path


def move_images_from_list(list_file, source_dir, dest_dir):
    """
    Move images listed in a text file
    
    Args:
        list_file: Text file with one filename per line
        source_dir: Source directory
        dest_dir: Destination directory
    """
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Read list of files
    with open(list_file, 'r') as f:
        filenames = [line.strip() for line in f if line.strip()]
    
    print(f"Moving {len(filenames)} images from {source_dir} to {dest_dir}")
    print("=" * 70)
    
    moved = 0
    not_found = 0
    errors = 0
    
    for filename in filenames:
        source_path = source_dir / filename
        dest_path = dest_dir / filename
        
        if not source_path.exists():
            print(f"✗ Not found: {filename}")
            not_found += 1
            continue
        
        try:
            shutil.move(str(source_path), str(dest_path))
            moved += 1
            if moved % 100 == 0:
                print(f"  Moved {moved}/{len(filenames)} images...")
        except Exception as e:
            print(f"✗ Error moving {filename}: {e}")
            errors += 1
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total files in list: {len(filenames)}")
    print(f"Successfully moved: {moved}")
    print(f"Not found: {not_found}")
    print(f"Errors: {errors}")
    print("=" * 70)


def move_by_pattern(source_dir, dest_dir, pattern, dry_run=False):
    """
    Move images matching a pattern
    
    Args:
        source_dir: Source directory
        dest_dir: Destination directory
        pattern: Glob pattern (e.g., "*without*", "*no_helmet*")
        dry_run: If True, only show what would be moved
    """
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    
    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Find matching files
    files = list(source_dir.glob(pattern))
    
    print(f"Found {len(files)} files matching pattern '{pattern}' in {source_dir}")
    
    if dry_run:
        print("\n[DRY RUN] Would move the following files:")
        print("=" * 70)
        for i, f in enumerate(files[:20], 1):  # Show first 20
            print(f"{i}. {f.name}")
        if len(files) > 20:
            print(f"... and {len(files) - 20} more files")
        print("=" * 70)
        print(f"\nTo actually move these files, run without --dry-run flag")
        return
    
    # Move files
    print(f"Moving to {dest_dir}...")
    print("=" * 70)
    
    moved = 0
    for f in files:
        try:
            dest_path = dest_dir / f.name
            shutil.move(str(f), str(dest_path))
            moved += 1
            if moved % 100 == 0:
                print(f"  Moved {moved}/{len(files)} files...")
        except Exception as e:
            print(f"✗ Error moving {f.name}: {e}")
    
    print("\n" + "=" * 70)
    print(f"Successfully moved {moved}/{len(files)} files")
    print("=" * 70)


def move_by_index_range(source_dir, dest_dir, start_idx, end_idx):
    """
    Move images by index range (useful for numbered files)
    
    Args:
        source_dir: Source directory
        dest_dir: Destination directory
        start_idx: Start index
        end_idx: End index (inclusive)
    """
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images and sort
    images = sorted(list(source_dir.glob('*.png')) + 
                   list(source_dir.glob('*.jpg')) +
                   list(source_dir.glob('*.jpeg')))
    
    if not images:
        print("No images found!")
        return
    
    # Validate range
    if start_idx < 0 or end_idx >= len(images):
        print(f"Invalid range! Available: 0 to {len(images)-1}")
        return
    
    # Move files
    files_to_move = images[start_idx:end_idx+1]
    print(f"Moving images {start_idx} to {end_idx} ({len(files_to_move)} files)")
    print(f"From: {source_dir}")
    print(f"To: {dest_dir}")
    print("=" * 70)
    
    moved = 0
    for f in files_to_move:
        try:
            dest_path = dest_dir / f.name
            shutil.move(str(f), str(dest_path))
            moved += 1
        except Exception as e:
            print(f"✗ Error moving {f.name}: {e}")
    
    print("\n" + "=" * 70)
    print(f"Successfully moved {moved}/{len(files_to_move)} files")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Batch move images")
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # From list command
    list_parser = subparsers.add_parser('list', help='Move images from a text file list')
    list_parser.add_argument('list_file', help='Text file with filenames (one per line)')
    list_parser.add_argument('--source', default='data/train/with_helmet', help='Source directory')
    list_parser.add_argument('--dest', default='data/train/without_helmet', help='Destination directory')
    
    # Pattern command
    pattern_parser = subparsers.add_parser('pattern', help='Move images matching a pattern')
    pattern_parser.add_argument('pattern', help='Glob pattern (e.g., "*without*")')
    pattern_parser.add_argument('--source', default='data/train/with_helmet', help='Source directory')
    pattern_parser.add_argument('--dest', default='data/train/without_helmet', help='Destination directory')
    pattern_parser.add_argument('--dry-run', action='store_true', help='Show what would be moved without moving')
    
    # Range command
    range_parser = subparsers.add_parser('range', help='Move images by index range')
    range_parser.add_argument('start', type=int, help='Start index (0-based)')
    range_parser.add_argument('end', type=int, help='End index (inclusive)')
    range_parser.add_argument('--source', default='data/train/with_helmet', help='Source directory')
    range_parser.add_argument('--dest', default='data/train/without_helmet', help='Destination directory')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        move_images_from_list(args.list_file, args.source, args.dest)
    elif args.command == 'pattern':
        move_by_pattern(args.source, args.dest, args.pattern, args.dry_run)
    elif args.command == 'range':
        move_by_index_range(args.source, args.dest, args.start, args.end)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

