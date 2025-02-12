import os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def count_yolo_labels(labels_dir):
    """
    Count occurrences of each class ID in YOLO format txt files.
    
    Args:
        labels_dir: Path to directory containing YOLO txt label files
    
    Returns:
        Dictionary with class IDs as keys and their counts as values
    """
    labels_path = Path(labels_dir)
    
    # Verify directory exists
    if not labels_path.exists():
        raise ValueError("Labels directory not found!")
    
    # Dictionary to store counts
    class_counts = defaultdict(int)
    total_objects = 0
    
    # Get list of all txt files
    txt_files = list(labels_path.glob('*.txt'))
    
    if not txt_files:
        raise ValueError("No txt files found in the directory!")
    
    print("\nAnalyzing YOLO label files...")
    
    # First pass: Count total lines for accurate progress bar
    total_lines = 0
    for txt_file in tqdm(txt_files, desc="Counting lines", unit="file"):
        with open(txt_file, 'r') as f:
            total_lines += sum(1 for line in f)
    
    # Second pass: Process each file and count class IDs
    with tqdm(total=total_lines, desc="Processing labels", unit="object") as pbar:
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:  # Skip empty lines
                            try:
                                # YOLO format: class_id x_center y_center width height
                                class_id = int(line.split()[0])
                                class_counts[class_id] += 1
                                total_objects += 1
                                pbar.update(1)
                            except (ValueError, IndexError) as e:
                                print(f"\nWarning: Invalid line in {txt_file.name}: {line}")
                                continue
            except Exception as e:
                print(f"\nError processing file {txt_file.name}: {e}")
                continue
    
    return class_counts, total_objects, len(txt_files)

def display_statistics(class_counts, total_objects, total_files):
    """Display the counting statistics in a formatted way."""
    print("\n" + "="*50)
    print("YOLO Labels Statistics")
    print("="*50)
    print(f"\nTotal files processed: {total_files}")
    print(f"Total objects detected: {total_objects}")
    print("\nClass ID Distribution:")
    print("-"*30)
    
    # Sort by class ID
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = (count / total_objects) * 100
        print(f"Class {class_id}: {count:,} objects ({percentage:.2f}%)")

def main():
    # Get labels directory path from user
    labels_dir = input("Enter the path to your YOLO labels directory: ")
    
    try:
        class_counts, total_objects, total_files = count_yolo_labels(labels_dir)
        display_statistics(class_counts, total_objects, total_files)
        print("\nAnalysis completed successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()