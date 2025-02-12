import os
from pathlib import Path
from tqdm import tqdm
import shutil

class YOLOLabelFilter:
    def __init__(self, dataset_path, allowed_classes=[0, 1, 2, 3]):
        self.dataset_path = Path(dataset_path)
        self.labels_path = self.dataset_path / 'labels'
        self.images_path = self.dataset_path / 'images'
        self.output_path = self.dataset_path / 'filtered_dataset'
        self.output_labels = self.output_path / 'labels'
        self.output_images = self.output_path / 'images'
        self.allowed_classes = set(allowed_classes)
        
        # Statistics
        self.stats = {
            'processed_files': 0,
            'filtered_files': 0,
            'removed_files': 0,
            'total_objects': 0,
            'kept_objects': 0,
            'removed_objects': 0
        }

    def setup_output_dirs(self):
        """Create output directories."""
        self.output_labels.mkdir(parents=True, exist_ok=True)
        self.output_images.mkdir(parents=True, exist_ok=True)

    def filter_label_file(self, label_path):
        """
        Filter a single label file to keep only allowed classes.
        Returns tuple (filtered_lines, had_allowed_classes)
        """
        filtered_lines = []
        had_allowed_classes = False
        total_objects = 0
        kept_objects = 0
        
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    total_objects += 1
                    line = line.strip()
                    if line:
                        class_id = int(line.split()[0])
                        if class_id in self.allowed_classes:
                            filtered_lines.append(line)
                            kept_objects += 1
                            had_allowed_classes = True
        except Exception as e:
            print(f"\nError reading file {label_path.name}: {e}")
            return [], False, 0, 0
            
        return filtered_lines, had_allowed_classes, total_objects, kept_objects

    def process_dataset(self):
        """Process all label files in the dataset."""
        # Verify directories exist
        if not self.labels_path.exists():
            raise ValueError("Labels directory not found!")
        if not self.images_path.exists():
            raise ValueError("Images directory not found!")
            
        # Create output directories
        self.setup_output_dirs()
        
        # Get list of label files
        label_files = list(self.labels_path.glob('*.txt'))
        if not label_files:
            raise ValueError("No label files found in the labels directory!")
        
        print("\nFiltering YOLO label files...")
        print(f"Keeping only classes: {sorted(self.allowed_classes)}")
        
        # Process each label file with progress bar
        for label_file in tqdm(label_files, desc="Processing files", unit="file"):
            self.stats['processed_files'] += 1
            
            # Filter the label file
            filtered_lines, had_allowed_classes, total_objects, kept_objects = self.filter_label_file(label_file)
            
            # Update statistics
            self.stats['total_objects'] += total_objects
            self.stats['kept_objects'] += kept_objects
            self.stats['removed_objects'] += (total_objects - kept_objects)
            
            # If file had allowed classes, save it and copy corresponding image
            if had_allowed_classes:
                self.stats['filtered_files'] += 1
                
                # Save filtered label file
                output_label_path = self.output_labels / label_file.name
                with open(output_label_path, 'w') as f:
                    f.write('\n'.join(filtered_lines))
                
                # Copy corresponding image if it exists
                image_name = label_file.stem + '.jpg'  # Assuming JPG format
                image_path = self.images_path / image_name
                if image_path.exists():
                    shutil.copy2(image_path, self.output_images / image_name)
                else:
                    print(f"\nWarning: Image not found for {label_file.name}")
            else:
                self.stats['removed_files'] += 1

    def print_summary(self):
        """Print summary of the filtering process."""
        print("\nFiltering Summary:")
        print("="*50)
        print(f"Total files processed: {self.stats['processed_files']}")
        print(f"Files kept: {self.stats['filtered_files']}")
        print(f"Files removed: {self.stats['removed_files']}")
        print(f"\nTotal objects in original dataset: {self.stats['total_objects']}")
        print(f"Objects kept: {self.stats['kept_objects']} ({(self.stats['kept_objects']/self.stats['total_objects']*100):.2f}%)")
        print(f"Objects removed: {self.stats['removed_objects']} ({(self.stats['removed_objects']/self.stats['total_objects']*100):.2f}%)")
        print(f"\nFiltered dataset saved to: {self.output_path}")

def main():
    # Get dataset path from user
    dataset_path = input("Enter the path to your dataset directory: ")
    
    try:
        filter_tool = YOLOLabelFilter(dataset_path)
        filter_tool.process_dataset()
        filter_tool.print_summary()
        print("\nFiltering completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()