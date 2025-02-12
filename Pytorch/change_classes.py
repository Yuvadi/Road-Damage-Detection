import os
from pathlib import Path
from tqdm import tqdm

class YOLOClassReplacer:
    def __init__(self, labels_dir, old_class, new_class):
        self.labels_dir = Path(labels_dir)
        self.old_class = str(old_class)  # Convert to string for comparison
        self.new_class = str(new_class)
        self.output_dir = self.labels_dir.parent / 'modified_labels'
        
        # Statistics
        self.stats = {
            'processed_files': 0,
            'modified_files': 0,
            'replaced_count': 0,
            'error_files': 0
        }

    def setup_output_dir(self):
        """Create output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def replace_class_in_file(self, file_path):
        """Replace class ID in a single file."""
        modified = False
        replacements = 0
        modified_lines = []
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if parts and parts[0] == self.old_class:
                            parts[0] = self.new_class
                            modified = True
                            replacements += 1
                        modified_lines.append(' '.join(parts))
                        
            return modified, replacements, modified_lines
                        
        except Exception as e:
            print(f"\nError processing file {file_path.name}: {e}")
            self.stats['error_files'] += 1
            return False, 0, []

    def process_directory(self):
        """Process all label files in the directory."""
        # Verify directory exists
        if not self.labels_dir.exists():
            raise ValueError(f"Labels directory not found: {self.labels_dir}")
            
        # Create output directory
        self.setup_output_dir()
        
        # Get list of label files
        label_files = list(self.labels_dir.glob('*.txt'))
        if not label_files:
            raise ValueError("No label files found in the directory!")
        
        print(f"\nReplacing class {self.old_class} with class {self.new_class}")
        print(f"Processing files in: {self.labels_dir}")
        
        # Process each file with progress bar
        for label_file in tqdm(label_files, desc="Processing files", unit="file"):
            self.stats['processed_files'] += 1
            
            # Replace class IDs in file
            modified, replacements, modified_lines = self.replace_class_in_file(label_file)
            
            # If file was modified, save the new version
            if modified:
                self.stats['modified_files'] += 1
                self.stats['replaced_count'] += replacements
                
                # Save modified file
                output_path = self.output_dir / label_file.name
                try:
                    with open(output_path, 'w') as f:
                        f.write('\n'.join(modified_lines))
                except Exception as e:
                    print(f"\nError saving file {output_path.name}: {e}")
                    self.stats['error_files'] += 1

    def print_summary(self):
        """Print summary of the replacement process."""
        print("\nReplacement Summary:")
        print("="*50)
        print(f"Total files processed: {self.stats['processed_files']}")
        print(f"Files modified: {self.stats['modified_files']}")
        print(f"Total replacements made: {self.stats['replaced_count']}")
        if self.stats['error_files'] > 0:
            print(f"Files with errors: {self.stats['error_files']}")
        print(f"\nModified files saved to: {self.output_dir}")

def main():
    # Get user input
    labels_dir = input("Enter the path to your labels directory: ")
    old_class = input("Enter the class ID to replace: ")
    new_class = input("Enter the new class ID: ")
    
    try:
        # Validate input
        if not old_class.isdigit() or not new_class.isdigit():
            raise ValueError("Class IDs must be numbers!")
            
        replacer = YOLOClassReplacer(labels_dir, old_class, new_class)
        replacer.process_directory()
        replacer.print_summary()
        print("\nClass replacement completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()