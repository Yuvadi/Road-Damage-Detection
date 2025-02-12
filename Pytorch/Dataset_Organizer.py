import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

def parse_xml_for_objects(xml_path):
    """
    Extract object names from XML file.
    Returns None if no valid objects found.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # Get all unique object names from the XML
        object_names = set()
        objects = root.findall('.//object/name')
        
        # If no objects found or all object names are empty
        if not objects or all(obj.text is None or obj.text.strip() == '' for obj in objects):
            return None
            
        for obj in objects:
            if obj.text and obj.text.strip():
                object_names.add(obj.text.strip())
        
        return object_names if object_names else None
    except ET.ParseError:
        return None

def organize_dataset(dataset_path):
    """
    Organize dataset into folders based on object classes from XML files.
    Creates a 'dirty_data' folder for files with missing or invalid object information.
    
    Args:
        dataset_path: Path to the dataset directory containing 'images' and 'labels' folders
    """
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / 'images'
    labels_dir = dataset_path / 'labels'
    
    # Verify directories exist
    if not images_dir.exists() or not labels_dir.exists():
        raise ValueError("Images or Labels directory not found!")
    
    # Dictionary to keep track of which files belong to which classes
    class_files = {}
    dirty_files = []
    
    # Create dirty_data directory structure
    dirty_data_dir = dataset_path / 'dirty_data'
    dirty_images_dir = dirty_data_dir / 'images'
    dirty_labels_dir = dirty_data_dir / 'labels'
    
    # Get total number of files for progress bar
    xml_files = list(labels_dir.glob('*.xml'))
    total_files = len(xml_files)
    dirty_count = 0
    
    print("\nAnalyzing XML files...")
    # Process each XML file with progress bar
    for xml_file in tqdm(xml_files, desc="Processing files", unit="file"):
        # Get corresponding image filename
        image_name = xml_file.stem + '.jpg'
        image_path = images_dir / image_name
        
        if not image_path.exists():
            print(f"\nWarning: Image {image_name} not found for {xml_file.name}")
            continue
        
        # Get object classes from XML
        object_classes = parse_xml_for_objects(xml_file)
        
        if object_classes is None or len(object_classes) == 0:
            # This is dirty data
            dirty_files.append((image_path, xml_file))
            dirty_count += 1
        else:
            # Add files to each class they belong to
            for obj_class in object_classes:
                if obj_class not in class_files:
                    class_files[obj_class] = []
                class_files[obj_class].append((image_path, xml_file))
    
    # Create class directories and copy files
    print("\nCopying files to class directories...")
    with tqdm(total=sum(len(files) for files in class_files.values()), desc="Copying class files", unit="file") as pbar:
        for class_name, files in class_files.items():
            # Create class directory structure
            class_dir = dataset_path / class_name
            class_images_dir = class_dir / 'images'
            class_labels_dir = class_dir / 'labels'
            
            # Create directories if they don't exist
            class_images_dir.mkdir(parents=True, exist_ok=True)
            class_labels_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy files to new locations
            for image_path, xml_path in files:
                try:
                    shutil.copy2(image_path, class_images_dir / image_path.name)
                    shutil.copy2(xml_path, class_labels_dir / xml_path.name)
                    pbar.update(1)
                except shutil.Error as e:
                    print(f"\nError copying files for class {class_name}: {e}")
    
    # Handle dirty data
    if dirty_files:
        print("\nCopying dirty data files...")
        # Create dirty data directories
        dirty_images_dir.mkdir(parents=True, exist_ok=True)
        dirty_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy dirty files with progress bar
        with tqdm(total=len(dirty_files), desc="Copying dirty files", unit="file") as pbar:
            for image_path, xml_path in dirty_files:
                try:
                    shutil.copy2(image_path, dirty_images_dir / image_path.name)
                    shutil.copy2(xml_path, dirty_labels_dir / xml_path.name)
                    pbar.update(1)
                except shutil.Error as e:
                    print(f"\nError copying dirty files: {e}")
    
    # Print summary
    print("\nDataset Organization Summary:")
    print(f"Total files processed: {total_files}")
    print(f"Clean classes found: {len(class_files)}")
    print(f"Files marked as dirty data: {dirty_count}")
    for class_name, files in class_files.items():
        print(f"Class '{class_name}': {len(files)} files")

def main():
    # Get dataset path from user
    dataset_path = input("Enter the path to your dataset directory: ")
    
    try:
        organize_dataset(dataset_path)
        print("\nDataset organization completed successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()