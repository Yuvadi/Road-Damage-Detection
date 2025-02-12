import os
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

class XMLToYOLOConverter:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.labels_path = self.dataset_path / 'labels'
        self.output_path = self.dataset_path / 'txt_labels'
        self.class_mapping = {}
        self.next_class_id = 0

    def get_class_id(self, class_name):
        if class_name not in self.class_mapping:
            self.class_mapping[class_name] = self.next_class_id
            self.next_class_id += 1
        return self.class_mapping[class_name]

    def convert_coordinates(self, size, box):
        
        dw = 1.0 / size[0]
        dh = 1.0 / size[1]
        
        # Extract coordinates
        xmin = float(box.find('xmin').text)
        xmax = float(box.find('xmax').text)
        ymin = float(box.find('ymin').text)
        ymax = float(box.find('ymax').text)
        
        # Calculate YOLO coordinates
        x_center = ((xmin + xmax) / 2.0) * dw
        y_center = ((ymin + ymax) / 2.0) * dh
        w = (xmax - xmin) * dw
        h = (ymax - ymin) * dh
        
        return x_center, y_center, w, h

    def convert_xml_to_yolo(self, xml_path):

        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image size
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        # Process each object
        yolo_lines = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            class_id = self.get_class_id(class_name)
            
            # Get bounding box
            bbox = obj.find('bndbox')
            x_center, y_center, w, h = self.convert_coordinates((width, height), bbox)
            
            # Format YOLO line: <class_id> <x_center> <y_center> <width> <height>
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
        
        return yolo_lines

    def process_dataset(self):
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Get list of XML files
        xml_files = list(self.labels_path.glob('*.xml'))
        if not xml_files:
            raise ValueError("No XML files found in the labels directory!")
        
        print("\nConverting XML files to YOLO format...")
        
        # Process each XML file with progress bar
        for xml_file in tqdm(xml_files, desc="Converting files", unit="file"):
            try:
                # Convert XML to YOLO format
                yolo_lines = self.convert_xml_to_yolo(xml_file)
                
                # Save YOLO format to txt file
                txt_filename = xml_file.stem + '.txt'
                txt_path = self.output_path / txt_filename
                
                with open(txt_path, 'w') as f:
                    f.write('\n'.join(yolo_lines))
                    
            except Exception as e:
                print(f"\nError processing {xml_file.name}: {e}")
                continue
        
        # Save class mapping
        self.save_class_mapping()
        
        return len(xml_files)

    def save_class_mapping(self):

        mapping_path = self.output_path / 'classes.txt'
        
        # Sort classes by their ID
        sorted_classes = sorted(self.class_mapping.items(), key=lambda x: x[1])
        
        with open(mapping_path, 'w') as f:
            for class_name, class_id in sorted_classes:
                f.write(f"{class_name}: {class_id}\n")

def main():
    # Get dataset path from user
    dataset_path = input("Enter the path to your dataset directory: ")
    
    try:
        converter = XMLToYOLOConverter(dataset_path)
        num_files = converter.process_dataset()
        
        print("\nConversion Summary:")
        print("="*50)
        print(f"Total files processed: {num_files}")
        print(f"Number of classes found: {len(converter.class_mapping)}")
        print("\nClass Mapping:")
        for class_name, class_id in sorted(converter.class_mapping.items(), key=lambda x: x[1]):
            print(f"  {class_name}: {class_id}")
        print(f"\nOutput files saved to: {converter.output_path}")
        print("\nConversion completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()