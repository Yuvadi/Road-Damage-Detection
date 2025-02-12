import os
from tqdm import tqdm

def convert_to_obb(yolo_label):
    # Assuming the YOLO label format is: class_index x_center y_center width height
    class_index, x_center, y_center, width, height = map(float, yolo_label.split())
    
    # Calculate the coordinates of the four corners of the bounding box
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center - height / 2
    x3 = x_center + width / 2
    y3 = y_center + height / 2
    x4 = x_center - width / 2
    y4 = y_center + height / 2
    
    return f"{int(class_index)} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}"

def convert_labels_to_obb(dataset_folder):
    labels_folder = os.path.join(dataset_folder, 'labels')
    obb_labels_folder = os.path.join(dataset_folder, 'obb_labels')
    
    # Create the obb_labels folder if it doesn't exist
    os.makedirs(obb_labels_folder, exist_ok=True)
    
    label_files = [f for f in os.listdir(labels_folder) if f.endswith('.txt')]
    
    for label_file in tqdm(label_files, desc="Converting labels"):
        with open(os.path.join(labels_folder, label_file), 'r') as f:
            yolo_labels = f.readlines()
        
        obb_labels = [convert_to_obb(label.strip()) for label in yolo_labels]
        
        with open(os.path.join(obb_labels_folder, label_file), 'w') as f:
            f.write('\n'.join(obb_labels))
    
    print(f"Converted labels saved in {obb_labels_folder}")

# Example usage
dataset_folder = 'Balanced-Dataset/val'
convert_labels_to_obb(dataset_folder)
