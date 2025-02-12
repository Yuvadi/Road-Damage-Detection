import os
import shutil

# Function to convert YOLO label format to EfficientNet format
def convert_yolo_to_eff(yolo_label, class_mapping):
    with open(yolo_label, 'r') as file:
        lines = file.readlines()
    eff_labels = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        class_name = class_mapping[class_id]
        # Create the EfficientNet label format here, e.g., "class_name x_center y_center width height"
        eff_label = f"{class_name} {x_center} {y_center} {width} {height}"
        eff_labels.append(eff_label)
    return eff_labels

# Load the class mappings from the YAML file (for simplicity, define it here)
class_mapping = {0: 'D10', 1: 'D00', 2: 'D20', 3: 'D40'}

# Paths
yolo_label_dir = './dataset/labels'  # Directory containing YOLO format labels
eff_label_dir = './dataset/Eff_labels'  # Directory to save EfficientNet format labels

# Create the Eff_labels directory if it doesn't exist
if not os.path.exists(eff_label_dir):
    os.makedirs(eff_label_dir)

# Iterate over all files in the YOLO label directory
for filename in os.listdir(yolo_label_dir):
    yolo_label_path = os.path.join(yolo_label_dir, filename)
    eff_label_path = os.path.join(eff_label_dir, filename)
    
    # Convert the label and save to the new directory
    eff_labels = convert_yolo_to_eff(yolo_label_path, class_mapping)
    with open(eff_label_path, 'w') as file:
        for eff_label in eff_labels:
            file.write(f"{eff_label}\n")

print("Conversion complete. EfficientNet labels saved in 'Eff_labels' directory.")
