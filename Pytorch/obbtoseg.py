import os
import yaml
from tqdm import tqdm

def obb_to_seg(obb_file, seg_file):
    with open(obb_file, 'r') as f:
        lines = f.readlines()

    with open(seg_file, 'w') as f:
        for line in lines:
            parts = line.strip().split()
            class_index = parts[0]
            coords = parts[1:]
            # Convert OBB to SEG format
            seg_coords = ' '.join(coords)
            f.write(f"{class_index} {seg_coords}\n")

def process_annotations(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    for file_name in tqdm(files, desc='Converting'):
        obb_file = os.path.join(input_dir, file_name)
        seg_file = os.path.join(output_dir, file_name)
        obb_to_seg(obb_file, seg_file)

def process_yaml(data_yaml):
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)

    # Directories for train, val, and test
    base_dir = os.path.dirname(data_yaml)
    train_dir = os.path.join(base_dir, 'train', 'labels')
    val_dir = os.path.join(base_dir, 'val', 'labels')
    test_dir = os.path.join(base_dir, 'test', 'labels')

    output_dirs = {
        'train': os.path.join(base_dir, 'seg_labels', 'train'),
        'val': os.path.join(base_dir, 'seg_labels', 'val'),
        'test': os.path.join(base_dir, 'seg_labels', 'test')
    }

    # Process annotations for train, val, and test
    process_annotations(train_dir, output_dirs['train'])
    process_annotations(val_dir, output_dirs['val'])
    process_annotations(test_dir, output_dirs['test'])

# Example usage
data_yaml = 'Balanced-Dataset/data.yaml' 
process_yaml(data_yaml)
