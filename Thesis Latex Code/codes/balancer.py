import os
import shutil
import random
from collections import defaultdict
from tqdm import tqdm

# Paths
data_path = "E:\Aditya_Thesis_Project_Data_Backup\combined dataset - Copy"
image_path = os.path.join(data_path, 'images')
label_path = os.path.join(data_path, 'labels')
output_path = './Balanced-Dataset'
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Create directories
os.makedirs(os.path.join(output_path, 'train/images'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'train/labels'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'val/images'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'val/labels'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'test/images'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'test/labels'), exist_ok=True)

# Read all labels and categorize by class
label_files = [f for f in os.listdir(label_path) if f.endswith('.txt')]
class_distribution = defaultdict(list)

for label_file in tqdm(label_files, desc="Reading labels"):
    with open(os.path.join(label_path, label_file), 'r') as file:
        lines = file.readlines()
        if not lines:  # Blank images
            class_distribution['blank'].append(label_file)
        for line in lines:
            class_id = line.split()[0]
            class_distribution[class_id].append(label_file)

# Get the minimum number of objects across all classes
min_objects = min(len(files) for files in class_distribution.values() if files != 'blank')

# Balance the dataset by oversampling/undersampling
balanced_files = []
for class_id, files in tqdm(class_distribution.items(), desc="Balancing dataset"):
    if class_id == 'blank':
        continue
    if len(files) > min_objects:
        balanced_files.extend(random.sample(files, min_objects))
    else:
        balanced_files.extend(files + random.choices(files, k=min_objects - len(files)))

# Include blank images
balanced_files.extend(class_distribution['blank'])

# Split the dataset
random.shuffle(balanced_files)
num_files = len(balanced_files)
train_files = balanced_files[:int(train_ratio * num_files)]
val_files = balanced_files[int(train_ratio * num_files):int((train_ratio + val_ratio) * num_files)]
test_files = balanced_files[int((train_ratio + val_ratio) * num_files):]

def copy_files(file_list, subset):
    for file in tqdm(file_list, desc=f"Copying files to {subset}"):
        image_file = file.replace('.txt', '.jpg')
        shutil.copy(os.path.join(image_path, image_file), os.path.join(output_path, f'{subset}/images', image_file))
        shutil.copy(os.path.join(label_path, file), os.path.join(output_path, f'{subset}/labels', file))

copy_files(train_files, 'train')
copy_files(val_files, 'val')
copy_files(test_files, 'test')

print("Dataset balanced and split successfully!")
