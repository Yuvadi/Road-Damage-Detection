import os
import cv2
from tqdm import tqdm

# Paths
base_path = './Balanced-Dataset'

# Directories to process
subsets = ['train', 'val', 'test']

# Function to convert images to grayscale
def convert_to_grayscale(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(image_path, gray_image)

# Process images
for subset in subsets:
    image_dir = os.path.join(base_path, subset, 'images')
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    
    for image_file in tqdm(image_files, desc=f"Converting {subset} images to grayscale"):
        image_path = os.path.join(image_dir, image_file)
        convert_to_grayscale(image_path)

print("All images have been converted to grayscale!")
