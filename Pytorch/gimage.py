import os
from PIL import Image
from tqdm import tqdm

def convert_to_grayscale(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return

    # Create a folder to save grayscale images
    grayscale_folder = os.path.join(folder_path, "grayscale")
    os.makedirs(grayscale_folder, exist_ok=True)

    # Get a list of image files in the folder
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    # Iterate through all image files with a loading bar
    for filename in tqdm(image_files, desc="Converting to grayscale"):
        file_path = os.path.join(folder_path, filename)
        
        # Open the image
        with Image.open(file_path) as img:
            # Convert the image to grayscale
            grayscale_img = img.convert("L")
            # Save the grayscale image
            grayscale_img.save(os.path.join(grayscale_folder, filename))

# Example usage
folder_path = "path/to/your/folder"
convert_to_grayscale(folder_path)
