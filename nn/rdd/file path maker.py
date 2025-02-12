import os

def create_train_txt(folder_path, output_file='E:/Aditya_Thesis_V2/nn/rdd/rdd_valid.txt'):
    # Get the list of all files in the folder
    files = os.listdir(folder_path)
    
    # Filter out only the image files (assuming jpg and png formats)
    image_files = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]
    
    # Create the output file and write the paths to the images
    with open(output_file, 'w') as f:
        for image in image_files:
            f.write(os.path.join(folder_path, image) + '\n')
    
    print(f'{output_file} has been created with {len(image_files)} image paths.')

# Example usage
folder_path = 'E:/Aditya_Thesis_V2/nn/rdd/valid'
create_train_txt(folder_path)
