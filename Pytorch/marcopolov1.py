import os
import cv2
import torch
from torchvision import transforms
from tqdm import tqdm

def log_image_names(folder_path, log_file_path):
    with open(log_file_path, 'a') as log_file:
        for entry in os.listdir(folder_path):
            if entry.endswith('.jpg'):
                log_file.write(entry + '\n')

def predict_images_torchscript(folder_path, log_file_path, predicted_folder, actual_predicted_folder):
    # Load TorchScript model
    model = torch.jit.load('last.torchscript')
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    for entry in os.listdir(folder_path):
        if entry.endswith('.jpg'):
            image_name = entry
            with open(log_file_path, 'r') as log_file:
                already_predicted = image_name in log_file.read()

            if not already_predicted:
                image = cv2.imread(os.path.join(folder_path, image_name))
                input_tensor = transform(image).unsqueeze(0)
                with torch.no_grad():
                    output = model(input_tensor)

                detected = True  # Replace with actual detection result based on output

                predicted_image_path = os.path.join(predicted_folder, image_name)
                cv2.imwrite(predicted_image_path, image)

                if detected:
                    actual_predicted_image_path = os.path.join(actual_predicted_folder, image_name)
                    cv2.imwrite(actual_predicted_image_path, image)

                with open(log_file_path, 'a') as log_file:
                    log_file.write(image_name + '\n')


folder_path = 'Balanced-Dataset/val/images'
log_file_path = 'testlog.txt'
predicted_folder = 'predicted'
actual_predicted_folder = 'actual_predicted'

os.makedirs(predicted_folder, exist_ok=True)
os.makedirs(actual_predicted_folder, exist_ok=True)

log_image_names(folder_path, log_file_path)
predict_images_torchscript(folder_path, log_file_path, predicted_folder, actual_predicted_folder)

