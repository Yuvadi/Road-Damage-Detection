import os
import cv2
from tqdm import tqdm
from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("runs/obb/train8/weights/last.pt")

# Path to the folder containing images
image_folder = "Balanced-Dataset/train/images"
output_video = "predictions_video.mp4"

# Get a list of all image files in the folder
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

# Sort images by name
images.sort()

# Initialize video writer
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))

# Process each image and write to video with progress bar
for image in tqdm(images, desc="Processing images"):
    img_path = os.path.join(image_folder, image)
    results = model(img_path)
    result = results[0]  # Access the first result in the list
    result.save("temp.jpg")
    frame = cv2.imread("temp.jpg")
    video.write(frame)

# Release the video writer
video.release()
