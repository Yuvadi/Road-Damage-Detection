import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import numpy as np

# Paths
data_path = './Balanced-Dataset'
train_image_path = os.path.join(data_path, 'train/images')
val_image_path = os.path.join(data_path, 'val/images')
test_image_path = os.path.join(data_path, 'test/images')
train_label_path = os.path.join(data_path, 'train/labels')
val_label_path = os.path.join(data_path, 'val/labels')
test_label_path = os.path.join(data_path, 'test/labels')

# Hyperparameters
batch_size = 16
learning_rate = 0.001
num_epochs = 25
image_size = 512
num_classes = 4  # Adjust this to match your number of classes

# Dataset class for YOLO labels
class YoloDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = glob.glob(f"{self.image_dir}/*.jpg")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        label_path = image_path.replace(self.image_dir, self.label_dir).replace('.jpg', '.txt')

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        with open(label_path, 'r') as file:
            labels = file.readlines()

        # Assuming the first class ID is representative for the image
        class_ids = [int(label.split()[0]) for label in labels]
        class_id = class_ids[0] if class_ids else -1  # Handle empty labels

        return image, class_id

# Data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to create DataLoader
def create_dataloader(image_path, label_path, batch_size):
    dataset = YoloDataset(image_path, label_path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Creating DataLoaders for train, validation, and test sets
train_loader = create_dataloader(train_image_path, train_label_path, batch_size)
val_loader = create_dataloader(val_image_path, val_label_path, batch_size)
test_loader = create_dataloader(test_image_path, test_label_path, batch_size)

# Model
weights = EfficientNet_B2_Weights.IMAGENET1K_V1
model = efficientnet_b2(weights=weights)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to('cpu')

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, targets in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
        images = images.to('cpu')
        targets = torch.tensor(targets).long().to('cpu')  # Convert targets to tensor
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validation"):
            images = images.to('cpu')
            targets = torch.tensor(targets).long().to('cpu')  # Convert targets to tensor
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * images.size(0)
    
    val_loss /= len(val_loader.dataset)
    print(f"Validation Loss: {val_loss:.4f}")

print("Training complete!")
