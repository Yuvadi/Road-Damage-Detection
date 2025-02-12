import torch
import torchvision
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image
import numpy as np

# Load your model
model = torch.load('E:/Aditya_Thesis_Project/signatrix_efficientdet_coco.pth')
model.eval()


# Initialize COCO ground truth api
coco = COCO('test/_annotations.coco.json')

# Load the test dataset
test_image_ids = coco.getImgIds()
test_images = coco.loadImgs(test_image_ids)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



def evaluate_model(model, coco, test_images, transform):
    results = []
    for img_info in test_images:
        img_path = 'test/' + img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image)
        
        # Process the outputs and convert to COCO format
        # Assuming the model outputs bounding boxes and class scores
        for i in range(len(outputs[0]['boxes'])):
            box = outputs[0]['boxes'][i].cpu().numpy()
            score = outputs[0]['scores'][i].cpu().numpy()
            label = outputs[0]['labels'][i].cpu().numpy()
            result = {
                'image_id': img_info['id'],
                'category_id': label,
                'bbox': box.tolist(),
                'score': score
            }
            results.append(result)
    
    return results

results = evaluate_model(model, coco, test_images, transform)

import json

with open('results.json', 'w') as f:
    json.dump(results, f)



coco_dt = coco.loadRes('results.json')
coco_eval = COCOeval(coco, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
