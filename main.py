import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from ultralytics import YOLO

from fins_training import val_transform

yolo_model = YOLO("yolo/best.pt")
dataset_dir = 'dataset/random'

import os
import rawpy
import imageio

# Convert CR2 to JPG
for image_file in os.listdir(dataset_dir):
    if image_file.endswith('.CR2'):
        print("Converting", image_file)
        raw_path = os.path.join(dataset_dir, image_file)
        jpg_path = os.path.join(dataset_dir, image_file.rsplit('.', 1)[0] + '.JPG')

        with rawpy.imread(raw_path) as raw:
            rgb = raw.postprocess()

        imageio.imsave(jpg_path, rgb)
        os.remove(raw_path)


device = torch.device("mps" if torch.mps.is_available() else "cpu")
checkpoint = torch.load("fins/fins.pt", map_location=device)
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, len(checkpoint['class_names']))
model = model.to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

for image_file in np.random.choice(os.listdir(dataset_dir), size=5):
    if not image_file.endswith('.JPG'):
        print("Skipping", image_file)
        continue

    image_path = os.path.join(dataset_dir, image_file)
    results = yolo_model.predict(image_path)

    image = Image.open(image_path).convert('RGB')
    plt.imshow(image)

    for result in results:
        print(type(result))
        bboxes = result.boxes.xyxy.numpy()
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox

            cropped_img = image.crop((x_min, y_min, x_max, y_max))
            input_img = val_transform(cropped_img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_img)
                _, pred = torch.max(output, 1)
                predicted_class = checkpoint['class_names'][pred.item()]

            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                 edgecolor='red', linewidth=2, fill=False)
            plt.gca().add_patch(rect)
            plt.text(x_min, y_min, predicted_class, fontsize=10, color='red')

    plt.show()
