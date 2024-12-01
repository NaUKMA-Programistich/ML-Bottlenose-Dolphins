import os

import click
import imageio
import numpy as np
import rawpy
import torch
from PIL import Image
from matplotlib import pyplot as plt
from ultralytics import YOLO

from fins_training import val_transform


CLASS_NAMES = ['1056', 'random']


def prepare_folder(data_folder):
    """Convert CR2 files in the data_folder to JPG format."""
    # Convert CR2 to JPG
    for image_file in os.listdir(data_folder):
        if image_file.endswith('.CR2'):
            print("Converting", image_file)
            raw_path = os.path.join(data_folder, image_file)
            jpg_path = os.path.join(
                data_folder,
                image_file.rsplit('.', 1)[0] + '.JPG'
            )

            with rawpy.imread(raw_path) as raw:
                rgb = raw.postprocess()

            imageio.imsave(jpg_path, rgb)
            os.remove(raw_path)

    return data_folder


@click.command()
@click.option('--data_folder', type=str, default='dataset/1056')
@click.option('--device', type=str, default='mps')
@click.option('--size', type=int, default=10)
def main(data_folder, device, size):
    print(f"Data folder: {data_folder}")
    print(f"Device: {device}")

    prepare_folder(data_folder)

    device = torch.device(device)
    model = torch.load('fins/fins.pt', map_location=device)
    model = model.to(device)
    model.eval()

    yolo_model = YOLO("yolo/best.pt")

    image_files = np.random.choice(os.listdir(data_folder), size=size)
    for image_file in image_files:
        image_path = os.path.join(data_folder, image_file)
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
                    predicted_class = CLASS_NAMES[pred.item()]

                rect = plt.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    edgecolor='red',
                    linewidth=2,
                    fill=False
                )
                plt.gca().add_patch(rect)
                plt.text(
                    x_min,
                    y_min,
                    predicted_class,
                    fontsize=10,
                    color='red'
                )

        plt.show()


if __name__ == '__main__':
    main()
