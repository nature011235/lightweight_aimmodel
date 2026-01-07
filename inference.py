import os
import random

import cv2
import numpy as np
import torch

from model.model import MobileNetAimBot

# config
IMG_DIR = "dataset/fps.v2i.yolov11/test/images"
IMG_PATH = "dataset/reallife_test/3.png"
CHECKPOINT = "checkpoints/2025-12-24_03-24-52/best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def visualize(pic_dir=None):
    model = MobileNetAimBot().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT))
    model.eval()

    image_path = IMG_PATH
    # preprocessing
    if pic_dir is not None:
        image_files = [
            f
            for f in os.listdir(pic_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        random_image_file = random.choice(image_files)
        image_path = os.path.join(pic_dir, random_image_file)
    print(image_path)
    raw_img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (256, 256))

    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_tensor = img_resized / 255.0
    img_tensor = (img_tensor - mean) / std
    img_tensor = (
        torch.from_numpy(img_tensor).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)
    )

    with torch.no_grad():
        coords = model(img_tensor)  # output -1 ~ 1

    x_norm = coords[0][0].item()
    y_norm = coords[0][1].item()

    # denormalize to original input
    pred_x = int((x_norm + 1) / 2 * 256)
    pred_y = int((y_norm + 1) / 2 * 256)

    # to bgr
    show_img = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

    cv2.circle(show_img, (pred_x, pred_y), 3, (0, 255, 0), -1)
    cv2.line(show_img, (pred_x - 10, pred_y), (pred_x + 10, pred_y), (0, 255, 0), 1)
    cv2.line(show_img, (pred_x, pred_y - 10), (pred_x, pred_y + 10), (0, 255, 0), 1)

    print(f"Model Predicted: ({pred_x}, {pred_y})")
    cv2.imshow("Prediction", show_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize()
