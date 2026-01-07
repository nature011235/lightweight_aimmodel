import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


class HeadHunterDataset(Dataset):
    def __init__(
        self,
        img_folder,
        label_folder,
        transform=None,
        add_img_folder=None,
        add_label_folder=None,
    ):
        self.transform = transform

        # save tuple (img_root, label_root, filename, target_head_id)
        self.data_infos = []

        # id 1 is head
        if os.path.exists(img_folder):
            files = sorted(
                [f for f in os.listdir(img_folder) if f.endswith((".jpg", ".png"))]
            )
            for f in files:
                self.data_infos.append((img_folder, label_folder, f, 1))

        # id 2 is head
        if add_img_folder is not None and add_label_folder is not None:
            if os.path.exists(add_img_folder):
                files_add = sorted(
                    [
                        f
                        for f in os.listdir(add_img_folder)
                        if f.endswith((".jpg", ".png"))
                    ]
                )
                for f in files_add:
                    self.data_infos.append((add_img_folder, add_label_folder, f, 0))

        print(f"total {len(self.data_infos)} pictures")

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        img_root, label_root, img_name, target_head_id = self.data_infos[idx]

        img_path = os.path.join(img_root, img_name)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(label_root, label_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"can't load picture: {img_path}, Error: {e}")
            image = Image.new("RGB", (256, 256))

        # label processing
        best_head = None
        min_dist_to_center = float("inf")

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    cls_id = int(parts[0])

                    if cls_id == target_head_id:
                        x_c = float(parts[1])
                        y_c = float(parts[2])

                        dist = (x_c - 0.5) ** 2 + (y_c - 0.5) ** 2

                        if dist < min_dist_to_center:
                            min_dist_to_center = dist
                            best_head = (x_c, y_c)

        # 3. 0~1 to -1~1
        if best_head is not None:
            norm_x = best_head[0] * 2 - 1
            norm_y = best_head[1] * 2 - 1
            label = torch.tensor([norm_x, norm_y], dtype=torch.float32)
        else:
            label = torch.tensor([0.0, 0.0], dtype=torch.float32)

        # transform
        if self.transform:
            image = self.transform(image)
        else:
            image = F.to_tensor(image)

        return image, label
