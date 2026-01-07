import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from src.dataset import HeadHunterDataset
from src.model.model import MobileNetAimBot

# config
DATASET_ROOT = "dataset/fps.v2i.yolov11"
BATCH_SIZE = 128
LEARNING_RATE = 2e-3
FINAL_LR = 2e-5
EPOCHS = 50
LR_COS_PERIOD = 5
IMG_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 8
ENCODER_PATH = "checkpoints/dae/2026-01-04_18-39-42/best_model.pth"


RESUME_PATH = None
# RESUME_PATH = "checkpoints/2025-12-27_02-30-00/last_checkpoint.pth"

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


CHECKPOINT_DIR = os.path.join("checkpoints", TIMESTAMP)
LOG_DIR = os.path.join("logs", TIMESTAMP)

#normalization 
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# transform
train_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"checkpoint: {CHECKPOINT_DIR}")
    print(f"log: {LOG_DIR}")

    writer = SummaryWriter(LOG_DIR)

    train_img_dir = os.path.join(DATASET_ROOT, "train", "images")
    train_lbl_dir = os.path.join(DATASET_ROOT, "train", "labels")
    val_img_dir = os.path.join(DATASET_ROOT, "valid", "images")
    val_lbl_dir = os.path.join(DATASET_ROOT, "valid", "labels")

    # dataset
    train_dataset = HeadHunterDataset(
        img_folder=train_img_dir,
        label_folder=train_lbl_dir,
        transform=train_transform,
    )

    valid_dataset = HeadHunterDataset(
        img_folder=val_img_dir, label_folder=val_lbl_dir, transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = MobileNetAimBot(defaultweight=True, en_weight=ENCODER_PATH).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # lr scheduler
    steps_per_epoch = len(train_loader)
    cycle_epochs = int(EPOCHS / LR_COS_PERIOD)
    total_steps_per_cycle = cycle_epochs * steps_per_epoch

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=total_steps_per_cycle, T_mult=1, eta_min=FINAL_LR
    )

    start_epoch = 0
    best_pixel_error = float("inf")
    global_step = 0

    # resume model
    if RESUME_PATH and os.path.exists(RESUME_PATH):
        print(f"loaded checkpoint: {RESUME_PATH}")
        checkpoint = torch.load(RESUME_PATH, map_location=DEVICE)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_pixel_error = checkpoint["best_pixel_error"]

        global_step = start_epoch * len(train_loader)

        print(f"start from {start_epoch} (Best Error: {best_pixel_error:.2f} px)...")
    else:
        print("start from none")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0.0

        current_lr = optimizer.param_groups[0]["lr"]
        loop = tqdm(
            train_loader,
            desc=f"Epoch [{epoch + 1}/{EPOCHS}] Train | LR: {current_lr:.2e}",
        )

        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # update lr
            scheduler.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

            # write tensorBoard batch
            writer.add_scalar("Loss/Train_Step", loss.item(), global_step)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step)
            global_step += 1

        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)

        # validate
        model.eval()
        val_loss = 0.0
        total_pixel_error = 0.0

        with torch.no_grad():
            for images, labels in tqdm(
                valid_loader, desc=f"Epoch [{epoch + 1}/{EPOCHS}] Valid"
            ):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)

                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # cal pixel error
                distance = torch.sqrt(torch.sum((outputs - labels) ** 2, dim=1))
                pixel_error = distance * (IMG_SIZE / 2)
                total_pixel_error += pixel_error.mean().item()

        avg_val_loss = val_loss / len(valid_loader)
        avg_pixel_error = total_pixel_error / len(valid_loader)

        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("Metric/Pixel_Error", avg_pixel_error, epoch)

        print(
            f" Epoch {epoch + 1} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | Avg Error: {avg_pixel_error:.2f} px"
        )

        # save model
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_pixel_error": best_pixel_error,
        }

        # last model
        last_ckpt_path = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth")
        torch.save(checkpoint, last_ckpt_path)

        # save best
        if avg_pixel_error < best_pixel_error:
            best_pixel_error = avg_pixel_error

            # for resume
            best_ckpt_path = os.path.join(CHECKPOINT_DIR, "best_checkpoint.pth")
            torch.save(checkpoint, best_ckpt_path)

            # goal model weight
            best_model_path = os.path.join(CHECKPOINT_DIR, "best_model_weights.pth")
            torch.save(model.state_dict(), best_model_path)

            print(f"New Best Model Saved! Error: {best_pixel_error:.2f} px")

    writer.close()
    print("train done")


if __name__ == "__main__":
    main()
