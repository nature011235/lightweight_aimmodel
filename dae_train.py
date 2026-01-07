import glob
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from src.model.component.dae import MobileNetDAE

# config
BATCH_SIZE = 1024
LR = 5e-3
FINAL_LR = 3e-5
EPOCHS = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256
NOISE_FACTOR = 0.3
DATASET_ROOT = "dataset/ILSVRC2012_img_train"
VAL_SPLIT = 0.1
LR_COS_PERIOD = 5
RESUME_PATH = None


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
CHECKPOINT_DIR = os.path.join("checkpoints", "dae", TIMESTAMP)
LOG_DIR = os.path.join("logs", "dae", TIMESTAMP)

train_transform = transforms.Compose(
    [
        transforms.Resize(int(IMG_SIZE * 1.14)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize(int(IMG_SIZE * 1.14)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
    ]
)


class UnlabeledImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPEG"]

        for ext in extensions:
            self.image_paths.extend(
                glob.glob(os.path.join(root_dir, "**", ext), recursive=True)
            )

        print(f"have {len(self.image_paths)} file")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        while True:
            img_path = self.image_paths[idx]
            try:
                image = Image.open(img_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                return image, 0
            except Exception as e:
                print(f"{e}")
                idx = (idx + 1) % len(self.image_paths)


def add_noise(img_tensor, noise_factor=0.3):
    noisy_img = img_tensor + noise_factor * torch.randn_like(img_tensor)
    noisy_img = torch.clip(noisy_img, 0.0, 1.0)
    return noisy_img


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"checkpoint: {CHECKPOINT_DIR}")
    print(f"Log: {LOG_DIR}")

    writer = SummaryWriter(LOG_DIR)
    train_dataset_full = UnlabeledImageDataset(DATASET_ROOT, transform=train_transform)
    val_dataset_full = UnlabeledImageDataset(DATASET_ROOT, transform=val_transform)

    dataset_size = len(train_dataset_full)
    val_size = 10000
    train_size = dataset_size - val_size
    generator = torch.Generator().manual_seed(42)

    train_subset, val_subset = random_split(
        range(dataset_size), [train_size, val_size], generator=generator
    )

    from torch.utils.data import Subset

    train_dataset = Subset(train_dataset_full, train_subset.indices)
    val_dataset = Subset(val_dataset_full, val_subset.indices)

    print(f"Split: Train={len(train_dataset)}, Val={len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    model = MobileNetDAE().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # lr scheduler setting
    steps_per_epoch = len(train_loader)
    cycle_epochs = int(EPOCHS / LR_COS_PERIOD)
    total_steps_per_cycle = cycle_epochs * steps_per_epoch  # total step

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=total_steps_per_cycle,
        T_mult=1,
        eta_min=FINAL_LR,
    )

    criterion = nn.MSELoss()
    normalizer = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    start_epoch = 0
    best_val_loss = float("inf")
    scaler = GradScaler()
    # loading model
    if RESUME_PATH and os.path.exists(RESUME_PATH):
        print(f"loading checkpoint: {RESUME_PATH}")
        checkpoint = torch.load(RESUME_PATH)

        model.load_state_dict(checkpoint["model_state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        print(f"start from epoch: {start_epoch}")
    else:
        print("Start from Scratch")

    step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0

        current_lr = optimizer.param_groups[0]["lr"]
        loop = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Start LR={current_lr:.2e}]"
        )

        for batch_idx, (images, _) in enumerate(loop):
            images = images.to(DEVICE)
            noisy_images = add_noise(images, noise_factor=NOISE_FACTOR).to(DEVICE)
            with autocast(device_type="cuda"):
                model_input = normalizer(noisy_images)
                outputs = model(model_input)
                loss = criterion(outputs, images)

            optimizer.zero_grad()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # lr update
            scheduler.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

            writer.add_scalar("Loss/train_step", loss.item(), step)  # steop loss
            current_step_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("LR", current_step_lr, step)

            step += 1

            # show picture output
            if batch_idx == 0:
                with torch.no_grad():
                    n_show = min(images.shape[0], 8)
                    clean = images[:n_show]
                    noisy = noisy_images[:n_show]
                    recon = outputs[:n_show]
                    grid_clean = make_grid(clean, nrow=n_show)
                    grid_noisy = make_grid(noisy, nrow=n_show)
                    grid_recon = make_grid(recon, nrow=n_show)
                    writer.add_image("Vis/Clean", grid_clean, epoch)
                    writer.add_image("Vis/Noisy", grid_noisy, epoch)
                    writer.add_image("Vis/Reconstructed", grid_recon, epoch)

        # validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(DEVICE)
                noisy_images = add_noise(images, noise_factor=NOISE_FACTOR).to(DEVICE)
                model_input = normalizer(noisy_images)
                outputs = model(model_input)
                loss = criterion(outputs, images)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        last_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | End LR: {last_lr:.2e}"
        )
        # write to tensorboard
        writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)
        writer.add_scalar("Loss/val_epoch", avg_val_loss, epoch)

        # saved model
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
        }
        last_path = os.path.join(CHECKPOINT_DIR, "last.pth")
        torch.save(checkpoint, last_path)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save(checkpoint, best_path)
            print(f"New Best Model Saved! Val Loss: {best_val_loss:.5f}")

    writer.close()
    print(f"Done, Best Val Loss: {best_val_loss:.5f}")
    print(f"checkpoint: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
