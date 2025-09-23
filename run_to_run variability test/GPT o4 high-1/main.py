# unet_segmentation/main.py

import os
import time
import torch
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchinfo import summary

from dataset import SegmentationDataset
from model import UNet
from train import (
    train_epoch, validate_epoch, test_model,
    save_losses, plot_losses, visualize_predictions
)

def paired_transforms(img, mask, size=(256,256)):
    # Resize, to tensor, normalize to [0,1]
    t_img = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    t_mask = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    return t_img(img), t_mask(mask)

if __name__ == "__main__":
    # --- Config ---
    IMG_DIR = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\images"
    MASK_DIR = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\Masks"
    SAVE_PATH = "D:\\qy44lyfe\\LLM segmentation\\Results\\Models Comparison\\Models run to run comparison\\GPT o4 high-1"
    os.makedirs(SAVE_PATH, exist_ok=True)

    BATCH_SIZE = 8
    LR = 1e-4
    EPOCHS = 20
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Dataset & Split ---
    full_ds = SegmentationDataset(IMG_DIR, MASK_DIR, transforms=paired_transforms)
    idxs = list(range(len(full_ds)))
    train_idxs, temp_idxs = train_test_split(idxs, test_size=0.2, random_state=42)
    val_idxs, test_idxs = train_test_split(temp_idxs, test_size=0.5, random_state=42)

    print(f"Train: {len(train_idxs)}, Val: {len(val_idxs)}, Test: {len(test_idxs)}")

    train_ds = Subset(full_ds, train_idxs)
    val_ds = Subset(full_ds, val_idxs)
    test_ds = Subset(full_ds, test_idxs)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Loaders -> Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

    # --- Model ---
    model = UNet().to(DEVICE)
    summary(model, input_size=(BATCH_SIZE,1,256,256))
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total learnable parameters: {total_params:,}")

    # --- Training Loop ---
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses, val_losses = [], []
    start_time = time.time()
    for epoch in range(1, EPOCHS+1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = validate_epoch(model, val_loader, criterion, DEVICE, SAVE_PATH, epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch}/{EPOCHS}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    duration = time.time() - start_time
    print(f"Total training time: {duration/60:.2f} minutes")

    # save model
    torch.save(model, os.path.join(SAVE_PATH, "unet_model.pth"))
    torch.save(model.state_dict(), os.path.join(SAVE_PATH, "unet_state_dict.pth"))

    # save losses
    save_losses(train_losses, os.path.join(SAVE_PATH, "train_losses.xlsx"))
    save_losses(val_losses, os.path.join(SAVE_PATH, "val_losses.xlsx"))
    plot_losses(train_losses, val_losses, SAVE_PATH)

    # --- Testing & Visualization ---
    samples = test_model(model, test_loader, DEVICE, SAVE_PATH)
    visualize_predictions(samples, SAVE_PATH)
