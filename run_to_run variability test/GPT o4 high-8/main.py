# main.py

import os
import random
import torch
from torch.utils.data import DataLoader, Subset
from torch import optim, nn
from sklearn.model_selection import train_test_split
from torchinfo import summary
import time

from dataset import GrayscaleSegmentationDataset
from model import UNet
from train import (
    train_epoch, validate_epoch, test_model,
    save_losses, save_batch_dices, plot_losses
)

if __name__ == "__main__":
    # --- CONFIG ---
    images_dir = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\images"
    masks_dir = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\Masks"    # or same as images_dir
    save_path = "D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT o4 high-8"
    os.makedirs(save_path, exist_ok=True)

    # hyperparams
    lr = 1e-3
    batch_size = 8
    num_epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- DATASET & SPLIT ---
    dataset = GrayscaleSegmentationDataset(images_dir, masks_dir, image_suffix='_m')
    indices = list(range(len(dataset)))
    train_idx, temp_idx = train_test_split(indices, train_size=0.8, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, train_size=0.5, random_state=42)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)
    test_ds  = Subset(dataset, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4)
    print(f"Loaders -> Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

    # --- MODEL & OPT ---
    model = UNet().to(device)
    summary(model, input_size=(batch_size,1,256,256))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # --- TRAINING LOOP ---
    train_losses, val_losses = [], []
    val_batch_dices = []
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        tl = train_epoch(model, train_loader, criterion, optimizer, device)
        vl, vd = validate_epoch(model, val_loader, criterion, device)
        train_losses.append(tl)
        val_losses.append(vl)
        val_batch_dices.append(vd)
        print(f"  Train Loss: {tl:.4f} | Val Loss: {vl:.4f} | Val Dice (mean per-batch): {sum(vd)/len(vd):.4f}")

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time//60:.0f}m {total_time%60:.0f}s")

    # --- SAVE MODEL & STATS ---
    torch.save(model, os.path.join(save_path, "unet_model_full.pth"))
    torch.save(model.state_dict(), os.path.join(save_path, "unet_model_state.pth"))

    save_losses(train_losses, save_path, "train_losses.xlsx", "Train Loss")
    save_losses(val_losses,   save_path, "val_losses.xlsx",   "Val Loss")
    save_batch_dices(val_batch_dices, save_path, "validation_dice_scores.xlsx")
    plot_losses(train_losses, val_losses, save_path)

    # --- TEST ---
    test_dices, samples = test_model(model, test_loader, device)
    print(f"Mean test Dice over batches: {sum(test_dices)/len(test_dices):.4f}")
    save_batch_dices([test_dices], save_path, "test_dice_scores.xlsx")

    # visualize 5 random samples
    import matplotlib.pyplot as plt
    rand_samples = random.sample(samples, 5)
    fig, axes = plt.subplots(5, 3, figsize=(12, 20))
    cols = ['Input', 'GroundTruth', 'Prediction']
    for i, (img, mask, pred, fname) in enumerate(rand_samples):
        for j, mat in enumerate([img, mask, pred]):
            ax = axes[i][j]
            ax.imshow(mat.squeeze(), cmap='gray')
            if i == 0:
                ax.set_title(cols[j])
            ax.axis('off')
            if j == 0:
                ax.set_ylabel(fname, rotation=0, labelpad=50, va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "test_predictions.png"))
    plt.close()
