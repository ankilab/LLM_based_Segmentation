# unet_segmentation/main.py

import os
import torch
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from torchinfo import summary
from dataset import SegmentationDataset
from model import UNet
from train import (train_epoch, validate_epoch, test_epoch,
                   plot_losses, visualize_predictions)
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import time

def main():
    # --- Configuration ---
    image_dir = 'D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\images'
    mask_dir  = 'D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\Masks'      # or None if same dir + suffix
    save_path = 'D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT o4 high-6'   # make sure exists
    os.makedirs(save_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 2 #50
    batch_size = 8
    lr = 1e-3

    # --- Dataset & Split ---
    full_ds = SegmentationDataset(image_dir, mask_dir, suffix='_seg.png')
    n = len(full_ds)
    idxs = list(range(n))
    train_idx, test_idx = train_test_split(idxs, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(full_ds, val_idx)
    test_ds  = Subset(full_ds, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4)

    # --- Model, Loss, Optimizer ---
    model = UNet(n_channels=1, n_classes=1).to(device)
    print(summary(model, input_size=(batch_size,1,256,256)))
    print(f"Total params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- Training Loop ---
    train_losses, val_losses = [], []
    start_time = time.time()
    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch}/{epochs}")
        tl = train_epoch(model, train_loader, criterion, optimizer, device)
        vl = validate_epoch(model, val_loader, criterion, device, save_path, epoch)
        train_losses.append(tl)
        val_losses.append(vl)
        # save per-epoch losses to Excel
        pd.DataFrame([list(range(1,epoch+1)), train_losses]).to_excel(
            os.path.join(save_path, 'train_losses.xlsx'), index=False, header=False
        )
        pd.DataFrame([list(range(1,epoch+1)), val_losses]).to_excel(
            os.path.join(save_path, 'val_losses.xlsx'), index=False, header=False
        )
        print(f"  Train Loss: {tl:.4f}  Val Loss: {vl:.4f}")

    total_time = time.time() - start_time
    print(f"Total training time: {total_time/60:.2f} minutes")

    # --- Save model ---
    torch.save(model, os.path.join(save_path, 'unet_model_full.pth'))
    torch.save(model.state_dict(), os.path.join(save_path, 'unet_model_state.pth'))

    # --- Plots & Evaluation ---
    plot_losses(train_losses, val_losses, save_path)
    validate_epoch(model, val_loader, criterion, device, save_path, epochs)  # regenerate dice xls
    test_epoch(model, test_loader, device, save_path)
    visualize_predictions(model, test_loader, device, save_path, num_samples=5)

if __name__ == '__main__':
    main()
