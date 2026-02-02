# main.py

import os
import time
import random
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from torchinfo import summary

from dataset import SegmentationDataset
from model import UNet
from train import (
    train_epoch, validate_epoch, test_model,
    save_losses_to_excel, save_dice_excel, plot_losses
)

if __name__ == "__main__":
    # Paths
    images_dir = 'D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\images'
    masks_dir  = 'D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\Masks'      # or same as images_dir
    save_path  = 'D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT o4 high-4'
    os.makedirs(save_path, exist_ok=True)

    # Hyperparameters
    lr = 1e-3
    batch_size = 8
    num_epochs = 25
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset + splits
    full_ds = SegmentationDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        mask_suffix='_m',       # set to your actual mask suffix
        resize=(256, 256)
    )
    idx = list(range(len(full_ds)))
    train_idx, temp_idx = train_test_split(idx, test_size=0.2, random_state=42, shuffle=True)
    val_idx, test_idx  = train_test_split(temp_idx, test_size=0.5, random_state=42, shuffle=True)

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(full_ds, val_idx)
    test_ds  = Subset(full_ds, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Dataloaders → Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

    # Model, optimizer, loss
    model     = UNet(in_channels=1, out_channels=1).to(device)
    summary(model, input_size=(batch_size,1,256,256))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Training loop
    train_losses = []
    val_losses   = []
    val_dice_all = []

    start_time = time.time()
    for epoch in range(1, num_epochs+1):
        print(f"\n=== Epoch {epoch}/{num_epochs} ===")
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_dice = validate_epoch(model, val_loader, criterion, device)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        val_dice_all.append(vl_dice)

        print(f"Epoch {epoch}: Train Loss={tr_loss:.4f}, Val Loss={vl_loss:.4f}, Val Dice Mean={vl_dice.mean():.4f}")

    print(f"\nTotal training time: {(time.time() - start_time)/60:.2f} minutes")

    # Save model
    torch.save(model,            f'{save_path}/unet_full.pth')
    torch.save(model.state_dict(), f'{save_path}/unet_state.pth')

    # Save losses & dice scores
    epochs = list(range(1, num_epochs+1))
    save_losses_to_excel(train_losses, epochs, save_path, 'train_losses.xlsx')
    save_losses_to_excel(val_losses,   epochs, save_path, 'val_losses.xlsx')

    val_dice_matrix = np.stack(val_dice_all, axis=0)
    save_dice_excel(val_dice_matrix, save_path, 'validation_dice_scores.xlsx')

    plot_losses(train_losses, val_losses, save_path)

    # Testing
    test_dice = test_model(model, test_loader, device)
    save_dice_excel(test_dice, save_path, 'test_dice_scores.xlsx')

    # Visualize 5 random test predictions
    os.makedirs(f'{save_path}/visuals', exist_ok=True)
    sample_idxs = random.sample(test_idx, min(5, len(test_idx)))

    fig, axes = plt.subplots(len(sample_idxs), 3, figsize=(9, 3*len(sample_idxs)))
    from train import dice_coef

    for i, idx in enumerate(sample_idxs):
        img, mask, name = full_ds[idx]
        inp = img.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = torch.sigmoid(model(inp)).cpu()[0,0]

        axes[i,0].imshow(img[0], cmap='gray')
        axes[i,0].set_title(f'Input\n{name}'); axes[i,0].axis('off')

        axes[i,1].imshow(mask[0], cmap='gray')
        axes[i,1].set_title('Ground Truth'); axes[i,1].axis('off')

        axes[i,2].imshow((pred>0.5).float(), cmap='gray')
        axes[i,2].set_title('Prediction'); axes[i,2].axis('off')

    plt.tight_layout()
    plt.savefig(f'{save_path}/visuals/test_predictions.png')
    plt.close()
