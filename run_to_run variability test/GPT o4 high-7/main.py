import os
import time
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, Resize, ToTensor
from torch.optim import Adam
from torchinfo import summary

from dataset import GrayscaleSegmentationDataset
from model import UNet
from train import train_epoch, validate_epoch, test_epoch, plot_losses

def paired_transform(image, mask, size=(256,256)):
    """
    Resize, ToTensor, and normalize to [0,1].
    image, mask: PIL.Images
    """
    transform = Compose([
        Resize(size),
        ToTensor(),
    ])
    return transform(image), transform(mask)

if __name__ == "__main__":
    # --- configs ---
    image_folder = 'D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\images'
    mask_folder  = 'D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\Masks'         # or 'path/to/masks'
    mask_suffix  = '_m.jpg'
    save_path    = 'D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT o4 high-7'
    batch_size   = 8
    lr           = 1e-4
    num_epochs   = 2 #50
    device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_path, exist_ok=True)

    # --- dataset & splits ---
    full_ds = GrayscaleSegmentationDataset(
        image_folder, mask_folder, mask_suffix,
        transform=paired_transform
    )
    indices = list(range(len(full_ds)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx  = train_test_split(test_idx,  test_size=0.5, random_state=42)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(full_ds, val_idx)
    test_ds  = Subset(full_ds, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4)

    # --- model, optimizer, summary ---
    model = UNet(n_channels=1, n_classes=1).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    summary(model, input_size=(batch_size,1,256,256))
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total learnable parameters: {total_params}")

    # --- training loop ---
    train_losses, val_losses = [], []
    t0 = time.time()
    for epoch in range(1, num_epochs+1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        tr_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate_epoch(model, val_loader, device, save_path, epoch)
        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        # save epoch losses to excel
        pd.DataFrame([list(range(1, epoch+1)), train_losses])\
          .to_excel(os.path.join(save_path, 'train_losses.xlsx'), index=False, header=False)
        pd.DataFrame([list(range(1, epoch+1)), val_losses])\
          .to_excel(os.path.join(save_path, 'val_losses.xlsx'),   index=False, header=False)

    total_time = time.time() - t0
    print(f"\nTotal training time: {total_time/60:.2f} minutes")

    # --- save model ---
    torch.save(model, os.path.join(save_path, 'unet_model_full.pth'))
    torch.save(model.state_dict(), os.path.join(save_path, 'unet_model_state.pth'))

    # --- plot losses ---
    plot_losses(train_losses, val_losses, save_path)

    # --- testing & visualization ---
    test_epoch(model, test_loader, device, save_path)

    # visualize 5 random samples
    import random
    import matplotlib.pyplot as plt
    from torch import sigmoid

    samples = random.sample(range(len(test_ds)), 5)
    fig, axes = plt.subplots(5, 3, figsize=(9, 15))
    fig.suptitle('Input | Ground Truth | Prediction', fontsize=16)
    for i, idx in enumerate(samples):
        img, mask, name = test_ds[idx]
        inp = img.unsqueeze(0).to(device)
        with torch.no_grad():
            out = sigmoid(model(inp)).cpu().squeeze(0)
        pred = (out > 0.5).float()

        axes[i,0].imshow(img.squeeze(), cmap='gray')
        axes[i,1].imshow(mask.squeeze(), cmap='gray')
        axes[i,2].imshow(pred.squeeze(),cmap='gray')
        for j, title in enumerate([name, name, name]):
            axes[i,j].set_title(title)
        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(save_path, 'test_visualization.png'))
    plt.close()
