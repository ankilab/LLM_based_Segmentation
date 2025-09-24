# main.py

import os
import random
import time
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
from torchvision import transforms
from torchinfo import summary

from dataset import GrayscaleSegmentationDataset
from model import UNet
from train import (
    train_epoch, validate_epoch, test_model,
    plot_losses
)

if __name__ == "__main__":
    # --- Config ---
    images_dir = 'D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\images'
    masks_dir = 'D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\Masks'      # or None if same folder
    save_path = 'D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT o4 high-10'
    os.makedirs(save_path, exist_ok=True)
    img_size = (256, 256)
    batch_size = 8
    lr = 1e-4
    num_epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Transforms ---
    img_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])

    # --- Dataset ---
    full_ds = GrayscaleSegmentationDataset(images_dir, masks_dir,
                                           transform=img_transform,
                                           mask_transform=mask_transform)
    N = len(full_ds)
    idxs = list(range(N))
    train_idx, temp_idx = train_test_split(idxs, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    print(f"Total: {N}, Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)
    test_ds = Subset(full_ds, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"Loaders -> Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    # --- Model, Optimizer, Loss ---
    model = UNet(n_channels=1, n_classes=1).to(device)
    summary(model, input_size=(batch_size,1,*img_size))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    # --- Training Loop ---
    train_losses, val_losses = [], []
    start_time = time.time()
    for epoch in range(1, num_epochs+1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        tl = train_epoch(model, train_loader, criterion, optimizer, device)
        vl = validate_epoch(model, val_loader, criterion, device, save_path, epoch)
        train_losses.append(tl)
        val_losses.append(vl)
        # save epoch losses to Excel
        import pandas as pd
        pd.DataFrame([train_losses], index=['train']).to_excel(os.path.join(save_path,'train_losses.xlsx'))
        pd.DataFrame([val_losses],   index=['val']).to_excel(os.path.join(save_path,'val_losses.xlsx'))
        print(f"Train Loss: {tl:.4f} | Val Loss: {vl:.4f}")

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time/60:.2f} minutes")

    # --- Save model ---
    torch.save(model, os.path.join(save_path, 'unet_model.pth'))
    torch.save(model.state_dict(), os.path.join(save_path, 'unet_state_dict.pth'))

    # --- Plot losses ---
    plot_losses(train_losses, val_losses, save_path)

    # --- Testing ---
    _ = test_model(model, test_loader, device, save_path)

    # --- Visualization of 5 samples ---
    import matplotlib.pyplot as plt
    import random

    model.eval()
    samples = random.sample(range(len(test_ds)), k=5)
    fig, axes = plt.subplots(5, 3, figsize=(9, 15))
    for i, idx in enumerate(samples):
        img, mask, name = test_ds[idx]
        inp = img.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = torch.sigmoid(model(inp)).cpu().squeeze()
        axes[i,0].imshow(img.squeeze(), cmap='gray');   axes[i,0].set_title('Input')
        axes[i,1].imshow(mask.squeeze(), cmap='gray');  axes[i,1].set_title('GT')
        axes[i,2].imshow((pred>0.5).float(), cmap='gray'); axes[i,2].set_title('Pred')
        for ax in axes[i]:
            ax.axis('off')
        axes[i,0].text(0, -5, name, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'test_visuals.png'))
    plt.close()
