# unet_segmentation/main.py

import os
import argparse
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import pandas as pd
from torch import optim, nn
from torchinfo import summary
from dataset import SegmentationDataset
from model import UNet
from train import (train_epoch, validate_epoch, test_model,
                   plot_losses)

def save_loss_excel(losses, filename):
    # first row epochs, second row losses
    df = pd.DataFrame([list(range(1, len(losses)+1)), losses])
    df.to_excel(filename, header=False, index=False)

def visualize_samples(samples, save_path):
    import matplotlib.pyplot as plt
    import random
    chosen = random.sample(samples, 5)
    fig, axes = plt.subplots(5, 3, figsize=(9, 15))
    col_titles = ['Input Image', 'Ground Truth', 'Prediction']
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title)
    for i, (img, mask, pred, name) in enumerate(chosen):
        img = img.squeeze(0).numpy()
        mask = mask.squeeze(0).numpy()
        pred = (pred.squeeze(0).numpy() > 0.5).astype(float)
        for j, data in enumerate([img, mask, pred]):
            ax = axes[i, j]
            ax.imshow(data, cmap='gray')
            ax.axis('off')
            if j == 0:
                ax.set_ylabel(name, rotation=0, labelpad=50, va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'test_samples.png'))
    plt.close()

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset
    ds = SegmentationDataset(args.images_dir, args.masks_dir, mask_suffix='_m', img_size=(args.img_size, args.img_size))
    n = len(ds)
    idx = list(range(n))
    # splits
    train_idx, temp_idx = train_test_split(idx, test_size=0.2, random_state=42, shuffle=True)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, shuffle=True)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    train_ds = Subset(ds, train_idx)
    val_ds   = Subset(ds, val_idx)
    test_ds  = Subset(ds, test_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Loaders ➤ Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

    # model
    model = UNet(n_channels=1, n_classes=1).to(device)
    summary(model, input_size=(args.batch_size, 1, args.img_size, args.img_size))
    print(f"Total params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # optimizer, criterion
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    # training loop
    train_losses, val_losses = [], []
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = validate_epoch(model, val_loader, criterion, device, args.save_path, epoch)
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} min")

    # save losses
    os.makedirs(args.save_path, exist_ok=True)
    save_loss_excel(train_losses, os.path.join(args.save_path, 'train_losses.xlsx'))
    save_loss_excel(val_losses,   os.path.join(args.save_path, 'val_losses.xlsx'))
    # save model
    torch.save(model, os.path.join(args.save_path, 'unet_full.pth'))
    torch.save(model.state_dict(), os.path.join(args.save_path, 'unet_state.pth'))
    # plot curves
    plot_losses(train_losses, val_losses, args.save_path)

    # testing
    samples = test_model(model, test_loader, device, args.save_path)
    visualize_samples(samples, args.save_path)


if __name__ == "__main__":
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, required=True, help='path to images folder')
    parser.add_argument('--masks_dir',  type=str, default=None, help='path to masks folder (if different)')
    parser.add_argument('--save_path',  type=str, required=True, help='where to save outputs')
    parser.add_argument('--img_size',   type=int, default=256, help='resize H and W')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--epochs',     type=int, default=50)
    args = parser.parse_args()
    main(args)
