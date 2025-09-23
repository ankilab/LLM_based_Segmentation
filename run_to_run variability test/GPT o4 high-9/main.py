# unet_segmentation/main.py

import os
import time
import torch
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torch import optim, nn
from torchinfo import summary

from dataset import SegmentationDataset
from model import UNet
from train import train_epoch, validate_epoch, test_model, plot_losses

def main(args):
    # prepare save directory
    os.makedirs(args.save_path, exist_ok=True)

    # build dataset
    dataset = SegmentationDataset(
        images_dir=args.images_dir,
        masks_dir=(args.masks_dir or None),
        mask_suffix=args.mask_suffix,
        img_size=(args.img_size, args.img_size)
    )
    N = len(dataset)
    indices = list(range(N))

    # split 80/10/10
    train_idx, temp_idx = train_test_split(indices,
                                           train_size=0.8,
                                           random_state=42,
                                           shuffle=True)
    val_idx, test_idx = train_test_split(temp_idx,
                                        test_size=0.5,
                                        random_state=42,
                                        shuffle=True)

    print(f"Total samples: {N}")
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # subsets and loaders
    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)
    test_ds  = Subset(dataset, test_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    print("Loader sizes:",
          len(train_loader.dataset),
          len(val_loader.dataset),
          len(test_loader.dataset))

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model
    model = UNet(n_channels=1, n_classes=1).to(device)
    # print summary
    summary(model, input_size=(args.batch_size, 1, args.img_size, args.img_size))
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total learnable parameters: {total_params}")

    # criterion & optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # training loop
    train_losses, val_losses = [], []
    start_time = time.time()
    for epoch in range(args.epochs):
        tl = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        vl = validate_epoch(model, val_loader, criterion, device, epoch, args.save_path)
        train_losses.append(tl)
        val_losses.append(vl)

    # end timer
    elapsed = time.time() - start_time
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    print(f"Training completed in {int(h)}h {int(m)}m {int(s)}s")

    # save losses to Excel
    df_train = pd.DataFrame([train_losses],
                            columns=[str(i+1) for i in range(len(train_losses))])
    df_train.to_excel(os.path.join(args.save_path, "train_losses.xlsx"),
                      index=False, header=True)

    df_val = pd.DataFrame([val_losses],
                          columns=[str(i+1) for i in range(len(val_losses))])
    df_val.to_excel(os.path.join(args.save_path, "val_losses.xlsx"),
                    index=False, header=True)

    # save model
    torch.save(model, os.path.join(args.save_path, "unet_full.pth"))
    torch.save(model.state_dict(), os.path.join(args.save_path, "unet_state.pth"))

    # plot losses
    plot_losses(train_losses, val_losses, args.save_path)

    # testing
    test_model(model, test_loader, device, args.save_path)

    # visualize 5 random predictions
    from random import sample
    import matplotlib.pyplot as plt

    sample_idxs = sample(test_idx, min(5, len(test_idx)))
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(9, 15))
    for row, idx in enumerate(sample_idxs):
        img, mask, name = dataset[idx]
        img_batch = img.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img_batch)
        pred_bin = (pred > 0.5).float().cpu().squeeze(0).squeeze(0)
        axes[row, 0].imshow(img.squeeze(0), cmap="gray")
        axes[row, 1].imshow(mask.squeeze(0), cmap="gray")
        axes[row, 2].imshow(pred_bin, cmap="gray")
        axes[row, 0].set_title("Input")
        axes[row, 1].set_title("GT")
        axes[row, 2].set_title("Pred")
        for col in range(3):
            axes[row, col].set_xlabel(name if col==0 else "")
            axes[row, col].axis("off")

    plt.tight_layout()
    fig.savefig(os.path.join(args.save_path, "test_predictions.png"), dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True,
                        help="folder with input grayscale PNGs")
    parser.add_argument("--masks_dir", type=str, default=None,
                        help="folder with mask PNGs (if None, use same folder with suffix)")
    parser.add_argument("--mask_suffix", type=str, default="_m.jpg",
                        help="suffix for mask filenames")
    parser.add_argument("--img_size", type=int, default=256,
                        help="resize images/masks to this size")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--save_path", type=str, default="./results",
                        help="where to save models, logs, plots, excels")
    args = parser.parse_args()
    main(args)
