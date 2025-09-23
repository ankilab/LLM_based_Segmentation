import os
import time
import torch
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchinfo import summary

from dataset import SegmentationDataset, ToTensorNormalize
from model import UNet
from train import (
    train_epoch, validate_epoch, test_model,
    plot_losses, visualize_predictions,
    save_epoch_losses
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--image_dir',   type=str, required=True)
    p.add_argument('--mask_dir',    type=str, default=None)
    p.add_argument('--mask_suffix', type=str, default='_m')
    p.add_argument('--save_path',   type=str, required=True)
    p.add_argument('--batch_size',  type=int, default=8)
    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--epochs',      type=int,   default=20) 
    p.add_argument('--img_size',    type=int,   nargs=2, default=[256,256])
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset & transforms
    full_ds = SegmentationDataset(
        args.image_dir, args.mask_dir, args.mask_suffix,
        transforms=ToTensorNormalize(tuple(args.img_size))
    )
    n = len(full_ds)
    idxs = list(range(n))
    # split 80/10/10
    idx_train, idx_temp = train_test_split(idxs, test_size=0.2, random_state=42, shuffle=True)
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=42, shuffle=True)

    print(f"Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")

    ds_train = Subset(full_ds, idx_train)
    ds_val   = Subset(full_ds, idx_val)
    ds_test  = Subset(full_ds, idx_test)

    loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    loader_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    loader_test  = DataLoader(ds_test,  batch_size=1,             shuffle=False, num_workers=4, pin_memory=True)

    print(f"DataLoaders -> train: {len(loader_train)}, val: {len(loader_val)}, test: {len(loader_test)}")

    # Model
    model = UNet().to(device)
    summary(model, input_size=(1,1,*args.img_size))
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_losses, val_losses = [], []
    start_time = time.time()

    for epoch in range(1, args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tl = train_epoch(model, loader_train, criterion, optimizer, device)
        vl, _ = validate_epoch(model, loader_val, criterion, device, args.save_path, epoch)
        train_losses.append(tl)
        val_losses.append(vl)
        print(f"  Train Loss: {tl:.4f} | Val Loss: {vl:.4f}")

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time/60:.2f} minutes")

    # Save losses as Excel
    save_epoch_losses(train_losses, os.path.join(args.save_path, 'train_losses.xlsx'))
    save_epoch_losses(val_losses,   os.path.join(args.save_path, 'val_losses.xlsx'))

    # Save model
    torch.save(model, os.path.join(args.save_path, 'unet_model.pth'))
    torch.save(model.state_dict(), os.path.join(args.save_path, 'unet_state_dict.pth'))

    # Plot train/val loss curve
    plot_losses(train_losses, val_losses, args.save_path)

    # Test & visualize
    results = test_model(model, loader_test, criterion, device, args.save_path)
    visualize_predictions(results, args.save_path, num_samples=5)

    print("All done. Outputs saved in:", args.save_path)
