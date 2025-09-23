# main.py
import os
import argparse
import time
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from sklearn.model_selection import train_test_split
from torchinfo import summary
import pandas as pd

from dataset import GrayscaleSegmentationDataset, PairedTransform
from model import UNet
from train import (
    train_epoch, validate_epoch, test_epoch,
    plot_losses, visualize_predictions
)

def paired_transform(resize=(256,256)):
    """
    Returns a function that takes (PIL_img, PIL_mask) and returns
    (tensor_img, tensor_mask).
    """
    tf_img = T.Compose([
        T.Resize(resize),
        T.ToTensor(),            # scales to [0,1]
    ])
    tf_mask = T.Compose([
        T.Resize(resize),
        T.ToTensor(),            # also [0,1]
    ])
    def _transform(img, mask):
        return tf_img(img), tf_mask(mask)
    return _transform

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, required=True,
                        help='path to images folder')
    parser.add_argument('--masks', type=str, default=None,
                        help='path to masks folder (or None to use suffix)')
    parser.add_argument('--suffix', type=str, default='_m',
                        help='mask filename suffix if masks in same folder')
    parser.add_argument('--save_path', type=str, default='./checkpoints',
                        help='where to save models and metrics')
    parser.add_argument('--epochs', type=int, default=2) #20
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--resize', type=int, nargs=2, default=(256,256))
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build full dataset
    full_ds = GrayscaleSegmentationDataset(
        args.images, mask_dir=args.masks,
        suffix=args.suffix, transform=paired_transform(tuple(args.resize))
    )
    n = len(full_ds)
    idxs = list(range(n))
    # 80/10/10 split
    idx_train, idx_tmp = train_test_split(idxs, test_size=0.2, shuffle=True, random_state=42)
    idx_val, idx_test = train_test_split(idx_tmp, test_size=0.5, shuffle=True, random_state=42)
    print(f"Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")

    ds_train = Subset(full_ds, idx_train)
    ds_val   = Subset(full_ds, idx_val)
    ds_test  = Subset(full_ds, idx_test)

    loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    loader_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False, num_workers=4)
    loader_test  = DataLoader(ds_test,  batch_size=1,               shuffle=False, num_workers=4)

    print(f"Train loader batches: {len(loader_train)}, Val: {len(loader_val)}, Test: {len(loader_test)}")

    # Model
    model = UNet(n_channels=1, n_classes=1).to(device)
    summary(model, input_size=(args.batch_size,1,*args.resize))
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total learnable parameters: {total_params}")

    # criterion and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses, val_losses = [], []
    start_time = time.time()
    for epoch in range(1, args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_epoch(model, loader_train, criterion, optimizer, device)
        val_loss   = validate_epoch(model, loader_val, criterion, device, args.save_path, epoch)
        print(f"  Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # save losses to excel each epoch
        pd.DataFrame([list(range(1, epoch+1)), train_losses]) \
          .to_excel(os.path.join(args.save_path, 'train_losses.xlsx'),
                    index=False, header=False)
        pd.DataFrame([list(range(1, epoch+1)), val_losses]) \
          .to_excel(os.path.join(args.save_path, 'val_losses.xlsx'),
                    index=False, header=False)

    # save model
    torch.save(model, os.path.join(args.save_path, 'unet_model_full.pth'))
    torch.save(model.state_dict(), os.path.join(args.save_path, 'unet_model_weights.pth'))

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time/60:.2f} minutes")

    # loss curve
    plot_losses(train_losses, val_losses, args.save_path)

    # testing
    imgs_all, masks_all, preds_all, img_names = test_epoch(model, loader_test, device, args.save_path)
    visualize_predictions(imgs_all, masks_all, preds_all, img_names, args.save_path)

    print("Done.")
