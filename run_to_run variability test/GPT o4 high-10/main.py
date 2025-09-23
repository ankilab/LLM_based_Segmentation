# unet_segmentation/main.py
import os
import time
import torch
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torch import optim
from torchinfo import summary

from dataset import SegmentationDataset
from model import UNet
from train import (train_epoch, validate_epoch, test_epoch,
                   plot_losses, visualize_predictions)

def save_losses_excel(losses, filename):
    # losses: list of floats
    df = pd.DataFrame([list(range(1, len(losses)+1)), losses])
    df.to_excel(filename, index=False, header=False)

if __name__ == "__main__":
    # paths & hyperparams
    data_dir  = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\images"    # <-- change
    masks_dir = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\Masks"              # same dir or separate
    SAVE_PATH = "D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT o4 high-10"
    os.makedirs(SAVE_PATH, exist_ok=True)

    lr          = 1e-4
    batch_size  = 8
    num_epochs  = 50
    img_size    = (256,256)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    full_ds = SegmentationDataset(data_dir, masks_dir, mask_suffix="_m", img_size=img_size)
    N = len(full_ds)
    idxs = list(range(N))
    # 80/10/10 split
    idx_train, idx_temp = train_test_split(idxs, test_size=0.2, random_state=42, shuffle=True)
    idx_val, idx_test    = train_test_split(idx_temp, test_size=0.5, random_state=42, shuffle=True)
    print(f"Train/Val/Test sizes: {len(idx_train)}/{len(idx_val)}/{len(idx_test)}")

    ds_train = Subset(full_ds, idx_train)
    ds_val   = Subset(full_ds, idx_val)
    ds_test  = Subset(full_ds, idx_test)
    print("Subsets created.")

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=4)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=4)
    dl_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=4)
    print("DataLoaders ready.")

    # model
    model = UNet(in_channels=1, out_channels=1).to(device)
    print(summary(model, input_size=(batch_size,1,*img_size)))
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total learnable parameters: {total_params}")

    # criterion & optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # training loop
    train_losses, val_losses, all_val_dice = [], [], []
    start_t = time.time()
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
        tr_loss = train_epoch(model, dl_train, criterion, optimizer, device)
        train_losses.append(tr_loss)
        save_losses_excel(train_losses, os.path.join(SAVE_PATH, "train_losses.xlsx"))

        val_loss = validate_epoch(model, dl_val, criterion, device,
                                  SAVE_PATH, epoch, all_val_dice)
        val_losses.append(val_loss)
        save_losses_excel(val_losses, os.path.join(SAVE_PATH, "val_losses.xlsx"))

        print(f"Epoch {epoch+1} ▶ Train Loss: {tr_loss:.4f}, Val Loss: {val_loss:.4f}")

    total_time = time.time() - start_t
    print(f"\nTraining completed in {total_time/60:.2f} minutes")

    # save model
    torch.save(model, os.path.join(SAVE_PATH, "unet_model.pth"))
    torch.save(model.state_dict(), os.path.join(SAVE_PATH, "unet_model_state.pth"))

    # final plots
    plot_losses(train_losses, val_losses, SAVE_PATH)

    # testing
    all_preds, img_names = test_epoch(model, dl_test, device, SAVE_PATH)
    visualize_predictions(all_preds, ds_test.dataset, img_names, SAVE_PATH)
