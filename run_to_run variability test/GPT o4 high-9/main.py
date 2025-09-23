# main.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torchinfo import summary
from dataset import GrayscaleSegmentationDataset
from model import UNet
from train import (train_epoch, validate_epoch, test_epoch,
                   plot_losses, plot_predictions)

if __name__ == "__main__":
    # --- User configs ---
    images_dir = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\images"
    masks_dir  = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\Masks"      # or set to None if same folder
    save_path  = "D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT o4 high-9"
    os.makedirs(save_path, exist_ok=True)

    # Hyperparameters
    epochs     = 20
    lr         = 1e-3
    batch_size = 8
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size   = (256, 256)

    # Transforms
    def paired_transform(pair):
        img, mask = pair
        img = img.resize(img_size)
        mask = mask.resize(img_size)
        return img, mask

    # Full dataset
    full_ds = GrayscaleSegmentationDataset(images_dir, masks_dir, transform=paired_transform)
    idxs = list(range(len(full_ds)))
    train_idxs, tmp_idxs = train_test_split(idxs, test_size=0.2, random_state=42)
    val_idxs, test_idxs = train_test_split(tmp_idxs, test_size=0.5, random_state=42)
    print(f"Sizes -> Train: {len(train_idxs)}, Val: {len(val_idxs)}, Test: {len(test_idxs)}")

    train_ds = Subset(full_ds, train_idxs)
    val_ds   = Subset(full_ds, val_idxs)
    test_ds  = Subset(full_ds, test_idxs)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    print(f"DataLoaders -> Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

    # Model, loss, optimizer
    model = UNet(n_channels=1, n_classes=1, bilinear=True)
    print(summary(model, input_size=(8, 1, 256, 256)))
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    train_losses, val_losses = [], []
    start_time = time.time()
    for epoch in range(1, epochs+1):
        print(f"\nEpoch {epoch}/{epochs}")
        t_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss = validate_epoch(model, val_loader, criterion, device, save_path, epoch)
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        # save per-epoch losses
        pd.DataFrame({"epoch":[epoch], "train_loss":[t_loss]}).to_excel(f"{save_path}/train_losses.xlsx",
                                                                      index=False, header=False, startrow=0, startcol=0, engine='openpyxl', mode='a')
        pd.DataFrame({"epoch":[epoch], "val_loss":[v_loss]}).to_excel(f"{save_path}/val_losses.xlsx",
                                                                      index=False, header=False, startrow=0, startcol=0, engine='openpyxl', mode='a')
        print(f"  Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time/60:.2f} minutes")

    # Save model
    torch.save(model,        f"{save_path}/unet_model_full.pth")
    torch.save(model.state_dict(), f"{save_path}/unet_model_state.pth")

    # Loss curves
    plot_losses(train_losses, val_losses, save_path)

    # Testing
    test_dices = test_epoch(model, test_loader, device, save_path)
    print(f"Mean test Dice: {sum(test_dices)/len(test_dices):.4f}")

    # Plot predictions
    plot_predictions(model, full_ds, device, save_path, n_samples=5)
