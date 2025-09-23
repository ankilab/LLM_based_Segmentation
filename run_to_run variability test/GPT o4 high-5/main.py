import os
import time
import torch
from torch.utils.data import Subset, DataLoader
from torchinfo import summary
from sklearn.model_selection import train_test_split
import numpy as np

from dataset import GrayscaleSegDataset
from model import UNet
from train import (
    train_one_epoch, validate, test,
    save_losses_excel, save_dice_excel,
    plot_losses, visualize_predictions
)

def collate_fn(batch):
    imgs, masks = zip(*batch)
    return torch.stack(imgs), torch.stack(masks)

if __name__ == "__main__":
    # --- Configuration ---
    image_dir = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\images"
    mask_dir = "D:\qy44lyfe\LLM segmentation\Data sets\Brain Meningioma\Masks"         # or same as image_dir
    save_path = "D:\qy44lyfe\LLM segmentation\Results\Models Comparison\Models run to run comparison\GPT o4 high-5"
    os.makedirs(save_path, exist_ok=True)

    # hyperparameters
    num_epochs = 20
    lr = 1e-3
    batch_size = 8
    img_size = (256, 256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Dataset & Split ---
    # transform function to resize and to tensor
    from torchvision.transforms import functional as TF
    # transform function to resize and to tensor
    def transform(img, mask):
        # Resize
        img = TF.resize(img, img_size)
        mask = TF.resize(mask, img_size, interpolation=TF.InterpolationMode.NEAREST)

        # To tensor
        img = TF.to_tensor(img)  # 1xHxW float
        arr = np.array(mask)  # HxW, values 0/255
        mask_tensor = torch.from_numpy((arr > 0).astype(np.float32))  # HxW float
        mask_tensor = mask_tensor.unsqueeze(0)  # 1xHxW

        return img, mask_tensor

    full_ds = GrayscaleSegDataset(image_dir, mask_dir, suffix="_m", transform=transform)
    n = len(full_ds)
    idxs = list(range(n))
    idx_train, idx_temp = train_test_split(idxs, test_size=0.2, random_state=42, shuffle=True)
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=42, shuffle=True)

    print(f"Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")

    ds_train = Subset(full_ds, idx_train)
    ds_val   = Subset(full_ds, idx_val)
    ds_test  = Subset(full_ds, idx_test)

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    loader_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    loader_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print("Dataloaders:", len(loader_train), len(loader_val), len(loader_test))

    # --- Model ---
    model = UNet(n_channels=1, n_classes=1).to(device)
    print(summary(model, input_size=(batch_size,1,*img_size)))
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total learnable parameters: {total_params}")

    # --- Training setup ---
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    all_val_dice = []

    start_time = time.time()
    for epoch in range(1, num_epochs+1):
        print(f"Epoch {epoch}/{num_epochs}")
        tl = train_one_epoch(model, loader_train, criterion, optimizer, device)
        vl, dice_batches = validate(model, loader_val, criterion, device)

        train_losses.append(tl)
        val_losses.append(vl)
        all_val_dice.append([float(d) for batch in dice_batches for d in batch])

        # save intermediate losses
        save_losses_excel(train_losses, save_path, "train_losses.xlsx")
        save_losses_excel(val_losses,   save_path, "val_losses.xlsx")
        save_dice_excel(all_val_dice,   save_path, "validation_dice_scores.xlsx")

        print(f"  Train Loss: {tl:.4f} | Val Loss: {vl:.4f}")

    # save model
    torch.save(model, os.path.join(save_path, "unet_model_full.pt"))
    torch.save(model.state_dict(), os.path.join(save_path, "unet_model_state_dict.pth"))

    total_time = time.time() - start_time
    print(f"Total training time: {total_time/60:.2f} minutes")

    # plot losses
    plot_losses(train_losses, val_losses, save_path)

    # --- Testing ---
    test_dice = test(model, loader_test, device)
    save_dice_excel([[float(d) for d in batch] for batch in test_dice], save_path, "test_dice_scores.xlsx")

    # visualize some predictions
    visualize_predictions(model, full_ds, device, save_path, n_samples=5)
