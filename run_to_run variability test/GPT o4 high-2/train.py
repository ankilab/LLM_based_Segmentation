# unet_segmentation/train.py

import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision.transforms import Normalize
from tqdm import tqdm
from dataset import SegmentationDataset
from model import UNet

def dice_coeff(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    inter = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2 * inter + eps) / (union + eps)
    return dice

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    pbar = tqdm(loader, desc="  train", leave=False)
    for imgs, masks, _ in pbar:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        pbar.set_postfix(loss=loss.item())
    return np.mean(losses)

def validate_epoch(model, loader, criterion, device):
    model.eval()
    losses = []
    all_dice = []
    pbar = tqdm(loader, desc="  val  ", leave=False)
    with torch.no_grad():
        for imgs, masks, _ in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            losses.append(loss.item())
            dice = dice_coeff(preds, masks).cpu().numpy()
            all_dice.append(dice)
            pbar.set_postfix(loss=loss.item(), dice=dice.mean())
    return np.mean(losses), all_dice

def test_epoch(model, loader, device):
    model.eval()
    all_dice = []
    filenames = []
    pbar = tqdm(loader, desc="  test ", leave=False)
    with torch.no_grad():
        for imgs, masks, names in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            dice = dice_coeff(preds, masks).cpu().numpy()
            all_dice.append(dice)
            filenames.extend(names)
            pbar.set_postfix(dice=dice.mean())
    return all_dice, filenames

def save_losses_to_excel(losses, path, name):
    # losses: list of floats
    df = pd.DataFrame([losses], columns=list(range(1, len(losses)+1)))
    df.to_excel(os.path.join(path, name), index=False)

def save_dice_to_excel(dice_list, path, name):
    # dice_list: list of arrays (batches)
    df = pd.DataFrame(dice_list,
                      index=list(range(1, len(dice_list)+1)),
                      columns=[f"batch_{i+1}" for i in range(len(dice_list[0]))])
    df.to_excel(os.path.join(path, name))

def plot_losses(train_losses, val_losses, path, name="loss_plot.png"):
    plt.figure()
    epochs = list(range(1, len(train_losses)+1))
    plt.plot(epochs, train_losses, label='train')
    plt.plot(epochs, val_losses, label='val')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(path, name))
    plt.close()

def save_model(model, path, name_prefix="unet"):
    os.makedirs(path, exist_ok=True)
    torch.save(model, os.path.join(path, f"{name_prefix}_full.pth"))
    torch.save(model.state_dict(), os.path.join(path, f"{name_prefix}_state.pth"))

def run_training(image_dir, mask_dir, save_dir,
                 epochs=20, batch_size=8, lr=1e-3,
                 img_size=(256,256), device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    # full dataset
    ds = SegmentationDataset(image_dir, mask_dir, img_size)
    idx = list(range(len(ds)))
    # split
    train_idx, temp_idx = train_test_split(idx, train_size=0.8, random_state=42, shuffle=True)
    val_idx, test_idx = train_test_split(temp_idx, train_size=0.5, random_state=42, shuffle=True)
    print(f"Train/Val/Test sizes: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    # subsets and loaders
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(Subset(ds, val_idx),   batch_size=batch_size)
    test_loader  = DataLoader(Subset(ds, test_idx),  batch_size=batch_size)
    # model, loss, opt
    model = UNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # print model summary
    from torchinfo import summary
    summary(model, input_size=(batch_size,1,*img_size))
    # training loop
    train_losses, val_losses = [], []
    val_dice_all = []
    start = time.time()
    for ep in range(1, epochs+1):
        print(f"Epoch {ep}/{epochs}")
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_dice = validate_epoch(model, val_loader, criterion, device)
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        val_dice_all.append(vl_dice)
    total_time = time.time() - start
    print(f"Total training time: {total_time/60:.2f} minutes")
    # save
    os.makedirs(save_dir, exist_ok=True)
    save_losses_to_excel(train_losses, save_dir, "train_losses.xlsx")
    save_losses_to_excel(val_losses, save_dir,   "val_losses.xlsx")
    save_dice_to_excel(val_dice_all, save_dir,   "validation_dice_scores.xlsx")
    save_model(model, save_dir)
    plot_losses(train_losses, val_losses, save_dir)
    # test
    test_dice, filenames = test_epoch(model, test_loader, device)
    save_dice_to_excel(test_dice, save_dir, "test_dice_scores.xlsx")
    # visualize 5 random
    visualize_predictions(model, test_loader, save_dir)
    return model

def visualize_predictions(model, loader, save_dir, n_samples=5, device='cuda'):
    import random
    import matplotlib.pyplot as plt
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    imgs_list, masks_list, preds_list, names = [], [], [], []
    with torch.no_grad():
        # collect at least n_samples
        for imgs, masks, fnames in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            out = torch.sigmoid(model(imgs))
            preds = (out>0.5).float()
            for i in range(imgs.size(0)):
                imgs_list.append(imgs[i,0].cpu().numpy())
                masks_list.append(masks[i,0].cpu().numpy())
                preds_list.append(preds[i,0].cpu().numpy())
                names.append(fnames[i])
            if len(imgs_list) >= n_samples:
                break
    fig, axes = plt.subplots(n_samples, 3, figsize=(9, 3*n_samples))
    for i in range(n_samples):
        axes[i,0].imshow(imgs_list[i], cmap='gray');   axes[i,0].set_title("Input")
        axes[i,1].imshow(masks_list[i], cmap='gray');  axes[i,1].set_title("GroundTruth")
        axes[i,2].imshow(preds_list[i], cmap='gray');  axes[i,2].set_title("Prediction")
        for j in range(3):
            axes[i,j].set_xticks([]); axes[i,j].set_yticks([])
        fig.suptitle("Examples: Input | GT | Pred", y=0.92)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "test_predictions.png"))
    plt.close()
